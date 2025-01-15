# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import gc
import inspect
import logging
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

import numpy as np
import torch
from fairseq2.gang import Gang
from numpy.random import RandomState

from lcm.datasets.base import DataLoader
from lcm.datasets.configs import EvaluationDataLoadingConfig, JSONDatasetConfig
from lcm.evaluation.api import (
    PREDICTION_TEXT_COLUMN,
    PREDICTION_TOKEN_COLUMN,
    PREDICTION_TOKEN_ID_COLUMN,
    EOSConfig,
    Example,
    ExampleFn,
    MetricFn,
    Prediction,
    Predictor,
    Prompt,
    Prompts,
    ScorerConfig,
    Task,
    TaskConfig,
    TaskResult,
)
from lcm.evaluation.metrics import get_scorer
from lcm.evaluation.utils.common import (
    ExampleSelector,
    parse_omega_list,
    run_metrics,
    text_index,
)
from lcm.evaluation.utils.data_utils import JSONTestDataLoader
from lcm.evaluation.utils.distributed import (
    clear_cuda,
    get_dp_group,
    get_mp_rank,
    mean_reduce_dict,
    rank_zero_print,
)
from lcm.utils.common import batched

logger = logging.getLogger(__name__)


@dataclass
class FewShotTaskConfig(TaskConfig):
    num_few_shot: int = 0
    few_shot_examples: Optional[List[Example]] = None
    few_shot_file: Optional[str] = None
    few_shot_strategy: Literal["first", "index", "random"] = "first"
    few_shot_indices: Optional[Sequence[int]] = None


@dataclass
class GenerationTaskConfig(FewShotTaskConfig):
    max_gen_len: int = 256
    """Maximum length of generated content (tokens, sentences, etc.)"""

    max_gen_len_ratio: Optional[float] = None
    """
    Maximum length of generated content as the ratio on the input length.
    This parameter takes precedence over `max_gen_len`
    """

    max_added_tokens: int = 100
    """Maximum number of tokens ADDED to the prompt. This parameter is
    useful for the models that repeats """

    min_gen_len: int = 1
    """Minimum length of generated content (tokens, sentences, etc.)"""

    max_prompt_len: int = 1024
    """Maxium input prompt length. Everything exceeding this will be trimmed"""

    num_generations: int = 1
    """Number of generations for each """

    prompt_func: Optional[Callable[[Any], Prompts]] = None
    preprocess_fn: Optional[ExampleFn] = None
    postprocess_fn: Optional[ExampleFn] = None
    metrics_to_report: Any = None
    metric_fns: Optional[Sequence[MetricFn]] = None
    model_based_metrics: Optional[List[ScorerConfig]] = None
    eos_config: Optional[EOSConfig] = None


def load_few_shots(
    config: FewShotTaskConfig, data_loader: Optional[DataLoader] = None
) -> Optional[ExampleSelector]:
    few_shot_examples = config.few_shot_examples
    if config.num_few_shot > 0:
        if few_shot_examples is None:
            assert config.few_shot_file, (
                f"Expect non-empty few_shot_file when few_shot = {config.num_few_shot}"
            )
            assert data_loader, "Expect non-empty data loader"
            assert isinstance(data_loader.data_config, EvaluationDataLoadingConfig), (
                f"unexpected data loading type: {type(data_loader.data_config)}"
            )
            few_shot_data_loader = JSONTestDataLoader(
                data_config=data_loader.data_config,
                dataset=JSONDatasetConfig(file_path=config.few_shot_file),
                gang=data_loader.gang,
            )
            few_shot_examples = [
                x for b in few_shot_data_loader.iterate_batches() for x in b
            ]

        return ExampleSelector(
            examples=few_shot_examples,
            num_examples=config.num_few_shot,
            select_strategy=config.few_shot_strategy,
            select_indices=config.few_shot_indices,
        )
    else:
        return None


def parse_prompt_func(
    prompt_func: Optional[Callable[..., Prompts]], config: TaskConfig
):
    if prompt_func:
        params = inspect.signature(prompt_func).parameters
        if "config" in params:
            return partial(prompt_func, config=config)  # type: ignore
        else:
            return prompt_func
    else:
        return lambda x: x


def maybe_chain(*funcs: Optional[Callable]) -> Optional[Callable]:
    real_funcs = [func for func in funcs if func]
    if real_funcs:

        def chained_func(x):
            for func in real_funcs:
                x = func(x)
            return x

        return chained_func
    return None


class GenerationTask(Task):
    def __init__(
        self,
        config: GenerationTaskConfig,
        data_loader: DataLoader,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        self.config = config
        self.data_loader = data_loader
        self.few_shot_selector = load_few_shots(config, data_loader=data_loader)
        self.prompt_fn = parse_prompt_func(config.prompt_func, config)
        self.metrics_to_report = parse_omega_list(self.config.metrics_to_report)

        self.logger = logger
        self.gang: Optional[Gang] = kwargs.get("gang", self.data_loader.gang)

    def run(  # type: ignore[override]
        self,
        predictor: Predictor,
        random_state: Optional[RandomState] = None,
        show_progress: bool = False,
        temperature: float = 0.0,
        disable_cache: bool = False,
        top_p: float = 0.0,
        top_k: int = 0,
        seed: int = 42,
        **kwargs,
    ) -> TaskResult:
        # Set up eval workers
        # In case we train on Ampere or later, use TF32.
        torch.set_float32_matmul_precision("high")

        dataset = self.data_loader.iterate_batches()
        raw_results: List[Example] = []
        accumulated_result: Dict[str, List[float]] = defaultdict(list)
        accumulated_indices: List[Sequence] = []
        accumulated_preds: List[Prediction] = []

        # Some prompt functions need to access the predictor info, for example,
        # its encoder for prompt the sonar embedding
        params = inspect.signature(self.prompt_fn).parameters
        if "predictor" in params:
            prompt_fn = partial(self.prompt_fn, predictor=predictor)
        else:
            prompt_fn = self.prompt_fn  # type: ignore[assignment]

        for batch in dataset:
            # Gotcha: LCM predictor does not work with preprocess_fn yet, as it
            # expects batched input
            for x in batch:
                if self.config.preprocess_fn:
                    x.update(self.config.preprocess_fn(x))
                if self.config.num_few_shot > 0:
                    assert self.few_shot_selector, "Cannot construct few shot selector"
                    x["few_shot"] = self.few_shot_selector(random_state=random_state)

            prompts = prompt_fn(batch)

            # Assume:
            # For n prompts and k generations, Predictions is a list of nxk items, where
            # n-subseuence has k items, corresponding to generations of one prompt
            n, k = len(prompts), self.config.num_generations
            indices = [list(range(i * k, (i + 1) * k)) for i in range(n)]
            predictions = predictor(
                prompts=prompts,
                max_prompt_len=self.config.max_prompt_len,
                min_gen_len=self.config.min_gen_len,
                max_gen_len=self.config.max_gen_len,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                disable_cache=disable_cache,
                show_progress=show_progress,
                return_logprobs=False,
                echo=True,
                seed=seed,
                num_generations=self.config.num_generations,
                max_gen_len_ratio=self.config.max_gen_len_ratio,
            )

            # Calculate metrics on MP rank 0 only
            if get_mp_rank() != 0:
                continue
            postprocess_fn = getattr(predictor, "post_process", None)

            batch_results = self.compute_metrics(
                predictions=predictions,
                hypothesis_indices=indices,
                input=batch,
                postprocess_fn=postprocess_fn,
                show_progress=show_progress,
            )
            raw_results.extend(batch)
            accumulated_indices.extend(indices)
            accumulated_preds.extend(predictions)

            for name, values in batch_results["metrics"].items():
                accumulated_result[name].extend(values)

        del predictor
        clear_cuda()

        extra_results = self.compute_model_based_metrics(
            predictions=accumulated_preds,
            hypothesis_indices=accumulated_indices,  # type: ignore
            input=raw_results,
            show_progress=show_progress,
        )
        accumulated_result.update(extra_results)

        # Calculate metrics on MP rank 0 only
        if get_mp_rank() != 0:
            return TaskResult(metrics={})
        avg_results = mean_reduce_dict(accumulated_result, group=get_dp_group())
        return TaskResult(metrics=avg_results, raw_results=raw_results)

    def compute_metrics(  # type: ignore
        self,
        predictions: Sequence[Prediction],
        hypothesis_indices: List[List[int]],
        input: Sequence[Example],
        postprocess_fn: Optional[Callable] = None,
        show_progress: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        num_gens = self.config.num_generations or 1
        assert len(predictions) == len(input) * num_gens, (
            f"For {self.config.num_generations} generations and {len(input)} inputs, "
            f"expect {len(input) * self.config.num_generations} predictions, "
            f"got length={len(predictions)}"
        )
        assert len(hypothesis_indices) == len(input), f"number of hypotheses {len(hypothesis_indices)} != input length {len(input)}"  # fmt: skip
        postprocess_fn = maybe_chain(self.config.postprocess_fn, postprocess_fn)
        batch_results: Dict[str, List[float]] = defaultdict(list)
        for idx, x in enumerate(input):
            preds: List[Prediction] = [predictions[i] for i in hypothesis_indices[idx]]
            x[PREDICTION_TEXT_COLUMN] = [p.text for p in preds]
            # if preds[0].embed is not None:
            #     x[PREDICTION_EMBED_COLUMN] = [p.embed for p in preds]
            if preds[0].token_ids is not None:
                x[PREDICTION_TOKEN_ID_COLUMN] = [p.token_ids for p in preds]
            if preds[0].tokens is not None:
                x[PREDICTION_TOKEN_COLUMN] = [p.tokens for p in preds]
            x.update(postprocess_fn(x) if postprocess_fn else {})
            if self.config.metric_fns:
                run_metrics(x, self.config.metric_fns, self.metrics_to_report)
                for name, value in x["metrics"].items():
                    batch_results[name].append(value)
            if show_progress:
                msg = ""
                if "_source_text_column" in x:
                    prompt = x["_source_text_column"]
                    if len(prompt) > 16:
                        prompt = prompt[:8] + ["....."] + prompt[-8:]
                    msg += f"Prompts: {prompt}"
                if "targets" in x:
                    msg += f"\nTargets: {x['targets']}"
                if "prediction" in x:
                    msg += f"\nPrediction: {x['prediction']}"
                if "metrics" in x:
                    msg += f"\nScores: {x['metrics']}"
                rank_zero_print(msg)

        return {"metrics": batch_results}

    def compute_model_based_metrics(
        self,
        predictions: Sequence[Prediction],
        hypothesis_indices: List[List[int]],
        input: Sequence[Example],
        show_progress: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        if self.config.model_based_metrics is None:
            return {}

        # Clean CUDA to save memory
        for x in input:
            for k in x.keys():
                if isinstance(x[k], torch.Tensor):
                    x[k] = x[k].cpu()
        clear_cuda()

        result: Dict[str, List[float]] = defaultdict(list)
        for metric_cfg in self.config.model_based_metrics:
            if (
                self.config.metrics_to_report
                and metric_cfg.scorer_type not in self.config.metrics_to_report
            ):
                continue
            metric_fn = get_scorer(
                metric_cfg, self.config.metrics_to_report, gang=self.gang
            )
            if metric_fn is None:  # metrics should not be reported
                continue

            # Different metric models require different memories and thus have
            # different batch sizes
            # batch_size = len(input)
            batch_size = 10
            if metric_cfg.params and "batch_size" in metric_cfg.params:
                batch_size = metric_cfg.params["batch_size"]

            for batch_examples in batched(input, batch_size=batch_size):
                batch_results = metric_fn(batch_examples, show_progress=show_progress)
                for name, values in batch_results.items():
                    result[name].extend(list(values))

                    # update per-example metrics
                    assert len(values) == len(batch_examples)
                    for example, value in zip(batch_examples, values):
                        if "metrics" not in example:
                            example["metrics"] = {}
                        example["metrics"][name] = value
            del metric_fn
            gc.collect()
            clear_cuda()
        return result


def nll_accuracy(x: Dict[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for name, nlls in x["nlls"].items():
        pred = int(np.argmin(nlls))
        metrics[f"acc_{name}"] = float(pred == x["target"])
        metrics[f"nll_{name}"] = nlls[pred]
        metrics[f"nll_{name}_target"] = nlls[x["target"]]
        # negative log-normalized probability of target -log(P[correct]/(sum_{choice} P[choice]))
        norm_factor = torch.logsumexp(-torch.tensor(nlls), dim=0).item()
        metrics[f"nll_{name}_target_norm"] = nlls[x["target"]] + norm_factor
    return metrics


@dataclass
class ChoiceTaskConfig(FewShotTaskConfig):
    prompt_func: Optional[Callable[..., str]] = None
    max_prompt_len: int = 1024
    max_gen_len: int = 0
    nll_completion: bool = False

    preprocess_fn: Optional[ExampleFn] = None
    postprocess_fn: Optional[ExampleFn] = None
    metric_fns: Optional[Sequence[MetricFn]] = (nll_accuracy,)


class ChoiceTask(Task):
    def __init__(
        self,
        config: ChoiceTaskConfig,
        data_loader: DataLoader,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ) -> None:
        self.config = config
        self.data_loader = data_loader
        self.few_shot_selector = load_few_shots(config, data_loader=data_loader)
        self.prompt_fn = parse_prompt_func(config.prompt_func, config)
        self.logger = logger
        self.gang: Optional[Gang] = kwargs.get("gang", self.data_loader.gang)
        self.metric_fns: Sequence[MetricFn] = config.metric_fns or []

    def run(  # type: ignore[override]
        self,
        predictor: Predictor,
        random_state: Optional[RandomState] = None,
        show_progress: bool = False,
        temperature: float = 1.0,
        disable_cache: bool = False,
        top_p: float = 0.6,
        top_k: int = 0,
        **kwargs: Any,
    ) -> TaskResult:
        # Set up eval workers
        # In case we train on Ampere or later, use TF32.
        torch.set_float32_matmul_precision("high")

        dataset = self.data_loader.iterate_batches()
        raw_results: List[Dict[str, Any]] = []
        accumulated_result: Dict[str, List[float]] = defaultdict(list)

        for batch in dataset:
            prompts: List[Prompt] = []
            indices: List[Sequence[int]] = []

            for x in batch:
                if self.config.preprocess_fn:
                    x.update(self.config.preprocess_fn(x))
                assert all(key in x for key in ("choice_texts", "target"))
                if self.config.num_few_shot > 0:
                    assert self.few_shot_selector, "Cannot construct few shot selector"
                    x["few_shot"] = self.few_shot_selector(random_state=random_state)
                x["prompts"] = [
                    self.prompt_fn(**x, choice_text=c) for c in x["choice_texts"]
                ]  # type: ignore
                x["completions"] = [f"Answer: {c}" for c in x["choice_texts"]]

            prev_index = len(prompts)
            prompts += x["prompts"] + (
                x["completions"] if self.config.nll_completion else []
            )
            indices.append(range(prev_index, len(prompts)))

            # TODO: Optimize the text embedding process of prompts, some thing like
            # `prompts = convert_prompts_lcm(prompts, self)`
            # So based on the task information and dataset, the system can look up
            # the cached sonarized dataset and return an LCMInput ?

            predictions: Sequence[Prediction] = predictor(
                prompts=prompts,
                max_prompt_len=self.config.max_prompt_len,
                max_gen_len=self.config.max_gen_len,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                echo=True,
                disable_cache=disable_cache,
                return_logprobs=True,
            )

            # Calculate metrics on MP rank 0 only
            if get_mp_rank() != 0:
                continue

            for idx, x in enumerate(batch):
                if self.config.postprocess_fn:
                    x.update(self.config.postprocess_fn(x))
                preds = [predictions[i] for i in indices[idx]]

                x["nlls"] = defaultdict(list)
                for cix, (text, pred) in enumerate(zip(x["choice_texts"], preds)):
                    assert pred.tokens and pred.logprobs and pred.text_offsets, pred
                    assert isinstance(pred.text, str), (
                        "multiple texts output is not supported in LLM predictor"
                    )
                    logprobs = pred.logprobs[
                        text_index(pred.text, pred.text_offsets, text)
                    ]
                    x["nlls"]["char"].append(-sum(logprobs) / len(text))
                    x["nlls"]["token"].append(-sum(logprobs) / len(logprobs))
                    x["nlls"]["raw"].append(-sum(logprobs))

                    if self.config.nll_completion:
                        assert len(preds) == 2 * len(x["choice_texts"])
                        compl = preds[cix + len(x["choice_texts"])]
                        assert compl.tokens and compl.logprobs and compl.text_offsets
                        assert isinstance(compl.text, str), (
                            "multiple texts output is not supported in LLM predictor"
                        )
                        slice = text_index(compl.text, compl.text_offsets, text)
                        nll_compl = -sum(logprobs) + sum(compl.logprobs[slice])
                        x["nlls"]["completion"].append(nll_compl)

                    x["metrics"] = {
                        k: v for fn in self.metric_fns for k, v in fn(x).items()
                    }
                    raw_results.append(x)

                    for name, values in x["metrics"].items():
                        accumulated_result[name].append(values)

        # Calculate metrics on MP rank 0 only
        if get_mp_rank() != 0:
            return TaskResult(metrics={})

        avg_results = mean_reduce_dict(accumulated_result, group=get_dp_group())
        return TaskResult(metrics=avg_results, raw_results=raw_results)
