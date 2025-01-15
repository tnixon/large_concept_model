# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import json
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple, Union

import dacite
import torch
from fairseq2.gang import FakeGang
from omegaconf import DictConfig, OmegaConf
from typing_extensions import Self

from lcm.datasets.configs import DatasetConfig, EvaluationDataLoadingConfig
from lcm.evaluation.api import AverageMetric, EOSConfig, PredictorConfig
from lcm.evaluation.predictors import build_predictor, get_config_cls
from lcm.evaluation.tasks import TaskRegistry, build_task
from lcm.evaluation.utils.common import (
    get_random_state,
    log_raw_results,
    set_torch_variables,
    setup_env,
)
from lcm.evaluation.utils.data_utils import load_jsonl
from lcm.evaluation.utils.distributed import get_mp_rank, rank_zero_info
from lcm.utils.common import promote_config

if TYPE_CHECKING:
    from logging import Logger

    from fairseq2.gang import Gang


@dataclass
class RunConfig:
    """
    Config values to construct config for a single task run
    """

    name: str
    """unique name of a task run"""

    task_name: str
    """Registered task name"""

    dump_dir: str
    """Where to store the output"""

    predictor: str
    """predictor as registered in lcm.evaluation.predictors"""

    params: Optional[Dict[str, Any]] = None
    """Task parameteres"""

    data_loading: Optional[EvaluationDataLoadingConfig] = None
    """The dataloading config"""

    dataset: Optional[DatasetConfig] = None
    """The dataset config"""

    dtype: str = "torch.float16"
    """dtype to load the model and data"""

    predictor_config: Optional[PredictorConfig] = None
    """predictor config"""

    seed: int = 42
    confidence_level: Optional[float] = None

    disable_cache: bool = False
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0

    metric_log_dir: Optional[str] = None
    tb_log_dir: Optional[str] = None
    no_resume: Optional[bool] = False

    metrics_to_report: Any = None
    """only report a subset of the registered metrics of the task"""

    show_progress: bool = False
    """Log the intermediate progress while evaluating"""

    log_raw_results: bool = True
    """If true, will output per-example scores in addition to the overall metrics"""

    log_only_text: bool = True
    """If true, only textual raw results logged, not the intermediate / output embeddings"""

    def _update_dataset_params(self):
        # If the task allows custom dataset argument, we will pass user-defined config to it,
        # and silently freeze the user-defined values. If there is no user-defined config,
        # an empty dataset config is set, in which case we do not need to freeze the values
        defaults = TaskRegistry.get_task_args(self.task_name)
        if "dataset" in defaults:
            cls_ = TaskRegistry.get_dataloader_type(self.task_name).dataset_config()
            self.params = self.params or {}
            self.params["dataset"] = self.dataset or cls_()
        elif self.dataset:
            raise ValueError(
                f"Task {self.task_name} accepts no custom dataset arguments, got {self.dataset}"
            )

    def __post_init__(self) -> None:
        if self.name and "/" in self.name:
            self.name = "_".join(s.split("/")[-1] for s in self.name.split(","))

        if self.data_loading is None:
            self.data_loading = EvaluationDataLoadingConfig()

        self.params = self.params or {}
        self._update_dataset_params()

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name == "dataset" and value is not None:
            self._update_dataset_params()

    @classmethod
    def from_dict(cls, config: Union[DictConfig, Dict]) -> Self:
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config)  # type: ignore

        predictor_config_converter = partial(
            promote_config, config_cls=get_config_cls(config["predictor"])
        )
        custom_hooks = dacite.Config(
            type_hooks={PredictorConfig: predictor_config_converter},
            cast=[Path],
        )

        return dacite.from_dict(data_class=cls, data=config, config=custom_hooks)


def run_task(
    run_config: RunConfig,
    logger: "Logger",
    gang: "Gang" = None,
    log_file_suffix: str = "",
) -> Tuple[Dict[str, AverageMetric], str]:
    run_name = run_config.name

    if run_config.dump_dir is not None:
        filename = os.path.join(
            run_config.dump_dir,
            "raw_results",
            f"{run_name}",
            f"{run_name}{log_file_suffix}",
        )
        result_file = os.path.join(run_config.dump_dir, "results", f"{run_name}.json")
        if not run_config.no_resume:
            if os.path.exists(result_file):  # In local mode
                rank_zero_info(f"Loading cached evaluation results from {result_file}")
                with open(result_file, encoding="utf-8") as f:
                    metrics = json.load(f)["results"]
                    metrics = {
                        k: AverageMetric(avg=v, count=0, square=0)
                        for k, v in metrics.items()
                    }  # type: ignore
                    return metrics, result_file
            if not log_file_suffix and os.path.exists(
                filename + ".json"
            ):  # In shard mode
                rank_zero_info(
                    f"Loading cached evaluation results from {filename}.json"
                )
                examples = load_jsonl(filename + ".json")
                metrics = {}
                for example in examples:
                    for k, v in example["metrics"]:
                        if k not in metrics:
                            metrics[k] = AverageMetric(avg=0, count=0, square=0)
                        metrics[k].update(value=v, count=1)
                return metrics, filename + ".json"

    # Set up the RANK and WORLD configs in each worker
    setup_env()
    set_torch_variables()
    torch.manual_seed(run_config.seed)

    gang = gang or FakeGang()
    assert isinstance(run_config.params, Mapping)

    task_config = TaskRegistry.get_config(run_config.task_name, **run_config.params)

    data_loader_type = TaskRegistry.get_dataloader_type(run_config.task_name)
    assert isinstance(run_config.data_loading, EvaluationDataLoadingConfig), (
        "data loading not specified"
    )
    task = build_task(
        task_config,
        data_loading_config=run_config.data_loading,
        data_loader_type=data_loader_type,
        gang=gang,
        dataset_config=run_config.dataset,
        logger=logger,
        dtype=run_config.dtype,
    )

    # Add the `eos_config` if present
    eos_config = run_config.params.get("eos_config", None)
    if eos_config:
        eos_config = EOSConfig(**eos_config)

    assert run_config.predictor_config, "Predictor config cannot be empty"
    predictor = build_predictor(
        run_config.predictor_config,
        gang=gang,
        eos_config=eos_config,
        dtype=run_config.dtype,
        top_p=run_config.top_p,
        top_k=run_config.top_k,
        temperature=run_config.temperature,
    )

    rank_zero_info(f"Predictor loaded: {predictor.__class__.__name__}", logger=logger)
    result = task.run(
        predictor=predictor,
        top_p=run_config.top_p,
        top_k=run_config.top_k,
        temperature=run_config.temperature,
        disable_cache=run_config.disable_cache,
        random_state=get_random_state(run_config.seed),
        show_progress=run_config.show_progress,
        seed=run_config.seed,
    )

    metrics = result.metrics

    # Compute confidence intervals
    cf = run_config.confidence_level
    if cf is not None:
        for key, avg_metric in list(metrics.items()):
            ci_lb, ci_ub = avg_metric.compute_ci(cf)
            if ci_lb and ci_ub:
                metrics[f"{key}_ci_lb_{cf}"] = ci_lb
                metrics[f"{key}_ci_ub_{cf}"] = ci_ub

    if (
        run_config.dump_dir is not None
        and get_mp_rank() == 0
        and result.raw_results
        and run_config.log_raw_results
    ):
        filename = os.path.join(
            run_config.dump_dir,
            "raw_results",
            f"{run_name}",
            f"{run_name}{log_file_suffix}",
        )
        log_raw_results(
            result.raw_results,
            filename,
            logger=logger,
            log_only_text=run_config.log_only_text,
        )

    torch.cuda.empty_cache()
    return metrics, result_file
