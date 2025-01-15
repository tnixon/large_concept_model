# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import pytest
from fairseq2.gang import FakeGang
from numpy.random import RandomState

from lcm.datasets.base import BaseDataLoader
from lcm.datasets.configs import (
    ColumnsNames,
    DataLoadingConfig,
    EvaluationDataLoadingConfig,
    ParquetDatasetConfig,
)
from lcm.evaluation.api import (
    Example,
    Prediction,
    Predictor,
    Prompts,
    ScorerConfig,
    TaskConfig,
    TaskResult,
)
from lcm.evaluation.predictors import build_predictor, get_config_cls
from lcm.evaluation.tasks import TaskRegistry, build_task, register_task
from lcm.evaluation.tasks.base import GenerationTask, GenerationTaskConfig
from lcm.evaluation.utils.common import set_torch_variables
from lcm.evaluation.utils.data_utils import ParquetTestDataLoader
from lcm.utils.common import Batched

TASK_NAME = "test_task"
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS", "") == "true"


@dataclass
class EmptyConfig: ...


class IdentityPredictor(Predictor):
    """A testing predictor that returns the exact input seq as output"""

    @staticmethod
    def from_config(config: EmptyConfig) -> "IdentityPredictor":  # type: ignore
        return IdentityPredictor()

    def __call__(
        self,
        prompts: Prompts,
        max_prompt_len: Optional[int] = None,
        max_gen_len: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        echo: bool = True,
        return_logprobs: bool = False,
        show_progress: bool = False,
        disable_cache: bool = False,
        **kwargs,
    ) -> Sequence[Prediction]:
        # Beam size of 1
        preds = [
            Prediction(text=" ".join(map(str, seq)), embed=seq, tokens=seq)
            for seq in prompts  # type: ignore
        ]
        return preds


@pytest.fixture
def predictor() -> Predictor:
    set_torch_variables()
    return IdentityPredictor.from_config(EmptyConfig())


@dataclass
class DummyGenerationTaskConfig(GenerationTaskConfig):
    """
    Task config. Current Task design requires that the config class name
    and the task name must match except for the suffix `Config`, so if
    the task is DummyGeneration, then the config should be DummyGenerationConfig,
    and that the config class and task class must be in the same Python module
    """

    ...


class DummyGenerationTask(GenerationTask):
    def compute_metrics(  # type: ignore
        self,
        predictions: Sequence[Prediction],
        hypothesis_indices: List[List[int]],
        input: Sequence[Example],
        show_progress: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Calculate the sum of element-wise subtraction between
        prediction and the batched. This should be 0 for
        IdentityPredictor
        """
        assert len(input) == len(predictions)
        loss = 0.0
        for example, pred in zip(input, predictions):  # type: ignore
            seq = example[ColumnsNames.source_column.value].as_py()
            loss += sum(seq) - sum(pred.embed)  # type: ignore

        return {"metrics": {"m1": [loss] * len(input)}}


def run_task(
    predictor: Predictor,
    task_name: str,
    data_loading_config: DataLoadingConfig,
    dataset_config: ParquetDatasetConfig,
) -> TaskResult:
    task = build_task(
        task_config=TaskRegistry.get_config(task_name),
        data_loading_config=data_loading_config,  # type: ignore[arg-type]
        data_loader_type=ParquetTestDataLoader,
        dataset_config=dataset_config,
        gang=FakeGang(),
    )
    assert isinstance(task, GenerationTask)
    random_state = RandomState(data_loading_config.seed)
    return task.run(predictor, random_state=random_state)


def dummy_prompt_func(batch: Batched, config: TaskConfig) -> Prompts:
    return [x[ColumnsNames.source_column.value].as_py() for x in batch]  # type: ignore


@register_task(TASK_NAME, data_loader_type=BaseDataLoader)  # type: ignore
def get_dummy_task_config():
    return DummyGenerationTaskConfig(
        dataset=None,  # type: ignore
        prompt_func=dummy_prompt_func,  # type: ignore[arg-type]
    )


@register_task(TASK_NAME + "_ppl", data_loader_type=BaseDataLoader)  # type: ignore
def get_dummy_task_with_perplexity_config():
    return DummyGenerationTaskConfig(
        dataset=None,  # type: ignore
        prompt_func=dummy_prompt_func,  # type: ignore[arg-type]
        model_based_metrics=[
            ScorerConfig(
                scorer_type="sentence_perplexity",
                inputs=ColumnsNames.source_text_column.value,  # type: ignore
            ),
        ],
    )


def test_dummy_task(simple_data_config, predictor):
    dlc, dsc = simple_data_config
    eval_dlc = EvaluationDataLoadingConfig(
        **asdict(dlc),
        max_samples=3,
    )
    avg_result = run_task(
        predictor,
        TASK_NAME,
        data_loading_config=eval_dlc,
        dataset_config=dsc,
    )
    assert "m1" in avg_result.metrics and avg_result.metrics["m1"].avg == 0  # type: ignore


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS, reason="GPT-2 medium is not accessible in GH runner"
)
def test_dummy_task_with_perplexity(simple_data_config, predictor):
    dlc, dsc = simple_data_config
    eval_dlc = EvaluationDataLoadingConfig(
        **asdict(dlc),
        max_samples=3,
    )
    avg_result = run_task(
        predictor,
        TASK_NAME + "_ppl",
        data_loading_config=eval_dlc,
        dataset_config=dsc,
    )
    assert (
        "sentence_perplexity" in avg_result.metrics
        and avg_result.metrics["sentence_perplexity"].avg > 0
    )  # type: ignore


def test_run_task_with_dynamic_predictor(simple_data_config):
    """Test that a predictor can be understood and load from the PredictorRegistry"""

    predictor_cfg = get_config_cls("dummy")()
    predictor = build_predictor(predictor_cfg)

    dlc, dsc = simple_data_config
    eval_dlc = EvaluationDataLoadingConfig(
        **asdict(dlc),
        max_samples=3,
    )
    avg_result = run_task(
        predictor,
        TASK_NAME,
        data_loading_config=eval_dlc,
        dataset_config=dsc,
    )
    assert "m1" in avg_result.metrics and avg_result.metrics["m1"].avg == 0.0, (
        avg_result.metrics
    )  # type: ignore
