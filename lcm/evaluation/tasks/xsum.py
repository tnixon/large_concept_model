# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#
#

from functools import partial
from pathlib import Path
from typing import Optional

from lcm.datasets.configs import JSONDatasetConfig
from lcm.evaluation.metrics.common import ngram_score, rouge_score
from lcm.evaluation.tasks import register_task
from lcm.evaluation.tasks.base import GenerationTaskConfig
from lcm.evaluation.utils.common import evaluate
from lcm.evaluation.utils.data_utils import (
    default_text_postprocess,
    default_text_prompt,
)

SPLITS = ["test", "validation", "train"]
FORMS = ["", "inverse_"]


@register_task("xsum_{form}llm.{split}", {"split": SPLITS, "form": FORMS})
def get_task_config_llm(
    dataset: JSONDatasetConfig,
    dataset_dir: str,
    split: str,
    form: str,
    min_gen_len: int = 4,
    max_gen_len: int = 512,
    max_gen_len_ratio: Optional[float] = None,
    max_prompt_len: int = 4096,
) -> GenerationTaskConfig:
    file_path = f"{dataset_dir}/xsum/{split}.jsonl"

    # In case the user specifies the directory that point directly to the task dir
    if not Path(file_path).exists():
        file_path = f"{dataset_dir}/{split}.jsonl"

    assert Path(file_path).exists(), f"{file_path} not found."

    dataset.file_path = file_path

    # Default prompt if not specified by the user. Use Llama-3.1 prompts by default
    if form != "inverse_":
        source_text_column = "document"
        target_text_column = "summary"
    else:
        source_text_column = "summary"
        target_text_column = "document"

    dataset.source_text_column = source_text_column
    dataset.target_text_column = target_text_column

    # Add original columns for judge tasks
    dataset.columns = [dataset.source_text_column, dataset.target_text_column]

    postprocess_fn = partial(
        default_text_postprocess, source_text_column=source_text_column
    )

    return GenerationTaskConfig(
        dataset=dataset,
        prompt_func=default_text_prompt,
        postprocess_fn=postprocess_fn,
        metric_fns=[
            evaluate(
                rouge_score,
                outputs=("rouge2", "rougeL"),
                types=("rouge2", "rougeL"),
            ),
            evaluate(
                ngram_score,
                inputs=("prediction", "source"),
                outputs=("ngram_overlap", "repetition_4"),
            ),
        ],
        min_gen_len=min_gen_len,
        max_gen_len=max_gen_len,
        max_gen_len_ratio=max_gen_len_ratio,
        max_prompt_len=max_prompt_len,
    )
