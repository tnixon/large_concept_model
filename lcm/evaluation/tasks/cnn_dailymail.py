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


@register_task("cnn_dailymail_{form}llm.{split}", {"split": SPLITS, "form": FORMS})
def get_task_config_llm(
    dataset: JSONDatasetConfig,
    dataset_dir: str,
    split: str,
    form: str,
    min_gen_len: int = 10,
    max_gen_len: int = 512,
    max_gen_len_ratio: Optional[float] = None,
    max_prompt_len: int = 4096,
) -> GenerationTaskConfig:
    file_path = f"{dataset_dir}/cnn_daily_mail/{split}.jsonl"

    # In case the user specifies the directory that point directly to the task dir
    if not Path(file_path).exists():
        file_path = f"{dataset_dir}/{split}.jsonl"

    assert Path(file_path).exists(), f"{file_path} not found."

    dataset.file_path = file_path

    if form != "inverse_":
        source_text_column = "article"
        target_text_column = "highlights"
        dataset.source_prefix_text = "[INST] Summarize the following article: "
        dataset.source_suffix_text = " [/INST]"
    else:
        source_text_column = "highlights"
        target_text_column = "article"
        dataset.source_prefix_text = ("[INST] Write an article from the following summary: ")  # fmt: skip
        dataset.source_suffix_text = " [/INST]"

    dataset.source_text_column = source_text_column
    dataset.target_text_column = target_text_column

    # Add original columns for judge tasks
    dataset.columns = [dataset.source_text_column, dataset.target_text_column]
    postprocess_fn = partial(default_text_postprocess, source_text_column=dataset.source_text_column)  # fmt: skip

    return GenerationTaskConfig(
        dataset=dataset,
        postprocess_fn=postprocess_fn,
        prompt_func=default_text_prompt,
        metric_fns=[
            evaluate(
                rouge_score,
                outputs=("rouge2", "rougeL", "rougeLsum"),
                types=("rouge2", "rougeL", "rougeLsum"),
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
