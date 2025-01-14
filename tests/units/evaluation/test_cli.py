# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import json
import logging
import sys

from lcm.evaluation.__main__ import cfg_from_cli
from lcm.evaluation.cli.local import main as local_main
from lcm.evaluation.utils.data_utils import load_jsonl
from lcm.utils.common import batched, setup_conf

logger = logging.getLogger("lcm.evaluation.test_cli")


def test_dynamic_prompt(tmp_path, simple_json_dataset, monkeypatch):
    """Test that the dynamic prompt can be set via source_prefix_text"""
    setup_conf()
    bsz = 5
    commands = [
        "lcm.evaluation.__main__.py",
        "--tasks",
        "dummy_json_generation",
        "--predictor",
        "dummy",
        "--data_loading.batch_size",
        str(bsz),
        "--dataset.file_path",
        str(simple_json_dataset),
    ]

    raw_data = load_jsonl(simple_json_dataset)

    # Default prompt defined in the task
    with monkeypatch.context() as m1:
        m1.setattr(
            sys,
            "argv",
            commands
            + [
                "--dump_dir",
                str(tmp_path / "test_1"),
                "--dataset.source_text_column",
                "input_text",
            ],
        )
        eval_config, _ = cfg_from_cli()
        local_main(eval_config, logger=logger)

        with open(
            tmp_path.joinpath("test_1", "results", "dummy_json_generation.json")
        ) as fh:
            result = json.load(fh)
            assert "results" in result and result["results"]["m1"] == 0.0

        default_prompts = [f"[INST] Prompt: {x['input_text']}" for x in raw_data]
        default_prompts = batched(default_prompts, batch_size=bsz)  # type: ignore
        results = load_jsonl(
            tmp_path.joinpath(
                "test_1",
                "raw_results",
                "dummy_json_generation",
                "dummy_json_generation.json",
            )
        )
        for prompt, result in zip(default_prompts, results):
            assert result["text_prompts"] == prompt, (
                f"Not match: {result['text_prompts']} != {prompt}"
            )
    # Custom prompt with prefix and suffix
    with monkeypatch.context() as m2:
        m2.setattr(
            sys,
            "argv",
            commands
            + [
                "--dump_dir",
                str(tmp_path / "test_2"),
                "--dataset.source_text_column",
                "input_text",
                "--dataset.source_prefix_text",
                "[Myprompt] ",
                "--dataset.source_suffix_text",
                "[/Myprompt]",
            ],
        )
        eval_config, _ = cfg_from_cli()
        local_main(eval_config, logger=logger)

        custom_prompts = [f"[Myprompt] {x['input_text']}[/Myprompt]" for x in raw_data]
        custom_prompts = batched(custom_prompts, batch_size=bsz)  # type: ignore
        results = load_jsonl(
            tmp_path.joinpath(
                "test_2",
                "raw_results",
                "dummy_json_generation",
                "dummy_json_generation.json",
            )
        )
        for prompt, result in zip(custom_prompts, results):
            assert result["text_prompts"] == prompt, (
                f"Not match: {result['text_prompts']} != {prompt}"
            )

    # Custom prompt with complex sequences of text
    with monkeypatch.context() as m3:
        m3.setattr(
            sys,
            "argv",
            commands
            + [
                "--dump_dir",
                str(tmp_path / "test_3"),
                "--dataset.source_sequences",
                '{"text_value": "[SEQ]"}',
                "--dataset.source_sequences",
                '{"text_column": "input_text"}',
                "--dataset.source_sequences",
                '{"text_value": "-"}',
                "--dataset.source_sequences",
                '{"text_column": "input_text"}',
                "--dataset.source_sequences",
                '{"text_value": "[/SEQ]"}',
            ],
        )
        eval_config, _ = cfg_from_cli()
        local_main(eval_config, logger=logger)

        custom_prompts = [
            f"[SEQ] {x['input_text']} - {x['input_text']} [/SEQ]" for x in raw_data
        ]
        custom_prompts = batched(custom_prompts, batch_size=bsz)  # type: ignore
        results = load_jsonl(
            tmp_path.joinpath(
                "test_3",
                "raw_results",
                "dummy_json_generation",
                "dummy_json_generation.json",
            )
        )
        for prompt, result in zip(custom_prompts, results):
            assert result["text_prompts"] == prompt, (
                f"Not match: {result['text_prompts']} != {prompt}"
            )
