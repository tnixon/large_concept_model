# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#


import datetime
import json
import logging
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

from stopes.core import Launcher, Requirements, StopesModule

from lcm.evaluation.run import RunConfig, run_task
from lcm.evaluation.tasks import build_dataset_config
from lcm.evaluation.utils.common import (
    flatten_dict,
    format_dict,
    log_config_metadata,
    log_final_results,
    write_to_json,
)
from lcm.evaluation.utils.distributed import get_gang, rank_zero_info


@dataclass
class RunModuleConfig(RunConfig):
    """
    Config of one eval run on SLURM nodes
    """

    requirements: Optional[Requirements] = None
    """Requirements to request a slurm node"""

    nshards: Optional[int] = None
    """Number of shards to parallize the SLURM array job"""

    os_environs: Optional[Dict[str, Any]] = None
    """OS environments passed from the launcher node to slurm nodes"""

    def __post_init__(self):
        super().__post_init__()
        if self.requirements is None:
            self.requirements = Requirements(
                gpus_per_node=1, cpus_per_task=4, timeout_min=150
            )
        assert self.nshards == 1 or self.requirements.gpus_per_node == 1, (
            "Only support single job on multiple-gpu nodes or multiple jobs on single-gpu nodes. "
            f"Got {self.nshards} shards and gpus_per_node = {self.requirements.gpus_per_node}"
        )


class EvalRunModule(StopesModule):
    def __init__(self, config: RunModuleConfig):
        self.config = config
        self.retry_counts = [0]

    def requirements(self) -> Requirements:
        return self.config.requirements  # type: ignore

    def name(self):
        return "_".join(
            [
                self.config.name,
                self.config.predictor,
                self.sha_key()[:10],
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            ]
        )

    def get_config_for_cache(self):
        config_for_cache = asdict(self.config)
        OVERWRITE_VALUE_FOR_CACHE = -1
        NO_HASHED_KEYS = [
            "requirements",
            "timeout_min",
            "no_resume",
            "show_progress",
            "dump_dir",
            "tb_log_dir",
            "metric_log_dir",
        ]

        def deep_overwrite(dct: dict):
            for k, v in dct.items():
                if isinstance(v, dict):
                    deep_overwrite(v)
                elif k in NO_HASHED_KEYS:
                    dct[k] = OVERWRITE_VALUE_FOR_CACHE

        deep_overwrite(config_for_cache)
        return config_for_cache

    def array(self) -> Union[None, List[int]]:
        nshards = getattr(self.config, "nshards", None)
        if not nshards:
            return None

        return list(range(nshards))

    def run(self, iteration_value: Optional[Any] = None, iteration_index: int = 0):
        logger = logging.getLogger("lcm.evaluation.arun")

        # Make sure the environment variables in the launcher node is propagated to worker nodes
        if self.config.os_environs:
            for k, v in self.config.os_environs.items():
                os.environ[k] = v

        log_config_metadata(
            self.config, self.config.task_name, self.config.params, logger=logger
        )

        if iteration_value is not None:
            assert isinstance(iteration_value, int) and self.config.nshards, (
                f"Invalid shard value ({self.config.nshards}) or iteration value ({iteration_value})"
            )
            assert self.config.data_loading, (
                f"Data loading is not specified: \n {self.config}"
            )
            self.config.data_loading.rank = iteration_value
            self.config.data_loading.world_size = int(self.config.nshards)

        gang = get_gang()
        logger.info(f"Running task {self.config.task_name} on {gang.device}")

        return run_task(
            self.config,
            logger,
            gang=gang,
            log_file_suffix=f"_{iteration_index}",
        )


def build_async_task(
    base_config: Dict[str, Any], task: Dict[str, Any]
) -> EvalRunModule:
    """Generate the list of async configs from the user input about the desired tasks"""

    run_config = deepcopy(base_config)

    assert "name" in task, f"Invalid task config: {task}\n Expect `name` attribute"
    _name = task.pop("name")

    run_config["task_name"] = _name
    run_config["name"] = f"{_name}.{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"  # fmt: skip

    if "metrics_to_report" in task:
        run_config["metrics_to_report"] = task.pop("metrics_to_report")

    dataset = task.pop("dataset", None)
    _params = run_config.get("params", {})
    run_config["params"] = {**_params, **task}
    run_config["dataset"] = build_dataset_config(_name, dataset)
    run_config["requirements"] = base_config.get("requirements", None)

    typed_run_config = RunModuleConfig.from_dict(run_config)

    # Share some environment variables between the launcher node and slurm nodes
    if os.getenv("HF_HOME"):
        typed_run_config.os_environs = {"HF_HOME": os.environ["HF_HOME"]}

    return EvalRunModule(typed_run_config)  # type: ignore[arg-type]


async def schedule_task(
    module: EvalRunModule,
    launcher: Launcher,
    logger: logging.Logger,
) -> Dict[str, float]:
    if module.config.dump_dir is not None:
        result_file = os.path.join(
            module.config.dump_dir, "results", f"{module.config.name}.json"
        )
        if not module.config.no_resume and os.path.exists(result_file):
            rank_zero_info(f"Loading cached evaluation results from {result_file}")
            with open(result_file) as f:
                metrics = json.load(f)["results"]
            return metrics

    result = await launcher.schedule(module)

    # `module` is an array job -> aggregate the raw results, not the locally
    # aggregated result or the confident interval
    if isinstance(result, list):
        (metrics, result_file), *remaining_result = result
        for _metrics, _ in remaining_result:
            for name, value in _metrics.items():
                if "ci_lb" in name or "ci_ub" in name:
                    continue
                metrics[name].update(value.value, value.count, value.square)
        result = (metrics, result_file)

    result_metrics, result_file = result
    assert isinstance(result_metrics, dict), (
        f"Expected Tuple[Dict[str, AverageMetrics], str], get {type(result_metrics)}"
    )

    metrics = {}
    cf = getattr(module.config, "confidence_level", None)
    for key, avg_metric in result_metrics.items():
        if "ci_lb" in key or "ci_ub" in key:
            continue
        if not hasattr(avg_metric, "value"):
            # Local confidence interval, ignore as we will recalculate it
            continue

        metrics[key] = avg_metric.value
        ci_lb, ci_ub = avg_metric.compute_ci(cf) if cf is not None else None, None

        if ci_lb and ci_ub:
            metrics[f"{key}_ci_lb_{cf}"] = ci_lb
            metrics[f"{key}_ci_ub_{cf}"] = ci_ub

    if module.config.dump_dir is not None:
        result_content = {
            "results": metrics,
            "configs": module.get_config_for_cache(),
        }
        logger.info(f"Writing metric results to {result_file}")
        write_to_json(result_content, result_file, indent=4)

    results = flatten_dict(metrics)  # type: ignore
    rank_zero_info(f"All evaluation results: {format_dict(results)}", logger=logger)
    if module.config.metric_log_dir is not None:
        log_final_results(
            results,
            module.config.predictor_config,
            module.config.tb_log_dir,
            module.config.metric_log_dir,
            logger=logger,
        )
    return metrics
