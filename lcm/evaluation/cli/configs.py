# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from dataclasses import dataclass, is_dataclass
from typing import Any, Dict, List, Optional, Union

import yaml

from lcm.datasets.configs import DatasetConfig, EvaluationDataLoadingConfig
from lcm.evaluation.api import PredictorConfig
from lcm.evaluation.cli.params import parse_args
from lcm.evaluation.run import RunConfig
from lcm.evaluation.tasks import TaskRegistry
from lcm.evaluation.utils.common import filter_by_pattern, unroll_configs

LauncherOptions = Union[str, Dict[str, Any]]


@dataclass
class CliConfig:
    tasks: str
    """
    Task names as used to register to TaskRegistry via register_task().
    In case of multiple tasks, use "task1,task2" or wildcard patterns
    """

    predictor: str
    """predictor name"""

    dump_dir: str
    """Output directory to which the average results and raw results are
    stored, under the names `result.tsv` and `result.sample.tsv` respectively"""

    dataset_args: Optional[List[str]] = None
    """User data config in form of [<param1>, <value1>, <param2>, <value2>, ...]"""

    dataloading_args: Optional[List[str]] = None
    """arguments to build the data loader"""

    dataset_dir: Optional[str] = None
    """Directory of the datasets that are accessed locally"""

    task_args: Optional[Dict[str, Any]] = None
    """Task params as defined in each TaskConfig"""

    predictor_config: Optional[PredictorConfig] = None
    """predictor config"""

    prompt_file: Optional[str] = None
    """A YAML file that consists of prompt settings for the task. Useful for very long prompts"""

    dry_run: bool = False
    """Do not run evaluation but show its configs set"""

    # ppl: Optional[PerplexityTaskConfig] = None
    launcher: Optional[str] = None

    job_args: Optional[Dict[str, Any]] = None
    """Specific arguments for stope modules (requirements, etc.)"""

    dtype: str = "torch.float32"
    """If the dtype cannot be inferred from predictor's model, use this one"""

    temperature: float = 0.0
    disable_cache: bool = False
    top_k: int = 0
    top_p: float = 0.0
    seed: int = 42
    confidence_level: Optional[float] = None

    metric_log_dir: Optional[str] = None
    tb_log_dir: Optional[str] = None
    no_resume: Optional[bool] = False
    log_only_text: bool = False
    show_progress: bool = False

    def __post_init__(self) -> None:
        self.metric_log_dir = self.metric_log_dir or self.dump_dir
        assert self.temperature >= 0.0, (
            f"Expect non-zero temperature, get {self.temperature}"
        )
        if self.temperature == 0:
            self.top_p = 0
            self.top_k = 0


def extract_args_from_yaml(cfg, file: Optional[str] = None) -> None:
    if not file:
        return

    with open(file, "r", encoding="utf-8") as fp:
        args_dict = yaml.safe_load(fp)

    for k, v in args_dict.items():
        setattr(cfg, k, v)

    return cfg


def parse_configs(cfg: CliConfig) -> List[RunConfig]:
    """Parse the CLi to get the individual task runs"""

    configs: List[RunConfig] = []
    params_dict = cfg.task_args or {}

    # In `task_args`, params that are not grouped under specific task are propagated
    # to all others
    names = filter_by_pattern(TaskRegistry.names(), cfg.tasks)
    common_args = {k: v for k, v in params_dict.items() if k not in names}
    for name in names:
        defaults = TaskRegistry.get_task_args(name)

        params = params_dict.get(name, {})
        for fname, kwargs in unroll_configs(defaults, params, prefix=name).items():
            # Get the custom data configs if specified from the CLI
            dataset_config_cls = TaskRegistry.get_dataloader_type(name).dataset_config()
            dataset_config: Optional[DatasetConfig] = None
            if cfg.dataset_args or cfg.prompt_file:
                if cfg.dataset_args:
                    dataset_config = parse_args(dataset_config_cls, cfg.dataset_args)  # type: ignore
                else:
                    dataset_config = dataset_config_cls()
                if cfg.prompt_file:
                    extract_args_from_yaml(dataset_config, cfg.prompt_file)

                # By setting the dataset config to be "silently frozen", we ensure the
                # user-defined config values are not overriden by the task function
                dataset_config.freeze()  # type: ignore

            # Tasks that use local datasets must know the root directory where the
            # files are stored. These tasks are recognized with a parameters "dataset_dir"
            # in the registering function
            if "dataset_dir" in defaults:
                assert cfg.dataset_dir, f"Expect param `dataset_dir` for task {fname}"
                kwargs["dataset_dir"] = cfg.dataset_dir

            kwargs = {**kwargs, **common_args}
            dataloading_config = (
                parse_args(EvaluationDataLoadingConfig, cfg.dataloading_args)
                if cfg.dataloading_args
                else None
            )

            run_config = RunConfig(
                name=fname,
                task_name=name,
                params=kwargs,
                dataset=dataset_config,
                data_loading=dataloading_config,
                dtype=cfg.dtype,
                dump_dir=cfg.dump_dir,
                predictor=cfg.predictor,
                predictor_config=cfg.predictor_config,
                seed=cfg.seed,
                confidence_level=cfg.confidence_level,
                temperature=cfg.temperature,
                disable_cache=cfg.disable_cache,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                metric_log_dir=cfg.metric_log_dir,
                tb_log_dir=cfg.tb_log_dir,
                no_resume=cfg.no_resume,
                log_only_text=cfg.log_only_text,
                show_progress=cfg.show_progress,
            )
            configs.append(run_config)

    if cfg.predictor in {"llm_inference", "llm_inference_reward"}:
        assert cfg.predictor_config
        max_total_len = 0
        for run_config in configs:
            t = run_config.params
            if isinstance(t, dict):
                _total_len = 1 + t.get("max_prompt_len", 0) + t.get("max_gen_len", 0)
            elif is_dataclass(t):
                _total_len = (
                    1 + getattr(t, "max_prompt_len", 0) + getattr(t, "max_gen_len", 0)
                )
            else:
                _total_len = 1
            if _total_len > max_total_len:
                max_total_len = _total_len

        setattr(cfg.predictor_config, "max_total_len", max_total_len)
    return configs
