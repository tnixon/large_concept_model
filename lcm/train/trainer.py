# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
#

import gc
import logging
import os
import sys
from abc import abstractmethod
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from functools import cached_property
from itertools import count
from pathlib import Path
from pprint import pformat
from typing import (
    Any,
    ContextManager,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
)

import torch
import yaml
from fairseq2.assets import AssetCard, AssetCardFieldNotFoundError
from fairseq2.checkpoint import FileCheckpointManager
from fairseq2.gang import FakeGang, Gang, ReduceOperation, all_sum
from fairseq2.logging import get_log_writer
from fairseq2.metrics import (
    LogMetricRecorder,
    MetricBag,
    MetricRecorder,
    TensorBoardRecorder,
    record_metrics,
)
from fairseq2.nn.ddp import to_ddp
from fairseq2.nn.fsdp import to_fsdp
from fairseq2.nn.utils.gradient import (
    check_gradient_norms,
    clip_gradient_norm,
    scale_gradients,
)
from fairseq2.nn.utils.module import (
    _get_named_modules,
    freeze_parameters,
    to_device,
)
from fairseq2.optim import AdamW, DynamicLossScaler
from fairseq2.optim.lr_scheduler import AbstractLRScheduler, get_effective_lr
from fairseq2.recipes.utils.log import log_model
from fairseq2.utils.profiler import Profiler, Stopwatch
from fairseq2.utils.rng import RngBag
from fairseq2.utils.state import StatefulObjectBag
from omegaconf import MISSING
from stopes.core import Requirements
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.profiler import record_function
from torcheval.metrics import Mean

from lcm.datasets.configs import DataLoadingConfig, ValidationDataLoadingConfig
from lcm.datasets.dataloading import ds_name
from lcm.train.metrics import (
    LCMWandBRecorder,
    flatten_dict,
)
from lcm.train.optim import build_lr_scheduler
from lcm.utils.data_utils import update_dataclass
from lcm.utils.distributed import (
    SUPPORTED_FSDP_MEMORY_POLICIES,
    SUPPORTED_FSDP_WRAP_POLICIES,
    get_fsdp_memory_policy,
    get_fsdp_wrap_policy,
    init_process_group,
)
from lcm.utils.logging import (
    log_env_variables,
    setup_additional_logging,
)

logger = get_log_writer(__name__)


@dataclass
class TrainingConfig:
    """Holds the configuration of a training job."""

    training_data: Any = MISSING
    """The datasets to train with."""

    validation_data: Any = MISSING
    """The datasets to validate on."""

    model_arch: Optional[str] = None
    """Starting architecture for the model to train"""

    model_arch_overrides: Optional[Dict] = None
    """Dict of parameters to overwrite in `model_arch`"""

    model_config_or_name: Optional[Any] = None
    """The model configuration or name to train.
        This option cannot be paired with model_arch + model_arch_overrides
        If provided, this option supersedes model_arch + model_arch_overrides
    """
    output_dir: Path = MISSING
    """The output directory to store checkpoints and logs."""

    log_folder: Optional[Path] = None
    """The executor's log directory where stdout/stderr will be redirected.
        We will use this directory to optionally enable ATEN and NCCL
        logging (if debug is True) """

    tb_dir: Optional[Path] = None
    """The output directory to store tensorbaord logs"""

    # defaults to "uncategorized"
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None

    requirements: Requirements = field(
        default_factory=lambda: Requirements(
            nodes=1,
            tasks_per_node=8,
            gpus_per_node=8,
            cpus_per_task=8,
            mem_gb=256,
            timeout_min=3 * 24 * 60,
            constraint="volta32gb",
        )
    )
    """The scheduling requirements for this trainer"""

    data_loading_config: DataLoadingConfig = MISSING

    validation_data_loading_config: ValidationDataLoadingConfig = field(
        default_factory=lambda: ValidationDataLoadingConfig()
    )

    criterion: Any = MISSING

    dtype: str = "torch.float32"
    """The data type of the model."""

    lr_schedule: str = "myle"
    """The learning rate schedule out of
        `noop`: no learning rate schedule, just use the initial learning rate,
        `myle`: inv-sqrt as implemented in Fairseq,
        `cosine` cosine annealing schedule,
        `wsd` for  Warmup-Stable-Decay (WSD) or tri-stage """

    lr: float = 0.004
    """The initial (post-warm-up) learning rate for AdamW."""

    start_lr: float = 1e-7
    """The initial warmup learning rate."""

    final_lr: float = 1e-5
    """The final learning rate."""

    lr_stage_ratios: List[float] = field(default_factory=lambda: [0.1, 0.4, 0.5])
    """The ratios of the wsd (tri-stage) learning rate scheduler."""

    num_lr_warmup_steps: int = 800
    """The number of warm-up steps for the learning rate."""

    weight_decay: float = 0.1
    """The weight decay coefficient of AdamW (PyTorch default: 1e-2, Fs2 default: 0.0)."""

    adam_betas: List[float] = field(default_factory=lambda: [0.9, 0.98])
    """The beta coefficients of AdamW used for computing running averages of gradient and its square."""

    adam_eps: float = 1e-6
    """The term added to the denominator in AdamW to improve numerical stability.
        Default in FS2 and PyTorch is 1e-8. Previous hard coded value in our trainer is 1e-6"""

    use_optimizer_in_fp32: bool = True
    """if True, the optimizer (AdamW) will be initialized with `use_fp32 = True`
        i.e. we will store the optimizer state in single precision and convert
        gradients on-the-fly to single precision for numerical stability"""

    max_steps: int = 10_000
    """The maximum number of training steps."""

    max_grad_norm: float = 1000
    """Maximal gradient norm, for gradient clipping.
       gradients are multiplied by `torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)`
       if max_norm is arbitrarily large, then we'll only report gradients norm
    """
    turn_off_grad_normalization: bool = False
    """If ``True``, Turn off gradient normalization"""

    gradient_accumulation: int = 1
    """The number of steps to accumulate gradients before an optimizer update."""

    validate_every_n_steps: int = 5000
    """The number of steps after which to validate the model."""

    checkpoint_every_n_steps: int = 5000
    """The number of steps after which to checkpoint."""

    keep_last_n_checkpoints: int = -1
    """The number of checkpoints to keep on disk."""

    save_model_every_n_steps: int = 5000
    """The number of steps after which to save a consolidated version of the model."""

    preserve_consolidated_models: bool = False
    """If `True`, only pt files excluding ones starting with `mdoel` will be deleted from the step checkpoint directory."""

    publish_metrics_every_n_steps: int = 1
    """The number of steps after which to publish training metrics."""

    gc_every_n_steps: int = 1000
    """The frequency of steps at which we collect garbage with `gc.collect()`."""

    seed: int = 2
    """The RNG seed to use while starting the job."""

    debug: bool = False
    """If ``True``, runs the trainer in debug mode"""

    profile: bool = False
    """If ``True``, runs the PyTorch profiler at the beginning of the training."""

    profiler_skip_first: int = 200

    profiler_active: int = 3
    """If profiling (``profile = True``), The profiler will skip the first ``skip_first`` steps, then do the active recording for the next ``active`` steps
    If planning to visualize the trace with tensorbaord, then ``active`` should be small (less than 10 steps), otherwise tb won't load!
    """
    loss_scaler_init_scale: float = 2.0**15
    """The initial scale for the gradient scaler, fairseq2's default is 2.0**15"""

    loss_scaler_scale_window: Optional[int] = None
    """The number of consecutive optimizer steps without inf/NaN gradients that must occur for the scale to be updated"""

    use_fsdp: bool = True
    """If ``True``, uses FSDP instead of DDP."""

    use_autocast: bool = False
    """If ``True``, wrap the forward pass in AMP autocast context.
        autocast is only needed if training with mixed precision.
        If training fails without it, check if some module with its weights is not properly cast
    """

    fsdp_wrap_granularity: SUPPORTED_FSDP_WRAP_POLICIES = "model"
    """The granularity at which to wrap the model."""

    fsdp_memory_policy: SUPPORTED_FSDP_MEMORY_POLICIES = "standard"
    """The FSDP memory policy."""

    fsdp_fp32_reduce: bool = False
    """ If ``True``, the gradients will be reduced in full precision even when dtype is `torch.float16`"""

    use_submitit: bool = True
    """If ``True``, setup the environment ti use submitit."""

    fake_gang_device: Optional[str] = None
    """If non-empty, the trainer will be set locally on a device, instead of distributed training."""

    experiment_name: Optional[str] = None
    """experiment name for job trackin, if None default to StopesModule naming"""

    raise_oom: bool = False
    """If ``True``, raise OOM errors when they occur, if ``False`` give it another try."""

    raise_nan_or_inf: bool = False
    """If ``True``, raise FloatingPointError with Nan/Inf losses, if ``False`` give it another try."""

    max_ooms: int = 10
    """If ```raise_oom`` is False, how many OOMs we can tolerate per rank before raising an error."""

    max_nans_or_infs: int = 10
    """If ```raise_nan_or_inf`` is False, how many Nan/Infs we can tolerate per rank before raising an error."""

    freeze_modules: Optional[List[str]] = None
    """Name of modules in the model to be frozen when training/finetuning"""

    freezing_strategy: Literal["none", "modules", "ffn", "ffn-adaln", "adaln"] = "none"
    """
    Freezing strategy to follow. Options are:
        1. none: Nothing will be frozen (default)
        2. modules: A list of modules to freeze will be read from `freeze_modules`
        3. ffn: All ffn sub-modules will be frozen
        4. ffn-adaln: all FFN and Adaln sub-modules will be frozen.
    """


class Trainer(StatefulObjectBag):
    config: TrainingConfig
    model: Module
    training_data_loader: Any
    validation_data_loader: Optional[Any]
    gang: Gang
    optimizer: Optimizer
    loss_scaler: DynamicLossScaler
    lr_scheduler: AbstractLRScheduler
    rng_bag: RngBag
    step_nr: int
    train_metric_bag: MetricBag
    valid_metric_bag: Mapping[str, MetricBag]
    metric_recorders: List[MetricRecorder]
    profiler: Profiler
    stopwatch: Stopwatch
    criterion: Any
    card_metdata: Dict
    _train_step_time: float
    _valid_step_time: float

    def __init__(
        self,
        config: TrainingConfig,
        model: Module,
        training_data_loader: Any,
        validation_data_loader: Optional[Any],
        gang: Gang,
        checkpoint_manager: FileCheckpointManager,
        rng_bag: RngBag,
        stopwatch: Stopwatch,
        card_metadata: Dict,
    ) -> None:
        super().__init__()

        self.config = config

        if self.config.debug:
            logger._logger.setLevel(logging.DEBUG)
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        self.card_metadata = card_metadata

        self.dtype = eval(config.dtype)

        self.model = model

        self.training_data_loader = training_data_loader

        # Skip saving and loading the state of validation dataloader
        self.register_non_stateful("validation_data_loader", validation_data_loader)

        self.gang = gang

        self.rng_bag = rng_bag

        self.step_nr = 1

        self.current_run_steps = 0

        self.checkpoint_manager = checkpoint_manager

        tb_dir = config.tb_dir or config.output_dir.joinpath("tb")

        self.metric_recorders = [LogMetricRecorder(logger)]

        if gang.rank == 0:
            self.metric_recorders.append(TensorBoardRecorder(tb_dir))
            self.metric_recorders.append(
                LCMWandBRecorder(
                    name=config.wandb_run_name,
                    project=config.wandb_project or "uncategorized",
                    output_dir=config.output_dir / "wandb",
                    config=self._tb_flat_config,
                )
            )

        self.profiler = Profiler(
            skip_first=config.profiler_skip_first,
            active=config.profiler_active,
            log_dir=tb_dir,
            gang=gang,
            enabled=config.profile,
        )

        self.stopwatch = stopwatch
        self._train_step_time = 0.0
        self._valid_step_time = 0.0

        self.criterion = None  # type: ignore

        self.loss_scaler = None  # type: ignore

    @property
    def is_fsdp(self) -> bool:
        return isinstance(self.model, FSDP)

    @property
    def is_ddp(self) -> bool:
        return isinstance(self.model, DDP)

    def setup(self) -> None:
        self.criterion = self.setup_criterion()

        self.setup_metric_bags()

        # Add the grad_norm metric to the training metric bag
        self.train_metric_bag.register_metric(
            "grad_norm", Mean(device=self.gang.device), persistent=False
        )
        self.train_metric_bag.register_metric(
            "raw_grad_norm", Mean(device=self.gang.device), persistent=False
        )

        self.setup_optimizer_and_lr_schedule()

    def setup_optimizer_and_lr_schedule(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            betas=tuple(self.config.adam_betas),  # type: ignore
            eps=self.config.adam_eps,
            use_fp32=self.config.use_optimizer_in_fp32,
            weight_decay=self.config.weight_decay,
        )
        logger.info(
            (
                f"Setting up AdamW optimizer with betas={self.config.adam_betas}, "
                f"base lr={self.config.lr} and weight decay={self.config.weight_decay} "
                f"and use_fp32={self.config.use_optimizer_in_fp32}"
            )
        )

        self.register_stateful("optimizer", optimizer)

        self.loss_scaler = DynamicLossScaler(
            optimizer,
            gang=self.gang,
            init_scale=self.config.loss_scaler_init_scale,
            min_scale=0.0001,
            scale_window=self.config.loss_scaler_scale_window,
            enabled=self.dtype == torch.float16,
        )

        if self.loss_scaler.is_enabled:
            logger.info(
                f"Initializing DynamicLossScaler with init_scale={self.config.loss_scaler_init_scale}"
            )

        lr_scheduler = build_lr_scheduler(
            optimizer=self.optimizer,
            schedule=self.config.lr_schedule,
            lr=self.config.lr,
            warmup_steps=self.config.num_lr_warmup_steps,
            start_lr=self.config.start_lr,
            final_lr=self.config.final_lr,
            max_steps=self.config.max_steps,
            stage_ratio=tuple(self.config.lr_stage_ratios),
        )

        # Saving the lr_scheduler as well to properly resume training
        self.register_stateful("lr_scheduler", lr_scheduler)

    @abstractmethod
    def setup_criterion(self):
        """Define a criterion (loss / objective function to optimize)"""

    def setup_metric_bags(self):
        """Setup metric bags for tracking"""

        self.train_metric_bag = MetricBag(self.gang)

        self.register_non_stateful(
            "valid_metric_bag",
            {
                ds_name(dataset): MetricBag(self.gang)
                for dataset in self.config.validation_data
            },
        )

    def checkpoint_and_raise(self, exc) -> None:
        # Checkpoint before exiting
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.warning(f"R{self.gang.rank} checkpoint_and_raise - error={exc}")
        if self.current_run_steps > 100:
            # avoid checkpoining for early failures
            self._checkpoint(crash=exc)
        raise exc

    @cached_property
    def _tb_flat_config(self):
        """
        Prepare the flat config that will be used as HParams
        to record training metadata, namely config and environment hashes.
        """

        dict_config = flatten_dict(asdict(self.config))

        # Merge the data lists:
        def get_data_signature(dataset):
            return ":".join(
                map(str, (dataset["name"], dataset["weight"], dataset["filters"]))
            )

        dict_config["training_data"] = "+".join(
            get_data_signature(dataset) for dataset in dict_config["training_data"]
        )
        dict_config["validation_data"] = "+".join(
            get_data_signature(dataset) for dataset in dict_config["validation_data"]
        )

        # value should be one of int, float, str, bool, or torch.Tensor
        allowed_types = (int, float, str, bool, torch.Tensor)
        config_keys = list(dict_config)
        for k in config_keys:
            if not isinstance(dict_config[k], allowed_types):
                del dict_config[k]

        return dict_config

    def run(self) -> None:
        """Run the trainer for up to `max_steps`"""

        logger.info(f"Running training on {self.gang.size} device(s).")

        data_iter = self.training_data_loader.iterate_batches()

        logger.info(
            f"R{self.gang.rank} - waiting for all ranks to prepare a data iterator!"
        )
        self.gang.barrier()

        # These counters are rank-specific
        ooms, nans_or_infs = 0, 0

        # TODO: validate before training
        # logger.info(f"Starting with validation at step={self.step_nr}")
        # self._validate()

        with self.profiler:
            while self.step_nr <= self.config.max_steps:
                with record_function(f"step_{self.step_nr}"):
                    try:
                        # Main training step: forward -> backward -> optimizer.step -> log
                        stepped = self._train_step(data_iter)

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            self._log_oom(e)
                            ooms += 1
                            if self.config.raise_oom or ooms > self.config.max_ooms:
                                # Previous behaviour, no retries but still checkpointing
                                self.checkpoint_and_raise(e)

                            logger.warning(
                                f"Attempting to recover from OOM on R{self.gang.rank} (OOMS={ooms})"
                            )
                            stepped = True
                            # reset optimizer
                            self.optimizer.zero_grad(set_to_none=True)

                            # rollback updates
                            self.train_metric_bag.rollback_updates()

                            # Empty CUDA cache before trying again
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        else:
                            # Other RuntimeErrors
                            self.checkpoint_and_raise(e)

                    except FloatingPointError as e:
                        if "Losses are Nan/Inf" in str(e):
                            self._log_nan_loss(e)
                            nans_or_infs += 1
                            if (
                                self.config.raise_nan_or_inf
                                or nans_or_infs > self.config.max_nans_or_infs
                            ):
                                self.checkpoint_and_raise(e)

                            logger.warning(
                                f"Attempting to recover from NaN/Inf loss on R{self.gang.rank} (NaNs/Infs={nans_or_infs})"
                            )
                            stepped = True
                            # reset optimizer
                            self.optimizer.zero_grad(set_to_none=True)

                            # rollback updates
                            self.train_metric_bag.rollback_updates()

                        else:
                            # Other FloatingPointErrors
                            self.checkpoint_and_raise(e)

                    except Exception as e:
                        self.checkpoint_and_raise(e)

                if stepped:
                    if self._should_publish_train_metrics():
                        self._publish_train_metrics()

                    if self._should_checkpoint():
                        self._checkpoint()

                    if self._should_validate():
                        self._validate()

                    if self._should_collect_garbage():
                        self._collect_garbage()

                    self.profiler.step()

                    self.step_nr += 1
                    self.current_run_steps += 1

                else:
                    logger.info(f"R{self.gang.rank} - Resetting the datapipeline")
                    self.training_data_loader.pipeline.reset()

                    logger.info(f"R{self.gang.rank} - Done resetting the datapipeline")
                    data_iter = self.training_data_loader.iterate_batches()

        self._save_model_card_for_last_checkpoint(to_checkpoint_dir=False)
        logger.info(f"Finished training after {self.step_nr - 1} step(s).")

        self.gang.close()

    def restore(self) -> None:
        logger.info("Attempting to load last checkpoint.")

        step_nr, checkpoint = self.checkpoint_manager.load_last_checkpoint()

        logger.info(f"Checkpoint loaded, restoring training from step {step_nr}.")

        self.load_state_dict(checkpoint)

        self.gang.barrier()

        logger.info("Training restored, resuming.")

        self.step_nr = step_nr + 1

    def _maybe_with_autocast(self) -> ContextManager[None]:
        # autocast is only needed if training with mixed precision.
        # If training fails without it, check if some module with its weights
        # is not properly cast
        if self.config.use_autocast:
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        else:
            return nullcontext()

    def _train_step(self, data_iter: Iterator) -> bool:
        step_nr = self.step_nr

        step_stopwatch = Stopwatch(start=True, device=self.gang.device)

        stepped = False

        # We have to retry the step in case of a gradient overflow.
        while not stepped:
            batches = []

            # Collect batches.
            with record_function(f"step_{step_nr}_data_load"):
                for _ in range(self.config.gradient_accumulation):
                    try:
                        batches.append(next(data_iter))
                    except StopIteration:
                        break

            if len(batches) != self.config.gradient_accumulation:
                logger.info(
                    f"R{self.gang.rank} -End of data reached at training step {step_nr}."
                )

                return False

            # create a copy of the current metrics
            # any update to the metrics from this point will either be committed with `commit_updates`
            # or ignored with `rollback_updates`
            self.train_metric_bag.begin_updates()

            num_targets = 0

            # Accumulate gradients.
            for batch_nr, batch in enumerate(batches):
                with self._maybe_no_sync(batch_nr, len(batches)):
                    with record_function(f"step_{step_nr}_{batch_nr}_forward"):
                        # autocast should wrap only the forward pass(es)
                        # of your network, including the loss computation(s).
                        # Backward passes under autocast are not recommended.
                        with self._maybe_with_autocast():
                            loss = self.criterion(batch)

                    if not (
                        torch.isfinite(loss.value).all() or self.loss_scaler.is_enabled
                    ):
                        raise FloatingPointError("Losses are Nan/Inf.")

                    # update metrics
                    self.train_metric_bag.update([loss])

                    with record_function(f"step_{step_nr}_{batch_nr}_backward"):
                        self.loss_scaler.backward(loss.value)

                    num_targets += loss.num_target_elements

            # Record and clip gradient norm
            grad_norm, raw_grad_norm = self.process_gradients(step_nr, num_targets)

            # Update parameters.
            with record_function(f"step_{step_nr}_optimizer"):
                # scale_result: LossScaleResult(old_scale: float, new_scale: float, overflow: bool, min_reached: bool)
                _, scale_result = self.loss_scaler.run_optimizer_step(step_nr)

            if scale_result.overflow:
                # Walk back the metrics update:
                self.train_metric_bag.rollback_updates()
                logger.debug(
                    f"R{self.gang.rank} rolled back update {self.train_metric_bag._original_metrics is None}"
                )

                if scale_result.min_reached:
                    logger.error(f"Loss has started exploding at step {step_nr}. Stopping training.")  # fmt: skip

                    raise FloatingPointError("The training loss has exploded.")

                logger.debug(f"Repeating training step {step_nr}.")

            else:
                self.lr_scheduler.step()

                stepped = True

            # Reset.
            self.optimizer.zero_grad(set_to_none=True)

        # Stepped = True:
        with record_function(f"step_{step_nr}_metrics"):
            # do something with losses and grad_norm

            self.train_metric_bag.commit_updates()

            # gradient norm is common to workers
            self.train_metric_bag.grad_norm.update(grad_norm)
            self.train_metric_bag.raw_grad_norm.update(raw_grad_norm)

            if self.gang.rank == 0:
                # update elapsed time once
                self._train_step_time += step_stopwatch.get_elapsed_time()

        del batches
        return stepped

    def _maybe_no_sync(self, batch_nr: int, num_batches: int) -> ContextManager[None]:
        if batch_nr < num_batches - 1 and self.gang.size > 1:
            return self.model.no_sync()
        return nullcontext()

    def normalize_gradients(self, num_targets: int) -> None:
        """
        :param num_target:
            The number of targets used in loss computation in this process.

        If reduction = sum:
            similar to fairseq2's `normalize_gradients`, will normalize the gradients of the model by ``world_size/num_targets``.
        If reduction = mean:
            will simply multiply by world size i.e undo DDP/FSDP's default normalization
        """
        reduction = self.criterion.reduction
        if reduction == "sum":
            total_num_targets = torch.tensor(
                num_targets, device=self.gang.device, dtype=torch.int64
            )

            self.gang.all_reduce(total_num_targets, ReduceOperation.SUM)

            # Both DDP and FSDP divide gradients by the world size which we also undo.
            if total_num_targets > 0:
                grad_scale = self.gang.size / total_num_targets
            else:
                # If total_num_targets == 0, gradients will be zeroes anyway
                grad_scale = self.gang.size

        else:
            grad_scale = self.gang.size

        scale_gradients(self.model, grad_scale)

    def process_gradients(
        self, step_nr: int, num_targets: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with record_function(f"step_{self.step_nr}_process_grads"):
            # Normalize gradients
            """
            Normalize and clip the gradients
            """
            # this raw grad norm is only used for debugging
            raw_grad_norm = clip_gradient_norm(
                self.model,
                max_norm=None,
            )

            if not self.config.turn_off_grad_normalization:
                self.normalize_gradients(num_targets=num_targets)

            # undo the GradScaler's scaling before clipping
            self.loss_scaler.unscale_gradients_()

            # Clip gradients
            # If DDP, we use torch.nn.utils.clip_grad_norm_, if FSDP,
            # we use torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_
            # this method handles the fact that gradients might be sharded across ranks.
            grad_norm = clip_gradient_norm(
                self.model,
                max_norm=self.config.max_grad_norm,
            )

            # Check for gradient consistency across workers:
            if not check_gradient_norms(grad_norm, self.gang, step_nr):
                raise FloatingPointError(
                    f"The gradients are inconsistent between processes at step {step_nr}. Training cannot continue."
                )

        return grad_norm, raw_grad_norm

    def _should_validate(self) -> bool:
        return self._should_do(self.config.validate_every_n_steps)

    def _should_collect_garbage(self) -> bool:
        return self._should_do(self.config.gc_every_n_steps)

    def _collect_garbage(self):
        logger.info("Collecting garbage...")
        gc.collect()

    @torch.inference_mode()
    def _validate(self) -> None:
        gc.collect()
        torch.cuda.empty_cache()

        if self.validation_data_loader is None:
            logger.info("Skip validation as the data loader is empty")
            return

        self.model.eval()

        logger.info(f"Starting validation after step {self.step_nr}.")

        self.validation_data_loader.pipeline.reset()

        data_iter = self.validation_data_loader.iterate_batches()
        data_dummy_iter = self.validation_data_loader.iterate_dummy_batches()

        logger.info(f"R{self.gang.rank} done creating the validation data iterator")

        for step_nr in count(start=1):
            step_stopwatch = Stopwatch(start=True, device=self.gang.device)

            try:
                batch = next(data_iter)
                true_batch = 1
            except StopIteration:
                batch = next(data_dummy_iter)
                true_batch = 0

            total_nb_batches = all_sum(self.gang, true_batch)

            if bool(total_nb_batches == 0):
                break
            # we apply model for all workers to avoid process groups sync issues
            loss = self.criterion(batch)

            if true_batch:
                self._valid_step_time += step_stopwatch.get_elapsed_time()
                self.valid_metric_bag[batch.name].update([loss])

        self._publish_validation_metrics()

        logger.info(
            f"R{self.gang.rank} Validation complete in {step_nr} steps, resuming training."
        )

        self.model.train()

    def _should_publish_train_metrics(self) -> bool:
        return self._should_do(self.config.publish_metrics_every_n_steps)

    def _set_elements_per_second(
        self, metric_values: Dict[str, Any], elapsed_time: float
    ) -> None:
        try:
            num_elements = metric_values[self.criterion.throughput_metric_name]
        except KeyError:
            return

        if not isinstance(num_elements, (int, float, torch.Tensor)):
            return

        if elapsed_time == 0.0:
            metric_values["elements_per_second"] = 0.0
        else:
            metric_values["elements_per_second"] = num_elements / elapsed_time

    def _publish_train_metrics(self) -> None:
        values = self.train_metric_bag.sync_and_compute_metrics()

        self.train_metric_bag.reset_non_persistent_metrics()

        # Only rank-0 to record and publish
        # since sync_and_compute_metrics's recipient rank is 0
        if self.gang.rank != 0:
            return

        assert values is not None

        values["lr"] = get_effective_lr(self.lr_scheduler)

        self._set_elements_per_second(values, self._train_step_time)

        if self.loss_scaler.is_enabled:
            values["grad_scale"] = self.loss_scaler.get_scale()

        values["wall_time"] = self.stopwatch.get_elapsed_time()
        values["elapsed_time"] = self._train_step_time

        record_metrics(self.metric_recorders, "Train", values, self.step_nr)

        self._train_step_time = 0.0

    def _publish_validation_metrics(self) -> None:
        values = {}
        for name, metric_bag in self.valid_metric_bag.items():
            values[name] = metric_bag.sync_and_compute_metrics()
            metric_bag.reset_non_persistent_metrics()

        # Only rank-0 to record and publish
        if self.gang.rank != 0:
            return

        for name, val in values.items():
            assert val is not None
            self._set_elements_per_second(val, self._valid_step_time)
            val["elapsed_time"] = self._valid_step_time
            val["wall_time"] = self.stopwatch.get_elapsed_time()
            valid_name = f"Valid | {name}"
            record_metrics(self.metric_recorders, valid_name, val, self.step_nr)

        # reset timers
        self._valid_step_time = 0.0

    def _should_checkpoint(self) -> bool:
        return self._should_do(self.config.checkpoint_every_n_steps)

    def _should_save_consolidated_model(self) -> bool:
        return self.is_fsdp and self._should_do(self.config.save_model_every_n_steps)

    def _checkpoint(self, crash=None) -> None:
        logger.info(f"Saving checkpoint at step {self.step_nr}")
        checkpoint = self.state_dict()

        metadata = {
            "config": self.config,
            "crash": crash,
        }

        self.checkpoint_manager.begin_checkpoint(self.step_nr)

        if self.is_fsdp:
            replicated_keys = None
        elif self.is_ddp:
            # If we do not shard, save the model and the optimizer only on rank 0.
            replicated_keys = {"model", "optimizer"}
        else:
            replicated_keys = {"*"}

        self.checkpoint_manager.save_state(checkpoint, replicated_keys=replicated_keys)

        self.checkpoint_manager.save_metadata(metadata)

        if self._should_save_consolidated_model():
            self._save_consolidated_model()

        # Create a model card only after creating model.pt
        # i.e., regular checkpointing with DDP or after consolidation with FSDP
        if not self.is_fsdp:
            self._save_model_card_for_last_checkpoint(to_checkpoint_dir=True)

        self.checkpoint_manager.commit_checkpoint()

        # Note that this logic looks at saved directories regardless of
        # the nature of the checkpointing, consolidated or not
        if self.config.keep_last_n_checkpoints != -1:
            self.checkpoint_manager.keep_last_n_checkpoints(
                self.config.keep_last_n_checkpoints,
                preserve_model=self.config.preserve_consolidated_models,
            )

        logger.info(f"Checkpoint saved by worker @rank={self.gang.rank}")

    def _save_consolidated_model(self) -> None:
        logger.info(f"Saving consolidated model at step {self.step_nr}.")
        self.checkpoint_manager.save_consolidated_fsdp_model(self.model)
        self._save_model_card_for_last_checkpoint(to_checkpoint_dir=True)
        logger.info("Consolidated model saved.")

    def _should_do(self, n_step: int) -> bool:
        return self.step_nr % n_step == 0

    def create_model_card_for_last_checkpoint(
        self, is_final: bool = False, **card_kwargs
    ) -> Optional[AssetCard]:
        """Create a model card based on the last saved checkpoint and the model config."""
        logger.warning(
            "Could not create a model card with a generic trainer.  Please use a model-specific one."
        )
        return None

    def _save_model_card_for_last_checkpoint(
        self, to_checkpoint_dir: bool = False
    ) -> None:
        """Save the model card for the last checkpoint to the checkpoint directory or the core output directory."""
        if self.gang.rank != 0:
            return

        if to_checkpoint_dir:
            current_step_nr = self.checkpoint_manager._checkpoint_step_nr
            output_dir = self.checkpoint_manager._checkpoint_dir.joinpath(
                f"step_{current_step_nr}.tmp"
            )
        else:
            output_dir = self.config.output_dir

        card = self.create_model_card_for_last_checkpoint(
            is_final=not to_checkpoint_dir
        )

        if card is not None:
            card_data = card._metadata  # TODO: use the exposed attribute when available
            with open(output_dir / "model_card.yaml", "w", encoding="utf-8") as outfile:
                yaml.dump(card_data, outfile, default_flow_style=False)
            logger.info(f"Model card saved in {output_dir}")

    def _log_oom(self, exc):
        logger.warning(
            f"OOM: Ran out of memory on R{self.gang.rank} with exception: {exc}"
        )

        if torch.cuda.is_available():
            for device_idx in range(torch.cuda.device_count()):
                logger.warning(torch.cuda.memory_summary(device=device_idx))

        sys.stderr.flush()

    def _log_nan_loss(self, exc):
        logger.warning(f"We hit a Nan/Inf Loss: raised with exception: {exc}")


class TrainerBuilder:
    def __init__(self, config: TrainingConfig):
        assert config.save_model_every_n_steps % config.checkpoint_every_n_steps == 0, (
            f"save_model_every_n_steps={config.save_model_every_n_steps} for saving consolidated models should be a multiplier of checkpoint_every_n_steps={config.checkpoint_every_n_steps}"
        )

        self.config = config

        self.stopwatch = Stopwatch(start=True)

        # In case we train on Ampere or later, use TF32.
        torch.set_float32_matmul_precision("high")

        if self.config.fake_gang_device is None:
            # By default, we work with a process group
            self.gang = init_process_group(config, logger=logger._logger)
        else:
            # For testing purposes, we use a fake gang on the chosen device
            self.gang = FakeGang(device=torch.device(self.config.fake_gang_device))

        self.gang_rank = self.gang.rank if self.gang else 0

        if self.gang.device.type == "cuda":
            # Setup ATEN and NCCL logging if in debug mode
            self._setup_additional_logging()

            # Dump environment variables:
            log_env_variables(self.gang.device)

        # A variable to carry fields necessary to build concise model cards
        self.card_metdata: Dict = {}

        if self.gang_rank == 0:
            logger.info(f"Job Config\n{pformat(config)}")

        self.device = self.gang.device

        rng_bag = RngBag.from_device_defaults(self.device)

        # Ensure that each run has deterministic behavior.
        rng_bag.manual_seed(config.seed)

        self.rng_bag = rng_bag

        self.dtype = eval(config.dtype)

        self.finetune: bool = False

        self.has_checkpoint: bool = False

    @property
    @abstractmethod
    def model_loader(self):
        """A fairseq2 ModelLoader"""

    @property
    def model_config_loader(self):
        """A fairseq2 ConfigLoader"""
        return self.model_loader._config_loader

    @abstractmethod
    def load_data(self):
        """Load training and validation data
        Returns one loader for training data and one for validation data
        """

    def create_model_config(self, set_finetune_flag: bool = False):
        """
        Given `model_config_or_name`, `model_arch` and `model_arch_overrides`
        create the model config dict
        if `set_finetune_flag` is `True` then the trainer's finetune flag will be set
        here inferred from the use of `model_config_or_name`
        """
        if self.config.model_config_or_name is not None:
            assert self.config.model_arch is None, (
                "We cannot set both `model_config_or_name` and `model_arch`"
            )

            if isinstance(self.config.model_config_or_name, str):
                # The config of a registered model i.e. we're finetuning
                logger.info(
                    f"Loading pretrained model from {self.config.model_config_or_name}"
                )

                model_config = self.model_config_loader(
                    self.config.model_config_or_name
                )
                finetune = True

                # Metadata for card creation
                source_card = self.model_config_loader._asset_store.retrieve_card(
                    self.config.model_config_or_name
                )
                try:
                    arch = source_card.field("model_arch").as_(str)
                except AssetCardFieldNotFoundError:
                    arch = None

                self.card_metadata = {
                    "model_config": model_config if arch is None else None,
                    "model_type": model_config.model_type,
                    "model_arch": arch,
                }

            else:
                # model_config_or_name is a dataclass
                logger.info(
                    "Creating a model from the provided config in model_config_or_name"
                )
                model_config = self.config.model_config_or_name

                self.card_metadata = {
                    "model_config": model_config,
                    "model_type": model_config.model_type,
                    "model_arch": None,
                }

                finetune = False

        elif self.config.model_arch is not None:
            assert (
                self.config.model_arch in self.model_config_loader._arch_configs.names()
            ), (
                f"Could not recognise {self.config.model_arch} as a registered architecture "
            )

            logger.info(
                f"Creating a model from registered arch {self.config.model_arch}"
            )

            finetune = False
            model_config = self.model_config_loader._arch_configs.get(
                self.config.model_arch
            )
            self.card_metadata = {
                "model_config": None,
                "model_type": model_config.model_type,
                "model_arch": self.config.model_arch,
            }

        # In all setups we can override some config parameters
        if self.config.model_arch_overrides is not None:
            try:
                update_dataclass(model_config, self.config.model_arch_overrides)

            except (TypeError, ValueError) as ex:
                raise ValueError(
                    "The model_arch_overrides contain one or more invalid keys"
                ) from ex

            self.card_metadata["model_arch"] = None
            self.card_metadata["model_config"] = model_config

            logger.info(
                f"Overwriting model config parameters with {self.config.model_arch_overrides}"
            )

        if set_finetune_flag:
            self.finetune = finetune

        return model_config

    def create_model(self):
        """
        Load the model to be trained.
        In case other models are developed following a different paradigm, we can create
        corresponding trainers by overriding `create_model`
        """
        logger.info("Initializing model.")

        model_config = self.create_model_config(set_finetune_flag=True)

        if self.gang_rank == 0:
            logger.info(f"Final model config:\n{pformat(model_config)}")

        model = self.model_loader._factory(
            model_config,
            device=self.device,
            dtype=self.dtype,
        )
        # log model before any wrapping:
        log_model(model, logger)

        return model

    def wrap_model_with_ddp(self, model) -> DDP:
        """Wrap the model with DDP"""

        try:
            ddp_model = to_ddp(
                model,
                self.gang,
            )

        except ValueError:
            logger.warning(
                "Using pytorch DDP instead of fairseq's `to_ddp`\
                - please check fairseq2 after a3de79dcc6a4ea34cde644e15b4056f1a808a6a8"
            )

            ddp_model = DDP(model)

        if self.gang_rank == 0:
            log_model(ddp_model, logger)

        return ddp_model

    def wrap_model_with_fsdp(self, model) -> FSDP:
        """Wrap the model with FSDP."""

        wrap_policy, ignored_modules = get_fsdp_wrap_policy(
            model, wrap_granularity=self.config.fsdp_wrap_granularity
        )
        memory_policy = get_fsdp_memory_policy(policy=self.config.fsdp_memory_policy)

        if self.dtype == torch.float32:
            mixed_precision_dtype = None
        else:
            mixed_precision_dtype = self.dtype

        skip_init = False
        broadcast_state = self.finetune and not self.has_checkpoint
        fp32_reduce = self.config.fsdp_fp32_reduce

        if self.gang.rank == 0:
            logger.info(
                (
                    f"FSDP init with: \n--- ignored_modules={ignored_modules}"
                    f"\n--- wrap_policy={wrap_policy}"
                    f"\n--- mixed_precision_dtype={mixed_precision_dtype}"
                    f"\n--- skip_init={skip_init}"
                    f"\n--- broadcast_state (FSDP's sync_module_states)={broadcast_state}"
                    f"\n--- fp32_reduce={fp32_reduce}"
                    f"\n--- memory_policy={memory_policy}"
                )
            )

        fsdp_model = to_fsdp(
            model,
            self.gang,
            wrap_policy,
            mixed_precision_dtype=mixed_precision_dtype,
            ignored_modules=ignored_modules,
            fp32_reduce=fp32_reduce,
            skip_init=skip_init,
            broadcast_state=broadcast_state,
            memory_policy=memory_policy,
        )

        if self.gang_rank == 0:
            log_model(fsdp_model, logger)

        return fsdp_model

    def maybe_load_model(self, model):
        """
        If we are finetuning and we don't have a checkpoint,
        load the pre-trained model and broadcast it to
        all gang processes from rank 0.
        """
        if not self.has_checkpoint and self.finetune:
            logger.info(f"Loading for finetuning: {self.config.model_config_or_name}")

            if self.gang_rank == 0:
                pretrained_model = self.model_loader(
                    model_name_or_card=self.config.model_config_or_name,
                    device=self.gang.device,
                    dtype=self.dtype,
                )  # type: ignore[arg-type]

                try:
                    model.load_state_dict(
                        pretrained_model.state_dict(),
                        strict=True,
                        assign=False,
                    )
                except (KeyError, ValueError) as ex:
                    raise ValueError(
                        f"The model state form {self.config.model_config_or_name} "
                        "cannot be loaded. See nested exception for details."
                    ) from ex

            self.gang.barrier()

            to_device(model, self.gang.device)

            logger.info(
                f"Done loading model for finetuning: {self.config.model_config_or_name}"
            )

        return model

    def maybe_freeze_parameters(self, model):
        assert (self.config.freezing_strategy == "modules") == (
            self.config.freeze_modules is not None
        ), (
            "For the `modules` freezing_strategy, we need a list of `freeze_modules`. "
            "If `freeze_modules` is provided, make sure to use freezing_strategy=modules"
        )

        if self.config.freezing_strategy == "none":
            return model

        if self.config.freezing_strategy == "modules":
            # Optionally freeze the parameters of sub-modules:
            if self.config.freeze_modules is not None:
                for module in self.config.freeze_modules:
                    logger.info(f"... Freezing module={module}")
                    freeze_parameters(getattr(model, module))
            return model

        if self.config.freezing_strategy == "ffn":
            for name, m in _get_named_modules(model):
                if "ffn" in name:
                    logger.info(f"... Freezing module={name}")
                    freeze_parameters(m)
            return model

        if self.config.freezing_strategy == "adaln":
            for name, m in _get_named_modules(model):
                if "modulator" in name:
                    logger.info(f"... Freezing module={name}")
                    freeze_parameters(m)
            return model

        if self.config.freezing_strategy == "ffn-adaln":
            for name, m in _get_named_modules(model):
                if "modulator" in name or "ffn" in name:
                    logger.info(f"... Freezing module={name}")
                    freeze_parameters(m)
            return model

        raise ValueError(f"Unknown freezing stratgey {self.config.freezing_strategy}")

    def _setup_additional_logging(self):
        if self.config.debug:
            assert self.config.log_folder is not None, (
                "Missing log_folder, \
            make sure the log_folder is properly set in the training config"
            )
            setup_additional_logging(log_folder=self.config.log_folder)

    @property
    def use_fsdp(self) -> bool:
        return self.config.use_fsdp

    @property
    def use_ddp(self) -> bool:
        """
        Whether DDP should be used.
        if selg.gang.size == 1: single worker, no parallelism
        if use_fsdp:  use FSDP instead
        """
        return not (self.gang.size == 1 or self.use_fsdp)

    @abstractmethod
    def build_trainer(self):
        """Build the trainer by loading data and
        setting up the model for training

        Returns trainer
        """
