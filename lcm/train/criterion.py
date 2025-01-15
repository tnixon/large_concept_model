# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal

from fairseq2.logging import get_log_writer
from omegaconf import MISSING
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP

from lcm.train.metrics import LossTerm

logger = get_log_writer(__name__)


@dataclass
class CriterionConfig:
    """A dataclass for criterion parameters"""

    name: str = MISSING
    """Name of the criterion, a unique identifier used in the CriterionsFactory"""

    reduction: Literal["sum", "mean"] = "sum"
    """How to reduce the loss across samples"""


class Criterion:
    """And abstract class for training criterions"""

    def __init__(
        self,
        config: CriterionConfig,
        model: Module,
    ):
        self.config = config

        self.model = model

        self.summands: List[str] = []
        """ A list of loss term names to track during training.
            This will create metric bags for each
        """

        self.reduction = config.reduction

    @property
    def throughput_metric_name(self) -> str:
        return "num_target_elements"

    @property
    def base_model(self):
        """A pointer to the unwrapped model if training with FSDP/DDP"""
        if isinstance(self.model, (DDP, FSDP)):
            _model = self.model.module
        else:
            _model = self.model
        return _model

    @abstractmethod
    def __call__(self, batch) -> LossTerm:
        """
        Computes the loss given an input batch.
        The model's forward pass is performed here
        """


class CriterionsFactory:
    """Factory for LCM criterions"""

    registry: Dict[str, Any] = {}

    @classmethod
    def build_criterion(cls, name: str, **kwargs) -> Any:
        """build the criterion of choice from within the trainer"""

        criterion_class = cls.registry[name]

        criterion = criterion_class(**kwargs)

        return criterion

    @classmethod
    def register(cls, name: str) -> Callable:
        """decorator for adding criterions to the registry"""

        def inner_wrapper(wrapped_class: Criterion) -> Callable:
            assert name not in cls.registry, (
                f"{name} is already register as a criterion"
            )
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper
