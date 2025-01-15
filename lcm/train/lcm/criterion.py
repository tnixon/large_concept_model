# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from fairseq2.logging import get_log_writer
from torch import Tensor

from lcm.datasets.batch import LCMInput, LCMStyle
from lcm.models.abstract_lcm import AbstractLCModel
from lcm.models.sonar_normalizer import SonarNormalizer
from lcm.train.criterion import Criterion, CriterionConfig
from lcm.train.metrics import LossTerm

logger = get_log_writer(__name__)


def compute_standard_mse(
    flattened_predictions: Tensor,
    flattened_target: Tensor,
    scales: Optional[Tensor] = None,
    normalizer: Optional[SonarNormalizer] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Computes MSE loss between predictions and targets.
    Note that, unlike regular MSE with mean/sum reduction, we first sum across channels
    before later reducing in the criterion.

    Parameters:
        flattened_predictions (Tensor): The predictions in (N, C)
        flattened_target (Tensor): The targets in (N, C)
        scales (Optional[Tensor]): If not None, each channel will be weighted by the corresponding scale.
        epsilon: A small epsilon to be added before taking the square root of the l2 distance
        normalizer (Optional[SonarNormalizer]): If a normalizer is provided,
                the predictions and targets will first be denormalized before computing the RMSE loss

    Returns:
        mse (Tensor): the MSE loss with optional scaling
        plain_mse (Tensor): The MSE loss without any scaling (for logging)
    """

    assert flattened_predictions.dim() == 2, (
        "Expecting two-dimensional predictions and targets. ",
        f"Found targets in {flattened_target.size()} and ",
        f"predictions in {flattened_predictions.size()}",
    )

    assert flattened_predictions.shape == flattened_target.shape, (
        "Expecting predictions and targets of the same shape ",
        f"Received predictions {flattened_predictions.shape} and targets {flattened_target.shape}",
    )

    if scales is not None:
        assert scales.dim() == 1, (
            "Expecting a uni-dimensional tensor of scales ",
            f"Found a tensor with dimension {scales.dim()}",
        )
        assert len(scales) == flattened_target.shape[-1], (
            "The provided scales should have the same size as the target channels. ",
            f"Found {len(scales)} expected {flattened_target.shape[-1]}",
        )

    if normalizer is not None:
        assert hasattr(normalizer, "denormalize"), (
            "The provided normalizer has not method `denormalize`"
        )
        flattened_predictions = normalizer.denormalize(flattened_predictions)
        flattened_target = normalizer.denormalize(flattened_target)

    full_mse = torch.nn.functional.mse_loss(
        flattened_predictions, flattened_target, reduction="none"
    )
    plain_mse = full_mse.sum(dim=-1)

    if scales is not None:
        full_mse = full_mse * scales.unsqueeze(0)
        mse = full_mse.sum(dim=-1)
    else:
        mse = plain_mse
    return mse, plain_mse


@dataclass
class LCMCriterionConfig(CriterionConfig):
    compute_rmse: bool = True
    """If `True` take the square-root of MSE.
    This is for now `True` by default for backward compatibility"""


class LCMCriterion(Criterion):
    """And abstract class for the LCM's criterions"""

    config: LCMCriterionConfig

    def __init__(
        self,
        config: LCMCriterionConfig,
        model: AbstractLCModel,
        style: LCMStyle = LCMStyle.UNSUPERVISED,
    ):
        super().__init__(config, model)

        self.style = style

        # Summands for log/tb recorders
        self.summands = ["mse_loss", "reconstruction_loss"]

        self.normalize_in_criterion = (
            self.base_model.config.sonar_normalizer_name is not None
        )

    @property
    def sonar_normalizer(self) -> Optional[SonarNormalizer]:
        if hasattr(self.base_model, "sonar_normalizer"):
            return self.base_model.sonar_normalizer

        elif hasattr(self.base_model, "frontend") and hasattr(
            self.base_model.frontend, "sonar_normalizer"
        ):
            return self.base_model.frontend.sonar_normalizer

        else:
            logger.warning(
                "Couldn't find the model's `sonar_normalizer`, defaulting to None"
            )
            return None

    @property
    def throughput_metric_name(self) -> str:
        return "num_target_elements"

    @abstractmethod
    def __call__(self, batch: LCMInput) -> LossTerm:
        """
        Computes the loss given an input batch.
        The model's forward pass is performed here
        Input batch is LCMInput  (see `lcm.datasets.batch`):
        """
