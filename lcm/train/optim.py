# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
#

from typing import Tuple

from fairseq2.logging import get_log_writer
from fairseq2.optim.lr_scheduler import (
    AbstractLRScheduler,
    CosineAnnealingLR,
    MyleLR,
    NoopLR,
    PolynomialDecayLR,
    TriStageLR,
)
from torch.optim import Optimizer

logger = get_log_writer(__name__)


def build_lr_scheduler(
    optimizer: Optimizer,
    lr: float,
    warmup_steps: int,
    start_lr: float = 1e-7,
    final_lr: float = 1e-5,
    max_steps: int = 10_000,
    stage_ratio: Tuple[float, ...] = (0.1, 0.4, 0.5),
    schedule: str = "myle",
) -> AbstractLRScheduler:
    assert schedule in [
        "noop",
        "myle",
        "cosine",
        "wsd",
        "polynomial",
    ], (
        f"Cannot recognize the learing rate schedule {schedule}, only noop, myle, cosine and wsd are supported"
    )

    assert lr > 0, "The learning reate should be strictly positive"

    lr_scheduler: AbstractLRScheduler

    if schedule == "noop":
        lr_scheduler = NoopLR(optimizer)

    elif schedule == "myle":
        lr_scheduler = MyleLR(
            optimizer,
            num_warmup_steps=warmup_steps,
            start_lr=[start_lr],
        )

    elif schedule == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            cycle_len=max_steps - warmup_steps + 1,
            num_warmup_steps=warmup_steps,
            start_lr=[start_lr],
            final_lr=[final_lr],
            cycle_mul=1.0,
            lr_mul=1.0,
        )

    elif schedule == "wsd":
        assert lr > start_lr, (
            f"the starting learning rate {start_lr} should be lesser than the main lr {lr}"
        )
        start_lr_scale = start_lr / lr

        assert lr > final_lr, (
            f"the final learning rate {final_lr} should be lesser than the main lr {lr}"
        )
        final_lr_scale = final_lr / lr

        lr_scheduler = TriStageLR(
            optimizer,
            max_steps,
            stage_ratio=stage_ratio,  # type: ignore
            start_lr_scale=start_lr_scale,
            final_lr_scale=final_lr_scale,
        )

    elif schedule == "polynomial":
        lr_scheduler = PolynomialDecayLR(
            optimizer,
            max_steps,
            warmup_steps,
            power=200,
            start_lr=start_lr,
            final_lr=final_lr,
        )

    return lr_scheduler
