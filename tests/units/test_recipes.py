# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import warnings
from pathlib import Path

import pytest
from fairseq2.assets.error import AssetError
from hydra import compose, initialize
from omegaconf import DictConfig
from torch.cuda import OutOfMemoryError

from lcm.train.common import Trainer, get_trainer

KEY_TRAINING_RECIPES = [
    "pretrain/two_tower",
    "pretrain/mse",
    "finetune/two_tower",
    "finetune/mse",
]


@pytest.mark.skip("Need to create a real datacards")
@pytest.mark.parametrize("conf_name", KEY_TRAINING_RECIPES)
def test_train_recipes(monkeypatch, conf_name, tmp_path, group="train"):
    """
    Make sure that the recipes are synced with changes from the trainers'
    signatures and the training configs
    """
    from lcm.utils.common import setup_conf
    from tests.common import DEBUG

    setup_conf()

    # The new dynamic loss scaler does not work with non-Cuda env, disable it
    monkeypatch.setattr(
        "lcm.train.trainer.Trainer.setup_optimizer_and_lr_schedule", lambda self: None
    )

    with initialize(
        version_base="1.2",
        config_path="../../recipes/train/",
    ):
        config = compose(
            config_name="defaults",
            overrides=[
                f"+{group}={conf_name}",
                f"trainer.output_dir={tmp_path}",
                f"++trainer.debug={DEBUG}",
                "++trainer.use_fsdp=false",
            ],
        )
        assert isinstance(config, DictConfig), (
            f"+{group}={conf_name} expect dict-type config, get {type(config)}."
        )

        try:
            trainer = get_trainer(config.trainer)
        except (ValueError, AssetError, OutOfMemoryError) as err:
            if isinstance(err, OutOfMemoryError):
                warnings.warn(
                    f"Ignoring the error because the model from {conf_name} is too big in the test machine"
                )
                return
            if isinstance(err, ValueError):
                main_errs = err.args[:1]
                main_err = " ".join(map(str, main_errs))
            else:
                main_err = err.args[0]
            if "The checkpoint" in main_err:
                warnings.warn(
                    f"Ignoring the error when initializing the trainer for recipe {conf_name}."
                    "Probably, it is because the initial checkpoint is missing in the test environment."
                    f"The error: {err}"
                )
                return
            else:
                raise

        assert isinstance(trainer, Trainer), (
            f"+{group}={conf_name} Error parsing recipe."
        )


def find_eval_recipes():
    recipes_dir = Path(__file__).parent.parent.parent / "recipes/eval/lcm"
    config_files = recipes_dir.glob("*.yaml")
    for config_file in config_files:
        yield config_file.stem
