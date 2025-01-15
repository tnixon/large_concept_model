#  Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
#

from inspect import signature
from typing import Any, Dict, Protocol, Union, runtime_checkable

import hydra
from omegaconf import DictConfig, OmegaConf, read_write

from lcm.utils.common import promote_config

TRAINER_KEY = "_trainer_"


@runtime_checkable
class Trainer(Protocol):
    """Abstract trainer in LCM"""

    def run(self) -> Any: ...


def _parse_training_config(train_config: DictConfig):
    """Return the TrainingConfig object from the omegaconf inputs"""
    # The train_config should have 2 keys "_target_" and "_trainer_"
    # the config is set to read-only within stopes module __init__
    assert TRAINER_KEY in train_config, (
        f"The trainer configuration is missing a {TRAINER_KEY} configuration, "
        "you need to specify a Callable to initialize your config."
    )
    trainer_cls_or_func = train_config.get(TRAINER_KEY)
    try:
        trainer_obj = hydra.utils.get_object(trainer_cls_or_func)
        sign = signature(trainer_obj)
        assert len(sign.parameters) == 1 and "config" in sign.parameters, (
            f'{trainer_cls_or_func} should take a single argument called "config"'
        )
        param_type = sign.parameters["config"].annotation

        OmegaConf.resolve(train_config)
        with read_write(train_config):
            del train_config._trainer_

        typed_config = promote_config(train_config, param_type)
        return trainer_obj, typed_config
    except Exception as ex:
        raise ValueError(
            f"couldnt parse the train config: {train_config}.", str(ex)
        ) from ex


def get_trainer(train_config: DictConfig) -> Trainer:
    trainer_obj, typed_config = _parse_training_config(train_config)
    return trainer_obj(typed_config)


def _is_missing(config: Union[DictConfig, Dict], attr: str) -> bool:
    if isinstance(config, Dict):
        return attr in config and config[attr]
    if OmegaConf.is_missing(config, attr):
        return True
    if not hasattr(config, attr) or not getattr(config, attr):
        return True
    return False
