# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import inspect
from typing import Optional, Type

from lcm.utils.common import promote_config

from ..api import Predictor, PredictorConfig

_PREDICTOR_CONFIG_MAP = {
    "dummy": "lcm.evaluation.predictors.dummy.DummyPredictorConfig",
    "dummy_judge": "lcm.evaluation.predictors.dummy.DummyJudgeConfig",
    "llama3": "lcm.evaluation.predictors.llama.HFLlamaPredictorConfig",
    "base_lcm": "lcm.evaluation.predictors.lcm.LCMConfig",
    "two_tower_diffusion_lcm": "lcm.evaluation.predictors.two_tower_diffusion_lcm.TwoTowerDiffusionLCMConfig",
    "huggingface": "lcm.evaluation.predictors.huggingface.HuggingfacePredictorConfig",
    "gemma": "lcm.evaluation.predictors.gemma.GemmaPredictorConfig",
}


def get_config_cls(name: str) -> Type[PredictorConfig]:
    if name not in _PREDICTOR_CONFIG_MAP:
        raise ValueError(f"No predictor registered under the name {name}")

    module_path, config_cls_name = _PREDICTOR_CONFIG_MAP[name].rsplit(".", 1)
    module = __import__(module_path, fromlist=[config_cls_name])
    return getattr(module, config_cls_name)


def build_predictor(
    predictor_config: PredictorConfig,
    predictor_type: Optional[str] = None,
    **kwargs,
) -> Predictor:
    """
    The factory function that loads the predictor from its config. The config can be
    a real config class, or a duck-typed config (e.g. loaded via Hydra)
    """
    if isinstance(predictor_config, PredictorConfig):
        config_cls = predictor_config.__class__
    else:
        assert predictor_type is not None, (
            f"Cannot infer predictor from config type {type(predictor_config)}"
        )
        config_cls = get_config_cls(predictor_type)
        predictor_config = promote_config(predictor_config, config_cls)

    predictor_cls: Predictor = config_cls.predictor_class()
    sig = inspect.signature(predictor_cls.from_config)
    params = sig.parameters.values()
    has_kwargs = any([True for p in params if p.kind == p.VAR_KEYWORD])
    if has_kwargs:
        return predictor_cls.from_config(predictor_config, **kwargs)
    else:
        return predictor_cls.from_config(predictor_config)
