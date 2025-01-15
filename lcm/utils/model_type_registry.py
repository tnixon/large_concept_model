#  Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
#


from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class ModelTypeConfig:
    """A container for all functions associated with a specific model type."""

    model_type: str
    config_loader: Callable
    model_factory: Callable
    model_loader: Callable


class ModelTypeRegistry:
    """
    Represents a registry of model types.
    In fairseq2 terms, "architecture" refers to a set of model hyperparameters,
    and "model type" refers to a more generic way of constructing the model with the given hyperparameters.
    """

    _configs: Dict[str, ModelTypeConfig]

    def __init__(self) -> None:
        self._configs = {}

    def register(self, model_type_config: ModelTypeConfig) -> None:
        """Register a new architecture.

        :param arch_name:
            The name of the architecture.
        :param config_factory:
            The factory to construct model configurations.
        """
        model_type = model_type_config.model_type
        assert model_type, (
            "To register a model type, the model_type parameter should be non-empty."
        )
        if model_type in self._configs:
            raise ValueError(
                f"`model_type` must be a unique model type name, but '{model_type}' is already registered."
            )
        self._configs[model_type] = model_type_config

    def get_config(self, model_type: str) -> ModelTypeConfig:
        """Return the ModelTypeConfig for the specified model type.

        :param model_type:
            The model type.
        """
        # we import lcm.modules at runtime in order to populate the registy and avoid cyclical imports

        try:
            return self._configs[model_type]
        except KeyError:
            raise ValueError(
                f"The registry of model types does not contain a model type named '{model_type}'."
            )

    def get_model_loader(self, model_type: str) -> Callable:
        """Get a model loader function for the given model type."""
        model_type_config = self.get_config(model_type)
        return model_type_config.model_loader


lcm_model_type_registry = ModelTypeRegistry()
