#  Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
#

import dataclasses
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import yaml
from fairseq2.assets import (
    AssetNotFoundError,
    InProcAssetMetadataProvider,
    default_asset_store,
)
from fairseq2.assets.card import AssetCard
from fairseq2.checkpoint import FileCheckpointManager
from fairseq2.gang import FakeGang
from fairseq2.models import get_model_family
from fairseq2.typing import DataType, Device

from lcm.models.abstract_lcm import AbstractLCModel, AbstractLCModelConfig
from lcm.utils.model_type_registry import lcm_model_type_registry

logger = logging.getLogger(__file__)


def create_model_card(
    checkpoint_path: Path,
    model_config: Union[Dict, AbstractLCModelConfig, None],
    model_type: str,  # TODO: take this parameter from the config
    model_name="on_the_fly_lcm",
    model_arch: Optional[str] = None,
    **additional_card_kwargs,
) -> AssetCard:
    """
    Create an LCModel card given the checkpoint path and model args
    Args:
        - `checkpoint_path`: Path to the checkpoint to evaluate
        - `model_config`: model parmeters
        the default arch
    """

    # Create a fairseq2 model card on the fly.
    # assert (
    # checkpoint_path.is_file()
    # ), f"Couldn't find the checkpoint at {checkpoint_path}"

    if isinstance(model_config, AbstractLCModelConfig):
        model_config = dataclasses.asdict(model_config)

    model_card_info = {
        "name": model_name,
        "model_family": model_type,
        "checkpoint": "file://" + checkpoint_path.as_posix(),
        **additional_card_kwargs,
    }

    if model_config is not None:
        model_card_info["model_config"] = model_config

    if model_arch is not None:
        model_card_info["model_arch"] = model_arch

    default_asset_store.metadata_providers.append(
        InProcAssetMetadataProvider([model_card_info])
    )
    return default_asset_store.retrieve_card(model_name)


def load_model_with_overrides(
    model_dir: Path,
    step: Optional[int] = None,
    model_type: Optional[str] = None,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
    model_filename: str = "model.pt",
):
    if step is not None:
        checkpoint_path = model_dir / f"checkpoints/step_{step}" / model_filename
    else:
        checkpoint_path = model_dir / model_filename

    # New checkpoint
    config_path = checkpoint_path.parent / "model_card.yaml"
    if config_path.exists():
        try:
            return load_model_from_card(
                config_path.as_posix(), device=device, dtype=dtype
            )
        except Exception as exc:
            logger.warning(
                f"Model card {config_path} exists but is not valid ({exc}). "
                "Try global config instead."
            )

    # Old checkpoint
    config_path = model_dir / "config_logs/all_config.yaml"
    if config_path.exists():
        assert model_type, f"Need explicit model_type for checkpoint {checkpoint_path}"
        with open(config_path, "r") as f:
            config = yaml.full_load(f)
            model_config = config["trainer"]["model_config_or_name"]

        temporary_card = create_model_card(
            checkpoint_path=checkpoint_path,
            model_config=model_config,
            model_type=model_type,
            model_arch=f"toy_{model_type}",
        )
        loader_fn = lcm_model_type_registry.get_model_loader(model_type=model_type)
        return loader_fn(temporary_card, device=device, dtype=dtype)  # type: ignore
    else:
        raise ValueError(f"{model_dir} is not a valid model directory")


def create_model_card_from_training_folder(
    folder: Union[str, Path],
    card_name: str,
    step_nr: Optional[int] = None,
) -> AssetCard:
    """
    Extract the model config and the last checkpoint path using the checkpoint manager.
    Create and return a model card
    """
    folder_path = Path(folder)
    assert folder_path.exists(), f"Model directory {folder} does not exist."
    cp_dir = folder_path / "checkpoints"

    gang = FakeGang()
    checkpoint_manager = FileCheckpointManager(cp_dir, gang)

    if step_nr is None:
        step_numbers = checkpoint_manager.get_step_numbers()
        if not step_numbers:
            raise ValueError(
                f"In {cp_dir}, no step number with model checkpoints was detected!"
            )
        step_nr = step_numbers[-1]
        logger.info(f"Automatically setting step number as {step_nr}")

    metadata = checkpoint_manager.load_metadata(step_nr)
    assert metadata is not None, "The checkpoint does not have metadata."

    training_config = metadata["config"]
    model_config = training_config.model_config_or_name

    cp_fn = checkpoint_manager._checkpoint_dir / f"step_{step_nr}" / "model.pt"
    assert cp_fn, (
        f"Checkpoint manager could not extract checkpoint path for step {step_nr}."
    )
    # TODO: deal with the fine-tuning case, where model_config is a string
    if isinstance(model_config, str):
        parent_card = default_asset_store.retrieve_card(model_config)
        model_config = parent_card._metadata["model_config_or_name"]
        model_type = parent_card._metadata["model_family"]
    else:
        model_type = model_config.model_type

    card = create_model_card(
        checkpoint_path=cp_fn.absolute(),
        model_config=model_config,
        model_type=model_type,
        model_arch=f"toy_{model_type}",  # TODO: get rid of the toy architecture when FS2 allows it
        model_name=card_name,
    )
    return card


def save_model_card(card: AssetCard, path: Union[str, Path]) -> None:
    """Save a model card as YAML."""
    card_data = card._metadata  # TODO: use the exposed attribute when available
    with open(path, "w", encoding="utf-8") as outfile:
        yaml.dump(card_data, outfile, default_flow_style=False)


def load_model_from_card(
    model_name: str,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> AbstractLCModel:
    """
    Load LC model from the given assed card or path.
    The parameter `model_name` can be interpreted in multiple ways:
    - as the name of the model card
    - as the path to the yaml file of the model card
    - as the path to the training directory of the model
    - as the path to the model checkpoint (within a training directory, because we need to find the config)
    """
    try:
        card = default_asset_store.retrieve_card(model_name)
    except AssetNotFoundError as err:
        path = Path(model_name)
        # If the card is not found, try looking it up by interpreting model_name as a path to the yaml card.
        if path.exists() and path.suffix == ".yaml":
            with open(path, "r", encoding="utf-8") as f:
                card_data = yaml.full_load(f)
                model_name = card_data["name"]
                card = AssetCard(card_data)
        # If the card is not found, try interpreting model_name as the model training directory
        elif (path / "checkpoints").exists():
            card = create_model_card_from_training_folder(
                path, card_name="temporary_card"
            )
        # If the card is not found, try interpreting model_name as the path to the checkpoint within a training directory
        elif (
            path.suffix == ".pt"
            and path.parent.name.startswith("step_")
            and path.parent.parent.name == "checkpoints"
        ):
            training_dir = path.parent.parent.parent
            step_nr = int(path.parent.name[5:])
            card = create_model_card_from_training_folder(
                training_dir, card_name="temporary_card", step_nr=step_nr
            )
        else:
            raise err
    logger.info(f"Card loaded: {card}")
    model_type = get_model_family(card)
    loader = lcm_model_type_registry.get_model_loader(model_type=model_type)
    model = loader(card, device=device, dtype=dtype)
    return model
