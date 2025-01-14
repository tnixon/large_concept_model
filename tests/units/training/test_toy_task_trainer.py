# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import math

import numpy as np
import torch
from fairseq2.models import get_model_family
from tqdm.auto import tqdm

from lcm.datasets.configs import (
    DataLoadingConfig,
)
from lcm.models.base_lcm import BaseLCModelConfig
from lcm.nn.transformer import TransformerConfig
from lcm.train.lcm.trainer import LCMTrainerBuilder, LCMTrainingConfig
from lcm.train.mse_lcm.criterion import ReconstructionCriterionConfig
from lcm.train.trainer import Trainer
from lcm.utils.card_utils import load_model_from_card, load_model_with_overrides
from lcm.utils.model_type_registry import lcm_model_type_registry
from tests.common import DEBUG, device


def get_eval_loss(trainer: Trainer) -> float:
    trainer.model.eval()
    trainer.validation_data_loader.pipeline.reset()  # type: ignore
    for mb in trainer.valid_metric_bag.values():
        mb.reset_metrics()
    for batch in tqdm(trainer.validation_data_loader.iterate_batches()):  # type: ignore
        loss = trainer.criterion(batch)
        trainer.valid_metric_bag[batch.name].update(
            [loss],
        )
    values = {
        name: mb.sync_and_compute_metrics()
        for name, mb in trainer.valid_metric_bag.items()
    }
    trainer.model.train()
    # taking average value from over all datasets
    return np.mean([x["loss"].item() for x in values.values()])  # type: ignore


def compare_models(loaded_model, trained_model):
    loaded_state_dict = loaded_model.state_dict()
    for param_name, param in trained_model.named_parameters():
        assert torch.allclose(
            param.data, loaded_state_dict[param_name].to(param.device)
        ), f"{param_name} differs after loading the model!"


def test_toy_mse_training(tmp_path, simple_train_dataset, simple_validation_dataset):
    """
    Test that the trainer can be built, that it can run, and that it saves the model well.
    """
    model_config_or_name = BaseLCModelConfig(lcm=TransformerConfig(num_layers=1))
    criterion_cfg = ReconstructionCriterionConfig(
        name="next_sentence_mse", reduction="mean"
    )

    train_dirname = tmp_path / "tmp_lcm_trainer_output"
    n_steps = 10
    training_cfg = LCMTrainingConfig(
        debug=DEBUG,
        fake_gang_device=device,
        model_config_or_name=model_config_or_name,
        use_fsdp=False,
        use_submitit=False,
        data_loading_config=DataLoadingConfig(batch_size=10),
        training_data=[simple_train_dataset],
        validation_data=[simple_validation_dataset],
        output_dir=train_dirname,
        criterion=criterion_cfg,
        num_lr_warmup_steps=n_steps // 3 + 1,
        max_steps=n_steps,
        checkpoint_every_n_steps=1,
        save_model_every_n_steps=1,
        lr=1e-6,
    )

    # Testing that the trainer is buildable
    builder = LCMTrainerBuilder(training_cfg)
    trainer = builder.build_trainer()

    # Testing that the training does happen and decreases the loss
    old_eval_loss = get_eval_loss(trainer)
    assert math.isfinite(old_eval_loss), "Old eval loss is not finite!"
    trainer.run()
    new_eval_loss = get_eval_loss(trainer)
    assert math.isfinite(new_eval_loss), "New eval loss is not finite!"
    assert new_eval_loss < old_eval_loss

    # testing that the checkpointing works
    step_id, state_dict = trainer.checkpoint_manager.load_last_checkpoint()
    assert step_id == n_steps, f"step_id={step_id} does not match n_steps={n_steps}"
    for param_name, param in trainer.model.named_parameters():
        assert torch.allclose(
            param.data, state_dict["model"][param_name].to(param.device)
        ), f"{param_name} differs in checkpoint!"

    # Testing that the model card has been created
    assert (train_dirname / "model_card.yaml").exists(), (
        f"The file {train_dirname}/model_card.yaml does not exist"
    )

    # Testing that the model card can be used to load the model correctly
    card = trainer.create_model_card_for_last_checkpoint()
    model_type = get_model_family(card)
    model_loader = lcm_model_type_registry.get_model_loader(model_type)
    loaded_model = model_loader(card)
    compare_models(loaded_model, trainer.model)

    # Test that the model card API works out-of-the-box
    loaded_model_1 = load_model_from_card(str(train_dirname / "model_card.yaml"))
    compare_models(loaded_model_1, trainer.model)

    loaded_model_2 = load_model_with_overrides(train_dirname)
    compare_models(loaded_model_2, trainer.model)
