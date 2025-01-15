# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from dataclasses import dataclass, field
from typing import Literal, Optional

from fairseq2.logging import get_log_writer
from fairseq2.typing import DataType, Device

from lcm.nn.denoisers.attention_masks import (
    NoAttentionMaskFactory,
    ShiftedCausalAttentionMaskFactory,
)
from lcm.nn.denoisers.lcm_denoiser import (
    LCMDenoiser,
    LCMDenoiserLayer,
)
from lcm.nn.initialization import parse_norm_order
from lcm.nn.normalization import parse_layer_norm_factory
from lcm.nn.projection import (
    Projection,
    ProjectionConfig,
)
from lcm.nn.timestep_encoder import DiTTimestepEncoder
from lcm.nn.transformer import TransformerConfig, TransformerFactory

logger = get_log_writer(__name__)


@dataclass
class DenoiserConfig(TransformerConfig):
    """Config for building the LCM's denoiser"""

    pos_embedding_style: Literal["rope", "sine", "learned", "none"] = "none"
    """By default, a denoiser does not have a positional embedder"""

    pre_denoiser: ProjectionConfig = field(default_factory=lambda: ProjectionConfig())
    """the initial projection at the top of the denoiser"""

    post_denoiser: ProjectionConfig = field(default_factory=lambda: ProjectionConfig())
    """the final output projection at the end of the denoiser"""

    timestep_embed_dim: int = 1024
    """Diffusion timestep embedding dimension"""


class LCMDenoiserTransformerFactory(TransformerFactory):
    """Denoiser with hybrid AdaLN and cross-attention"""

    config: DenoiserConfig

    def __init__(
        self,
        model_dim: int,
        max_seq_len: int,
        num_diffusion_train_timesteps: int,
        config: DenoiserConfig,
        input_dim: int = 1024,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param model_dim:
            The hidden model dimension of the Transformer
        :params max_seqs_len:
            Maximum supported sequence length by the model
        :param config:
            The configuration.
        :param input_dim:
            The input embedding dimension i.e `sonar_embed_dim``
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        super().__init__(
            model_dim=model_dim,
            max_seq_len=max_seq_len,
            config=config,
            device=device,
            dtype=dtype,
        )

        self.input_dim = input_dim

        self.num_diffusion_train_timesteps = num_diffusion_train_timesteps

    def build_cross_attention_mask(self):
        return ShiftedCausalAttentionMaskFactory()

    def build_timestep_embedder(self):
        return DiTTimestepEncoder(
            embedding_dim=self.config.timestep_embed_dim,
            dtype=self.dtype,
            device=self.device,
        )

    def build_initial_proj(self) -> Projection:
        # We will be concatenating context and timesteps embeddings
        assert self.config.timestep_embed_dim == self.model_dim, (
            "Since the timestep embeddings will be added to the sequence of "
            "conditioning variables, they need to be of the same dimension. "
            f"Found timestep_embed_dim={self.config.timestep_embed_dim} "
            f"and model_dim={self.model_dim}"
        )

        return Projection(
            output_dim=self.model_dim,
            input_dim=self.input_dim,
            config=self.config.pre_denoiser,
            device=self.device,
            dtype=self.dtype,
        )

    def build_final_proj(self) -> Projection:
        return Projection(
            output_dim=self.input_dim,
            input_dim=self.model_dim,
            config=self.config.post_denoiser,
            device=self.device,
            dtype=self.dtype,
        )

    def build_model(self) -> LCMDenoiser:
        """Build the denoiser with its layers and initial/final projections"""
        embed_time = self.build_timestep_embedder()

        layers = [self.build_layer() for _ in range(self.config.num_layers)]

        norm_order = parse_norm_order(self.config.norm_order_style)

        # Self-attention here does not contextualize
        self_attn_mask_factory = NoAttentionMaskFactory()

        cross_attention_mask_factory = self.build_cross_attention_mask()

        layer_norm_factory = parse_layer_norm_factory(
            self.config.layer_normalization_style
        )

        pos_encoder = self.build_pos_encoder()

        return LCMDenoiser(
            embed_time=embed_time,
            layers=layers,
            initial_proj=self.build_initial_proj(),
            final_proj=self.build_final_proj(),
            dropout_p=self.config.final_dropout_p,
            norm_order=norm_order,
            layer_norm_factory=layer_norm_factory,
            self_attn_mask_factory=self_attn_mask_factory,
            cross_attention_mask_factory=cross_attention_mask_factory,
            pos_encoder=pos_encoder,
            device=self.device,
            dtype=self.dtype,
        )

    def build_layer(self) -> LCMDenoiserLayer:
        """Build a Transformer decoder layer based on the provided config."""

        assert isinstance(self.config, DenoiserConfig), (
            "Expecting a DenoiserConfig in the DenoiserTransformerFactory"
        )

        self_attn = self.build_attention()

        cross_attn = self.build_attention()

        ffn = self.build_ffn()

        norm_order = parse_norm_order(self.config.norm_order_style)

        layer_norm_factory = parse_layer_norm_factory(
            self.config.layer_normalization_style
        )

        modulator_input_dim = self_attn.model_dim

        layer = LCMDenoiserLayer(
            self_attn=self_attn,
            cross_attention=cross_attn,
            ffn=ffn,
            modulator_input_dim=modulator_input_dim,
            dropout_p=self.config.dropout_p,
            norm_order=norm_order,
            layer_norm_factory=layer_norm_factory,
            device=self.device,
            dtype=self.dtype,
        )
        return layer
