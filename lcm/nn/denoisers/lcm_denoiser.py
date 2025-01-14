# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from typing import Iterable, Optional, Tuple, cast

import torch
import torch.nn as nn
from fairseq2.nn import PositionEncoder
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.normalization import LayerNorm
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.transformer import (
    AttentionMask,
    AttentionMaskFactory,
    FeedForwardNetwork,
    LayerNormFactory,
    MultiheadAttention,
    TransformerDecoderLayer,
    TransformerNormOrder,
    create_standard_layer_norm,
)
from fairseq2.typing import DataType, Device, override
from torch import Tensor
from torch.nn import Dropout, Module, ModuleList
from torch.nn.parameter import Parameter

from lcm.nn.projection import Projection
from lcm.nn.timestep_encoder import DiTTimestepEncoder


class AdaLNModulator(Module):
    """An adaptive LayerNorm modulator to estimate
    shift, gate and scale for all 3 sub-modules."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        super().__init__()

        self.activate = nn.SiLU()
        self.fc = nn.Linear(
            input_dim,
            9 * output_dim,
            bias=True,
            device=device,
            dtype=dtype,
        )

    def reset_parameters(self):
        # zero-init
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, context: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        (modulate_san, modulate_cross_attention, modulate_ffn) = self.fc(
            self.activate(context)
        ).chunk(3, dim=-1)
        return modulate_san, modulate_cross_attention, modulate_ffn


class LCMDenoiser(Module):
    """
    The main denoiser module of the two-tower diffusion LCM.
    """

    model_dim: int
    layers: ModuleList
    self_attn_mask_factory: AttentionMaskFactory
    layer_norm: Optional[LayerNorm]
    dropout_p: float
    norm_order: TransformerNormOrder
    cross_attention_mask_factory: AttentionMaskFactory

    def __init__(
        self,
        embed_time: DiTTimestepEncoder,
        layers: Iterable[TransformerDecoderLayer],
        initial_proj: Projection,
        final_proj: Projection,
        *,
        self_attn_mask_factory: AttentionMaskFactory,
        cross_attention_mask_factory: AttentionMaskFactory,
        dropout_p: float = 0.0,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        pos_encoder: Optional[PositionEncoder] = None,
        layer_norm_factory: Optional[LayerNormFactory] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param layers:
            The decoder layers.
        :param self_attn_mask_factory:
            The self attention mask factory.
        :param cross_attention_mask_factory:
            The cross attention mask factory.
        :param dropout_p:
            The dropout probability on decoder outputs.
        :param norm_order:
            The Layer Normalization order.
        :param: pos_encoder:
            An optional positional encoding module
        :param layer_norm_factory:
            The factory to construct the Layer Normalization module.
        """
        layer_list = ModuleList(layers)

        if not layer_list:
            raise ValueError("`layers` must be non-empty.")

        model_dim = layer_list[0].model_dim

        super().__init__()

        self.model_dim = model_dim

        self.embed_time = embed_time

        self.initial_proj = initial_proj

        self.final_proj = final_proj

        self.pos_encoder = pos_encoder

        if layer_norm_factory is None:
            layer_norm_factory = create_standard_layer_norm

        self.self_attn_mask_factory = self_attn_mask_factory

        self.layers = layer_list

        if norm_order != TransformerNormOrder.POST:
            self.layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)
        else:
            self.register_module("layer_norm", None)

        if dropout_p > 0.0:
            self.dropout = Dropout(dropout_p)
        else:
            self.register_module("dropout", None)

        self.norm_order = norm_order

        self.cross_attention_mask_factory = cross_attention_mask_factory

    def forward(
        self,
        seqs: Tensor,
        diffusion_timesteps: Tensor,
        padding_mask: Optional[PaddingMask],
        conditioning_variables: Optional[Tensor] = None,
        conditioning_variables_padding_mask: Optional[PaddingMask] = None,
        source_lengths: Optional[Tensor] = None,
        cf_guidance_prob: float = 0.0,
        *,
        state_bag: Optional[IncrementalStateBag] = None,
        inference: Optional[bool] = False,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """
        Arguments:
            - seqs (`Tensor`): the sequence of latents to denoise
            - diffusion_timesteps (`Tensor`) the indices of the diffusion timesteps
                to be embedded and fed as a conditioning variable.
            - padding_mask (`PaddingMask`) mask of padded positions in the latents (seqs)

            - conditioning_variables (`Tensor`) the sequence of conditioning
                variables that will be combined with the timestep embedding to
                guide the diffusion process
            - conditioning_variables_padding_mask (`PaddingMask`) the mask of padded
                positions in `conditioning_variables`
            - source_lengths (`Optional[Tensor]`) the lengths of the source embeddings
                in `conditioning_variables` to properly shift the cross-attention mask
            - cf_guidance_prob: probability rate with which to drop all conditioning variables when denoising
            - state_bag (`IncrementalStateBag`) the incremental state bag of the denoiser to enable kv-caching
            - inference (`bool`) if `True` the cross-attention mask will be adjusted accordingly
        """

        emb_timesteps = self.embed_time(diffusion_timesteps)
        assert conditioning_variables is not None, (
            "Expected conditioning_variables, found None"
        )

        assert conditioning_variables is not None, (
            "Mypy - Expecting non-None conditioning_variables"
        )

        conditioning_variables = torch.cat(
            [
                torch.zeros_like(conditioning_variables[:, 0:1]),
                conditioning_variables,
            ],
            dim=1,
        )

        if conditioning_variables_padding_mask is not None:
            # shift by the length of the prepended timesteps
            conditioning_variables_padding_mask = PaddingMask(
                conditioning_variables_padding_mask._seq_lens + 1,
                conditioning_variables_padding_mask._batch_seq_len + 1,
            )

        # project to model_dim and add optional position codes:
        seqs = self.initial_proj(seqs)

        if self.pos_encoder is not None:
            seqs = self.pos_encoder(seqs, padding_mask)

        self_attn_mask = self.self_attn_mask_factory(
            seqs, keys=seqs, training=self.training, state_bag=state_bag
        )

        assert conditioning_variables is not None
        cross_attention_mask = self.cross_attention_mask_factory(
            seqs,
            keys=conditioning_variables,
            source_lengths=source_lengths,
            cf_guidance_prob=cf_guidance_prob,
            training=self.training,
            state_bag=state_bag,
            inference=inference,  # type: ignore
        )

        for layer_idx, layer in enumerate(self.layers):
            layer_output, layer_padding_mask = layer(
                seqs=seqs,
                padding_mask=padding_mask,
                self_attn_mask=self_attn_mask,
                emb_timesteps=emb_timesteps,
                conditioning_variables=conditioning_variables,
                conditioning_variables_padding_mask=conditioning_variables_padding_mask,
                cross_attention_mask=cross_attention_mask,
                state_bag=state_bag,
            )

            seqs, padding_mask = layer_output, layer_padding_mask

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        seqs = self.final_proj(seqs)

        return seqs, padding_mask


class LCMDenoiserLayer(TransformerDecoderLayer):
    """A single layer of the hybrid denoiser"""

    self_attn: MultiheadAttention
    self_attn_norm: Optional[LayerNorm]
    self_attn_dropout: Optional[Dropout]
    self_attn_layer_norm: LayerNorm
    cross_attention: MultiheadAttention
    cross_attention_dropout: Optional[Dropout]
    cross_attention_layer_norm: Optional[LayerNorm]
    ffn: FeedForwardNetwork
    ffn_dropout: Optional[Dropout]
    residual_scale: Optional[Parameter]
    ffn_layer_norm: LayerNorm
    norm_order: TransformerNormOrder

    def __init__(
        self,
        self_attn: MultiheadAttention,
        ffn: FeedForwardNetwork,
        cross_attention: MultiheadAttention,
        *,
        scale_residual: bool = False,
        dropout_p: float = 0.0,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        layer_norm_factory: Optional[LayerNormFactory] = None,
        modulator_input_dim: Optional[int] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param self_attn:
            The self attention layer.
        :param cross_attention:
            The cross attention layer if denoiser-type is `cross-attention`.
        :param ffn:
            The feed-forward network.
        :param scale_residual:
            If ``True``, scales residuals before adding them to the output of
            the feed-forward network as described in
            :cite:t:`https://doi.org/10.48550/arxiv.2110.09456`.
        :param dropout_p:
            The dropout probability on outputs of the attention layers and the
            feed-forward network.
        :param norm_order:
            The Layer Normalization order.
        :param layer_norm_factory:
            The factory to construct the Layer Normalization modules.
        """
        model_dim = self_attn.model_dim

        super().__init__(model_dim)

        if layer_norm_factory is None:
            layer_norm_factory = create_standard_layer_norm

        self_attn_layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)

        if norm_order != TransformerNormOrder.POST:
            self.self_attn_layer_norm = self_attn_layer_norm

        self.self_attn = self_attn

        if norm_order == TransformerNormOrder.PRE_WITH_NORMFORMER:
            self.self_attn_norm = layer_norm_factory(
                model_dim, device=device, dtype=dtype
            )
        else:
            self.register_module("self_attn_norm", None)

        if dropout_p > 0.0:
            self.self_attn_dropout = Dropout(dropout_p)
        else:
            self.register_module("self_attn_dropout", None)

        if norm_order == TransformerNormOrder.POST:
            self.self_attn_layer_norm = self_attn_layer_norm

        # Deal with the cross-attention layers:
        if cross_attention is None:
            self.register_module("cross_attention", None)
            self.register_module("cross_attention_layer_norm", None)
        else:
            cross_attention_layer_norm = layer_norm_factory(
                model_dim, device=device, dtype=dtype
            )

            if norm_order != TransformerNormOrder.POST:
                self.cross_attention_layer_norm = cross_attention_layer_norm

            self.cross_attention = cross_attention

            if dropout_p > 0.0:
                self.cross_attention_dropout = Dropout(dropout_p)
            else:
                self.register_module("cross_attention_dropout", None)

            if norm_order == TransformerNormOrder.POST:
                self.cross_attention_layer_norm = cross_attention_layer_norm
        # / deal with cross-attention

        ffn_layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)

        if norm_order != TransformerNormOrder.POST:
            self.ffn_layer_norm = ffn_layer_norm

        self.ffn = ffn

        if dropout_p > 0.0:
            self.ffn_dropout = Dropout(dropout_p)
        else:
            self.register_module("ffn_dropout", None)

        if norm_order == TransformerNormOrder.POST:
            self.ffn_layer_norm = ffn_layer_norm

        self.norm_order = norm_order

        # Add a modulator:
        modulator_input_dim = modulator_input_dim or model_dim
        self.modulator = AdaLNModulator(
            input_dim=modulator_input_dim,
            output_dim=model_dim,
            device=device,
            dtype=dtype,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        # Zero-out the modulators:
        self.modulator.reset_parameters()

    @override
    def forward(  # type: ignore
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        conditioning_variables: Tensor,
        emb_timesteps: Tensor,
        self_attn_mask: Optional[AttentionMask] = None,
        conditioning_variables_padding_mask: Optional[PaddingMask] = None,
        cross_attention_mask: Optional[AttentionMask] = None,
        *,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        # Get modulator output:
        (modulate_san, modulate_cross_attention, modulate_ffn) = self.modulator(
            emb_timesteps
        )

        seqs = self._forward_self_attn(
            seqs=seqs,
            padding_mask=padding_mask,
            modulators=modulate_san,
            self_attn_mask=self_attn_mask,
            state_bag=state_bag,
        )

        seqs = self._forward_cross_attention(
            seqs=seqs,
            padding_mask=padding_mask,
            conditioning_variables=conditioning_variables,
            modulators=modulate_cross_attention,
            cross_attention_mask=cross_attention_mask,
            key_padding_mask=conditioning_variables_padding_mask,
            state_bag=state_bag,
        )

        seqs = self._forward_ffn(
            seqs=seqs,
            modulators=modulate_ffn,
        )

        return seqs, padding_mask

    def _forward_self_attn(
        self,
        seqs: Tensor,
        modulators: Tensor,
        padding_mask: Optional[PaddingMask],
        self_attn_mask: Optional[AttentionMask],
        state_bag: Optional[IncrementalStateBag],
    ) -> Tensor:
        residual = seqs

        assert self.norm_order != TransformerNormOrder.POST, (
            "DiT AdaLN expect pre-normalization"
        )

        if self.norm_order != TransformerNormOrder.POST:
            seqs = self.self_attn_layer_norm(seqs)

        # split modulators into shift, scale and gate:
        shift, scale, gate = modulators.chunk(3, dim=-1)

        # modulate the input:
        seqs = seqs * (1 + scale) + shift

        seqs = self.self_attn(
            seqs,
            padding_mask,
            keys=seqs,
            key_padding_mask=None,
            values=seqs,
            attn_mask=self_attn_mask,
            state_bag=state_bag,
        )

        if self.self_attn_norm is not None:
            seqs = self.self_attn_norm(seqs)

        if self.self_attn_dropout is not None:
            seqs = self.self_attn_dropout(seqs)

        # Scale the residual with the gate weights
        seqs = residual + gate * seqs

        return seqs

    def _forward_cross_attention(
        self,
        seqs: Tensor,
        modulators: Tensor,
        padding_mask: Optional[PaddingMask],
        conditioning_variables: Optional[Tensor],
        key_padding_mask: Optional[PaddingMask],
        cross_attention_mask: Optional[AttentionMask],
        state_bag: Optional[IncrementalStateBag],
    ) -> Tensor:
        if conditioning_variables is None:
            raise ValueError(
                "`conditioning_variables` must not be `None` for cross attention."
            )

        residual = seqs

        assert self.norm_order != TransformerNormOrder.POST, (
            "DiT AdaLN expect pre-normalization"
        )

        if self.norm_order != TransformerNormOrder.POST:
            seqs = cast(LayerNorm, self.cross_attention_layer_norm)(seqs)

        # split modulators into shift, scale and gate:
        shift, scale, gate = modulators.chunk(3, dim=-1)

        # modulate the input:
        seqs = seqs * (1 + scale) + shift

        seqs = self.cross_attention(
            seqs,
            padding_mask,
            keys=conditioning_variables,
            key_padding_mask=key_padding_mask,
            attn_mask=cross_attention_mask,
            values=conditioning_variables,
            state_bag=state_bag,
        )

        if self.cross_attention_dropout is not None:
            seqs = self.cross_attention_dropout(seqs)

        # Scale the residual with the gate weights
        seqs = residual + gate * seqs

        return seqs

    def _forward_ffn(self, seqs: Tensor, modulators: Tensor) -> Tensor:
        assert self.norm_order != TransformerNormOrder.POST, (
            "DiT AdaLN expects pre-normalization"
        )
        residual = seqs

        if self.norm_order != TransformerNormOrder.POST:
            seqs = self.ffn_layer_norm(seqs)

        # split modulators into shift, scale and gate:
        shift, scale, gate = modulators.chunk(3, dim=-1)

        # modulate the input:
        seqs = seqs * (1 + scale) + shift

        seqs = self.ffn(seqs)

        if self.ffn_dropout is not None:
            seqs = self.ffn_dropout(seqs)

        # Scale the branch with the gate weights
        seqs = residual + gate * seqs

        return seqs
