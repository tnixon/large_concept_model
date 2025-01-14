# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from fairseq2.logging import get_log_writer
from fairseq2.nn import PositionEncoder
from fairseq2.nn.position_encoder import (
    LearnedPositionEncoder,
    RotaryEncoder,
    SinusoidalPositionEncoder,
)
from fairseq2.nn.projection import Linear
from fairseq2.nn.transformer import (
    FeedForwardNetwork,
    GLUFeedForwardNetwork,
    MultiheadAttention,
    StandardFeedForwardNetwork,
    TransformerDecoderLayer,
    create_default_sdpa,
)
from fairseq2.typing import DataType, Device

from lcm.nn.initialization import (
    SUPPORTED_INIT_TYPES,
    get_init_fn,
    parse_activation_fn,
    parse_norm_order,
)
from lcm.nn.normalization import SUPPORTED_LN_TYPES, parse_layer_norm_factory
from lcm.nn.transformer import LCMStandardTransformerDecoderLayer
from lcm.nn.transformer.attention import (
    FullAttentionState,
    QKNormMultiheadAttention,
)

SUPPORTED_NORM_ORDERS = Literal["pre", "post", "normformer"]


logger = get_log_writer(__name__)


@dataclass
class TransformerConfig:
    """A config object to group all config
    hyper-parameters of a LCMTransformerDecoder"""

    num_layers: int = 2

    num_attn_heads: int = 8

    # Dropout rates
    dropout_p: float = 0.1
    """ The dropout probability outputs of the attention layers and the
        feed-forward network (before joining the residual stream)"""

    final_dropout_p: float = 0.1
    """ The dropout probability on decoder outputs"""

    attention_dropout_p: float = 0.0
    """the dropout rate on attention weights in SDPA"""

    # FFN
    ffn_inner_dim: int = 1024 * 4

    use_swiglu: bool = False
    """Use GLUFeedForwardNetwork instead of regular FFN blocks"""

    ffn_inner_activation_name: str = "relu"

    """The activation to apply to outputs of the FFN inner projection layer.
    Default is `relu `i.e., `torch.nn.ReLU`. This is only relevant when `use_swiglu= False`"""

    # positional embedding
    pos_embedding_style: Literal["rope", "sine", "learned", "none"] = "learned"

    """If `rope`: a rotary positional encoder in used in the attention layers.
       If `sine`: Sinusoidal positional embeddings will be added in
            the frontend before heading into the decoder
       If `learned`: Learned positional embeddings will be added in
            the frontend before heading into the decoder.
       If `None`: no positional embeddings will be used (e.g. in the case
           of unconditional diffusion of a single vector)."""

    rope_theta: float = 10_000.0
    """ The coefficient of the long-term decay of RoPE embeddings."""

    # Normalization
    layer_normalization_style: SUPPORTED_LN_TYPES = "standard"

    norm_order_style: SUPPORTED_NORM_ORDERS = "pre"
    """LayerNorm order in the transformer decoder,
        default is pre-normalization (`pre`). Other options are post-normalization (`post`)
        and normformer-style normalization (`normformer`)"""

    final_norm_order_style: Optional[SUPPORTED_NORM_ORDERS] = None
    """Controls lcm-level norm-order, using ``post`` here with a ``pre`` layer-level norm-order
        means that we will skip the last layernorm in the stack"""

    enable_qk_layernorm: bool = False
    """If ``True``, LayerNorms will be applied to queries and keys in self-attention layers
        QK-LayerNorm described in https://arxiv.org/pdf/2302.05442 and subsequent work
        is recommended to alleviate Transformer training instabilities
    """
    mha_qkv_weight_normalization: bool = False
    """if ``True`` wrap the K/Q/V linears of MHA in weight normalization"""

    mha_output_weight_normalization: bool = False
    """if ``True`` wrap the output projection of MHA with weight normalization.
    This is a temporary fix to resume training some models and will be removed"""

    # Miscellaneous
    mha_output_proj_bias: bool = False
    """If ``True`` add a bias term to the MHA output projection"""

    scale_residual: Optional[float] = None
    """scale to multiply the residual in the Transformer decoder"""

    attention_output_init_fn: SUPPORTED_INIT_TYPES = "xavier"


class TransformerFactory:
    def __init__(
        self,
        model_dim: int,
        max_seq_len: int,
        config: TransformerConfig,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param model_dim:
            The hidden model dimension of the Transformer
        :params max_seq_len:
            Maximum supported sequence length by the model
        :param config:
            The configuration.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.config = config
        self.device, self.dtype = device, dtype

    def build_layer(self) -> TransformerDecoderLayer:
        """Build a Transformer decoder layer based on the provided config."""

        self_attn = self.build_attention()

        ffn = self.build_ffn()

        norm_order = parse_norm_order(self.config.norm_order_style)

        layer_norm_factory = parse_layer_norm_factory(
            self.config.layer_normalization_style
        )

        layer = LCMStandardTransformerDecoderLayer(
            self_attn=self_attn,
            encoder_decoder_attn=None,
            ffn=ffn,
            dropout_p=self.config.dropout_p,
            norm_order=norm_order,
            layer_norm_factory=layer_norm_factory,
            scale_residual=self.config.scale_residual is not None,
            device=self.device,
            dtype=self.dtype,
        )
        # reset residual_scale
        if layer.residual_scale is not None:
            assert self.config.scale_residual is not None, (
                f"Layer has a resiudal scale but scale={self.config.scale_residual}"
            )
            torch.nn.init.constant_(layer.residual_scale, self.config.scale_residual)
            logger.info(
                f"Initializing the residual scale at {self.config.scale_residual}"
            )
        return layer

    def build_pos_encoder(self) -> Optional[PositionEncoder]:
        """Build the positional encoder (learned or sinusoidal, if any)
        that will be used in the frontend"""
        pos_encoder: Optional[PositionEncoder]

        if self.config.pos_embedding_style == "learned":
            pos_encoder = LearnedPositionEncoder(
                self.model_dim,
                self.max_seq_len,
                device=self.device,
                dtype=self.dtype,
            )
        elif self.config.pos_embedding_style == "sine":
            pos_encoder = SinusoidalPositionEncoder(
                self.model_dim,
                self.max_seq_len,
                device=self.device,
            )

        else:
            pos_encoder = None

        return pos_encoder

    def build_attention_pos_encoder(self) -> Optional[PositionEncoder]:
        """Build the position encoder that can
        potentially be used in the MHA module"""

        pos_encoder: Optional[PositionEncoder]

        if self.config.pos_embedding_style == "rope":
            pos_encoder = RotaryEncoder(
                encoding_dim=self.model_dim // self.config.num_attn_heads,
                max_seq_len=self.max_seq_len,
                theta=self.config.rope_theta,
                device=self.device,
            )
        else:
            pos_encoder = None
        return pos_encoder

    def build_attention(self) -> MultiheadAttention:
        """Build a Transformer multi-head attention layer."""

        # allow for a different kv_dim
        kv_dim = self.model_dim

        # fairseq2.nn.transformer.attention.TorchSDPA
        sdpa = create_default_sdpa(attn_dropout_p=self.config.attention_dropout_p)

        init_fn = get_init_fn(self.config.attention_output_init_fn)

        # How does Rope play with encoder-decoder attention?
        pos_encoder = self.build_attention_pos_encoder()

        layer_norm_factory = parse_layer_norm_factory(
            self.config.layer_normalization_style
        )

        # build output_proj:
        output_proj = Linear(
            self.model_dim,
            self.model_dim,
            bias=self.config.mha_output_proj_bias,
            init_fn=init_fn,
            device=self.device,
            dtype=self.dtype,
        )
        if self.config.mha_output_weight_normalization:
            output_proj = torch.nn.utils.parametrizations.weight_norm(output_proj)

        return QKNormMultiheadAttention(
            self.model_dim,
            self.config.num_attn_heads,
            kv_dim=kv_dim,
            pos_encoder=pos_encoder,
            sdpa=sdpa,
            output_proj=output_proj,
            enable_qk_layernorm=self.config.enable_qk_layernorm,
            weight_normalization=self.config.mha_qkv_weight_normalization,
            layer_norm_factory=layer_norm_factory,
            state_factory=FullAttentionState,
            device=self.device,
            dtype=self.dtype,
        )

    def build_ffn(self) -> FeedForwardNetwork:
        """Build a Transformer feed-forward network."""
        if self.config.use_swiglu:
            # Default gate_activation is torch.nn.SiLU
            return GLUFeedForwardNetwork(
                self.model_dim,
                self.config.ffn_inner_dim,
                bias=True,
                inner_dim_scale=2 / 3,
                inner_dim_to_multiple=256,
                device=self.device,
                dtype=self.dtype,
            )

        ffn_inner_activation = parse_activation_fn(
            self.config.ffn_inner_activation_name
        )
        norm_order = parse_norm_order(self.config.norm_order_style)

        return StandardFeedForwardNetwork(
            self.model_dim,
            self.config.ffn_inner_dim,
            inner_activation=ffn_inner_activation,
            bias=True,
            norm_order=norm_order,
            device=self.device,
            dtype=self.dtype,
        )
