# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from fairseq2.config_registry import ConfigRegistry
from fairseq2.logging import get_log_writer
from fairseq2.nn.padding import PaddingMask, get_seq_lens
from fairseq2.nn.transformer import CausalAttentionMaskFactory
from fairseq2.typing import DataType, Device
from torch import Tensor

from lcm.datasets.batch import EmbeddingsBatch
from lcm.models.abstract_lcm import (
    AbstractLCModel,
    AbstractLCModelBuilder,
    AbstractLCModelConfig,
)
from lcm.models.sonar_normalizer.builder import SonarNormalizer
from lcm.models.two_tower_diffusion_lcm.frontend import (
    EncoderFrontend,
    EncoderFrontendConfig,
)
from lcm.nn.denoisers import (
    DenoiserConfig,
    LCMDenoiser,
    LCMDenoiserTransformerFactory,
)
from lcm.nn.incremental_state import LCMIncrementalStateBag
from lcm.nn.initialization import parse_norm_order
from lcm.nn.normalization import parse_layer_norm_factory
from lcm.nn.schedulers import DDIMScheduler, DDIMSchedulerConfig
from lcm.nn.transformer import (
    LCMTransformerDecoder,
    TransformerConfig,
    TransformerFactory,
)

logger = get_log_writer(__name__)


TWO_TOWER_DIFFUSION_LCM_MODEL_TYPE = "two_tower_diffusion_lcm"


@dataclass
class TwoTowerDiffusionLCModelConfig(AbstractLCModelConfig):
    model_type: str = TWO_TOWER_DIFFUSION_LCM_MODEL_TYPE

    max_seq_len: int = 2048

    model_dim: int = 1024

    frontend: EncoderFrontendConfig = field(
        default_factory=lambda: EncoderFrontendConfig()
    )
    """ The fronted config. This module maps from `sonar_embed_dim` to `model_dim`
        and potentially adds positional embeddings"""

    context_encoder: TransformerConfig = field(
        default_factory=lambda: TransformerConfig()
    )
    """The context encoder config. This is causal Transformer decoder"""

    noise_scheduler: DDIMSchedulerConfig = field(
        default_factory=lambda: DDIMSchedulerConfig()
    )
    """The config of the noise scheduler.
       See lcm/diffusion_schedulers/ddim for more"""

    denoiser: DenoiserConfig = field(default_factory=lambda: DenoiserConfig())
    """the config of the denoiser"""

    trained_with_cf_guidance: bool = False
    """If `True`, the model will be trained with classifier-free guidance i.e.,
       unconditional embedding generation.
       The CF-guidance probability is set in
       DiffusionLCMCriterionConfig.cf_guidance_probability"""


lcm_archs = ConfigRegistry[TwoTowerDiffusionLCModelConfig]()
lcm_arch = lcm_archs.decorator


class TwoTowerDiffusionLCModel(AbstractLCModel):
    """Class for a diffusion-based LCM model"""

    config: TwoTowerDiffusionLCModelConfig

    def __init__(
        self,
        config: TwoTowerDiffusionLCModelConfig,
        sonar_normalizer: SonarNormalizer,
        encoder_frontend: EncoderFrontend,
        context_encoder: LCMTransformerDecoder,
        denoiser: LCMDenoiser,
        noise_scheduler: DDIMScheduler,
    ) -> None:
        super().__init__(config)

        self.model_dim = context_encoder.model_dim

        self.sonar_embed_dim = config.sonar_embed_dim

        self.sonar_normalizer = sonar_normalizer

        self.encoder_frontend = encoder_frontend
        """The frontend of the context encoder.
        This frontend simply applies a pre-linear projection
        (to increase dimensionality) then adds positional embeddings"""

        self.context_encoder = context_encoder
        """A causal Transformer decoder"""

        self.noise_scheduler = noise_scheduler
        """The diffusion noise scheduler"""

        self.denoiser = denoiser

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()
        return f"{s}, dtype={self.dtype}"

    def forward(
        self,
        batch: EmbeddingsBatch,
        noisy_batch: EmbeddingsBatch,
        cf_guidance_prob: float = 0.0,
    ) -> EmbeddingsBatch:
        """
        Arguments:
            - batch (`EmbeddingsBatch`): The clean batch of embeddings to encode the context.
               If `unsupervised` this is the source embeddings.
               If `supervised` this is the source+target embeddings.

           - noisy_batch (`EmbeddingsBatch`): the embeddings noised by the noise scheduler
                If `unsupervised` this is noised source embeddings.
                If `supervised` this is noised target-only embeddings.

           - cf_guidance_prob: probability of training without any guiding context
        """
        # Get source lengths if any:
        source_lengths = batch.source_lengths

        # Encode as context:
        context = self.encode(batch)

        # Predict denoised output
        output_batch = self.denoise(
            noisy_batch=noisy_batch,
            context=context,
            source_lengths=source_lengths,
            cf_guidance_prob=cf_guidance_prob,
        )
        return output_batch

    def encode(
        self,
        batch: EmbeddingsBatch,
        state_bag: Optional[LCMIncrementalStateBag] = None,
        **kwargs,
    ) -> EmbeddingsBatch:
        """
        The main context encoder that takes in a sequence of sonar embeddings in B, T, D
        and returns a sequence of the same shape after causal contextualization.

        Main modules:
            `frontend`: linear projection to model_dim + optional positional embeddings,
            `context_encoder`: Causal Transformer decoder to causally encode the context
        """
        # Frontend
        seqs, padding_mask = self.encoder_frontend(
            batch.seqs,
            batch.padding_mask,
            state_bag=state_bag,
            **kwargs,
        )

        # Main Transformer
        seqs, padding_mask = self.context_encoder(
            seqs,
            padding_mask,
            state_bag=state_bag,
            **kwargs,
        )

        return EmbeddingsBatch(seqs=seqs, padding_mask=padding_mask)

    def denoise(
        self,
        noisy_batch: EmbeddingsBatch,
        context: EmbeddingsBatch,
        source_lengths: Optional[Tensor] = None,
        cf_guidance_prob: float = 0.0,
        state_bag: Optional[LCMIncrementalStateBag] = None,
        inference: bool = False,
    ) -> EmbeddingsBatch:
        """Diffuse a noised sonar embedding conditioned on the encoded context"""
        seqs, padding_mask = self.denoiser(
            seqs=noisy_batch.seqs,
            diffusion_timesteps=noisy_batch.diffusion_timesteps,
            padding_mask=noisy_batch.padding_mask,
            conditioning_variables=context.seqs,
            conditioning_variables_padding_mask=context.padding_mask,
            source_lengths=source_lengths,
            cf_guidance_prob=cf_guidance_prob,
            inference=inference,
        )
        return EmbeddingsBatch(seqs=seqs, padding_mask=padding_mask)

    def prep_for_denoising(self, decoding_options):
        """This setup is done once when we initialize the generator"""
        self.guidance_scale = decoding_options.guidance_scale
        self.guidance_rescale = decoding_options.guidance_rescale
        self.initial_noise_scale = decoding_options.initial_noise_scale
        self.timesteps = decoding_options.inference_timesteps
        self.clip_noise = decoding_options.clip_noise
        self.ddim_eta = decoding_options.ddim_eta
        self.epsilon_scaling = decoding_options.epsilon_scaling

        # if guidance_scale > 1.0 we will duplicate batches
        self.do_classifier_free_guidance = self.guidance_scale != 1.0

        # Setup the diffusion training-like noise scheduler
        # by updating the timesteps according to the decoding `inference_timesteps`
        self.noise_scheduler.set_timesteps(self.timesteps, device=self.device)

        # Override the initial noise scale
        self.noise_scheduler.init_noise_sigma = self.initial_noise_scale
        # Override thresholding options:
        if decoding_options.thresholding:
            self.noise_scheduler.config.thresholding = decoding_options.thresholding
            self.noise_scheduler.config.dynamic_thresholding_ratio = (
                decoding_options.dynamic_thresholding_ratio
            )
            self.noise_scheduler.config.sample_max_value = (
                decoding_options.sample_max_value
            )

    def sample_initial_noise_vectors(self, batch_size: int):
        # Check that we called `prep_for_denoising`:
        assert hasattr(self, "clip_noise"), (
            "The model is not properly set for decoding, make sure to call `model.prep_for_denoising()`"
        )

        # Sample a noise vector for next embedding prediction
        latents = torch.randn(
            batch_size, 1, self.config.sonar_embed_dim, device=self.device
        )

        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.noise_scheduler.init_noise_sigma

        # clip?
        latents = latents.clip(-self.clip_noise, self.clip_noise)
        return latents

    @torch.inference_mode()
    def predict_next_sentence(  # type: ignore
        self,
        batch: EmbeddingsBatch,
        context: EmbeddingsBatch,
        temperature: float = 1.0,
        state_bag: Optional[LCMIncrementalStateBag] = None,
        context_state_bag: Optional[LCMIncrementalStateBag] = None,
        **kwargs,
    ) -> Tuple[EmbeddingsBatch, EmbeddingsBatch]:
        assert context_state_bag is not None, (
            "Expected a state_bag to incrementally encode the context"
        )

        if self.do_classifier_free_guidance:
            logger.debug("Running inference with CF-guidance...")
            return self.predict_next_sentence_with_cf_guidance(
                batch=batch,
                context=context,
                temperature=temperature,
                state_bag=state_bag,
                context_state_bag=context_state_bag,
                **kwargs,
            )

        # Normalize the input embeddings if we're expected to
        # normalize outside of the model's forward pass
        if self.sonar_normalizer is not None:
            batch = batch.normalize_seqs(self.sonar_normalizer)

        # Encode context:
        new_context = self.encode(batch, context_state_bag)
        context_state_bag.increment_step_nr(1)

        # Append to context
        context = EmbeddingsBatch(torch.cat((context.seqs, new_context.seqs), dim=1))

        # Sample latents:
        latents = self.sample_initial_noise_vectors(batch_size=batch.seqs.size(0))

        # Denoise
        diffusion_timesteps_schedule = self.noise_scheduler.timesteps

        for diffusion_timestep in diffusion_timesteps_schedule:
            input_batch = EmbeddingsBatch(
                seqs=latents,
                diffusion_timesteps=diffusion_timestep.long().repeat(
                    (latents.shape[0], 1)
                ),
            )
            # Get model output
            model_prediction = self.denoise(
                noisy_batch=input_batch,
                context=context,
                state_bag=None,
                inference=True,
            )

            scheduler_outputs = self.noise_scheduler.step(
                model_output=model_prediction.seqs,
                timestep=diffusion_timestep,
                sample=latents,
                eta=self.ddim_eta,
                epsilon_scaling=self.epsilon_scaling,
            )

            # setup latents for the next diffusion step
            latents = scheduler_outputs.prev_sample
            # clip?
            latents = latents.clip(-self.clip_noise, self.clip_noise)

        # Take the final predicted denoised sample (x_0 in the ddim paper) and denormalize if needed:
        final_seqs = scheduler_outputs.pred_original_sample

        final_seqs = self.sonar_normalizer.denormalize(final_seqs)

        return EmbeddingsBatch(final_seqs, None), context

    @torch.inference_mode()
    def predict_next_sentence_with_cf_guidance(  # type: ignore
        self,
        batch: EmbeddingsBatch,
        context: EmbeddingsBatch,
        temperature: float = 1.0,
        state_bag: Optional[LCMIncrementalStateBag] = None,
        context_state_bag: Optional[LCMIncrementalStateBag] = None,
        **kwargs,
    ) -> Tuple[EmbeddingsBatch, EmbeddingsBatch]:
        assert context_state_bag is not None, (
            "Expected a state_bag to incrementally encode the context"
        )

        # Normalize the input embeddings if we're expected to
        # normalize outside of the model's forward pass
        if self.sonar_normalizer is not None:
            batch = batch.normalize_seqs(self.sonar_normalizer)

        # Encode context:
        new_context = self.encode(batch, context_state_bag)
        context_state_bag.increment_step_nr(1)

        # Append to context
        context = EmbeddingsBatch(torch.cat((context.seqs, new_context.seqs), dim=1))

        # Sample latents:
        latents = self.sample_initial_noise_vectors(batch_size=batch.seqs.size(0))

        # Denoise
        diffusion_timesteps_schedule = self.noise_scheduler.timesteps

        # Duplicate the context and its padding mask, the second half will be ignored
        _seq_lens = get_seq_lens(context.seqs, context.padding_mask)

        # add zeros:
        _seq_lens = torch.concat((_seq_lens, torch.zeros_like(_seq_lens)), dim=0)

        context = EmbeddingsBatch(
            torch.concat((context.seqs, torch.zeros_like(context.seqs)), dim=0),
            PaddingMask(_seq_lens, batch_seq_len=context.seqs.size(1)),
        )

        batch_multiplier = 2
        for diffusion_timestep in diffusion_timesteps_schedule:
            is_max_diffusion_step = (
                diffusion_timestep == self.noise_scheduler.num_diffusion_train_steps - 1
            )

            input_batch = EmbeddingsBatch(
                torch.concat(batch_multiplier * [latents], dim=0),
                diffusion_timesteps=diffusion_timestep.long().repeat(
                    (latents.shape[0] * batch_multiplier, 1)
                ),
            )

            model_prediction = self.denoise(
                noisy_batch=input_batch,
                context=context,
                state_bag=None,
                inference=True,
            )

            # If at the max step, do not step in the epsilon_scheduler
            if is_max_diffusion_step:
                # if beta_prod_t (denominator) is null i.e.,
                # the diffusion timestep is at its max value (num_training_stesp-1)
                # no denoising will be performed.

                # Note that since the batch might be doubled because
                # we're doing classifier-free guidance, we chunk the model output
                # by batch_multiplier. If not at max_diffusion_step
                # this chunking is performed in apply_classifier_free_guidance
                scheduler_outputs = self.noise_scheduler.step(
                    model_output=model_prediction.seqs.chunk(batch_multiplier)[0],
                    timestep=diffusion_timestep,
                    sample=latents,
                    eta=self.ddim_eta,
                    epsilon_scaling=self.epsilon_scaling,
                )
            else:
                # Predict the noise residual according to the prediction type
                predicted_noise = self.noise_scheduler.get_epsilon(
                    model_output=model_prediction.seqs,
                    sample=input_batch.seqs,
                    timestep=diffusion_timestep,
                )

                if self.do_classifier_free_guidance:
                    # Perform guidance if trained with cf-guidance:
                    # The returned predicted noise will combine the conditional and
                    # unconditional predictions  i.e., from (2 x batch_size, 1, C)
                    # to: (batch_size, 1, C)
                    predicted_noise = self.apply_classifier_free_guidance(
                        predicted_noise
                    )

                # The cf-guidance operates on predicted noises and although we
                # can go back and forth between epsilon and predicted sample
                # once we combine cond and uncond we cannot go back to predicted_x0

                # compute the previous noisy sample x_t -> x_t-1
                scheduler_outputs = self.noise_scheduler.step(
                    model_output=predicted_noise,
                    timestep=diffusion_timestep,
                    sample=latents,
                    eta=self.ddim_eta,
                    epsilon_scaling=self.epsilon_scaling,
                    prediction_type="epsilon",
                )

            # setup latents for the next diffusion step
            latents = scheduler_outputs.prev_sample
            # clip?
            latents = latents.clip(-self.clip_noise, self.clip_noise)

        # Take the final predicted denoised sample (x_0 in the ddim paper) and denormalize if needed:
        final_seqs = scheduler_outputs.pred_original_sample

        final_seqs = self.sonar_normalizer.denormalize(final_seqs)

        return EmbeddingsBatch(final_seqs, None), context

    def apply_classifier_free_guidance(self, predicted_noise: Tensor) -> Tensor:
        """ "
        Apply Classifier-Free Guidance with Rescale as introduced in Algorithm 2 of https://arxiv.org/pdf/2305.08891
        `pos` would be the conditional prediction `cond_prediction`
        and `neg` the unconditional prediction `uncond_prediction`:
        The batch during prefilling is prepared with the conditioning prefix in
        the first half
        """
        # Chunk and follow algorithm 2
        cond_prediction, uncond_prediction = predicted_noise.chunk(2)

        # Regular classifier-free guidance:
        guided_noise_prediction = uncond_prediction + self.guidance_scale * (
            cond_prediction - uncond_prediction
        )

        # Rescale classifier-free guidance to prevent over-exposure
        # Calculate standard deviations.
        std_pos = cond_prediction.std(dim=-1, keepdim=True)
        std_cfg = guided_noise_prediction.std(dim=-1, keepdim=True)

        # Apply guidance rescale with fused operations.
        factor = std_pos / std_cfg
        factor = self.guidance_rescale * factor + (1 - self.guidance_rescale)

        return factor * guided_noise_prediction


class TwoTowerDiffusionLCModelBuilder(AbstractLCModelBuilder):
    """Builds modules of a diffusion-based LCM"""

    config: TwoTowerDiffusionLCModelConfig
    denoiser_factory: LCMDenoiserTransformerFactory

    def __init__(
        self,
        config: TwoTowerDiffusionLCModelConfig,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        super().__init__(config=config, device=device, dtype=dtype)

        self.context_encoder_factory = TransformerFactory(
            model_dim=self.config.model_dim,
            max_seq_len=self.config.max_seq_len,
            config=self.config.context_encoder,
            device=device,
            dtype=dtype,
        )

        self.denoiser_factory = LCMDenoiserTransformerFactory(
            model_dim=self.config.model_dim,
            num_diffusion_train_timesteps=self.config.noise_scheduler.num_diffusion_train_steps,
            max_seq_len=self.config.max_seq_len,
            config=self.config.denoiser,
            input_dim=self.config.sonar_embed_dim,
            device=device,
            dtype=dtype,
        )

    def build_model(self) -> TwoTowerDiffusionLCModel:
        """Build a model."""

        sonar_normalizer = self.build_sonar_normalizer()
        assert sonar_normalizer is not None, (
            "TwoTowerDiffusionLCModel expects a `sonar_normalizer`"
        )

        # the context encoder
        encoder_frontend = self.build_frontend()

        context_encoder = self.build_context_encoder()

        # the denoiser
        denoiser = self.build_denoiser()

        noise_scheduler = self.build_noise_scheduler()

        return TwoTowerDiffusionLCModel(
            config=self.config,
            sonar_normalizer=sonar_normalizer,
            context_encoder=context_encoder,
            encoder_frontend=encoder_frontend,
            denoiser=denoiser,
            noise_scheduler=noise_scheduler,
        )

    def build_frontend(self) -> EncoderFrontend:
        """Build the context encoder front-end."""

        return EncoderFrontend(
            sonar_embed_dim=self.config.sonar_embed_dim,
            model_dim=self.config.model_dim,
            config=self.config.frontend,
            pos_encoder=self.context_encoder_factory.build_pos_encoder(),
            device=self.device,
            dtype=self.dtype,
        )

    def build_context_encoder(self) -> LCMTransformerDecoder:
        """Build the context encoder."""

        config = self.config.context_encoder

        num_layers = config.num_layers
        assert num_layers > 0, "The context encoder needs a non-zero number of layers"

        layers = [self.context_encoder_factory.build_layer() for _ in range(num_layers)]

        self_attn_mask_factory = CausalAttentionMaskFactory()

        if config.final_norm_order_style is None:
            # The final norm order style will be that of
            # the layer-level norm order
            final_norm_order = parse_norm_order(config.norm_order_style)
        else:
            final_norm_order = parse_norm_order(config.final_norm_order_style)

        layer_norm_factory = parse_layer_norm_factory(config.layer_normalization_style)

        return LCMTransformerDecoder(
            layers,
            self_attn_mask_factory=self_attn_mask_factory,
            norm_order=final_norm_order,
            layer_norm_factory=layer_norm_factory,
            dropout_p=config.final_dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_noise_scheduler(self) -> DDIMScheduler:
        return DDIMScheduler(self.config.noise_scheduler)

    def build_denoiser(self) -> LCMDenoiser:
        """Build a Transformer for diffusing noised latents."""
        return self.denoiser_factory.build_model()


def create_two_tower_diffusion_lcm_model(
    config: TwoTowerDiffusionLCModelConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> TwoTowerDiffusionLCModel:
    """Create a DiffusionLCM model.
    :param config:
        The configuration.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    return TwoTowerDiffusionLCModelBuilder(
        config,
        device=device,
        dtype=dtype,  # type: ignore
    ).build_model()
