# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from fairseq2.generation.generator import (
    GenerationCounters,
    Hypothesis,
    SequenceGeneratorOutput,
)
from fairseq2.logging import get_log_writer

from lcm.datasets.batch import EmbeddingsBatch, PaddingMask
from lcm.inference.lcm.generator import (
    LCMGenerator,
    LCMGeneratorOptions,
)
from lcm.models.abstract_lcm import AbstractLCModel
from lcm.models.two_tower_diffusion_lcm.builder import TwoTowerDiffusionLCModel
from lcm.nn.incremental_state import LCMIncrementalStateBag

logger = get_log_writer(__name__)


@dataclass
class DiffusionLCMGeneratorOptions(LCMGeneratorOptions):
    """Holds the options to pass to a diffusion-based sequence generator."""

    guidance_scale: float = 1.0
    """The weight of the regular classifier-free guidance.
        Here `guidance_scale` is defined as the guidance weight `w` of
        Equation (2) of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf.
        `guidance_scale = 1` corresponds to doing no classifier free guidance.
        A higher guidance scale value encourages the model to generate outputs
        closely related to the `prompt` at the expense of lower quality."""

    guidance_rescale: float = 0.0
    """The rescaling factor for Classifier-Free Guidance with Rescale
        (Algorithm 2 - https://arxiv.org/pdf/2305.08891)"""

    ddim_eta: float = 0.0
    """The weight of noise for added noise in diffusion step.
    It controls the level of interpolation between a deterministic
    DDIM (at eta=0.0) and a stochastic DDPM (at eta = 1.0)
    See section 5 of the DDIM paper https://arxiv.org/pdf/2010.02502 """

    epsilon_scaling: Optional[float] = None
    """epsilon_scaling: Optional[float] if not None, the predicted epsilon will
    be scaled down by the provided factor as
    introduced in https://arxiv.org/pdf/2308.15321""" ""

    initial_noise_scale: float = 1.0
    """For Diffusion models, scaling of initial noise"""

    inference_timesteps: int = 100
    """For Diffusion models, number of denoising timesteps"""

    clip_noise: int = 100
    """For Diffusion models, factor to clip noise of the sampling steps"""

    thresholding: bool = False
    """Whether to use the "dynamic thresholding" method.
    This is unsuitable for latent-space diffusion models such as Stable Diffusion."""

    dynamic_thresholding_ratio: float = 0.995
    """The ratio for the dynamic thresholding method. Valid only when `thresholding=True`."""

    sample_max_value: float = 6.0
    """The threshold value for dynamic thresholding. Valid only when `thresholding=True`."""


class TwoTowerDiffusionLCMGenerator(LCMGenerator):
    """Generates with a Two-tower Diffusion LCM model."""

    options: DiffusionLCMGeneratorOptions

    def __init__(
        self,
        model: AbstractLCModel,
        options: Optional[LCMGeneratorOptions] = None,
        eos_vec: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(model, options, eos_vec)

        assert isinstance(self.model, TwoTowerDiffusionLCModel), (
            "The TwoTowerDiffusionLCMGenerator expects a Diffusion LCM"
        )

        logger.info(
            f"Setting up the model with decoding_options: {options} -- {type(options)}"
        )
        model.prep_for_denoising(options)

    @torch.inference_mode()
    def __call__(
        self,
        batch_input: EmbeddingsBatch,
        max_gen_len: Optional[int] = None,
        min_gen_len: Optional[int] = None,
        temperature: float = 0.0,
        disable_cache: bool = False,
        **kwargs,
    ) -> SequenceGeneratorOutput:
        """
        :param input:
            `bacth_input` embedded and padded tensor sequence of the inputs
            `max_gen_len` max length to be generated for the given input
            `min_gen_len` minimum length to be generated for the given input
            `disable_cache` if True, do not use kv-caching
            `temperature` temperature to control the generation
        :returns:
            The output of the LCM generator, consists of :math:`N` lists of
            hypotheses for :math:`N` prompts. Each list has 1 Hypothesis
            (beam size = 1), of which `seq` has the  *Shape:* math:`(S+T, D)`
            (:math:`S` is the prompt length, :math:`T` the length of the
            generated sequence after the prompt and :math:`D` the model
            dimension.)

        """
        if self.options.seed:
            torch.manual_seed(self.options.seed)

        # Setup the variables
        batch_size, self.max_prompt_len, embed_dim = batch_input.seqs.size()
        prompt_padding_mask = batch_input.padding_mask
        if prompt_padding_mask is None:
            self.min_prompt_len = self.max_prompt_len
            self.prompt_padding_mask = None
            self.prompt_seq_lens = None
        else:
            self.prompt_seq_lens = prompt_padding_mask.seq_lens
            assert self.prompt_seq_lens is not None, (
                "Expecting a valid `self.prompt_seq_lens` Tensor, found `None`"
            )
            self.min_prompt_len = int(torch.min(self.prompt_seq_lens, dim=0)[0].item())

            # Keep the materialized mask
            self.prompt_padding_mask = prompt_padding_mask.materialize()

        if not max_gen_len:
            max_gen_len = self.max_seq_len

        # Make sure we do not accidentally set a max_gen_len that exceeds
        # the generator's model capability
        assert max_gen_len <= self.max_seq_len, (
            f"Generator can generate up to {self.max_seq_len} sequences, max_gen_len={max_gen_len}"
        )
        self.max_gen_len = max_gen_len

        if not min_gen_len:
            min_gen_len = self.min_seq_len

        assert min_gen_len > 0, (
            f"min_gen_len must be greater than or equal to 1, min_gen_len={min_gen_len}"
        )
        self.min_gen_len = min_gen_len

        if temperature == 0.0:
            # If the call doesn't pass a specific temperature,
            # use the default one from the decoding options
            temperature = self.options.lcm_temperature

        # Holds the generated sequences, scores and sample-dependent variables
        dtype = self.model.dtype
        device = batch_input.seqs.device
        self.temperature = temperature

        if disable_cache:
            self.state_bag = None
            self.context_state_bag = None
        else:
            self.state_bag = LCMIncrementalStateBag(
                self.max_prompt_len + self.max_gen_len
            )
            self.context_state_bag = LCMIncrementalStateBag(
                self.max_prompt_len + self.max_gen_len
            )

        # reserving full sequences capacity
        self.seqs = torch.zeros(
            (batch_size, self.max_prompt_len + self.max_gen_len, embed_dim),
            device=device,
            dtype=dtype,
        )
        self.step_scores = torch.zeros(
            (batch_size, self.max_prompt_len + self.max_gen_len),
            device=device,
        )
        self.lengths = torch.zeros(batch_size, dtype=torch.int, device=device) - 1

        # Hold the samples indices to return in order
        self.sample_indices = torch.arange(batch_size, device=device)
        # Output buffer
        self.hypotheses: List[List[Hypothesis]] = [[] for _ in range(batch_size)]

        # Bootstrap the sequences with the provided prompt.
        self.seqs[:, : self.max_prompt_len] = batch_input.seqs[:, : self.max_prompt_len]
        self.step_nr = self.min_prompt_len

        # A context we keep growing in each decoding step
        self.prefill()

        for self.step_nr in range(
            self.min_prompt_len, self.max_prompt_len + self.max_gen_len
        ):
            if not self._step():
                break

        return SequenceGeneratorOutput(self.hypotheses, counters=GenerationCounters())

    def state_bag_reorder(self, new_order: torch.Tensor) -> None:
        if self.state_bag is not None:
            self.state_bag.reorder(new_order)

        if self.context_state_bag is not None:
            self.context_state_bag.reorder(new_order)

    @torch.inference_mode()
    def prefill(self, **kwargs) -> None:
        """encode the prefix with the context encoder"""

        assert self.context_state_bag is not None, (
            "Expecting a context state bag to prefill"
        )

        context: EmbeddingsBatch

        prefill_len = self.step_nr - 1
        if prefill_len > 0:
            # normalize then encode
            input_seqs = self.seqs[:, :prefill_len]
            if self.model.config.sonar_normalizer_name is not None:
                input_seqs = self.model.sonar_normalizer.normalize(input_seqs)

            context = self.model.encode(
                EmbeddingsBatch(input_seqs, None),
                state_bag=self.context_state_bag,
                **kwargs,
            )

            self.context_state_bag.increment_step_nr(prefill_len)

        else:
            logger.warning(
                f"Skipping prefill since only a context size of {self.step_nr} is provided in the prefix"
            )
            context = EmbeddingsBatch(
                torch.empty(
                    (self.seqs.shape[0], 0, self.model.model_dim),
                    dtype=self.seqs.dtype,
                    device=self.seqs.device,
                )
            )

        self.context = context

    @torch.inference_mode()
    def _decode(
        self,
        seqs: torch.Tensor,
        padding_mask: Optional[PaddingMask] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output, context = self.model.predict_next_sentence(
            batch=EmbeddingsBatch(seqs, padding_mask),
            context=self.context,
            temperature=self.temperature,
            state_bag=self.state_bag,
            context_state_bag=self.context_state_bag,
            **kwargs,
        )
        self.context = context

        # Dummy scores
        scores = torch.zeros(seqs.shape[:-1])
        return output.seqs, scores

    def _step(self) -> bool:
        # Generate the next step output.

        if self.state_bag is None:
            # Without a state_bag, we're forwarding the full prefix
            # Encode the full context:

            model_output, step_score = self._decode(
                seqs=self.seqs[:, : self.step_nr],
                padding_mask=None,
            )
        else:
            # Since we're using a state_bag, we're only forwarding the last embedding
            model_output, step_score = self._decode(
                seqs=self.seqs[:, self.step_nr - 1 : self.step_nr],
                padding_mask=None,
            )

            self.state_bag.increment_step_nr()

        # model_output: EmbeddingBag
        return self.finalize_step(model_output, step_score)

    def finalize_step(
        self, model_output: torch.Tensor, step_score: torch.Tensor
    ) -> bool:
        """Post-processing and finalizing a step
        by checking all stopping criteria
        Takes the model's outputed embeddings (model_output)
        and their associated scores (step_score)
        If we're stepping, return True, else return False
        """
        already_finished = self.lengths > -1
        should_finish_now = torch.zeros_like(already_finished)

        model_last_output = model_output[:, -1]
        device = model_last_output.device

        # Ignore prompt positions between min-max prompt_len
        must_keep_going = None
        if self.step_nr < self.max_prompt_len:
            assert self.prompt_padding_mask is not None, (
                f"If self.prompt_padding_mas is None, then self.step_nr should start from self.max_prompt_len={self.max_prompt_len} - currently self.step_nr = {self.step_nr}"
            )
            mask = self.prompt_padding_mask[:, self.step_nr]
            model_last_output[mask] = self.seqs[mask, self.step_nr]
            must_keep_going = mask

        # Check stopping based on EOS similarity.
        if self.eos_threshold is not None and self.eos_vec is not None:
            sim2eos = torch.nn.functional.cosine_similarity(
                self.eos_vec.to(device), model_last_output
            )
            logger.debug(f"Similarity to eos vector: {sim2eos} vs {self.eos_threshold}")
            should_finish_now = should_finish_now | sim2eos.ge(self.eos_threshold)

        # Check stopping based on repetition.
        if (
            self.options.stop_on_repetition_cosine_threshold is not None
            and self.step_nr > 0
        ):
            sim2prev = torch.nn.functional.cosine_similarity(
                self.seqs[:, self.step_nr - 1], model_last_output
            )
            logger.debug(
                f"Similarity to prev vector: {sim2prev} vs {self.options.stop_on_repetition_cosine_threshold}"
            )
            should_finish_now = should_finish_now | sim2prev.ge(
                self.options.stop_on_repetition_cosine_threshold
            )

        if must_keep_going is not None:
            logger.debug(
                f"Must keep going (to cover max_prompt_len={self.max_prompt_len}) is not None = {must_keep_going}"
            )
            should_finish_now = should_finish_now & ~must_keep_going

        # Keep going if output is shorter than min_gen_len:
        if self.prompt_seq_lens is not None:
            longuer_than_min_gen_len = (self.step_nr - self.prompt_seq_lens).ge(
                self.min_gen_len
            )
        else:
            longuer_than_min_gen_len = (
                self.step_nr - self.max_prompt_len
            ) >= self.min_gen_len

        logger.debug(
            f"Longuer than min_gen_len ({self.min_gen_len}) = {longuer_than_min_gen_len}"
        )
        should_finish_now = should_finish_now & longuer_than_min_gen_len
        stopped_on_eos = should_finish_now

        # Stop hypotheses that reached max_gen_len
        if self.prompt_seq_lens is not None:
            exceeds_max_gen_len = (self.step_nr - self.prompt_seq_lens + 1).ge(
                self.max_gen_len
            )
            logger.debug(
                f"step: {self.step_nr}; max_gen_len: {self.max_gen_len}; promt_lens: {self.prompt_seq_lens}; steps exceeded: {self.max_gen_len + self.prompt_seq_lens}"
            )

        else:
            exceeds_max_gen_len = (
                self.step_nr - self.max_prompt_len + 1
            ) >= self.max_gen_len
            logger.debug(
                f"step: {self.step_nr}; max_gen_len: {self.max_gen_len}; promt_lens: None (unique length: {self.max_prompt_len}); steps exceeded: {self.max_prompt_len + self.max_gen_len}"
            )

        logger.debug(
            f"Stopping criteria: {should_finish_now}; exceeds max len: {exceeds_max_gen_len}; already finished: {already_finished}"
        )

        should_finish_now = should_finish_now | exceeds_max_gen_len

        # Assign lengths to the sequences that have just finished.
        should_finish_now = should_finish_now & ~already_finished
        self.lengths[should_finish_now] = self.step_nr + 1

        # Record the current step.
        self.seqs[:, self.step_nr] = model_last_output.squeeze(1)
        self.step_scores[:, self.step_nr - self.min_prompt_len] = step_score[:, -1]

        #  Save completed hypsptheses
        finished_mask = self.lengths.ne(-1)
        finished_indices = finished_mask.nonzero()

        # Remove finished hypotheses and reorder variables/state_bag if any are left
        if len(finished_indices) > 0:
            for idx in finished_indices:
                self.finish_sequence(int(idx), is_eos=bool(stopped_on_eos[int(idx)]))

        active_mask = ~finished_mask
        active_indices = active_mask.nonzero().squeeze(-1)

        if len(active_indices) == 0:
            return False

        self.reorder_state(active_indices)

        return True

    def finish_sequence(self, idx: int, is_eos: bool = False) -> None:
        seq_len = int(self.lengths[idx].item())

        if self.options.trim_hypotheses and self.lengths[idx].item() > -1 and is_eos:
            seq_len = int(self.lengths[idx].item()) - int(
                not self.options.include_eos_token
            )

        sample_idx = int(self.sample_indices[idx])
        self.hypotheses[sample_idx] = [
            Hypothesis(
                seq=self.seqs[idx, :seq_len],
                score=None,
                step_scores=self.step_scores[idx],  # Trim it as well?
            )
        ]

    def reorder_state(self, new_order: torch.Tensor) -> None:
        self.state_bag_reorder(new_order)

        self.context = EmbeddingsBatch(
            self.context.seqs.index_select(dim=0, index=new_order),
            self.context.padding_mask,
        )

        self.seqs = self.seqs.index_select(dim=0, index=new_order)

        self.sample_indices = self.sample_indices.index_select(dim=0, index=new_order)

        self.step_scores = self.step_scores.index_select(dim=0, index=new_order)

        self.lengths = self.lengths.index_select(dim=0, index=new_order)

        if self.prompt_padding_mask is not None:
            self.prompt_padding_mask = self.prompt_padding_mask.index_select(
                dim=0, index=new_order
            )

        if self.prompt_seq_lens is not None:
            self.prompt_seq_lens = self.prompt_seq_lens.index_select(
                dim=0, index=new_order
            )
