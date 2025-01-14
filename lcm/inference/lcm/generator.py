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
from lcm.models.abstract_lcm import AbstractLCModel
from lcm.nn.incremental_state import LCMIncrementalStateBag

logger = get_log_writer(__name__)


"""
This generator follows the style of existing generators in Fairseq2
"""


@dataclass
class LCMGeneratorOptions:
    """Holds the options to pass to a sequence generator."""

    max_seq_len: int = 200
    """The hard limit on maximum length of generated sequences."""

    min_seq_len: int = 1
    """The minimum length of generated sequences."""

    eos_threshold: Optional[float] = 0.9
    """Threshold for cosine similarity to the EOS vector"""

    sample_latent_variable: bool = True
    """When using VAE models, whether to return the mean or sample"""

    stop_on_repetition_cosine_threshold: Optional[float] = None
    """Stop the generation when the similarity of two consecutive concepts is above the threshold."""

    include_eos_token: bool = False
    """Whether the eos token should be included in the hypotheses (matters only if they are trimmed)."""

    trim_hypotheses: bool = False
    """Whether the tokens after the EOS token should be included in the hypotheses."""

    seed: Optional[int] = None
    """Seed to make generation deterministic"""

    lcm_temperature: float = 1.0
    """Temperature for decoding in the LCM"""


class LCMGenerator:
    """Generates with an LCM model."""

    def __init__(
        self,
        model: AbstractLCModel,
        options: Optional[LCMGeneratorOptions] = None,
        eos_vec: Optional[torch.Tensor] = None,
    ) -> None:
        """
        :param model:
            The LC model to use for generation.
        """
        model.eval()
        self.model = model

        if options is None:
            options = LCMGeneratorOptions()

        self.eos_vec = eos_vec
        if self.eos_vec is None and options.eos_threshold:
            logger.warning(
                f"eos_threshold is set to {options.eos_threshold}, but eos_vec is not provided"
            )
        if options.eos_threshold:
            logger.debug(f"The eos_vec in generator has been set to {self.eos_vec}")

        self.options = options

        self.max_seq_len = options.max_seq_len
        self.min_seq_len = options.min_seq_len

        assert self.min_seq_len >= 1, (
            f"min_seq_len must be greater than or equal to 1, min_seq_len={options.min_seq_len}"
        )

        self.eos_threshold = options.eos_threshold

        self.seqs: torch.Tensor
        self.step_nr = 0
        self.min_prompt_len: int
        self.max_prompt_len: int
        self.sample_indices: torch.Tensor
        self.state_bag: Optional[LCMIncrementalStateBag] = None
        self.prompt_seq_lens: Optional[torch.Tensor] = None
        self.prompt_padding_mask: Optional[torch.Tensor] = None
        self.lengths: torch.Tensor
        self.step_scores: torch.Tensor

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
            `temperature` temperature to control the generation
            `disable_cache` if True, do not use kv-caching
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

        self.temperature = temperature

        for k, v in kwargs.items():
            if hasattr(self.options, k) and v:
                setattr(self.options, k, v)

        # Holds the generated sequences, scores and sample-dependent variables
        dtype = self.model.dtype
        device = batch_input.seqs.device

        if disable_cache:
            self.state_bag = None
        else:
            self.state_bag = LCMIncrementalStateBag(
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
        self.prefill(**kwargs)

        for self.step_nr in range(
            self.min_prompt_len, self.max_prompt_len + self.max_gen_len
        ):
            if not self._step():
                break

        return SequenceGeneratorOutput(self.hypotheses, counters=GenerationCounters())

    @torch.inference_mode()
    def prefill(self, **kwargs) -> None:
        """The initial forward pass in the decoder with the prefix/prompt
        to populate the KV-cache"""

        if self.state_bag is None:
            return

        # Prefilling with -1 since the next call to step will use the last token in the prefix
        prefill_len = self.step_nr - 1

        if prefill_len > 0:
            _ = self._decode(
                self.seqs[:, :prefill_len],
                padding_mask=None,
            )
            self.state_bag.increment_step_nr(prefill_len)  # type: ignore
        else:
            logger.warning(
                f"Skipping prefill since only a context size of {self.step_nr} is provided in the prefix"
            )

    @torch.inference_mode()
    def _decode(
        self,
        seqs: torch.Tensor,
        padding_mask: Optional[PaddingMask],
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.model.predict_next_sentence(
            EmbeddingsBatch(seqs, padding_mask),
            sample=self.options.sample_latent_variable,
            temperature=self.temperature,
            state_bag=self.state_bag,
            **kwargs,
        )

        # Dummy scores
        scores = torch.zeros(seqs.shape[:-1])
        return output.seqs, scores

    def _step(self) -> bool:
        # Generate the next step output.

        if self.state_bag is None:
            # Without a state_bag, we're forwarding the full prefix
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
            longer_than_min_gen_len = (self.step_nr - self.prompt_seq_lens).ge(
                self.min_gen_len
            )
        else:
            longer_than_min_gen_len = (
                self.step_nr - self.max_prompt_len
            ) >= self.min_gen_len

        logger.debug(
            f"Longer than min_gen_len ({self.min_gen_len}) = {longer_than_min_gen_len}"
        )
        should_finish_now = should_finish_now & longer_than_min_gen_len
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

    def state_bag_reorder(self, new_order: torch.Tensor) -> None:
        if self.state_bag is not None:
            self.state_bag.reorder(new_order)

    def reorder_state(self, new_order: torch.Tensor) -> None:
        self.state_bag_reorder(new_order)

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
