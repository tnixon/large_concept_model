# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from typing import List, Optional, Tuple

import torch
from fairseq2.generation.generator import (
    GenerationCounters,
    Hypothesis,
    SequenceGeneratorOutput,
)
from fairseq2.logging import get_log_writer

from lcm.datasets.batch import EmbeddingsBatch, PaddingMask
from lcm.inference.lcm.generator import LCMGeneratorOptions
from lcm.inference.two_tower_diffusion_lcm import (
    TwoTowerDiffusionLCMGenerator,
)
from lcm.models.abstract_lcm import AbstractLCModel
from lcm.nn.incremental_state import LCMIncrementalStateBag

logger = get_log_writer(__name__)


class TwoTowerDiffusionLCMScorer(TwoTowerDiffusionLCMGenerator):
    """Score by generating in teacher-forcing mode with a Two-tower Diffusion LCM model."""

    def __init__(
        self,
        model: AbstractLCModel,
        options: Optional[LCMGeneratorOptions] = None,
        eos_vec: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(model, options, eos_vec)

    @torch.inference_mode()
    def __call__(  # type: ignore
        self,
        batch_input: EmbeddingsBatch,
        max_gen_len: Optional[int] = None,
        min_gen_len: Optional[int] = None,
        min_context_len: int = 1,
        temperature: float = 0.0,
        disable_cache: bool = False,
    ) -> SequenceGeneratorOutput:
        """
        :param input:
            `bacth_input` embedded and padded tensor sequence of the inputs
            `max_gen_len` max length to be generated for the given input
            `min_gen_len` minimum length to be generated for the given input
            `disable_cache` if True, do not use kv-caching
        :returns:
            The output of the LCM generator, consists of :math:`N` lists of
            hypotheses for :math:`N` documents. Each list has 1 Hypothesis
            (beam size = 1), of which `seq` has the  *Shape:* math:`(T, D)`
            (:math:`T` the length of the document and :math:`D` the model
            dimension.)

        """
        if self.options.seed:
            torch.manual_seed(self.options.seed)

        # Setup the variables
        self.min_context_len = min_context_len
        batch_size, self.max_text_len, embed_dim = batch_input.seqs.size()
        text_padding_mask = batch_input.padding_mask
        if text_padding_mask is None:
            self.text_padding_mask = None
            self.text_seq_lens = self.max_text_len * torch.ones(
                batch_size,
                dtype=torch.long,
                device=batch_input.seqs.device,
            )
        else:
            self.text_seq_lens = text_padding_mask.seq_lens
            assert self.text_seq_lens is not None, (
                "Expecting a valid `self.text_seq_lens` Tensor, found `None`"
            )

            # Keep the materialized mask
            self.text_padding_mask = text_padding_mask.materialize()

        if not max_gen_len:
            max_gen_len = self.max_seq_len

        max_gen_len = min(max_gen_len, self.max_text_len - self.min_context_len)
        assert max_gen_len is not None, "max_gen_len is None"

        # Make sure we do not accidentally set a max_gen_len that exceeds
        # the generator's model capability
        assert max_gen_len <= self.max_seq_len, (
            f"Generator can generate up to {self.max_seq_len} sequences, max_gen_len={max_gen_len}"
        )
        self.max_gen_len = max_gen_len

        if not min_gen_len:
            min_gen_len = self.min_seq_len

        assert min_gen_len is not None, "A `min_gen_len` is required"

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
            self.state_bag = LCMIncrementalStateBag(self.max_text_len)
            self.context_state_bag = LCMIncrementalStateBag(self.max_text_len)

        # reserving full sequences capacity
        self.seqs = batch_input.seqs
        self.preds = torch.zeros(
            (batch_size, self.max_text_len - self.min_context_len, embed_dim),
            device=device,
            dtype=dtype,
        )

        self.step_scores = torch.zeros(
            (batch_size, self.max_text_len),
            device=device,
        )
        # Hold the samples indices to return in order
        self.sample_indices = torch.arange(batch_size, device=device)
        # Output buffer
        self.hypotheses: List[List[Hypothesis]] = [[] for _ in range(batch_size)]

        # the sequences with the provided prompt.
        self.step_nr = self.min_context_len

        # A context we keep growing in each decoding step
        self.prefill()

        for self.step_nr in range(self.min_context_len, self.max_text_len):
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

        # FIXME for this model we can prefill with self.step_nr
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

        # Dummy score
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
        model_last_output = model_output[:, -1]
        must_keep_going = self.text_seq_lens.gt(self.step_nr + 1)
        should_finish_now = ~must_keep_going

        # Record the current step prediction.
        self.preds[:, self.step_nr - self.min_context_len] = model_last_output.squeeze(
            1
        )
        self.step_scores[:, self.step_nr - self.min_context_len] = step_score[:, -1]

        #  Save completed hypsptheses
        finished_indices = should_finish_now.nonzero()

        # Remove finished hypotheses and reorder variables/state_bag if any are left
        if len(finished_indices) > 0:
            for idx in finished_indices:
                self.finish_sequence(int(idx))

        active_mask = must_keep_going
        active_indices = active_mask.nonzero().squeeze(-1)

        if len(active_indices) == 0:
            return False

        self.reorder_state(active_indices)

        return True

    def finish_sequence(self, idx: int) -> None:  # type: ignore
        seq_len = int(self.text_seq_lens[idx].item())
        sample_idx = int(self.sample_indices[idx])
        self.hypotheses[sample_idx] = [
            Hypothesis(
                seq=self.preds[idx, : seq_len - self.min_context_len],
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
        self.preds = self.preds.index_select(dim=0, index=new_order)

        self.sample_indices = self.sample_indices.index_select(dim=0, index=new_order)

        self.step_scores = self.step_scores.index_select(dim=0, index=new_order)

        if self.text_padding_mask is not None:
            self.text_padding_mask = self.text_padding_mask.index_select(
                dim=0, index=new_order
            )

        self.text_seq_lens = self.text_seq_lens.index_select(dim=0, index=new_order)
