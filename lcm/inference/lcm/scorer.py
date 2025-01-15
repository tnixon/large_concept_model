# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
#

from typing import List, Optional

import torch
from fairseq2.generation.generator import (
    GenerationCounters,
    Hypothesis,
    SequenceGeneratorOutput,
)

from lcm.datasets.batch import EmbeddingsBatch
from lcm.inference.lcm.generator import LCMGenerator, LCMGeneratorOptions
from lcm.nn.incremental_state import LCMIncrementalStateBag


class LCMScorer(LCMGenerator):
    """Generates with an LCM model in teacher-forcing mode."""

    options: LCMGeneratorOptions

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
            dimension

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
        else:
            self.state_bag = LCMIncrementalStateBag(self.max_text_len)

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
        self.prefill()

        for self.step_nr in range(self.min_context_len, self.max_text_len):
            if not self._step():
                break

        return SequenceGeneratorOutput(self.hypotheses, counters=GenerationCounters())

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

        #  Save completed hypotheses
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

    def finish_sequence(self, idx: int, is_eos: bool = False) -> None:
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

        self.seqs = self.seqs.index_select(dim=0, index=new_order)
        self.preds = self.preds.index_select(dim=0, index=new_order)

        self.sample_indices = self.sample_indices.index_select(dim=0, index=new_order)

        self.step_scores = self.step_scores.index_select(dim=0, index=new_order)

        if self.text_padding_mask is not None:
            self.text_padding_mask = self.text_padding_mask.index_select(
                dim=0, index=new_order
            )

        self.text_seq_lens = self.text_seq_lens.index_select(dim=0, index=new_order)
