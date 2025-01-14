# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.
#
#

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from fairseq2.logging import get_log_writer
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import PaddingMask, pad_seqs
from fairseq2.typing import Device
from torch import Tensor
from torch.nn import Module

from lcm.utils.common import Batched

logger = get_log_writer(__name__)


DOC_LENGTHS = "__doc_lengths"


class LCMStyle(Enum):
    """Specifies a style for preparing the LCM input."""

    SUPERVISED = 1
    """For when the model is fed supervised data with source & target sentences."""

    UNSUPERVISED = 2
    """For when the model is fed unsupervised data with source sentences only."""

    PACKED_UNSUPERVISED = 3
    """For when the model is fed ``packed`` unsupervised data with source sentences only.
    This means that we will look for document_lengths and propagate them to the
    packed causal masked attention and the packed position encoders"""


@dataclass
class EmbeddingsBatch:
    """Represents a sequence of embeddings batch.
    Resembles Fairseq2's SequenceBatch with additional properties"""

    seqs: Tensor
    """The sequences. *Shape:* :math:`(B,S,D)`, where :math:`B` is the batch
    size, :math:`S` is the sequence length (in sentences per document),
    and :math:`D` the embedding dimension
    """

    padding_mask: Optional[PaddingMask] = None
    """The padding mask of ``seqs``. *Shape:* :math:`(B,S)`, where :math:`B` is
    the batch size and :math:`S` is the sequence length."""

    diffusion_timesteps: Optional[Tensor] = None
    """Diffusion timesteps of noising process of ``seqs``. *Shape:* :math:`(B,S)`, where :math:`B` is
    the batch size and :math:`S` is the sequence length."""

    document_lengths: Optional[Tensor] = None
    """Lengths of the documents (in sentences) present in the batch
    Shape: (len_doc, )
    """

    source_lengths: Optional[Tensor] = None
    """Lengths of source part for each element in batch, so that `seqs[i, :source_lengths[i]]` corresponds to source for each i in [0, batch_size).
    Shape: (batch_size, )
    """

    def __post_init__(self):
        if self.document_lengths is not None:
            assert self.document_lengths.sum() == self.seqs.size(
                1
            ) or 2 * self.document_lengths.sum() == self.seqs.size(1), (
                "The legnths do no sum up to the sequence length "
                "(nor half the length for doubled diffusion sequences). "
                f"We have seqs.size={self.seqs.size()} and lengths={self.document_lengths} "
                f"summing to {self.document_lengths.sum()}"
            )

    def __len__(self) -> int:
        return self.batch_size

    @property
    def batch_size(self) -> int:
        """The size of the batch."""
        return self.seqs.size(0)

    @property
    def shape(self) -> torch.Size:
        """The shape of the batch."""
        return self.seqs.shape

    @property
    def device(self) -> Device:
        """The device of the batch."""
        return self.seqs.device

    def clone(self):
        return deepcopy(self)

    def __getitem__(self, i: int) -> Any:
        raise NotImplementedError(
            "Access to each item in EmbeddingsBatch not allowed yet"
        )

    def unbatch(self) -> List[Tensor]:
        if self.padding_mask is None:
            return list(self.seqs)
        else:
            return [
                tt[:length] for tt, length in zip(self.seqs, self.padding_mask.seq_lens)
            ]

    def get_last_element(self) -> Tensor:
        if self.padding_mask:
            return self.seqs[
                torch.arange(len(self.padding_mask.seq_lens), device=self.seqs.device),
                (self.padding_mask.seq_lens - 1),
            ]
        else:
            return self.seqs[:, -1]

    def set_last_element(self, element: Tensor) -> None:
        element = element.to(self.seqs.device)
        if self.padding_mask:
            for i, slen in enumerate(self.padding_mask.seq_lens):
                self.seqs[i, slen - 1] = element[i]
        else:
            self.seqs[:, -1] = element

    def normalize_seqs(self, normalizer: Optional[Module]) -> "EmbeddingsBatch":
        if normalizer is None:
            logger.warning(
                "The normalizer is None, as such, the features will remain unchanged"
            )
            return self

        return EmbeddingsBatch(
            seqs=normalizer.normalize(self.seqs),
            padding_mask=self.padding_mask,
            diffusion_timesteps=self.diffusion_timesteps,
            document_lengths=self.document_lengths,
            source_lengths=self.source_lengths,
        )

    def denormalize_seqs(self, normalizer: Optional[Module]) -> "EmbeddingsBatch":
        if normalizer is None:
            logger.warning(
                "The normalizer is None, as such, the features will remain unchanged"
            )
            return self

        return EmbeddingsBatch(
            seqs=normalizer.denormalize(self.seqs),
            padding_mask=self.padding_mask,
            diffusion_timesteps=self.diffusion_timesteps,
            document_lengths=self.document_lengths,
            source_lengths=self.source_lengths,
        )

    def double_seqs(self) -> "EmbeddingsBatch":
        """
        performs sequence elements repeatition in sequence dim :
        1, 2, 3 -> 1, 1, 2, 2, 3, 3
        x, y -> x, x, y, y
        """
        if self.padding_mask is not None:
            doubled_padding_mask = PaddingMask(
                seq_lens=2 * self.padding_mask._seq_lens,
                batch_seq_len=2 * self.padding_mask._batch_seq_len,
            )
        else:
            doubled_padding_mask = None

        return EmbeddingsBatch(
            seqs=torch.repeat_interleave(self.seqs, 2, dim=1),
            padding_mask=doubled_padding_mask,
            diffusion_timesteps=self.diffusion_timesteps,
            document_lengths=self.document_lengths,
            source_lengths=(
                torch.repeat_interleave(self.source_lengths, 2, dim=0)
                if self.source_lengths is not None
                else None
            ),
        )

    def flatten_to_sentences(self) -> Tensor:
        """Flatten the sequence of embeddings
        from B, S, D to B*~S, D after removing the padded positions
        """

        embed_dim = self.seqs.size(-1)

        if self.padding_mask is not None:
            seq_lens = self.padding_mask.seq_lens

            embeds_mask = self.padding_mask.materialize().unsqueeze(-1)

            # Remove padded positions and reshape as B*~S, D
            flat_embeds = torch.masked_select(self.seqs, embeds_mask).reshape(
                (-1, embed_dim)
            )

            # split per document/paragraph
            flat_embeds_per_doc = list(torch.split(flat_embeds, seq_lens.tolist()))

            # Concatenate back
            flat_embeds = torch.concat(flat_embeds_per_doc)

        else:
            embeds = self.seqs

            flat_embeds = embeds.reshape((-1, embed_dim))

        return flat_embeds


@dataclass
class LCMInput(Batched):
    """Dataclass for a pair of source/target sequences of SONAR embeddings"""

    source: List[Tensor]
    """source: SONAR embeddings of the source text
            i.e [X^1 in (N_1, D), ... X^M in (N_M, D)]"""

    target: Union[None, List[Tensor]]
    """target: If supervised data:  SONAR embeddings of the target text"""

    tokens: Union[None, SequenceBatch] = None
    """tokens: Tokenized flattened sentences for the SONAR decoder
        (see the dataloader `_prepare_subword_tokens`)"""

    target_tokens: Union[None, SequenceBatch] = None
    """target_tokens: a sequence of the same shape as target_tokens, but shifted, to serve as the target.
        (see the dataloader `_prepare_subword_tokens`)"""

    name: Optional[str] = None
    """
    dataset name from which input is coming from
    """
    batch: Optional[Dict[str, Any]] = None
    """raw batch of dataloader used for tracking and debugging"""

    def __post_init__(self):
        assert self.source is not None

        length = len(self.source)

        assert (self.target is None) or (len(self.target) == length), (
            f"all elements in LCMInput should be of the same length, got {len(self.target)} and {length}"
        )

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, i: int) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Return the content of item in the batch
        """
        if self.target is None:
            return self.source[i]
        else:
            return self.source[i], self.target[i]

    def prepare_input(
        self,
        style: LCMStyle = LCMStyle.UNSUPERVISED,
    ) -> EmbeddingsBatch:
        """
        Adds special tokens to the source (& target) and prepares
        the EmbeddingsBatch (tensor & its padding mask) that will be
        forwarded in the LCM model.

        `style`: LCMStyle is either supervised or
                unsupervised (requires target embeddings)
        """

        if style == LCMStyle.UNSUPERVISED:
            return get_embeddings_sequence(src_seqs=self.source)

        elif style == LCMStyle.PACKED_UNSUPERVISED:
            # If using PACKED_UNSUPERVISED, document_lengths will be added to `EmbeddingsBatch`
            document_lengths = None
            if self.batch is not None and self.batch.get(DOC_LENGTHS, None) is not None:
                # document_lengths will only be consumed if the batch_size is 1
                assert len(self.batch[DOC_LENGTHS]) == 1, "Expecting batch size of 1"

                document_lengths = self.batch[DOC_LENGTHS][0].type(torch.int64)

            return get_embeddings_sequence(
                src_seqs=self.source,
                document_lengths=document_lengths,
            )

        elif style == LCMStyle.SUPERVISED:
            assert self.target is not None, (
                "Missing target embeddings for a supervised batch"
            )
            return get_embeddings_sequence(
                src_seqs=self.source,
                tgt_seqs=self.target,
            )

        raise ValueError(f"Unsupported style={style} - could not prepare input")

    def prepare_target_mask(
        self,
        embeddings: EmbeddingsBatch,
        style: LCMStyle,
        min_context_size: Optional[int] = None,
    ) -> Tensor:
        """Prepare a target mask signaling what positions
            we should predict and optimize the model for

        Args:
            - min_context_size: the minimum context used to predict the next
                concept (only used for unuspervised training)

        """

        batch_size, maxlen, _ = embeddings.seqs.size()

        device = embeddings.seqs.device

        if style == LCMStyle.UNSUPERVISED:
            # A target mask for unsupervised next sentence prediction
            # All positions are optimized/predicted starting from min_context_size
            target_mask = torch.ones(
                (batch_size, maxlen),
                dtype=torch.bool,
                device=device,
            )
            if min_context_size is not None:
                target_mask[:, : min(min_context_size, target_mask.size(1))] = False

        elif style == LCMStyle.PACKED_UNSUPERVISED:
            # A target mask for unsupervised next sentence prediction when the data is packed
            # All positions are optimized starting from min_context_size in each document
            document_lengths = embeddings.document_lengths
            if document_lengths is not None:  # training

                def get_document_target_mask(doc_length):
                    mask = torch.ones(doc_length, dtype=torch.bool, device=device)
                    mask[: min(min_context_size, doc_length)] = False
                    return mask

                target_mask = torch.cat(
                    [get_document_target_mask(length) for length in document_lengths]
                ).unsqueeze(0)

            else:  # validation with unpacked data:
                target_mask = torch.ones(
                    (batch_size, maxlen),
                    dtype=torch.bool,
                    device=device,
                )
                if min_context_size is not None:
                    target_mask[:, : min(min_context_size, target_mask.size(1))] = False

        elif style == LCMStyle.SUPERVISED:
            # A target mask for target prediction
            indices = torch.arange(maxlen, device=device).expand(batch_size, -1)

            source_lengths = torch.tensor(
                [seq.size(0) for seq in self.source],
                device=device,
            )

            target_mask = indices >= source_lengths.unsqueeze(1).expand(-1, maxlen)

        # Factor in padded positions:
        if embeddings.padding_mask is not None:
            target_mask = target_mask * embeddings.padding_mask.materialize()

        return target_mask.detach()


def get_embeddings_sequence(
    src_seqs: List[Tensor],
    tgt_seqs: Optional[List[Tensor]] = None,
    document_lengths: Optional[Tensor] = None,
    double_target: bool = False,
) -> EmbeddingsBatch:
    seqs_lst: List[Tensor] = []
    for src_seq, tgt_seq in zip(src_seqs, tgt_seqs or [None] * len(src_seqs)):  # type: ignore
        embeds: List[Tensor] = []
        device, dtype = src_seq.device, src_seq.dtype

        # mandatory src_sec
        embeds.append(src_seq)

        # supervised tgt_seq
        if tgt_seq is not None:
            tgt_seq = tgt_seq.to(device).type(dtype)

            if double_target:
                embeds.append(torch.repeat_interleave(tgt_seq, 2, dim=0))
            else:
                embeds.append(tgt_seq)

        seqs_lst.append(torch.concat(embeds))

    seqs, padding_mask = pad_seqs(seqs_lst)

    if document_lengths is not None:
        document_lengths = document_lengths.to(seqs.device)

    if tgt_seqs is not None:
        source_lengths = torch.tensor(
            [seq.size(0) for seq in src_seqs], device=seqs.device
        )
    else:
        source_lengths = None

    output = EmbeddingsBatch(
        seqs,
        padding_mask=padding_mask,
        document_lengths=document_lengths,
        source_lengths=source_lengths,
    )

    return output
