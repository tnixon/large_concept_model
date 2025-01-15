#  Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
#

import os
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from fairseq2.generation import BeamSearchSeq2SeqGenerator, TextTranslator
from fairseq2.logging import get_log_writer
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.nllb import load_nllb_tokenizer
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.transformer import load_transformer_model
from fairseq2.nn.padding import pad_seqs
from fairseq2.typing import CPU, DataType, Device
from sacrebleu import CHRF
from sonar.models.sonar_text import load_sonar_tokenizer
from stopes.pipelines.monolingual.utils.word_tokenization import get_word_tokenizer
from tqdm.auto import tqdm, trange
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPT2LMHeadModel,
)

from lcm.evaluation.api import PREDICTION_COLUMN, Scorer
from lcm.evaluation.utils.hf import infer_cache_dir

logger = get_log_writer(__name__)


def count_nonzero_non_nan(tensor):
    # Create a mask for non-zero and non-NaN values
    mask = (tensor != 0) & (~torch.isnan(tensor))
    # Count the number of True values in the mask
    count = torch.sum(mask)
    return count


class FluencyClassifierScorer(Scorer):
    """
    A class for applying a fluency classifier (e.g. based on a corpus of linguistic acceptability) to sentences.
    """

    def __init__(
        self,
        model_name: str = "cointegrated/roberta-large-cola-krishna2020",
        outputs: str = "sentence_fluency",
        class_id: int = 0,
        device: Device = CPU,
        dtype: Optional[DataType] = None,
        batch_size: int = 16,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            outputs=outputs,
            device=device,
            dtype=dtype,
            **kwargs,
        )
        self.class_id = class_id
        self.batch_size = batch_size

    def init_model(self):
        cache_dir = self.kwargs.get("cache_dir", infer_cache_dir())
        token = os.environ.get("HF_AUTH_TOKEN", None)
        offload_folder = self.kwargs.get("offload_folder", None)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            token=token,
            clean_up_tokenization_spaces=False,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            # device_map="auto",
            cache_dir=cache_dir,
            token=token,
            offload_folder=offload_folder,
        ).to(self.device)

    def score_texts(
        self,
        texts: Sequence[str],
        references: Optional[Sequence[str]] = None,
        show_progress: bool = False,
    ) -> np.ndarray:  # type: ignore
        """Output an array of predicted probabilities of being fluent for each text."""
        results = []
        range_fn = trange if show_progress else range
        for i in range_fn(0, len(texts), self.batch_size):  # type: ignore
            batch = self.tokenizer(
                texts[i : i + self.batch_size],
                max_length=self.tokenizer.model_max_length,
                padding=True,
                truncation=True,
                return_overflowing_tokens=True,
                stride=8,  # overlap between truncated and overflowing sequences
                return_tensors="pt",
            ).to(self.device)
            num_samples = len(texts[i : i + self.batch_size])
            overflow_map = batch["overflow_to_sample_mapping"]
            del batch["overflow_to_sample_mapping"]
            with torch.inference_mode():
                out = self.model(**batch)
                proba = torch.softmax(out.logits, axis=-1)[:, self.class_id]  # type: ignore
                # average over overflown tokens  # TODO: vectorize
                sample_proba = []
                for sample_id in range(num_samples):
                    sample_proba.append(
                        proba[overflow_map == sample_id].mean(dim=0).item()
                    )
                results.append(np.array(sample_proba))
        return np.concatenate(results)


class PerplexityScorer(Scorer):
    """
    A class for computing sentence perplexities with a language model (e.g. GPT2)
    """

    def __init__(
        self,
        model_name: str = "gpt2-medium",
        outputs: str = "sentence_perplexity",
        class_id: int = 0,
        device: Device = CPU,
        dtype: Optional[DataType] = None,
        batch_size: int = 4,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            outputs=outputs,
            device=device,
            dtype=dtype,
            **kwargs,
        )
        self.class_id = class_id
        self.batch_size = batch_size

    def init_model(self):
        cache_dir = self.kwargs.get("cache_dir", infer_cache_dir())
        token = os.environ.get("HF_AUTH_TOKEN", None)
        offload_folder = self.kwargs.get("offload_folder", None)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            token=token,
            clean_up_tokenization_spaces=False,
        )
        assert self.model_name == "gpt2-medium"

        # Special tokens for GPT2
        self.tokenizer.padding_side = "right"
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.model = (
            GPT2LMHeadModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                # device_map="auto",
                cache_dir=cache_dir,
                token=token,
                offload_folder=offload_folder,
            )
            .to(self.device)
            .eval()
        )

        # Resize embeddings
        _ = self.model.resize_token_embeddings(len(self.tokenizer))
        # Initialize with eos/bos/unk weights
        self.model.transformer.wte.weight.data[-1] = (
            self.model.transformer.wte.weight.data[self.tokenizer.eos_token_id]
        )

    def score_texts(
        self,
        texts: Sequence[str],
        references: Optional[Sequence[str]] = None,
        show_progress: bool = False,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
    ) -> np.ndarray:  # type: ignore
        """
        Output an array of perplexities of being fluent for each text.
        """
        # Add special tokens:
        if bos_token != "":
            bos_token = bos_token or getattr(self.tokenizer, "bos_token", "\n")
        if eos_token != "":
            eos_token = eos_token or getattr(self.tokenizer, "eos_token", "\n")
        assert eos_token is not None and bos_token is not None, (
            "Expecting eos and bos tokens, for perplexity without any surrounding tokens, use eos_token='' and bos_token=''"
        )
        logger.info(
            f"Computing perplexity with bos_token={repr(bos_token)} and eos_token={repr(eos_token)}"
        )

        texts = [bos_token + text + eos_token for text in texts]
        results = []
        range_fn = trange if show_progress else range
        with torch.inference_mode():
            for i in range_fn(0, len(texts), self.batch_size):
                num_samples = len(texts[i : i + self.batch_size])
                inputs = self.tokenizer(
                    texts[i : i + self.batch_size],
                    max_length=self.tokenizer.model_max_length,
                    padding=True,
                    truncation=True,
                    stride=8,
                    return_overflowing_tokens=True,
                    return_tensors="pt",
                ).to(self.device)

                overflow_map = inputs["overflow_to_sample_mapping"]
                del inputs["overflow_to_sample_mapping"]

                out = self.model(**inputs)
                shift_logits = out.logits[..., :-1, :].contiguous()
                shift_labels = inputs.input_ids[..., 1:].contiguous()

                flat_logits = shift_logits.view(-1, self.model.config.vocab_size)
                flat_labels = shift_labels.view(
                    -1,
                )
                loss = torch.nn.functional.cross_entropy(
                    flat_logits,
                    target=flat_labels,
                    ignore_index=self.tokenizer.pad_token_id,
                    reduction="none",
                ).view(shift_labels.shape)
                # average over overflown tokens  # TODO: vectorize
                for sample_id in range(num_samples):
                    ppl = loss[overflow_map == sample_id].nansum() / (
                        count_nonzero_non_nan(loss[overflow_map == sample_id]) + 1e-6
                    )
                    results.append(ppl.item())

        return np.array(results)


class RoundTripTranslationScorer(Scorer):
    """
    A class for computing sentence round-trip-translation reconstructability
    """

    def __init__(
        self,
        model_name: str = "nllb-200_dense_distill_600m",
        outputs: str = "round_trip_translation",
        class_id: int = 0,
        device: Device = CPU,
        dtype: DataType = torch.float32,
        batch_size: int = 16,
        lang: str = "eng_Latn",
        other_lang: str = "rus_Cyrl",
        max_seq_len: int = 128,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            outputs=outputs,
            device=device,
            dtype=dtype,
            **kwargs,
        )
        self.class_id = class_id
        self.lang = lang
        self.other_lang = other_lang
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    def init_model(self):
        self.model = load_transformer_model(
            self.model_name, device=self.device, dtype=self.dtype
        )
        self.tokenizer = load_nllb_tokenizer(self.model_name)

    def get_nllb_loss(
        self, src_texts: List[str], tgt_texts: List[str], src_lang: str, tgt_lang: str
    ):
        """Compute the mean per-observation loss for an NLLB translation model.
        The first target token (language code) is excluded from the final computation.
        """
        src_enc = self.tokenizer.create_encoder(
            task="translation", lang=src_lang, mode="source", device=self.device
        )
        tgt_enc = self.tokenizer.create_encoder(
            task="translation", lang=tgt_lang, mode="target", device=self.device
        )
        src_seqs = [src_enc(text) for text in src_texts]
        tgt_seqs = [tgt_enc(text) for text in tgt_texts]

        # do the padding of source, target prefix, and target label sentences
        source_padded, source_padding_mask = pad_seqs(src_seqs)
        target_prefix_seqs, target_prefix_padding_mask = pad_seqs(
            [seq[:-1] for seq in tgt_seqs]
        )
        target_labels_seqs, _ = pad_seqs([seq[1:] for seq in tgt_seqs])
        # The target_labels_padding_mask mask can be None (e.g. when abels_seqs consists of
        # sequences of the same length) So we should count the sequence length from the original tensor
        labels_seq_len = torch.IntTensor([seq[1:].size(-1) for seq in tgt_seqs])

        target_labels_seqs[:, 0] = 0  # ignore the loss for the language id token

        batch = Seq2SeqBatch(
            source_seqs=source_padded,
            source_padding_mask=source_padding_mask,
            target_seqs=target_prefix_seqs,
            target_padding_mask=target_prefix_padding_mask,
        )
        with torch.inference_mode():
            out = self.model(batch)
            # out.logits.shape torch.Size([16, 17, 256206])
            loss = torch.nn.functional.cross_entropy(
                out.logits.view(-1, out.logits.shape[-1]),
                target_labels_seqs.view(-1),
                reduction="none",
                ignore_index=0,
            ).view(out.logits.shape[:2])
            loss_per_token = loss.sum(-1) / (labels_seq_len - 1)  # type: ignore

        return loss_per_token.cpu().numpy().astype(np.float32)

    def backtranslate(
        self, texts: Sequence[str]
    ) -> Tuple[List[str], List[str], np.ndarray]:
        src_lang, tgt_lang = self.lang, self.other_lang
        translations = []
        back_translations = []
        losses = []
        assert isinstance(self.model, EncoderDecoderModel), (
            f"Unsupported type: {type(self.model)}"
        )
        generator = BeamSearchSeq2SeqGenerator(
            self.model, echo_prompt=True, max_seq_len=self.max_seq_len
        )
        translator1 = TextTranslator(
            generator,
            self.tokenizer,
            source_lang=src_lang,
            target_lang=tgt_lang,
        )
        translator2 = TextTranslator(
            generator,
            self.tokenizer,
            source_lang=tgt_lang,
            target_lang=src_lang,
        )
        for i in trange(0, len(texts), self.batch_size):
            texts0 = texts[i : i + self.batch_size]
            texts1 = translator1.batch_translate(texts0)[0]
            texts2 = translator2.batch_translate(texts1)[0]
            translations.extend(texts1)
            back_translations.extend(texts2)
            losses.extend(self.get_nllb_loss(texts1, texts0, tgt_lang, src_lang))  # type: ignore
        return translations, back_translations, np.array(losses)

    def score_texts(
        self,
        texts: Sequence[str],
        references: Optional[Sequence[str]] = None,
        show_progress: bool = False,
    ) -> np.ndarray:  # type: ignore
        """Return round-trip chrF++ scores for each sentence"""
        _, backtranslated, _ = self.backtranslate(texts)
        chrf_eval = CHRF(word_order=2)
        results = np.array(
            [
                chrf_eval.sentence_score(new, [old]).score
                for new, old in tqdm(zip(backtranslated, texts))
            ]
        )
        return results


def repeating_share(tokens: List[str]) -> float:
    ngrams = tokens
    if len(ngrams) == 0:
        return 0
    return 1.0 - len((set(ngrams))) / len(ngrams)


class WordRepetitionScorer(Scorer):
    def __init__(
        self,
        inputs: Union[str, Tuple[str, ...]] = PREDICTION_COLUMN,
        outputs: str = "word_repetition",
        language: str = "eng_Latn",
        **kwargs,
    ):
        self.tokenizer = get_word_tokenizer(language[:3])
        if isinstance(inputs, str):
            self.inputs = (inputs,)
        else:
            self.inputs = inputs

        if not outputs:
            outputs = self.__class__.__name__
        if isinstance(outputs, str):
            self.outputs = (outputs,)
        else:
            self.outputs = outputs

    def init_model(self):
        pass

    def score_texts(
        self,
        texts: Sequence[str],
        references: Optional[Sequence[str]] = None,
        show_progress: bool = False,
    ) -> np.ndarray:  # type: ignore
        """
        Output the list of fractions of repeating words in each text.
        """
        results = []
        itr = tqdm(texts) if show_progress else texts
        with torch.inference_mode():
            for sent in itr:  # type: ignore
                tokens = self.tokenizer.tokenize(sent)
                results.append(repeating_share(tokens))
        return np.array(results)


class TokenRepetitionScorer(Scorer):
    def __init__(
        self,
        inputs: Tuple[str, ...] = PREDICTION_COLUMN,  # type: ignore
        outputs: str = "token_repetition",
        language: str = "eng_Latn",
        tokenizer_name: str = "text_sonar_basic_encoder",
        **kwargs,
    ):
        self.tokenizer = load_sonar_tokenizer(tokenizer_name).create_encoder(
            lang=language
        )
        if isinstance(inputs, str):
            self.inputs = (inputs,)
        else:
            self.inputs = inputs

        if not outputs:
            outputs = self.__class__.__name__
        if isinstance(outputs, str):
            self.outputs = (outputs,)
        else:
            self.outputs = outputs

    def score_texts(
        self,
        texts: Sequence[str],
        references: Optional[Sequence[str]] = None,
        show_progress: bool = False,
    ) -> np.ndarray:  # type: ignore
        """
        Output the list of fractions of repeating tokens in each text.
        """
        results = []
        itr = tqdm(texts) if show_progress else texts
        with torch.inference_mode():
            for sent in itr:  # type: ignore
                # we skip the first token (language code) and the last one (EOS)
                tokens = self.tokenizer.encode_as_tokens(sent)[1:-1]
                results.append(repeating_share(tokens))
        return np.array(results)
