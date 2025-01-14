#  Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
#

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from bert_score import BERTScorer
from fairseq2.generation import BeamSearchSeq2SeqGenerator, TextTranslator
from fairseq2.logging import get_log_writer
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.nllb import load_nllb_tokenizer
from fairseq2.models.transformer import load_transformer_model
from fairseq2.typing import CPU, DataType, Device
from rouge_score import rouge_scorer
from tqdm.auto import trange

from lcm.datasets.sentence_splitting import get_split_algo
from lcm.evaluation.api import Scorer

logger = get_log_writer(__name__)


class TranslatedRougeScorer(Scorer):
    """
    A class for comparing the texts with ROUGE after translating them into English.
    By default, a heavy `nllb-200_dense_3b` is used as the translator.
    It can be replaced with a smaller `nllb-200_dense_distill_600m`, if needed.
    """

    def __init__(
        self,
        model_name: str = "nllb-200_dense_3b",
        outputs: Tuple[str, str, str] = ("eng_rouge2", "eng_rougeL", "eng_rougeLsum"),
        device: Device = CPU,
        dtype: DataType = torch.float32,
        batch_size: int = 16,
        tgt_lang: str = "eng_Latn",
        src_lang: str = "ell_Grek",
        max_seq_len: int = 128,
        **kwargs,
    ):
        self.tgt_lang = tgt_lang
        self.src_lang = src_lang
        super().__init__(
            model_name=model_name,
            outputs=outputs,
            device=device,
            dtype=dtype,
            **kwargs,
        )
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    def init_model(self):
        self.model = load_transformer_model(
            self.model_name, device=self.device, dtype=self.dtype
        )
        self.tokenizer = load_nllb_tokenizer(self.model_name)
        self.splitter = get_split_algo(self.src_lang[:3], "default")

    def translate(
        self,
        texts: Sequence[str],
        show_progress: bool = False,
    ) -> List[str]:
        src_lang, tgt_lang = self.src_lang, self.tgt_lang
        sent_translations = []
        assert isinstance(self.model, EncoderDecoderModel), (
            f"Unsupported type: {type(self.model)}"
        )
        generator = BeamSearchSeq2SeqGenerator(
            self.model, echo_prompt=True, max_seq_len=self.max_seq_len
        )
        translator = TextTranslator(
            generator,
            self.tokenizer,
            source_lang=src_lang,
            target_lang=tgt_lang,
        )
        # split the input documents into individual sentences, so that NLLB could translate them correctly
        sentenized = [list(self.splitter(text)) for text in texts]
        sentences = [sent for doc in sentenized for sent in doc]
        range_fn = trange if show_progress else range
        for i in range_fn(0, len(sentences), self.batch_size):  # type: ignore
            inputs = sentences[i : i + self.batch_size]
            outputs = translator.batch_translate(inputs)[0]
            sent_translations.extend(outputs)

        # grouping the translated sentences back into documents
        translations_sentenized = []
        pointer = 0
        for doc in sentenized:
            translations_sentenized.append(
                sent_translations[pointer : pointer + len(doc)]
            )
            pointer += len(doc)
        translations = [" ".join(doc) for doc in translations_sentenized]
        return translations

    def score_texts(
        self,
        texts: Sequence[str],
        references: Optional[Sequence[str]] = None,
        show_progress: bool = False,
    ) -> np.ndarray:  # type: ignore
        """Return round-trip chrF++ scores for each sentence"""
        assert references is not None
        texts_eng = self.translate(texts, show_progress=show_progress)
        references_eng = self.translate(references, show_progress=show_progress)

        types = ["rouge2", "rougeL", "rougeLsum"]
        scorer = rouge_scorer.RougeScorer(rouge_types=types, split_summaries=True)
        scores = []
        for pred, ref in zip(texts_eng, references_eng):
            scores_bag = scorer.score(target=ref, prediction=pred)
            scores.append([scores_bag[s].fmeasure for s in types])
        results = np.array(scores)
        return results


class BertScoreScorer(Scorer):
    """
    A class for comparing the texts BertScore (by default, using a multilingual BERT model).
    """

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        outputs: Tuple[str] = ("bertscore",),
        device: Device = CPU,
        dtype: DataType = torch.float32,
        batch_size: int = 16,
        **kwargs,
    ):
        self.batch_size = batch_size
        super().__init__(
            model_name=model_name,
            outputs=outputs,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    def init_model(self):
        model_name = self.model_name
        model_path = model_name

        self.scorer = BERTScorer(
            model_type=model_path, batch_size=self.batch_size, device=self.device
        )

    def score_texts(
        self,
        texts: Sequence[str],
        references: Optional[Sequence[str]] = None,
        show_progress: bool = False,
    ) -> np.ndarray:  # type: ignore
        """Return round-trip chrF++ scores for each sentence"""
        assert references is not None
        single_references = []
        for refs_set in references:
            if isinstance(refs_set, str):
                single_references.append(refs_set)
            elif isinstance(refs_set, list):
                refs_set.append(refs_set[0])
            else:
                # we don't know what it is; let bertscorer decide if it is a text
                single_references.append(refs_set)
        precisions, recalls, fscores = self.scorer.score(texts, single_references)
        results = fscores.cpu().numpy()
        return results
