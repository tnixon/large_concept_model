# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import logging
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Union

import torch
from fairseq2.gang import FakeGang
from fairseq2.generation.generator import SequenceGeneratorOutput
from sonar.inference_pipelines.text import (
    EmbeddingToTextModelPipeline,
    TextToEmbeddingModelPipeline,
)
from stopes.core.utils import batch
from stopes.pipelines.monolingual.utils.sentence_split import get_split_algo

from lcm.datasets.batch import EmbeddingsBatch, get_embeddings_sequence
from lcm.datasets.configs import SonarDecoderConfig, SonarEncoderConfig
from lcm.inference.lcm.generator import LCMGenerator, LCMGeneratorOptions
from lcm.utils.card_utils import load_model_from_card
from lcm.utils.common import Batched, torch_type

from ..api import EOSConfig, Prediction, PredictorConfig, Prompts

logger = logging.getLogger(__name__)


@dataclass(unsafe_hash=True)
class LCMConfig(LCMGeneratorOptions, PredictorConfig):
    """Holds the configuration of a fairseq2-trained LCM eval."""

    model_card: str = ""
    """Loading model using model card"""

    decoder_config: SonarDecoderConfig = field(
        default_factory=lambda: SonarDecoderConfig()
    )
    """decoder config to generate text from the embedding"""

    encoder_config: SonarEncoderConfig = field(
        default_factory=lambda: SonarEncoderConfig()
    )
    """Optional encoder config to encode the input text (if not yet encoded)"""

    generator_batch_size: Optional[int] = None
    """Size of the internal batch to run the generator. By default this equals to the
    batch size of the task"""

    @classmethod
    def predictor_class(cls):
        return LCMPredictor


LCMOutput = Union[torch.Tensor, EmbeddingsBatch, SequenceGeneratorOutput]


class LCMPredictor:
    """
    A predictor that wraps LCMGenerator and format the output for evaluation
    """

    def __init__(
        self,
        config: LCMConfig,
        **kwargs: Any,
    ):
        self.config = config

        # Run's temperature alwayys take precedence over LCM temperature.
        # The exception is when temperature = 0.0, then we skip it
        if "temperature" in kwargs and kwargs["temperature"] > 0:
            self.config.lcm_temperature = kwargs["temperature"]

        # Register the run config into the predictor, so that log messages
        # about the predictor config are more accurate
        for k, v in kwargs.items():
            if hasattr(self.config, k) and v:
                setattr(self.config, k, v)

        self.gang = kwargs.get("gang", FakeGang())
        self.device = kwargs.get("device", self.gang.device)
        self.dtype = torch_type(kwargs.get("dtype", None))

        model = load_model_from_card(
            self.config.model_card,
            device=self.device,
            dtype=self.dtype,
        )
        self.eos_vec = self.get_eos(kwargs.get("eos_config", None))
        self.build_generator(model)

        self.text_decoder = EmbeddingToTextModelPipeline(
            decoder=self.config.decoder_config.decoder,
            tokenizer=self.config.decoder_config.tokenizer,
            device=self.device,
            dtype=self.dtype,
        )
        self.gang.barrier()

    def get_eos(self, eos_config: Optional[EOSConfig]) -> Optional[torch.Tensor]:
        if not eos_config:
            return None

        if eos_config.ckpt and Path(eos_config.ckpt).exists():
            eos_vec = torch.load(eos_config.ckpt, map_location=self.device)
        elif eos_config.text:
            eos_vec = self.encoder.predict(
                [eos_config.text], source_lang=self.config.encoder_config.lang
            ).squeeze()
        return eos_vec.to(dtype=self.dtype)

    def build_generator(self, model):
        self.generator = LCMGenerator(
            model=model, options=self.config, eos_vec=self.eos_vec
        )

    @staticmethod
    def from_config(config: LCMConfig, **kwargs) -> "LCMPredictor":
        return LCMPredictor(config=config, **kwargs)

    def format_predictions(
        self,
        lcm_output: LCMOutput,
        return_logprobs: bool,
        prompt_len: torch.Tensor,
        **kwargs: Any,
    ) -> Sequence[Prediction]:
        assert isinstance(lcm_output, SequenceGeneratorOutput)
        preds: List[Prediction] = []

        # re-batch the LCM output to be able to sonar-decode at once
        embeds: List[torch.Tensor] = []

        offsets = [0]
        for pl, hyps in zip(prompt_len, lcm_output.hypotheses):
            # Get the best hypothesis
            seq = hyps[0].seq

            # seq.shape = length of generated sentences x embed_dim
            assert len(seq.shape) == 2
            lcm_output_len = seq.size(0)
            next_seq = seq[pl:lcm_output_len]
            embeds.append(next_seq)
            offsets.append(offsets[-1] + next_seq.shape[0])

        # Decode the text with the same sonar decoder
        lcm_texts = self.text_decoder.predict(
            torch.cat(embeds),
            target_lang=self.config.decoder_config.lang,
            max_seq_len=self.config.decoder_config.max_tokens_in_sentence,
            temperature=self.config.decoder_config.temperature,
        )

        for i, embed in enumerate(embeds):
            lcm_text = lcm_texts[offsets[i] : offsets[i + 1]]
            hyps = lcm_output.hypotheses[i]
            if return_logprobs:
                assert hyps[0].step_scores, (
                    "hypothesis must have non-empty `step_scores` "
                    "if the param `return_logprobs` is set"
                )
                lprobs = hyps[0].step_scores.tolist()
            else:
                lprobs = None
            pred = Prediction(
                text=lcm_text,
                embed=embed,
                logprobs=lprobs,
            )
            preds.append(pred)
        return preds

    @cached_property
    def encoder(self) -> TextToEmbeddingModelPipeline:
        return TextToEmbeddingModelPipeline(
            encoder=self.config.encoder_config.encoder,
            tokenizer=self.config.encoder_config.tokenizer,
            device=self.gang.device,
        )

    @cached_property
    def split_text(self) -> Callable[[str], Iterable[str]]:
        return get_split_algo(self.config.encoder_config.lang, "default")

    @torch.inference_mode()
    def __call__(
        self,
        prompts: Prompts,
        max_prompt_len: Optional[int] = None,
        min_gen_len: Optional[int] = None,
        max_gen_len: Optional[int] = None,
        temperature: float = 0.0,
        disable_cache: bool = False,
        greedy: bool = True,
        top_p: float = 0.0,
        top_k: int = 0,
        echo: bool = True,
        return_logprobs: bool = False,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> Sequence[Prediction]:
        assert greedy, "Non-greedy generation for continuous space not yet supported"  # fmt: skip

        if max_gen_len is None:
            max_gen_len = self.config.max_seq_len

        assert isinstance(prompts, Batched), (
            f"Expect sequence of prompts, get {type(prompts)}"
        )

        # Extract the input embeddings
        seqs: List[torch.Tensor] = []

        # Input is not pre-sonarized. Encode the text on the fly
        if isinstance(prompts[0], str):
            assert isinstance(prompts, Sequence)  # mypy
            for prompt in prompts:
                sentences = self.split_text(str(prompt))
                prompt_embs = self.encoder.predict(
                    sentences, source_lang=self.config.encoder_config.lang
                )
                seqs.append(prompt_embs)
        else:
            assert isinstance(prompts, list) and isinstance(prompts[0], torch.Tensor), (
                f"Expect sonarized prompts in the form or List[torch.Tensor], get {type(prompts)}"
            )
            seqs = prompts

        if max_prompt_len:
            seqs = [src[:max_prompt_len, :] for src in seqs]

        if self.config.generator_batch_size:
            seqs_batch = batch(seqs, batch_size=self.config.generator_batch_size)
        else:
            seqs_batch = [seqs]

        predictions: List[Prediction] = []
        for seq in seqs_batch:
            embed_seq = get_embeddings_sequence(src_seqs=seq)
            if embed_seq.padding_mask:
                prompt_len = embed_seq.padding_mask.seq_lens
            else:
                prompt_len = torch.tensor(
                    [embed_seq.seqs.size(1)] * embed_seq.seqs.size(0)
                )
            if "max_gen_len_ratio" in kwargs and kwargs["max_gen_len_ratio"]:
                max_gen_len = max(1, int(prompt_len * kwargs["max_gen_len_ratio"]))

            output = self.generator(
                embed_seq,
                min_gen_len=min_gen_len,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                disable_cache=disable_cache,
            )
            preds = self.format_predictions(
                output,
                return_logprobs=return_logprobs,
                prompt_len=prompt_len,
            )
            predictions.extend(preds)
        return predictions
