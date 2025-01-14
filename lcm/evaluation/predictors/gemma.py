# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from dataclasses import dataclass

from lcm.evaluation.api import (
    PREDICTION_COLUMN,
    Example,
)
from lcm.evaluation.predictors.huggingface import (
    HuggingfacePredictor,
    HuggingfacePredictorConfig,
)


@dataclass
class GemmaPredictorConfig(HuggingfacePredictorConfig):
    @classmethod
    def predictor_class(cls):
        return GemmaPredictor


class GemmaPredictor(HuggingfacePredictor):
    @staticmethod
    def from_config(config: GemmaPredictorConfig, **kwargs) -> "GemmaPredictor":  # type: ignore
        predictor = HuggingfacePredictor.from_config(config, **kwargs)
        return GemmaPredictor(predictor.model, predictor.tokenizer, config)

    def post_process(self, x: Example) -> Example:
        """Handle the cleaning of response from Gemma models"""

        pred = x.pop(PREDICTION_COLUMN)

        # Pretrained model
        if not self.config.model_name.endswith("-it"):
            # Clean prediction for summarization task based on some default prompts
            if "Write me a summary" in pred:
                response_idx = pred.find(
                    "\n\nResponse:", pred.rfind("Write me a summary")
                )
                if response_idx > 0:
                    pred = pred[response_idx + len("\n\nResponse:") :].strip()

        # Instruction-fine-tuned models
        else:
            if "\n\nmodel\n\n" in pred:
                pred = pred[pred.rfind("\n\nmodel\n") + len("\nmodel\n") :].strip()
            if (
                pred.startswith("Sure, here")
                or pred.startswith("Sure here")
                or pred.startswith("## Summary")
            ):
                colon_idx = pred.find(":")
                pred = pred[colon_idx + 1 :].strip()

        return {
            PREDICTION_COLUMN: pred,
            **x,
        }
