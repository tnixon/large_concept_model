# Copyright (c) Meta Platforms, Inc. and affiliates.
#

# flake8: noqa
import inspect
from typing import List, Optional

from fairseq2.gang import Gang

from lcm.evaluation.api import Scorer, ScorerConfig
from lcm.evaluation.metrics.common import (
    bleu,
    exact_match,
    exact_match_f1,
    f1,
    rouge_score,
    sentence_bleu,
)
from lcm.evaluation.metrics.similarity import (
    edit_distance,
    l2_distance,
    longest_common_substring,
    memorization_score,
    mse_constrative_accuracy,
    nltk_sentence_bleu,
)

_SCORER_MAP = {
    "sentence_fluency": "lcm.evaluation.metrics.sentence_fluency.FluencyClassifierScorer",
    "sentence_perplexity": "lcm.evaluation.metrics.sentence_fluency.PerplexityScorer",
    "round_trip_translation": "lcm.evaluation.metrics.sentence_fluency.RoundTripTranslationScorer",
    "word_repetition": "lcm.evaluation.metrics.sentence_fluency.WordRepetitionScorer",
    "token_repetition": "lcm.evaluation.metrics.sentence_fluency.TokenRepetitionScorer",
    "seahorse": "lcm.evaluation.metrics.seahorse.SeahorseScorer",
    "momentum_coherence": "lcm.evaluation.metrics.coherence_metrics.MomentumCoherenceProcessor",
    "translated_rouge": "lcm.evaluation.metrics.multilingual_similarity.TranslatedRougeScorer",
    "bertscore": "lcm.evaluation.metrics.multilingual_similarity.BertScoreScorer",
}


def get_scorer(
    config: ScorerConfig,
    metrics_to_report: Optional[List] = None,
    gang: Optional[Gang] = None,
) -> Optional[Scorer]:
    scorer_type = config.scorer_type
    if scorer_type not in _SCORER_MAP:
        raise ValueError(f"No metrics found for {scorer_type}")

    module_path, config_cls_name = _SCORER_MAP[scorer_type].rsplit(".", 1)
    module = __import__(module_path, fromlist=[config_cls_name])
    scorer_cls = getattr(module, config_cls_name)
    assert issubclass(scorer_cls, Scorer), f"Unsupported scorer type: {scorer_cls}"
    defaults = inspect.signature(scorer_cls.__init__).parameters

    # Mark the metric that we don't want to calculate
    if "outputs" in defaults:
        output_columns = defaults["outputs"].default
    else:
        assert config.model_name, (
            f"Cannot resolve output name for the scorer type {scorer_cls}"
        )
        output_columns = scorer_cls.default_outputs(config.model_name)

    if isinstance(output_columns, str):
        output_columns = [output_columns]
    elif isinstance(output_columns, tuple):
        output_columns = list(output_columns)
    if metrics_to_report:
        for i, metric_name in enumerate(output_columns):
            if metric_name not in metrics_to_report:
                output_columns[i] = None

    if all(c is None for c in output_columns):
        return None

    kwargs = {"model_name": "", "outputs": tuple(output_columns)}

    if config.model_name:
        kwargs["model_name"] = config.model_name
    elif "model_name" in defaults:
        kwargs["model_name"] = defaults["model_name"].default

    if config.inputs:
        kwargs["inputs"] = config.inputs
    elif "inputs" in defaults:
        kwargs["inputs"] = defaults["inputs"].default

    _params = config.params or {}
    has_kwargs = len({k: v for k, v in defaults.items() if v.kind == v.VAR_KEYWORD}) > 0
    for k, v in _params.items():
        if k in defaults:
            v = v or defaults[k].default
        elif not has_kwargs:
            continue
        kwargs[k] = v
    return scorer_cls(gang=gang, **kwargs)
