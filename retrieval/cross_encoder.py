"""Cross-encoder reranker for scoring query-chunk relevance."""

import logging
import math

from langsmith import traceable

from config.pipeline_config import CROSS_ENCODER_MODEL

logger = logging.getLogger(__name__)

_model = None


def get_cross_encoder():
    """Return the shared cross-encoder model, loading it on first call."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder model: %s", CROSS_ENCODER_MODEL)
        _model = CrossEncoder(CROSS_ENCODER_MODEL)
    return _model


@traceable(name="cross_encoder_score")
def score_pairs(query: str, texts: list[str]) -> list[float]:
    """Score each (query, text) pair and return as a value within [0, 1] (sigmoid)."""
    if not texts:
        return []
    model = get_cross_encoder()
    pairs = [(query, text) for text in texts]
    logits = model.predict(pairs)
    return [1.0 / (1.0 + math.exp(-float(s))) for s in logits]
