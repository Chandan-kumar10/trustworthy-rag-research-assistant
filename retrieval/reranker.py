import numpy as np
from sentence_transformers import CrossEncoder
from config import RERANK_MODEL

_model = None


def _load():
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def rerank(query: str, candidates: list, top_k: int = 6):
    """
    Score each (query, chunk) pair with a cross-encoder.
    rerank_score is between 0 and 1 (sigmoid), used later for abstention.
    """
    if not candidates:
        return []

    model = _load()
    pairs = [(query, c["text"]) for c in candidates]
    logits = np.array(model.predict(pairs), dtype=float)
    probs = 1 / (1 + np.exp(-logits))

    for c, p in zip(candidates, probs):
        c["rerank_score"] = float(p)

    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]