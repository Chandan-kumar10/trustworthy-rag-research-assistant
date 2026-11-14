import json
import re
from rank_bm25 import BM25Okapi
from config import CHUNKS_FILE

STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "by", "with", "as", "at", "that", "this", "it", "from",
    "what", "how", "why", "which", "does", "do", "can",
}

_chunks = None
_bm25 = None


def _tokenize(text: str):
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def _load():
    global _chunks, _bm25
    if _bm25 is None:
        with open(CHUNKS_FILE, encoding="utf-8") as f:
            _chunks = json.load(f)
        _bm25 = BM25Okapi([_tokenize(c["text"]) for c in _chunks])
    return _chunks, _bm25


def reset():
    """Call after uploads/deletes so the index is rebuilt from chunks.json."""
    global _chunks, _bm25
    _chunks, _bm25 = None, None


def bm25_search(query: str, k: int = 20, paper_ids=None):
    chunks, bm25 = _load()
    scores = bm25.get_scores(_tokenize(query))
    allowed = set(paper_ids) if paper_ids is not None else None

    idx = [
        i for i in range(len(scores))
        if scores[i] > 0 and (allowed is None or chunks[i]["meta"]["paper_id"] in allowed)
    ]
    idx.sort(key=lambda i: scores[i], reverse=True)

    return [
        {
            "id": chunks[i]["id"],
            "text": chunks[i]["text"],
            "meta": chunks[i]["meta"],
            "score": float(scores[i]),
        }
        for i in idx[:k]
    ]