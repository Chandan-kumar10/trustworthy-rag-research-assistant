import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, QUERY_PREFIX

_model = None
_col = None


def _load():
    global _model, _col
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
        _col = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION_NAME)
    return _model, _col


def get_model():
    return _load()[0]


def get_collection():
    return _load()[1]


def dense_search(query: str, k: int = 20, paper_ids=None):
    """Meaning-based search. paper_ids (optional) limits search to those papers."""
    model, col = _load()
    kwargs = {}
    if paper_ids is not None:
        if not paper_ids:
            return []
        kwargs["where"] = {"paper_id": {"$in": list(paper_ids)}}

    emb = model.encode([QUERY_PREFIX + query], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=emb, n_results=k, **kwargs)

    results = []
    for cid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        results.append({"id": cid, "text": doc, "meta": meta, "score": 1 - dist})
    return results