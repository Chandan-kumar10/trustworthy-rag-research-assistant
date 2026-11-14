from retrieval.vector_store import dense_search
from retrieval.bm25_search import bm25_search


def hybrid_search(query: str, k: int = 20, pool: int = 50, rrf_k: int = 60, paper_ids=None):
    dense = dense_search(query, pool, paper_ids)
    sparse = bm25_search(query, pool, paper_ids)

    fused = {}
    for results in (dense, sparse):
        for rank, item in enumerate(results, start=1):
            if item["id"] not in fused:
                fused[item["id"]] = {**item, "score": 0.0}
            fused[item["id"]]["score"] += 1.0 / (rrf_k + rank)

    ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:k]