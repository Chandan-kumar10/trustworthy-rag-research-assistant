import sys
from retrieval.vector_store import dense_search
from retrieval.bm25_search import bm25_search
from retrieval.hybrid import hybrid_search
from retrieval.reranker import rerank

query = sys.argv[1] if len(sys.argv) > 1 else "How does reranking improve RAG performance?"


def show(name, results, score_key="score", n=5):
    print(f"\n===== {name} =====")
    for r in results[:n]:
        m = r["meta"]
        print(f"{r[score_key]:.3f} | {m['title'][:60]} | {m['section']} | p.{m['page']}")


print(f"Query: {query}")
show("DENSE ONLY", dense_search(query, 20))
show("BM25 ONLY", bm25_search(query, 20))

hybrid = hybrid_search(query, 20)
show("HYBRID (RRF)", hybrid)

final = rerank(query, hybrid, top_k=6)
show("HYBRID + RERANKER", final, score_key="rerank_score", n=6)

print("\n===== TOP PASSAGE AFTER RERANKING =====")
print(final[0]["text"])