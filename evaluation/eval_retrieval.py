import json
from retrieval.vector_store import dense_search
from retrieval.bm25_search import bm25_search
from retrieval.hybrid import hybrid_search
from retrieval.reranker import rerank

TESTSET = "evaluation/testset.json"
K_VALUES = [1, 3, 5, 10]
POOL = 20


def paper_ranking(results):
    """Ordered list of unique paper_ids from a ranked chunk list."""
    seen, order = set(), []
    for r in results:
        pid = r["meta"]["paper_id"]
        if pid not in seen:
            seen.add(pid)
            order.append(pid)
    return order


def methods(q):
    yield "Dense only", dense_search(q, POOL)
    yield "BM25 only", bm25_search(q, POOL)
    hyb = hybrid_search(q, POOL)
    yield "Hybrid (RRF)", hyb
    yield "Hybrid + Reranker", rerank(q, [dict(c) for c in hyb], top_k=POOL)


def main():
    data = json.load(open(TESTSET, encoding="utf-8"))
    answerable = [d for d in data if d["answerable"]]
    print(f"Evaluating {len(answerable)} answerable questions...\n")

    stats = {}
    for i, item in enumerate(answerable, start=1):
        target = item["expected_paper_id"]
        for name, results in methods(item["question"]):
            ranking = paper_ranking(results)
            s = stats.setdefault(name, {"hits": {k: 0 for k in K_VALUES}, "rr": 0.0})
            if target in ranking:
                rank = ranking.index(target) + 1
                s["rr"] += 1.0 / rank
                for k in K_VALUES:
                    if rank <= k:
                        s["hits"][k] += 1
        print(f"  done {i}/{len(answerable)}", end="\r")

    n = len(answerable)
    print("\n")
    header = f"{'Method':<20}" + "".join(f"Recall@{k:<4}" for k in K_VALUES) + "MRR"
    print(header)
    print("-" * len(header))
    for name, s in stats.items():
        row = f"{name:<20}" + "".join(f"{s['hits'][k] / n:<11.3f}" for k in K_VALUES)
        print(row + f"{s['rr'] / n:.3f}")

    json.dump(stats, open("evaluation/retrieval_results.json", "w"), indent=2)
    print("\nSaved: evaluation/retrieval_results.json")


if __name__ == "__main__":
    main()