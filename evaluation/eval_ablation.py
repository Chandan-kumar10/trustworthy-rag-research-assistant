import json
import math
import os
import sys

from retrieval.vector_store import dense_search
from retrieval.reranker import rerank
from generation.prompts import build_user_prompt
from generation.llm_client import ask_llm
from generation.answer import answer_question
from trust.citation_verifier import verify_answer
from config import ABSTAIN_THRESHOLD

TESTSET = "evaluation/testset.json"
CACHE = "evaluation/ablation_cache.json"
OUT = "evaluation/ablation_results.json"

# Plain RAG baseline: no refusal rule on purpose
BASE_SYSTEM = (
    "Answer the question using the numbered context passages. "
    "Cite the passage numbers like [1] or [2][3]. Be concise."
)
JUDGE_SYSTEM = "You are a strict grader. Reply with one word only."


def load_cache():
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def judge_correct(question, reference, candidate):
    prompt = (
        f"Question: {question}\nReference answer: {reference}\nCandidate answer: {candidate}\n\n"
        "Does the candidate answer state the key facts of the reference answer "
        "without contradicting it? Reply CORRECT or INCORRECT."
    )
    out = ask_llm(JUDGE_SYSTEM, prompt, max_tokens=600).upper()
    return "INCORRECT" not in out and "CORRECT" in out


def judge_declined(question, candidate):
    prompt = (
        f"Question: {question}\nResponse: {candidate}\n\n"
        "Does the response give a substantive answer to the question, or does it say the "
        "information is not available or cannot be answered from the context? "
        "Reply ANSWERED or DECLINED."
    )
    return "DECLINED" in ask_llm(JUDGE_SYSTEM, prompt, max_tokens=600).upper()


def summ(v):
    return {"supported": v["supported"], "total": v["total"]} if v else None


def evaluate(item, cache):
    q = item["question"]
    if q in cache:
        return cache[q]

    # Versions A, B, C share one plain-RAG answer (dense top 6, no refusal rule)
    chunks = dense_search(q, 6)
    a_text = ask_llm(BASE_SYSTEM, build_user_prompt(q, chunks))
    sources = [{"n": i, "text": c["text"]} for i, c in enumerate(chunks, start=1)]

    rr = rerank(q, [dict(c) for c in chunks], top_k=len(chunks))
    top = rr[0]["rerank_score"] if rr else 0.0

    try:
        a_ver = summ(verify_answer(a_text, sources))
    except Exception:
        a_ver = None

    # Version D: the full system
    d = answer_question(q, mode="papers", verify=True)

    rec = {
        "answerable": item["answerable"],
        "top_score": round(top, 3),
        "a_text": a_text,
        "a_ver": a_ver,
        "d_kind": d["kind"],
        "d_text": d["answer"],
        "d_ver": summ(d.get("verification")),
    }
    if item["answerable"]:
        rec["a_correct"] = judge_correct(q, item["expected_answer"], a_text)
        rec["d_correct"] = (
            judge_correct(q, item["expected_answer"], d["answer"])
            if d["kind"] == "grounded" else False
        )
    else:
        rec["a_declined"] = judge_declined(q, a_text)

    cache[q] = rec
    save_cache(cache)
    return rec


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def fmt(k, n):
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.0f}%  (95% CI {100 * lo:.0f}-{100 * hi:.0f}%)"


def vshare(vers):
    vs = [v for v in vers if v and v["total"]]
    if not vs:
        return "n/a"
    avg = sum(v["supported"] / v["total"] for v in vs) / len(vs)
    return f"{100 * avg:.0f}% (n={len(vs)} answers)"


def main():
    with open(TESTSET, encoding="utf-8") as f:
        data = json.load(f)
    ans = [d for d in data if d["answerable"]]
    un = [d for d in data if not d["answerable"]]
    n_ans = int(sys.argv[1]) if len(sys.argv) > 1 else len(ans)
    n_un = int(sys.argv[2]) if len(sys.argv) > 2 else len(un)
    ans, un = ans[:n_ans], un[:n_un]

    cache = load_cache()
    items = ans + un
    for i, item in enumerate(items, start=1):
        print(f"[{i}/{len(items)}] {item['question'][:70]}")
        evaluate(item, cache)

    A = [cache[x["question"]] for x in ans]
    U = [cache[x["question"]] for x in un]
    thr = ABSTAIN_THRESHOLD
    na, nu = len(A), len(U)

    a_correct = sum(r["a_correct"] for r in A)
    b_correct = sum(r["a_correct"] and r["top_score"] >= thr for r in A)
    d_correct = sum(r["d_correct"] for r in A)
    b_wrong_ref = sum(r["top_score"] < thr for r in A)
    d_wrong_ref = sum(r["d_kind"] != "grounded" for r in A)

    a_hall = sum(not r["a_declined"] for r in U)
    b_hall = sum(r["top_score"] >= thr and not r["a_declined"] for r in U)
    d_hall = sum(r["d_kind"] == "grounded" for r in U)

    print("\n" + "=" * 78)
    print(f"Answerable questions: {na} | Unanswerable questions: {nu} | threshold {thr}")
    print("=" * 78)
    print("\nA. Plain RAG (dense, no abstention, no verifier)")
    print("   Correct answers      :", fmt(a_correct, na))
    print("   Hallucinated answers :", fmt(a_hall, nu))
    print("\nB. A + abstention")
    print("   Correct answers      :", fmt(b_correct, na))
    print("   Wrongly refused      :", fmt(b_wrong_ref, na))
    print("   Hallucinated answers :", fmt(b_hall, nu))
    print("\nC. A + verifier (same answers as A, with verification)")
    print("   Verified share, answerable            :", vshare([r["a_ver"] for r in A]))
    print("   Verified share, hallucinated answers  :",
          vshare([r["a_ver"] for r in U if not r["a_declined"]]))
    print("\nD. Full system (hybrid + reranker + strict prompt + abstention + verifier)")
    print("   Correct answers      :", fmt(d_correct, na))
    print("   Wrongly refused      :", fmt(d_wrong_ref, na))
    print("   Hallucinated answers :", fmt(d_hall, nu))
    print("   Verified share, answerable            :",
          vshare([r["d_ver"] for r in A if r["d_kind"] == "grounded"]))
    print("   Verified share, hallucinated answers  :",
          vshare([r["d_ver"] for r in U if r["d_kind"] == "grounded"]))

    summary = {
        "n_answerable": na, "n_unanswerable": nu, "threshold": thr,
        "A": {"correct": a_correct, "hallucinated": a_hall},
        "B": {"correct": b_correct, "wrongly_refused": b_wrong_ref, "hallucinated": b_hall},
        "D": {"correct": d_correct, "wrongly_refused": d_wrong_ref, "hallucinated": d_hall},
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {OUT} (details and raw answers in {CACHE})")


if __name__ == "__main__":
    main()