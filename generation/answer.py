from retrieval.hybrid import hybrid_search
from retrieval.reranker import rerank
from generation.prompts import (
    SYSTEM_PROMPT, build_user_prompt,
    WEB_SYSTEM_PROMPT, GENERAL_SYSTEM_PROMPT, build_web_prompt,
)
from generation.llm_client import ask_llm
from generation.web_search import web_search
from trust.citation_verifier import verify_answer
from config import TOP_K_RETRIEVE, TOP_K_RERANK, ABSTAIN_THRESHOLD

MAX_PER_PAPER = 2
NO_EVIDENCE = "Insufficient evidence in the indexed papers to answer this question."


def _diversify(chunks, max_per_paper=MAX_PER_PAPER):
    counts, out = {}, []
    for c in chunks:
        pid = c["meta"]["paper_id"]
        if counts.get(pid, 0) < max_per_paper:
            out.append(c)
            counts[pid] = counts.get(pid, 0) + 1
    return out


def _papers_answer(question, paper_ids, verify=True):
    candidates = hybrid_search(question, TOP_K_RETRIEVE, paper_ids=paper_ids)
    reranked = rerank(question, candidates, top_k=TOP_K_RETRIEVE)
    chunks = _diversify(reranked)[:TOP_K_RERANK]

    if not chunks or chunks[0]["rerank_score"] < ABSTAIN_THRESHOLD:
        return {"answer": NO_EVIDENCE, "sources": [], "kind": "none"}

    text = ask_llm(SYSTEM_PROMPT, build_user_prompt(question, chunks))
    if "insufficient evidence" in text.lower():
        return {"answer": NO_EVIDENCE, "sources": [], "kind": "none"}

    sources = [
        {
            "n": i, "title": c["meta"]["title"], "section": c["meta"]["section"],
            "page": c["meta"]["page"], "url": c["meta"]["url"],
            "score": round(c["rerank_score"], 3), "text": c["text"],
        }
        for i, c in enumerate(chunks, start=1)
    ]

    verification = None
    if verify:
        try:
            verification = verify_answer(text, sources)
        except Exception as e:
            print("Verification failed:", e)
    return {"answer": text, "sources": sources, "kind": "grounded", "verification": verification}


def _general_answer(question):
    results = web_search(question)
    if results:
        text = ask_llm(WEB_SYSTEM_PROMPT, build_web_prompt(question, results))
        if "not sure" in text.lower()[:40]:
            return {"answer": NO_EVIDENCE, "sources": [], "kind": "none"}
        sources = [
            {
                "n": i, "title": r["title"], "section": "Web", "page": "n/a",
                "url": r["url"], "score": round(float(r.get("score", 0)), 3),
                "text": r["content"],
            }
            for i, r in enumerate(results, start=1)
        ]
        return {"answer": text, "sources": sources, "kind": "web"}

    text = ask_llm(GENERAL_SYSTEM_PROMPT, question, max_tokens=600)
    if "not sure" in text.lower()[:40]:
        return {"answer": NO_EVIDENCE, "sources": [], "kind": "none"}
    return {"answer": text, "sources": [], "kind": "general"}


def answer_question(question: str, paper_ids=None, mode: str = "papers", verify: bool = True):
    """mode: 'papers' (default, used by evaluation), 'hybrid', or 'general'."""
    if mode == "general":
        return _general_answer(question)
    res = _papers_answer(question, paper_ids, verify)
    if res["kind"] == "none" and mode == "hybrid":
        return _general_answer(question)
    return res