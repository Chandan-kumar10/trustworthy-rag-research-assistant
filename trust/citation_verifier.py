import re
import numpy as np
from sentence_transformers import CrossEncoder

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
# Label order for this model: contradiction, entailment, neutral
_model = None


def _load():
    global _model
    if _model is None:
        _model = CrossEncoder(NLI_MODEL)
    return _model


def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def split_claims(answer: str):
    """
    One claim per line/bullet, then per sentence, then split compound
    claims on ';' and ' - '. Keeps the [n] citation numbers of each claim.
    Skips markdown tables and short intro headings.
    """
    claims = []
    for line in answer.replace("**", "").split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("|") or re.fullmatch(r"[-|:\s]+", line):
            continue  # markdown table rows/separators are not claims
        if line.endswith(":") and len(line) < 120:
            continue  # intro headings like "depends on ...:"

        line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", "", line)
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[(])", line)
        for p in parts:
            cites = [int(n) for n in re.findall(r"\[(\d+)\]", p)]
            clean = re.sub(r"\s*\[\d+\]", "", p).strip(" -*\u2022")
            for piece in re.split(r";\s+|\s[\u2013\u2014-]\s", clean):
                piece = piece.strip(" -*\u2022,")
                if len(piece) > 25:
                    claims.append({"text": piece, "cites": sorted(set(cites))})
    return claims


def _windows(passage, size=60, step=30):
    words = passage.split()
    if not words:
        return []
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)]




def _best_entailment(claim, passage_items):
    """passage_items: list of (source_n, passage_text). Returns (score, source_n, window_text)."""
    model = _load()
    pairs, meta = [], []
    for n, p in passage_items:
        for w in _windows(p):
            pairs.append((w, claim))
            meta.append((n, w))
    if not pairs:
        return 0.0, None, ""
    probs = _softmax(np.array(model.predict(pairs), dtype=float))
    ent = probs[:, 1]
    best = int(ent.argmax())
    return float(ent[best]), meta[best][0], meta[best][1]

def verify_answer(answer: str, sources: list, supported_at=0.4, partial_at=0.1):
    """status: supported | partial | unsupported. Also returns the best evidence window."""
    by_n = {s["n"]: s["text"] for s in sources}
    results = []
    for c in split_claims(answer):
        cited = [(n, by_n[n]) for n in c["cites"] if n in by_n]
        scope = cited if cited else list(by_n.items())
        score, ev_n, ev_text = _best_entailment(c["text"], scope)

        if score >= supported_at:
            status = "supported"
        elif score >= partial_at:
            status = "partial"
        else:
            status = "unsupported"
        results.append({
            "text": c["text"], "cites": c["cites"],
            "score": round(score, 3), "status": status,
            "had_citation": bool(c["cites"]),
            "evidence_n": ev_n if status != "unsupported" else None,
            "evidence": ev_text if status != "unsupported" else "",
        })

    supported = sum(r["status"] == "supported" for r in results)
    partial = sum(r["status"] == "partial" for r in results)
    return {
        "claims": results,
        "supported": supported,
        "partial": partial,
        "unsupported": len(results) - supported - partial,
        "total": len(results),
    }