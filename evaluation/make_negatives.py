import json
import random
import re

from trust.citation_verifier import _best_entailment

LABELS = "evaluation/verify_labels.json"
OUT = "evaluation/negatives.json"
random.seed(7)

SWAPS = [
    ("increase", "decrease"), ("improves", "worsens"), ("higher", "lower"),
    ("outperforms", "underperforms"), ("eliminating", "requiring"),
    ("eliminates", "requires"), ("without", "with"), ("simple", "complex"),
    ("fast", "slow"), ("reference-free", "reference-based"),
    ("Monte Carlo Tree Search", "Genetic Algorithm"), ("MCTS", "beam search"),
    ("dense", "sparse"), ("supervised", "unsupervised"),
]


def change_number(text):
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    nums = [n for n in nums if len(n) <= 5]
    if not nums:
        return None
    n = random.choice(nums)
    new = str(int(float(n) * 2 + 3)) if "." not in n else str(round(float(n) * 2 + 1.5, 1))
    return text.replace(n, new, 1)


def swap_word(text):
    random.shuffle(SWAPS)
    for a, b in SWAPS:
        if a in text:
            return text.replace(a, b, 1)
    return None


def main():
    rows = json.load(open(LABELS, encoding="utf-8"))
    good = [r for r in rows if r["label"] is True]
    out = []
    for r in good:
        for fn in (change_number, swap_word):
            bad = fn(r["claim"])
            if not bad or bad == r["claim"]:
                continue
            passages = r["cited_passages"]
            score, _, _ = _best_entailment(bad, list(enumerate(passages)))
            out.append({
                "original": r["claim"],
                "claim": bad,
                "verifier_score": round(score, 3),
                "label": False,
            })
            break
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Saved {len(out)} synthetic negatives to {OUT}")


if __name__ == "__main__":
    main()