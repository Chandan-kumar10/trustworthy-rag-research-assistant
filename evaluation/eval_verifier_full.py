import json

REAL = "evaluation/verify_labels.json"
NEG = "evaluation/negatives.json"


def load():
    real = [r for r in json.load(open(REAL, encoding="utf-8")) if r["label"] is not None]
    neg = json.load(open(NEG, encoding="utf-8"))
    return real, neg


def stats(rows, thr):
    tp = sum(1 for r in rows if r["verifier_score"] >= thr and r["label"])
    fp = sum(1 for r in rows if r["verifier_score"] >= thr and not r["label"])
    fn = sum(1 for r in rows if r["verifier_score"] < thr and r["label"])
    tn = sum(1 for r in rows if r["verifier_score"] < thr and not r["label"])
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    return p, rc, f1, tp, fp, fn, tn


def main():
    real, neg = load()
    rows = real + neg
    pos = sum(1 for r in rows if r["label"])
    print(f"Statements: {len(rows)} (supported {pos}, unsupported {len(rows) - pos}; "
          f"{len(neg)} synthetic negatives)\n")
    print(f"{'Thr':<7}{'Precision':<11}{'Recall':<9}{'F1':<8}TP  FP  FN  TN")
    for thr in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        p, rc, f1, tp, fp, fn, tn = stats(rows, thr)
        mark = "  <- current" if thr == 0.4 else ""
        print(f"{thr:<7}{p:<11.3f}{rc:<9.3f}{f1:<8.3f}{tp:<4}{fp:<4}{fn:<4}{tn}{mark}")

    wrong_pass = [r for r in neg if r["verifier_score"] >= 0.4]
    print(f"\nSynthetic negatives wrongly marked Verified at 0.4: {len(wrong_pass)}/{len(neg)}")
    for r in wrong_pass:
        print("  -", r["claim"][:120])


if __name__ == "__main__":
    main()