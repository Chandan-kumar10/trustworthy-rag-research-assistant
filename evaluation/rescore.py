import json

from trust.citation_verifier import _best_entailment

REAL = "evaluation/verify_labels.json"
NEG = "evaluation/negatives.json"
THRS = [0.1, 0.2, 0.3, 0.4, 0.5]


def score(claim, passages):
    s, _, _ = _best_entailment(claim, list(enumerate(passages, start=1)))
    return s


def metrics(rows, key, thr):
    tp = sum(r[key] >= thr and r["label"] for r in rows)
    fp = sum(r[key] >= thr and not r["label"] for r in rows)
    fn = sum(r[key] < thr and r["label"] for r in rows)
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return p, rc, tp, fp, fn


def main():
    real = [r for r in json.load(open(REAL, encoding="utf-8")) if r["label"] is not None]
    neg = json.load(open(NEG, encoding="utf-8"))
    passages_of = {r["claim"]: r["cited_passages"] for r in real}

    rows = []
    for r in real:
        rows.append({"claim": r["claim"], "label": r["label"], "old": r["verifier_score"],
                     "new": score(r["claim"], r["cited_passages"])})
    skipped = 0
    for n in neg:
        ps = passages_of.get(n["original"])
        if not ps:
            skipped += 1
            continue
        rows.append({"claim": n["claim"], "label": False, "old": n["verifier_score"],
                     "new": score(n["claim"], ps)})

    pos = sum(r["label"] for r in rows)
    print(f"Statements: {len(rows)} (supported {pos}, unsupported {len(rows) - pos}); "
          f"negatives skipped (no matching source): {skipped}\n")
    print(f"{'Thr':<6}{'old P':<8}{'old R':<8}{'new P':<8}{'new R':<8}{'FP old->new':<14}{'TP old->new'}")
    for t in THRS:
        po, ro, tpo, fpo, _ = metrics(rows, "old", t)
        pn, rn, tpn, fpn, _ = metrics(rows, "new", t)
        print(f"{t:<6}{po:<8.3f}{ro:<8.3f}{pn:<8.3f}{rn:<8.3f}{str(fpo) + '->' + str(fpn):<14}{tpo}->{tpn}")

    print("\nStatements whose status flips at 0.4:")
    for r in rows:
        if (r["old"] >= 0.4) != (r["new"] >= 0.4):
            print(f"  label={r['label']} old={r['old']:.3f} new={r['new']:.3f} | {r['claim'][:90]}")

    json.dump(rows, open("evaluation/rescored.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()