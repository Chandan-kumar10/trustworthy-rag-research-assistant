import json
from generation.answer import answer_question

TESTSET = "evaluation/testset.json"
OUT = "evaluation/verify_labels.json"
NUM_QUESTIONS = 12


def main():
    data = json.load(open(TESTSET, encoding="utf-8"))
    qs = [d for d in data if d["answerable"]][:NUM_QUESTIONS]

    rows = []
    for i, d in enumerate(qs, start=1):
        res = answer_question(d["question"], mode="papers", verify=True)
        v = res.get("verification")
        if res["kind"] != "grounded" or not v:
            print(f"skip {i}: not grounded")
            continue
        by_n = {s["n"]: s for s in res["sources"]}
        for c in v["claims"]:
            cited = [by_n[n]["text"] for n in c["cites"] if n in by_n]
            rows.append({
                "question": d["question"],
                "claim": c["text"],
                "cites": c["cites"],
                "cited_passages": cited if cited else [s["text"] for s in res["sources"]],
                "verifier_score": c["score"],
                "verifier_status": c["status"],
                "label": None,
            })
        print(f"done {i}/{len(qs)}")

    json.dump(rows, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved {len(rows)} statements to {OUT}")


if __name__ == "__main__":
    main()