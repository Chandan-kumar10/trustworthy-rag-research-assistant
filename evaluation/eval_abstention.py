import json
from generation.answer import answer_question, NO_EVIDENCE

TESTSET = "evaluation/testset.json"


def main():
    data = json.load(open(TESTSET, encoding="utf-8"))
    unans = [d for d in data if not d["answerable"]]
    ans = [d for d in data if d["answerable"]]

    correct_refusals = 0
    print("UNANSWERABLE questions (should refuse):")
    for d in unans:
        res = answer_question(d["question"])
        refused = res["answer"] == NO_EVIDENCE
        correct_refusals += refused
        print(f"  {'OK  ' if refused else 'FAIL'} | {d['question']}")

    wrong_refusals = 0
    print("\nANSWERABLE questions (should NOT refuse):")
    for d in ans[:10]:
        res = answer_question(d["question"])
        refused = res["answer"] == NO_EVIDENCE
        wrong_refusals += refused
        print(f"  {'FAIL' if refused else 'OK  '} | {d['question'][:80]}")

    print(f"\nCorrect refusals: {correct_refusals}/{len(unans)}")
    print(f"Wrongly refused answerable: {wrong_refusals}/{min(10, len(ans))}")


if __name__ == "__main__":
    main()