import json
import random
import re
from config import METADATA_FILE
from generation.llm_client import ask_llm

NUM_PAPERS = 25
OUT_FILE = "evaluation/testset_draft.json"

SYSTEM = (
    "You create exam questions from a research paper abstract. "
    "Return ONLY valid JSON, no extra text."
)


def make_prompt(title, abstract):
    return (
        f"Title: {title}\nAbstract: {abstract}\n\n"
        "Write 2 specific questions that this abstract clearly answers. "
        "Each question must make sense on its own WITHOUT mentioning 'this paper' or 'the abstract', "
        "and must include the key name/topic from the paper so it is unambiguous. "
        'Return JSON like: [{"question": "...", "expected_answer": "one short sentence"}, ...]'
    )


def main():
    papers = json.load(open(METADATA_FILE, encoding="utf-8"))
    random.seed(42)
    chosen = random.sample(papers, min(NUM_PAPERS, len(papers)))

    items = []
    for p in chosen:
        try:
            raw = ask_llm(SYSTEM, make_prompt(p["title"], p["abstract"]), max_tokens=1500)
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            qa_list = json.loads(m.group(0))
        except Exception as e:
            print(f"Skipped {p['paper_id']}: {e}")
            continue

        for qa in qa_list[:2]:
            items.append({
                "question": qa["question"],
                "expected_paper_id": p["paper_id"],
                "expected_paper_title": p["title"],
                "expected_answer": qa["expected_answer"],
                "answerable": True,
            })
        print(f"OK: {p['paper_id']}")

    json.dump(items, open(OUT_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nDraft saved: {OUT_FILE} ({len(items)} questions)")


if __name__ == "__main__":
    main()