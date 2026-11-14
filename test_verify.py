from generation.answer import answer_question

q = "How should documents be chunked for RAG?"
res = answer_question(q, mode="papers")
print("ANSWER:\n", res["answer"], "\n")

v = res.get("verification")
if not v:
    print("No verification (answer was not grounded).")
else:
    print(f"Supported {v['supported']} | Partial {v['partial']} | Unsupported {v['unsupported']} | Total {v['total']}\n")
    for c in v["claims"]:
        cit = c["cites"] if c["had_citation"] else "none"
        print(f"[{c['status'].upper():11}] score={c['score']} cites={cit}")
        print("   ", c["text"][:130])