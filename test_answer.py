import sys
from generation.answer import answer_question

q = sys.argv[1] if len(sys.argv) > 1 else "Who is virat kohli?"
res = answer_question(q)

print("\nQUESTION:", q)
print("\nANSWER:\n", res["answer"])
print("\nSOURCES:")
for s in res["sources"]:
    print(f"[{s['n']}] {s['title']} | {s['section']} | p.{s['page']} | score={s['score']}")