import json

c = json.load(open("evaluation/ablation_cache.json", encoding="utf-8"))
t = {d["question"]: d for d in json.load(open("evaluation/testset.json", encoding="utf-8"))}

both = a_only = d_only = neither = 0
for q, r in c.items():
    if r["answerable"]:
        a, d = r["a_correct"], r["d_correct"]
        both += a and d
        a_only += a and not d
        d_only += d and not a
        neither += (not a) and (not d)
print(f"Paired: both correct {both}, only A {a_only}, only D {d_only}, neither {neither}\n")

print("== Answerable: A and D disagree (check the judge) ==")
for q, r in c.items():
    if r["answerable"] and r["a_correct"] != r["d_correct"]:
        print(f"\nQ: {q}\nExpected: {t[q]['expected_answer']}")
        print(f"A (judge={r['a_correct']}): {r['a_text'][:300]}")
        print(f"D (judge={r['d_correct']}): {r['d_text'][:300]}")

print("\n== Unanswerable: plain RAG did NOT decline ==")
for q, r in c.items():
    if not r["answerable"] and not r["a_declined"]:
        print(f"\nQ: {q}\nA: {r['a_text'][:500]}\nVerified: {r['a_ver']}")