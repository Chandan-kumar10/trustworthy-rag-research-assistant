from retrieval.vector_store import dense_search
from trust.citation_verifier import verify_answer

q = "What is the capital of France?"
chunks = dense_search(q, 6)
sources = [{"n": i, "text": c["text"]} for i, c in enumerate(chunks, start=1)]
v = verify_answer("Paris is the capital of France [1].", sources)
c = v["claims"][0]
print("status:", c["status"], "score:", c["score"], "evidence from source:", c["evidence_n"])
print("\nEVIDENCE WINDOW:\n", c["evidence"])
print("\nSOURCE TEXT:\n", sources[(c["evidence_n"] or 1) - 1]["text"][:700])