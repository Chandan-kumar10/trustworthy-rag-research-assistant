from trust.citation_verifier import verify_answer

sources = [{
    "n": 1,
    "text": "CARROT is a retrieval optimization framework for RAG. Experimental results demonstrate "
            "that CARROT achieves up to a 30% improvement over baseline models, highlighting the "
            "framework's effectiveness, scalability, and suitability."
}]

tests = [
    ("SAHI      ", "CARROT achieves up to a 30% improvement over baseline models [1]."),
    ("GALAT NUM ", "CARROT achieves up to a 63% improvement over baseline models [1]."),
    ("BAHAR KI  ", "CARROT was trained on medical images collected from hospitals [1]."),
]

for tag, t in tests:
    c = verify_answer(t, sources)["claims"][0]
    print(f"{tag} -> {c['status'].upper():12} score={c['score']}")
    print("           ", t, "\n")