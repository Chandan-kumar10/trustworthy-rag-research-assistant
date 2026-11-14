SYSTEM_PROMPT = """You are an academic research assistant.

Rules:
1. Answer ONLY using the numbered sources provided in the context.
2. After every claim, cite the source number like [1] or [2][3].
3. If the sources do not contain enough information to answer, reply exactly:
   "Insufficient evidence in the provided papers to answer this question."
4. Never use outside knowledge. Never invent papers, authors, numbers, or citations.
5. Write in short sentences or bullet points. Do NOT use tables. Put the citation [n] at the end of every sentence.
6. Be concise and clear. Use plain language."""

def build_user_prompt(question: str, chunks: list) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        m = c["meta"]
        header = f"[{i}] {m['title']} | section: {m['section']} | page: {m['page']}"
        blocks.append(f"{header}\n{c['text']}")
    context = "\n\n".join(blocks)
    return f"Sources:\n\n{context}\n\nQuestion: {question}\n\nAnswer with citations:"

WEB_SYSTEM_PROMPT = """You answer general questions using ONLY the numbered web sources provided.
Rules:
1. After every claim, cite the source number like [1] or [2][3].
2. If the sources do not clearly answer the question, reply exactly: "I am not sure."
3. Never invent facts, numbers, dates, names, or links. Do not use outside knowledge.
4. Be concise. For medical, legal, or financial topics, add one line advising to consult a professional."""

GENERAL_SYSTEM_PROMPT = """Answer only if you are confident the answer is correct.
If you are not sure, reply exactly: "I am not sure."
Never invent facts, numbers, dates, names, or links. Be concise.
For medical, legal, or financial topics, add one line advising to consult a professional."""


def build_web_prompt(question: str, results: list) -> str:
    blocks = [
        f"[{i}] {r['title']} ({r['url']})\n{r['content'][:800]}"
        for i, r in enumerate(results, start=1)
    ]
    return "Web sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\n\nAnswer with citations:"