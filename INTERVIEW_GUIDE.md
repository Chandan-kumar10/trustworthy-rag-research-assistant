# Interview and Explanation Guide

## 1. The 30-second pitch

"I built a RAG-based research assistant that answers questions only from research papers. Every claim carries a citation, an NLI model automatically checks whether the cited passage supports the claim, and the system refuses to answer when the evidence is missing. I evaluated retrieval, abstention and the verifier with my own measured numbers."

## 2. The 2-minute explanation

1. Problem: LLMs hallucinate and invent references, which is unacceptable for academic use.
2. Solution: Retrieval-Augmented Generation. The model is not trained by me; I give it retrieved passages and instruct it to answer only from them.
3. Retrieval: dense embeddings plus BM25, merged with Reciprocal Rank Fusion, then reranked with a cross-encoder.
4. Trust: abstain below a score threshold, cite every claim, verify citations with an NLI model, highlight the evidence.
5. Evaluation: ablation table for retrieval, abstention test, verifier precision and recall.

## 3. Key concepts in simple words

- **RAG**: look up relevant text first, then let the LLM answer using that text. No retraining needed; adding a paper only means indexing it.
- **Chunking**: papers are split into ~400-token pieces so retrieval can find the exact relevant part. Section names and page numbers are stored for citations.
- **Embedding**: a vector that represents the meaning of a text. Similar meaning, nearby vectors.
- **Dense search**: finds chunks by meaning. Misses exact names sometimes.
- **BM25**: keyword search. Strong on exact terms and names, weak on paraphrase.
- **RRF (Reciprocal Rank Fusion)**: combines two ranked lists by summing 1/(60 + rank) for each item.
- **Cross-encoder reranker**: reads the question and a chunk together and scores relevance. More accurate than embeddings but slower, so it only reranks the top 20.
- **Abstention threshold**: if the best reranker score is below 0.4, answer "No evidence found" without calling the LLM.
- **NLI (Natural Language Inference)**: a model that decides whether a premise entails, contradicts or is neutral to a statement. Used as the verifier.
- **Precision / recall (verifier)**: precision = of statements marked Verified, how many were truly supported. Recall = of truly supported statements, how many were marked Verified.

## 4. My results (say these honestly)

- 50 arXiv papers, 1508 chunks.
- Hybrid + reranker: Recall@1 0.971, MRR 0.985 (34 questions).
- Abstention: 10/10 out-of-corpus refused, 0/10 wrongly refused.
- Verifier at 0.4: precision 0.833, recall 0.476 on 31 statements (8 synthetic negatives, 2 accepted wrongly).

## 5. Likely interview questions and answers

**Q: Did you train a model?**
No. I use pre-trained models (embedding, reranker, NLI, and an LLM through Groq). The contribution is the pipeline, the trust layer and the evaluation.

**Q: Why hybrid retrieval?**
Dense search captures meaning, BM25 captures exact terms. They fail on different queries, so fusion is more robust. In my test BM25 was already strong because questions came from abstracts with specific keywords, so the gain from hybrid was small. I report that honestly.

**Q: Why a reranker?**
The first stage is fast but coarse. A cross-encoder compares question and passage jointly and orders the top candidates better. It also gives a calibrated 0 to 1 score that I use for abstention.

**Q: How do you reduce hallucination?**
Strict prompt (answer only from sources, cite every sentence), abstention threshold, a check that the LLM itself refuses when sources are insufficient, and post-hoc NLI verification of each statement.

**Q: How does the verifier work?**
The answer is split into statements. Each statement is compared with its cited passage, split into overlapping 60-word windows, using an NLI model. The best entailment probability gives the status: supported (at least 0.4), partial, or could not verify. The best window is highlighted as evidence.

**Q: How accurate is the verifier?**
Precision 0.83 and recall 0.48 on a small set. It is conservative. It missed many correct paraphrased claims, and it accepted two synthetic errors where a single number or word changed. I do not claim it guarantees correctness; the UI says "Could not verify" rather than "wrong".

**Q: What are the weaknesses?**
Small, abstract-derived test set; one annotator; synthetic negatives; tested only on CS papers; text only; English only; no conversation memory; passages are sent to an external API.

**Q: What did you debug along the way?**
- The arXiv library version removed `download_pdf`, so downloads silently skipped and the index was empty. I traced it from 0 chunks back to 0 papers.
- Abstention returned sources for refused answers; fixed by checking the LLM's refusal.
- The verifier gave low scores when the LLM wrote citations in a different bracket style, so citations were not parsed.
- A first verifier test failed because the source passage never named the entity used in the claim ("CARROT" vs "the framework"): NLI needs matching entities.
- Groq free-tier token limits caused 429 errors; added retry with waiting.

**Q: How would you improve it?**
Add a number/entity consistency check, bigger human-written test set, multiple annotators, conversation memory with query rewriting, OCR and table support, and a local LLM for private documents.

**Q: Is it production ready?**
No. It is a research prototype: single user, local deployment, small evaluation.

**Q: Why did you add web search? Does it fit the project?**
It is an optional extra. Its answers are not grounded in the indexed papers, are not evaluated, and are labelled as such. The core and the evaluated behaviour is "Papers only".

## 6. Demo script (3 minutes)

1. Papers only, All papers: "What optimization technique does CARROT use to find the best chunk combination order?" Show answer, citations, "X of Y verified" badge, click a citation and show the highlighted evidence.
2. Papers only: "Who is Virat Kohli?" Show "No evidence found".
3. Upload a PDF, scope "My uploads", ask a question about it and open the source drawer.
4. Open the evaluation tables and state the limitations.

## 7. Things not to claim

- Do not say the system is always correct or "hallucination free".
- Do not say the verifier has perfect precision; the final measured value is 0.83.
- Do not claim it works for every field; it was tested on CS papers.
- Do not describe the web mode as part of the evaluated system.
