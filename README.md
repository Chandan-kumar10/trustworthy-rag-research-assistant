# Trustworthy RAG-Based Academic Research Assistant

A question answering system for research papers. It answers **only from the indexed papers**, attaches a citation to every claim, automatically checks whether each cited passage actually supports the claim, and refuses to answer when the evidence is missing.

## Why this project

General chatbots can invent facts and fake references. For academic work that is a real problem. This project is built around three rules:

1. Answer only from retrieved passages (Retrieval-Augmented Generation).
2. Show the proof: paper, section, page and the exact passage for every citation.
3. Say "No evidence found" instead of guessing.

## Features

- Hybrid retrieval (dense embeddings + BM25) with a cross-encoder reranker
- Grounded answers with numbered citations `[1][2]`
- Abstention: refuses when the best retrieved passage scores below a threshold
- Citation verification with an NLI model, shown as a badge ("4 of 5 statements verified")
- Evidence highlighting inside the source passage
- PDF upload: ask questions over your own papers (scope: "My uploads")
- Web UI: dark/light theme, sidebar, chat history, source drawer, export
- Evaluation scripts: retrieval ablation, abstention, verifier precision/recall

## Architecture

```
Offline (ingestion)
  arXiv / uploaded PDF -> parse (PyMuPDF) -> section-aware chunks -> embeddings (bge) -> ChromaDB
                                                                 \-> BM25 index (chunks.json)

Online (a question)
  question
    -> dense search (ChromaDB) + BM25 keyword search
    -> Reciprocal Rank Fusion -> top 20
    -> cross-encoder reranker -> top 6 (max 2 chunks per paper)
    -> top score < threshold ?  yes -> "No evidence found"
    -> LLM (Groq) writes the answer with [n] citations from the 6 passages
    -> answer shown immediately
    -> NLI verifier checks each statement against its cited passage (runs after the answer)
    -> UI shows verification badge and highlights the evidence
```

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| PDF parsing | PyMuPDF |
| Chunking | LangChain RecursiveCharacterTextSplitter (1600 chars, 200 overlap), section-aware |
| Embeddings | BAAI/bge-base-en-v1.5 (local) |
| Vector store | ChromaDB (cosine) |
| Keyword search | rank-bm25 |
| Reranker | BAAI/bge-reranker-base (local cross-encoder) |
| Answer LLM | openai/gpt-oss-120b via Groq API |
| Verifier | cross-encoder/nli-deberta-v3-base (local) |
| Backend | FastAPI + Uvicorn |
| Frontend | Single-page HTML/CSS/JS (no framework) |

Only the answer LLM is a remote service. Search, ranking and verification run locally.

## Project structure

```
rag/
  config.py                 paths, model names, thresholds
  ingestion/                fetch_papers, parse_pdf, chunker, build_index, upload
  retrieval/                vector_store, bm25_search, hybrid, reranker
  generation/               prompts, llm_client, answer, web_search (optional)
  trust/                    citation_verifier
  api/main.py               FastAPI endpoints
  app/static/index.html     web interface
  evaluation/               testset.json, eval_retrieval, eval_abstention,
                            make_verify_labels, eval_verifier, make_negatives,
                            eval_verifier_full
  test_*.py                 small manual test scripts
```

## Setup

```bash
git clone <your-repo-url>
cd <repo>
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Create a `.env` file in the project root (never commit it):

```
GROQ_API_KEY=your_groq_key
# optional, only for the optional web mode
TAVILY_API_KEY=your_tavily_key
```

Keep the project in a short path (for example `C:\rag`). Long Windows paths can break `pip install`.

## Build the corpus

```bash
python -m ingestion.fetch_papers     # downloads papers from arXiv (default: 50)
python -m ingestion.chunker          # parse + chunk -> data/chunks.json
python -m ingestion.build_index      # embed + store in ChromaDB
```

Warning: `chunker` and `build_index` rebuild from scratch and remove papers added through the upload feature.

## Run

```bash
uvicorn api.main:app --port 8000
```

Open http://localhost:8000.

- **Scope**: "All papers" (built-in corpus + your uploads) or "My uploads" (only your PDFs).
- **Mode**: "Papers only" is the intended, evaluated behaviour. The "Hybrid" and "General (web)" modes are optional extras: they use web search, their answers are not grounded in your papers and are not covered by the evaluation.

## Evaluation

All numbers below were measured on this project's own data.

### Retrieval (34 answerable questions, paper-level hit)

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| Dense only | 0.941 | 0.941 | 0.971 | 1.000 | 0.950 |
| BM25 only | 0.971 | 1.000 | 1.000 | 1.000 | 0.985 |
| Hybrid (RRF) | 0.941 | 1.000 | 1.000 | 1.000 | 0.971 |
| Hybrid + reranker | 0.971 | 1.000 | 1.000 | 1.000 | 0.985 |

Caveat: the questions were generated from paper abstracts and often contain paper-specific keywords, so BM25 performs strongly and the methods are close to each other.

### Abstention (threshold 0.4)

- Out-of-corpus questions correctly refused: 10 / 10
- In-corpus questions wrongly refused: 0 / 10

### Citation verifier (threshold 0.4)

Evaluated on 31 statements: 21 supported, 2 unsupported (manually labelled) and 8 synthetic negatives (a number or key word altered in a supported claim).

| Metric | Value |
|---|---|
| Precision | 0.833 |
| Recall | 0.476 |
| Synthetic negatives wrongly accepted | 2 / 8 |

The verifier is conservative: it rejects negation flips and large numeric changes but misses many correct, heavily paraphrased claims, and it accepted claims where only one number ("41" to "85") or one adjective ("fast" to "slow") was changed. "Could not verify" in the UI therefore does not mean the statement is wrong.

## Limitations

- Small test set; questions derived from abstracts.
- Verifier labels were produced by one person; negatives are synthetic and easier than real hallucinations.
- Tested only on computer science / RAG papers.
- Text only: tables, figures and formulas are not understood. Scanned PDFs are not supported.
- English only.
- Each question is answered independently (no conversation memory).
- Free-tier Groq rate limits can slow answers.
- Retrieved passages are sent to the Groq API; do not upload confidential documents unless that is acceptable.
- Chat history is stored in the browser only.

## Future work

- Conversation memory with query rewriting
- Number and entity consistency check in the verifier
- Larger, human-written test set and multi-annotator labels
- Table-aware parsing and OCR for scanned PDFs
- Local LLM option (Ollama) for private documents

## License

Add a license of your choice (for example MIT).
