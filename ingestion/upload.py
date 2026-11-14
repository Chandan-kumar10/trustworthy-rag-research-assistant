import json
from pathlib import Path

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DATA_DIR, CHUNKS_FILE, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LEN
from ingestion.parse_pdf import parse_pdf
from retrieval.vector_store import get_model, get_collection
from retrieval.bm25_search import reset as reset_bm25

UPLOAD_DIR = DATA_DIR / "uploads"
UPLOADS_FILE = DATA_DIR / "uploads.json"


def list_uploads():
    if not UPLOADS_FILE.exists():
        return []
    with open(UPLOADS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _save_uploads(items):
    UPLOADS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(UPLOADS_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def _guess_title(path: Path, fallback: str) -> str:
    try:
        doc = fitz.open(path)
        title = (doc.metadata or {}).get("title", "").strip()
        if len(title) > 5:
            doc.close()
            return title[:150]
        first_page = doc[0].get_text().split("\n") if len(doc) else []
        doc.close()
        for line in first_page:
            line = line.strip()
            if 10 < len(line) < 150:
                return line
    except Exception:
        pass
    return fallback


def process_pdf(path: Path, original_name: str) -> dict:
    """Parse, chunk, embed and index one uploaded PDF. path must be <paper_id>.pdf"""
    paper_id = path.stem
    title = _guess_title(path, Path(original_name).stem)

    sections = parse_pdf(path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", ". ", " "],
    )

    chunks = []
    for si, sec in enumerate(sections):
        for ci, text in enumerate(splitter.split_text(sec["text"])):
            if len(text) < MIN_CHUNK_LEN:
                continue
            chunks.append({
                "id": f"{paper_id}_{si}_{ci}",
                "text": text,
                "meta": {
                    "paper_id": paper_id,
                    "title": title,
                    "authors": "Uploaded by you",
                    "year": "n/a",
                    "url": "",
                    "section": sec["section"],
                    "page": sec["page"],
                },
            })

    if not chunks:
        raise ValueError("No readable text found (scanned PDF?)")

    model, col = get_model(), get_collection()
    for i in range(0, len(chunks), 64):
        batch = chunks[i:i + 64]
        texts = [c["text"] for c in batch]
        emb = model.encode(texts, normalize_embeddings=True).tolist()
        col.add(
            ids=[c["id"] for c in batch],
            documents=texts,
            embeddings=emb,
            metadatas=[c["meta"] for c in batch],
        )

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        all_chunks = json.load(f)
    all_chunks.extend(chunks)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    reset_bm25()

    record = {
        "paper_id": paper_id,
        "title": title,
        "filename": original_name,
        "chunks": len(chunks),
    }
    items = list_uploads()
    items.append(record)
    _save_uploads(items)
    return record


def delete_paper(paper_id: str) -> bool:
    items = list_uploads()
    if not any(p["paper_id"] == paper_id for p in items):
        return False

    get_collection().delete(where={"paper_id": paper_id})

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        all_chunks = json.load(f)
    all_chunks = [c for c in all_chunks if c["meta"]["paper_id"] != paper_id]
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    reset_bm25()

    (UPLOAD_DIR / f"{paper_id}.pdf").unlink(missing_ok=True)
    _save_uploads([p for p in items if p["paper_id"] != paper_id])
    return True