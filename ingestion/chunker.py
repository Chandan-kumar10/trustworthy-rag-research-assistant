import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import (
    RAW_PDF_DIR, METADATA_FILE, CHUNKS_FILE,
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LEN,
)
from ingestion.parse_pdf import parse_pdf


def build_chunks():
    with open(METADATA_FILE, encoding="utf-8") as f:
        papers = json.load(f)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", ". ", " "],
    )

    chunks = []
    for p in papers:
        pdf_path = RAW_PDF_DIR / p["filename"]
        try:
            sections = parse_pdf(pdf_path)
        except Exception as e:
            print(f"Parse failed for {p['paper_id']}: {e}")
            continue

        for si, sec in enumerate(sections):
            for ci, text in enumerate(splitter.split_text(sec["text"])):
                if len(text) < MIN_CHUNK_LEN:
                    continue
                chunks.append({
                    "id": f"{p['paper_id']}_{si}_{ci}",
                    "text": text,
                    "meta": {
                        "paper_id": p["paper_id"],
                        "title": p["title"],
                        "authors": p["authors"],
                        "year": p["year"],
                        "url": p["url"],
                        "section": sec["section"],
                        "page": sec["page"],
                    },
                })

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print(f"Papers: {len(papers)} | Chunks created: {len(chunks)}")
    return chunks


if __name__ == "__main__":
    build_chunks()