import json
import requests
import arxiv
from config import RAW_PDF_DIR, METADATA_FILE, ARXIV_QUERY, MAX_PAPERS


def download_pdf(url: str, path):
    resp = requests.get(url, timeout=60, headers={"User-Agent": "rag-student-project"})
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)


def fetch(query: str = ARXIV_QUERY, max_results: int = MAX_PAPERS):
    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []
    for r in client.results(search):
        paper_id = r.get_short_id().replace("/", "_")
        filename = f"{paper_id}.pdf"
        path = RAW_PDF_DIR / filename

        if not path.exists():
            try:
                pdf_url = r.pdf_url or r.entry_id.replace("/abs/", "/pdf/")
                download_pdf(pdf_url, path)
                print(f"Downloaded: {r.title}")
            except Exception as e:
                print(f"Skipped {paper_id}: {e}")
                continue

        papers.append({
            "paper_id": paper_id,
            "filename": filename,
            "title": r.title,
            "authors": ", ".join(a.name for a in r.authors[:6]),
            "year": r.published.year,
            "url": r.entry_id,
            "abstract": r.summary.replace("\n", " "),
        })

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"\nTotal papers saved: {len(papers)}")


if __name__ == "__main__":
    fetch()