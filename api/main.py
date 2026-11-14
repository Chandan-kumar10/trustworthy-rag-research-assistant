import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import BASE_DIR
from ingestion import upload as up
from generation.answer import answer_question
from trust.citation_verifier import verify_answer

app = FastAPI(title="Research Assistant")
INDEX_HTML = BASE_DIR / "app" / "static" / "index.html"
MAX_MB = 25


class AskBody(BaseModel):
    question: str
    scope: str = "all"
    mode: str = "hybrid"


class VerifyBody(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/")
def home():
    return FileResponse(INDEX_HTML)


@app.get("/api/papers")
def papers():
    return {"papers": up.list_uploads()}


@app.post("/api/upload")
def upload(files: list[UploadFile] = File(...)):
    up.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for f in files:
        name = f.filename or "paper.pdf"
        if not name.lower().endswith(".pdf"):
            results.append({"filename": name, "ok": False, "error": "only PDF allowed"})
            continue

        content = f.file.read()
        if len(content) > MAX_MB * 1024 * 1024:
            results.append({"filename": name, "ok": False, "error": f"file above {MAX_MB} MB"})
            continue

        dest = up.UPLOAD_DIR / f"up_{uuid.uuid4().hex[:8]}.pdf"
        dest.write_bytes(content)
        try:
            rec = up.process_pdf(dest, name)
            results.append({"filename": name, "ok": True, "paper": rec})
        except Exception as e:
            dest.unlink(missing_ok=True)
            results.append({"filename": name, "ok": False, "error": str(e)})
    return {"results": results}


@app.delete("/api/papers/{paper_id}")
def delete(paper_id: str):
    if not up.delete_paper(paper_id):
        raise HTTPException(404, "Paper not found")
    return {"ok": True}


@app.post("/api/ask")
def ask(body: AskBody):
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "Please type a question.")
    if len(q) > 1000:
        raise HTTPException(400, "Question is too long (max 1000 characters).")

    paper_ids = None
    if body.scope == "mine":
        paper_ids = [p["paper_id"] for p in up.list_uploads()]
        if not paper_ids:
            raise HTTPException(400, "No uploaded papers yet. Upload a PDF or switch to 'All papers'.")

    try:
        # verify=False: answer returns fast, the UI calls /api/verify afterwards
        return answer_question(q, paper_ids, body.mode, verify=False)
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@app.post("/api/verify")
def verify(body: VerifyBody):
    try:
        return {"verification": verify_answer(body.answer, body.sources)}
    except Exception as e:
        raise HTTPException(500, f"Verification failed: {e}")