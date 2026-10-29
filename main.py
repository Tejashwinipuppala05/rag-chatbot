import os
import shutil
import tempfile
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag.load_index()  # reuse the saved FAISS index if one exists
    yield


app = FastAPI(
    title="RAG Document Q&A API",
    description="Upload PDFs, index them, and ask grounded questions using a local Retrieval-Augmented Generation workflow.",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    k: int = Field(3, ge=1, le=5, description="Number of chunks to retrieve")


@app.get("/health")
def health():
    return {"status": "ok", "index_loaded": rag.index_ready()}


# These are plain `def` (not `async def`) on purpose: embedding and generation
# are slow, blocking CPU work, and FastAPI runs plain `def` endpoints in a
# thread pool so they don't freeze the event loop.
@app.post("/ingest")
def ingest(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        pages, chunks = rag.ingest_pdf(tmp_path, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        os.remove(tmp_path)

    return {"filename": file.filename, "pages": pages, "chunks": chunks}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        answer, sources = rag.answer(req.question, req.k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"question": req.question, "answer": answer, "sources": sources}


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
