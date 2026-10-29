"""RAG core: PDF ingestion -> FAISS retrieval -> answer generation."""

import os
from functools import lru_cache
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline

INDEX_DIR = Path(os.getenv("INDEX_DIR", "faiss_index"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEN_MODEL = os.getenv("GEN_MODEL", "google/flan-t5-base")

_vectorstore = None


@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@lru_cache(maxsize=1)
def get_generator():
    return pipeline("text2text-generation", model=GEN_MODEL)


def index_ready() -> bool:
    return _vectorstore is not None


def load_index():
    """Load a previously saved FAISS index from disk, if there is one."""
    global _vectorstore
    if INDEX_DIR.exists():
        _vectorstore = FAISS.load_local(
            str(INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True
        )


def ingest_pdf(pdf_path: str, source_name: str):
    """Load a PDF, split it into chunks, embed them and save them to FAISS."""
    global _vectorstore

    docs = PyPDFLoader(pdf_path).load()
    for doc in docs:
        doc.metadata["source"] = source_name

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError("No text could be extracted from this PDF (is it a scanned image?)")

    if _vectorstore is None:
        _vectorstore = FAISS.from_documents(chunks, get_embeddings())
    else:
        _vectorstore.add_documents(chunks)
    _vectorstore.save_local(str(INDEX_DIR))

    return len(docs), len(chunks)


def retrieve(question: str, k: int = 3):
    if _vectorstore is None:
        raise ValueError("No documents have been ingested yet. Upload a PDF to /ingest first.")
    return _vectorstore.similarity_search(question, k=k)


def build_prompt(question: str, chunks) -> str:
    context = "\n\n".join(c.page_content for c in chunks)
    return (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say \"I don't know based on the provided documents.\"\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def answer(question: str, k: int = 3):
    """Retrieve the top-k chunks, then generate an answer grounded in them."""
    chunks = retrieve(question, k)
    prompt = build_prompt(question, chunks)
    generator = get_generator()
    result = generator(prompt, max_new_tokens=200, truncation=True)
    text = result[0]["generated_text"].strip()

    sources = []
    for c in chunks:
        page = c.metadata.get("page")
        sources.append(
            {
                "source": c.metadata.get("source"),
                "page": page + 1 if isinstance(page, int) else page,
                "text": c.page_content[:200],
            }
        )
    return text, sources
