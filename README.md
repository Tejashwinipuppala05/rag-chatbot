# RAG Document Q&A API

A lightweight FastAPI project that lets you upload a PDF, index it with FAISS, and ask grounded questions using a Retrieval-Augmented Generation flow.

## Features

- PDF upload support via FastAPI
- Text splitting and chunking with LangChain
- Embedding storage and similarity search using FAISS
- Hugging Face generation model for answer synthesis
- Saved index on disk for reuse across restarts
- Minimal web UI for upload and chat

## Project flow

```text
PDF -> PyPDFLoader -> RecursiveCharacterTextSplitter -> FAISS embeddings index
Question -> similarity search -> retrieved chunks -> prompt -> text generation
```

## Tech stack

- Python
- FastAPI
- LangChain
- FAISS
- Hugging Face Transformers
- PyPDF

## Setup

### 1) Create a virtual environment

```bash
python -m venv venv
```

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Start the app

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

Then open:

- http://127.0.0.1:8000
- http://127.0.0.1:8000/docs

> The first run downloads the Hugging Face models, so it may take a few minutes.

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Checks whether the app is running and whether an index is loaded |
| POST | `/ingest` | Uploads a PDF and indexes it |
| POST | `/ask` | Asks a question about the currently indexed document |

### Example ingest

```bash
curl -X POST http://127.0.0.1:8000/ingest -F "file=@your_document.pdf"
```

### Example question

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?", "k": 3}'
```

## Important notes

- This is a local demo project and is not a full multi-user production app.
- It uses a shared FAISS index for the current workspace.
- The generation model is intentionally small for CPU compatibility.
- Scanned PDF pages without real text will not work well.

## File overview

- `main.py` — FastAPI app and routes
- `rag.py` — retrieval and generation logic
- `static/index.html` — browser UI
- `requirements.txt` — package dependencies
- `faiss_index/` — generated vector database files

## License

This project is ready to be pushed to GitHub. You can add your own LICENSE file if you want to publish it publicly.
