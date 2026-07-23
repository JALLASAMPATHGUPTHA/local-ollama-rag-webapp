# Local Ollama RAG

A small Retrieval-Augmented Generation project that runs fully locally with
Ollama models.

## What It Does

- Reads `.txt` and `.md` files from `docs/`
- Splits them into overlapping chunks
- Creates embeddings with Ollama
- Stores vectors in `storage/vector_store.json`
- Retrieves the most relevant chunks for a question
- Sends the retrieved context to a local Ollama chat model
- Includes a light web interface for upload, indexing, and Q&A

## Requirements

1. Install Python 3.10 or newer.
2. Install and run [Ollama](https://ollama.com/).
3. Pull one chat model. This project defaults to `llama3:latest` for both chat
   and embeddings because it works with Ollama's local embeddings API:

```powershell
ollama pull llama3
```

## Website

Run the local web interface:

```powershell
python web.py
```

Then open:

```text
http://127.0.0.1:8000
```

The website lets you upload `.txt` and `.md` files, rebuild the local index, and
ask questions from the browser.

## CLI

Put your `.txt` or `.md` files in `docs/`, then build the vector store:

```powershell
python rag.py ingest
```

Ask a question:

```powershell
python rag.py ask "What is this project about?"
```

Start an interactive chat:

```powershell
python rag.py chat
```

## Configuration

You can override defaults with environment variables:

```powershell
$env:OLLAMA_CHAT_MODEL="llama3:latest"
$env:OLLAMA_EMBED_MODEL="llama3:latest"
$env:OLLAMA_HOST="http://localhost:11434"
```

For better retrieval quality, you can also install a dedicated embedding model:

```powershell
ollama pull nomic-embed-text
$env:OLLAMA_EMBED_MODEL="nomic-embed-text"
```

## Project Layout

```text
.
|-- docs/                  # Add your local documents here
|-- static/                # Website CSS and JavaScript
|-- storage/               # Generated vector store
|-- templates/             # Website HTML
|-- rag.py                 # CLI app
|-- web.py                 # Local website server
`-- README.md
```

## Notes

This is intentionally small and readable. For larger projects, the next step is
usually swapping the JSON store for Chroma, LanceDB, FAISS, or SQLite with a
vector extension.
