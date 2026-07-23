from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
STORAGE_DIR = ROOT / "storage"
STORE_PATH = STORAGE_DIR / "vector_store.json"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3:latest")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "llama3:latest")

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("RAG_TOP_K", "4"))


@dataclass
class Chunk:
    id: str
    source: str
    text: str
    embedding: list[float]


def post_ollama(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_HOST}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_HOST}. "
            "Make sure Ollama is running."
        ) from exc


def embed(text: str) -> list[float]:
    response = post_ollama(
        "/api/embeddings",
        {
            "model": EMBED_MODEL,
            "prompt": text,
        },
    )
    embedding = response.get("embedding")
    if not embedding:
        raise RuntimeError(f"Ollama did not return an embedding: {response}")
    return embedding


def chat(question: str, contexts: list[Chunk]) -> str:
    context_text = "\n\n".join(
        f"[Source: {chunk.source}]\n{chunk.text}" for chunk in contexts
    )
    prompt = f"""
    Answer the user's question using only the context below.
    If the context does not contain the answer, say you do not know.
    Cite source filenames in your answer when useful.

    Context:
    {context_text}

    Question:
    {question}
    """.strip()

    response = post_ollama(
        "/api/chat",
        {
            "model": CHAT_MODEL,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful local RAG assistant.",
                },
                {"role": "user", "content": prompt},
            ],
        },
    )
    message = response.get("message", {})
    answer = message.get("content")
    if not answer:
        raise RuntimeError(f"Ollama did not return a chat response: {response}")
    return answer.strip()


def iter_documents() -> Iterable[Path]:
    for pattern in ("*.txt", "*.md"):
        yield from sorted(DOCS_DIR.rglob(pattern))


def read_document(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")

    normalized = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = end - overlap
    return chunks


def load_store() -> list[Chunk]:
    if not STORE_PATH.exists():
        raise FileNotFoundError(
            f"No vector store found at {STORE_PATH}. Run `python rag.py ingest` first."
        )

    data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    return [Chunk(**item) for item in data["chunks"]]


def save_store(chunks: list[Chunk]) -> None:
    STORAGE_DIR.mkdir(exist_ok=True)
    STORE_PATH.write_text(
        json.dumps(
            {
                "embedding_model": EMBED_MODEL,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "chunks": [asdict(chunk) for chunk in chunks],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve(question: str, chunks: list[Chunk], top_k: int = TOP_K) -> list[Chunk]:
    question_embedding = embed(question)
    ranked = sorted(
        chunks,
        key=lambda chunk: cosine_similarity(question_embedding, chunk.embedding),
        reverse=True,
    )
    return ranked[:top_k]


def ingest() -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    chunks: list[Chunk] = []
    documents = list(iter_documents())

    if not documents:
        print(f"No .txt or .md files found in {DOCS_DIR}")
        return

    for document in documents:
        text = read_document(document)
        if not text:
            continue

        relative_source = str(document.relative_to(ROOT))
        for index, chunk_text in enumerate(split_text(text), start=1):
            chunk_id = f"{relative_source}#{index}"
            print(f"Embedding {chunk_id}")
            chunks.append(
                Chunk(
                    id=chunk_id,
                    source=relative_source,
                    text=chunk_text,
                    embedding=embed(chunk_text),
                )
            )

    save_store(chunks)
    print(f"Saved {len(chunks)} chunks to {STORE_PATH}")


def ask(question: str) -> None:
    chunks = load_store()
    contexts = retrieve(question, chunks)
    answer = chat(question, contexts)

    print("\nAnswer:")
    print(textwrap.fill(answer, width=100))
    print("\nSources:")
    for chunk in contexts:
        print(f"- {chunk.source} ({chunk.id})")


def interactive_chat() -> None:
    print("Local RAG chat. Type `exit` or `quit` to stop.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            ask(question)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Small local RAG app using Ollama.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Embed documents and build the vector store.")

    ask_parser = subparsers.add_parser("ask", help="Ask one question.")
    ask_parser.add_argument("question", help="Question to answer from local documents.")

    subparsers.add_parser("chat", help="Start an interactive RAG chat.")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "ingest":
            ingest()
        elif args.command == "ask":
            ask(args.question)
        elif args.command == "chat":
            interactive_chat()
        else:
            parser.error(f"Unknown command: {args.command}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
