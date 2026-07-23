from __future__ import annotations

import cgi
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import rag


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
TEMPLATES_DIR = ROOT / "templates"
ALLOWED_EXTENSIONS = {".txt", ".md"}


def json_bytes(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


def safe_filename(name: str) -> str:
    candidate = Path(name).name.strip().replace(" ", "_")
    return "".join(char for char in candidate if char.isalnum() or char in "._-")


class RAGRequestHandler(BaseHTTPRequestHandler):
    server_version = "LocalRAG/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_file(TEMPLATES_DIR / "index.html", "text/html; charset=utf-8")
            return

        if parsed.path.startswith("/static/"):
            self.send_static_file(parsed.path.removeprefix("/static/"))
            return

        self.send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/upload":
            self.handle_upload()
            return

        if parsed.path == "/api/ingest":
            self.handle_ingest()
            return

        if parsed.path == "/api/ask":
            self.handle_ask()
            return

        self.send_json({"error": "not found"}, status=404)

    def handle_upload(self) -> None:
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                    "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                },
            )
            files = form["files"] if "files" in form else []
            if not isinstance(files, list):
                files = [files]

            saved = []
            rag.DOCS_DIR.mkdir(exist_ok=True)
            for item in files:
                if not item.filename:
                    continue

                filename = safe_filename(item.filename)
                if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
                    continue

                target = rag.DOCS_DIR / filename
                with target.open("wb") as output:
                    output.write(item.file.read())
                saved.append(str(target.relative_to(ROOT)))

            if not saved:
                self.send_json(
                    {"error": "Upload at least one .txt or .md file."},
                    status=400,
                )
                return

            chunk_count = self.rebuild_index()
            self.send_json(
                {
                    "message": "Files uploaded and indexed.",
                    "files": saved,
                    "chunks": chunk_count,
                }
            )
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def handle_ingest(self) -> None:
        try:
            chunk_count = self.rebuild_index()
            self.send_json({"message": "Documents indexed.", "chunks": chunk_count})
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def handle_ask(self) -> None:
        try:
            payload = self.read_json()
            question = str(payload.get("question", "")).strip()
            if not question:
                self.send_json({"error": "Question is required."}, status=400)
                return

            chunks = rag.load_store()
            contexts = rag.retrieve(question, chunks)
            answer = rag.chat(question, contexts)
            self.send_json(
                {
                    "answer": answer,
                    "sources": [
                        {
                            "id": chunk.id,
                            "source": chunk.source,
                            "preview": chunk.text[:240],
                        }
                        for chunk in contexts
                    ],
                }
            )
        except FileNotFoundError:
            self.send_json(
                {"error": "No index found. Upload documents or click Rebuild Index."},
                status=400,
            )
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def rebuild_index(self) -> int:
        rag.ingest()
        return len(rag.load_store())

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def send_static_file(self, relative: str) -> None:
        path = (STATIC_DIR / relative).resolve()
        try:
            path.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self.send_json({"error": "not found"}, status=404)
            return

        if not path.exists() or not path.is_file():
            self.send_json({"error": "not found"}, status=404)
            return

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_file(path, content_type)

    def send_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: dict, status: int = 200) -> None:
        data = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), RAGRequestHandler)
    print("Local RAG website running at http://127.0.0.1:8000")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
