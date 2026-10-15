"""Dependency-free JSON API and static frontend server."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .explain import explain
from .persistence import load_artifact


def create_server(artifact: Path, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    vectorizer, model = load_artifact(artifact)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: object, content_type: str = "application/json") -> None:
            body = json.dumps(payload).encode() if content_type == "application/json" else payload
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if urlparse(self.path).path == "/api/health":
                self._send(200, {"status": "ok", "model": "cinescope-logistic-regression"})
                return
            index = Path(__file__).resolve().parents[1] / "web" / "index.html"
            self._send(200, index.read_bytes(), "text/html; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/predict":
                self._send(404, {"error": "not found"})
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                text = payload["text"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("text must be a non-empty string")
                probability = model.predict_proba_one(vectorizer.transform([text])[0])
                self._send(200, {"label": "positive" if probability >= 0.5 else "negative",
                                 "probability": round(probability, 4),
                                 "explanation": explain(text, vectorizer, model)})
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                self._send(400, {"error": str(error)})

        def log_message(self, *_args) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)


def run(artifact: Path, host: str = "127.0.0.1", port: int = 8000) -> None:
    print(f"CINESCOPE listening on http://{host}:{port}")
    create_server(artifact, host, port).serve_forever()
