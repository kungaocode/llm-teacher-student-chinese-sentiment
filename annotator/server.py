"""Zero-dependency HTTP server for the local annotation web app."""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import unquote, urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotator.store import AnnotationStore, AnnotationStoreError


STATIC_DIR = Path(__file__).resolve().parent / "static"
_MAX_BODY_BYTES = 64 * 1024


def _handler_for(store: AnnotationStore):
    class AnnotationHandler(BaseHTTPRequestHandler):
        server_version = "SentimentAnnotator/1.0"

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/health":
                self._send_json(200, {"ok": True})
                return
            if path == "/api/session":
                self._handle(lambda: store.session())
                return
            if path.startswith("/api/"):
                self._send_json(404, {"error": f"unknown API route: {path}"})
                return
            self._serve_static(path)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/annotations":
                self._handle(self._annotation)
                return
            if path == "/api/export/csv":
                self._handle(lambda: store.export_csv())
                return
            self._send_json(404, {"error": f"unknown API route: {path}"})

        def do_DELETE(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            prefix = "/api/annotations/"
            if path.startswith(prefix):
                row_id = unquote(path[len(prefix):])
                self._handle(lambda: store.clear(row_id))
                return
            self._send_json(404, {"error": f"unknown API route: {path}"})

        def _annotation(self) -> dict:
            payload = self._read_json()
            if not isinstance(payload, dict):
                raise AnnotationStoreError("request body must be a JSON object")
            if "id" not in payload:
                raise AnnotationStoreError("request body is missing 'id'")
            label = payload.get("label")
            if label not in store.labels:
                raise AnnotationStoreError(
                    "label must be one of: " + ", ".join(store.labels)
                )
            return store.annotate(payload["id"], label)

        def _handle(self, action) -> None:
            try:
                payload = action()
            except AnnotationStoreError as exc:
                self._send_json(400, {"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:  # pragma: no cover - safety net for the local server
                self._send_json(500, {"error": f"internal server error: {exc}"})
            else:
                self._send_json(200, {"ok": True, "result": payload})

        def _read_json(self):
            raw_length = self.headers.get("Content-Length", "0")
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise AnnotationStoreError("invalid Content-Length") from exc
            if length <= 0 or length > _MAX_BODY_BYTES:
                raise AnnotationStoreError("invalid request body size")
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise AnnotationStoreError("request body must be valid UTF-8 JSON") from exc

        def _serve_static(self, request_path: str) -> None:
            relative = "index.html" if request_path == "/" else request_path.lstrip("/")
            if relative not in {"index.html", "app.js", "styles.css"}:
                self._send_json(404, {"error": "not found"})
                return
            path = STATIC_DIR / relative
            if not path.is_file():
                self._send_json(404, {"error": f"missing static asset: {relative}"})
                return
            body = path.read_bytes()
            content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            if content_type.startswith("text/") or content_type in (
                "application/javascript",
                "text/javascript",
            ):
                content_type += "; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            print(f"[annotator] {self.address_string()} {fmt % args}")

    return AnnotationHandler


def build_server(
    store: AnnotationStore,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    """Create a threaded HTTP server without starting it."""
    return ThreadingHTTPServer((host, port), _handler_for(store))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local sentiment annotation web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--annotator", default="a", help="annotation identity (default: a)")
    parser.add_argument("--gold", default=None, help="override data/splits/gold.jsonl")
    parser.add_argument("--open", action="store_true", help="open the app in the default browser")
    args = parser.parse_args(argv)

    try:
        store = AnnotationStore(ROOT, args.annotator, gold_path=args.gold)
        total = len(store.session()["items"])
    except (AnnotationStoreError, ValueError) as exc:
        print(f"[annotator] cannot start: {exc}")
        return 1

    try:
        server = build_server(store, args.host, args.port)
    except OSError as exc:
        print(f"[annotator] cannot listen on {args.host}:{args.port}: {exc}")
        print("[annotator] choose another port with --port.")
        return 1

    actual_port = server.server_address[1]
    display_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{display_host}:{actual_port}"
    print(f"[annotator] {total} gold rows loaded")
    print(f"[annotator] annotator={store.annotator}  url={url}")
    print("[annotator] press Ctrl+C to stop")

    if args.open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[annotator] stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
