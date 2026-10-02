#!/usr/bin/env python3
"""
Lightweight sync server for Mati content packages.

A dependency-free HTTP server (stdlib only) that serves the content of a
`dist/` directory produced by tools/package_builder.py.

Endpoints:
    GET /catalog                      -> JSON catalog of all packages
    GET /manifest/<package_id>        -> JSON manifest for one package
    GET /chunk/<package_id>/<hash>    -> single chunk file (supports HTTP Range)
    GET /diff?package=<id>&have=h1,h2 -> JSON list of missing chunk hashes

Usage:
    python tools/sync_server.py --root dist --host 0.0.0.0 --port 8000
"""

import argparse
import json
import logging
import re
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CATALOG_NAME,
    chunk_file,
    manifest_path,
    package_dir,
    read_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sync_server")

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class SyncHandler(BaseHTTPRequestHandler):
    root: Path = Path("dist")
    server_version = "MatiSync/1.0"

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        parsed = urllib.parse.urlparse(self.path)
        parts = [urllib.parse.unquote(p) for p in parsed.path.strip("/").split("/") if p]

        try:
            if not parts:
                return self._send_json(200, {"service": "mati-sync", "status": "ok"})

            if parts[0] == "catalog":
                return self._serve_catalog()

            if parts[0] == "manifest" and len(parts) == 2:
                return self._serve_manifest(parts[1])

            if parts[0] == "chunk" and len(parts) == 3:
                return self._serve_chunk(parts[1], parts[2])

            if parts[0] == "diff" and len(parts) == 1:
                return self._serve_diff(parsed.query)

            return self._send_json(404, {"error": "not found"})
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Request failed: %s", self.path)
            self._send_json(500, {"error": str(exc)})

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    def _serve_catalog(self) -> None:
        catalog = self.root / CATALOG_NAME
        if not catalog.exists():
            return self._send_json(404, {"error": "catalog not found"})
        self._send_json(200, read_json(catalog))

    def _serve_manifest(self, package_id: str) -> None:
        mp = manifest_path(package_dir(self.root, package_id))
        if not mp.exists():
            return self._send_json(404, {"error": f"package {package_id} not found"})
        self._send_json(200, read_json(mp))

    def _serve_chunk(self, package_id: str, chunk_hash: str) -> None:
        cf = chunk_file(package_dir(self.root, package_id), chunk_hash)
        if not cf.exists():
            return self._send_json(404, {"error": "chunk not found"})

        size = cf.stat().st_size
        start, end = 0, size - 1
        status = 200

        range_header = self.headers.get("Range")
        if range_header:
            match = _RANGE_RE.search(range_header)
            if match:
                r_start, r_end = match.group(1), match.group(2)
                start = int(r_start) if r_start else 0
                if r_end:
                    end = min(int(r_end), size - 1)
                else:
                    end = size - 1
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                status = 206

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()

        with open(cf, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                block = f.read(min(64 * 1024, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)

    def _serve_diff(self, query: str) -> None:
        params = urllib.parse.parse_qs(query)
        package_id = (params.get("package") or [""])[0]
        have = set((params.get("have") or [""])[0].split(","))
        have.discard("")

        mp = manifest_path(package_dir(self.root, package_id))
        if not mp.exists():
            return self._send_json(404, {"error": f"package {package_id} not found"})

        manifest = read_json(mp)
        missing = [
            c["hash"] for c in manifest["chunks"]
            if c["hash"] not in have
        ]
        self._send_json(200, {
            "package_id": package_id,
            "have_count": len(have & {c["hash"] for c in manifest["chunks"]}),
            "missing_count": len(missing),
            "missing": missing,
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _send_json(self, status: int, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # quieter logging
        logger.info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mati sync server")
    parser.add_argument("--root", type=Path, default=Path("dist"), help="dist directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if not (args.root / CATALOG_NAME).exists():
        logger.error("No catalog.json found under %s. Run package_builder first.", args.root)
        sys.exit(1)

    SyncHandler.root = args.root.resolve()
    httpd = ThreadingHTTPServer((args.host, args.port), SyncHandler)
    logger.info("Sync server serving %s on http://%s:%d",
                args.root.resolve(), args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
