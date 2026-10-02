#!/usr/bin/env python3
"""
Validates scripts/ingest_content.py --ingest-from-manifest --dry-run against a
synced package directory (no ML dependencies required for dry-run).
"""

import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import package_builder  # noqa: E402
import sync_client  # noqa: E402
import sync_server  # noqa: E402


class _Quiet(sync_server.SyncHandler):
    def log_message(self, fmt, *args):
        pass


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="mati_ingest_test_"))
    try:
        dist = tmp / "dist"
        dest = tmp / "dest"
        package_builder.build_demo(dist, 3, int(50 * 1024 * 1024))
        package_builder.write_catalog(dist)

        sync_server.SyncHandler.root = dist.resolve()
        httpd = sync_server.ThreadingHTTPServer(("127.0.0.1", 0), _Quiet)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            client = sync_client.SyncClient(
                server=f"http://127.0.0.1:{port}", dest=dest)
            results = client.sync_all()
            client.report(results)
            assert all(r["failed"] == 0 for r in results.values())
        finally:
            httpd.shutdown()
            httpd.server_close()

        # Dry-run import via the real CLI.
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "ingest_content.py"),
               "--ingest-from-manifest", str(dest), "--dry-run"]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=str(REPO_ROOT))
        combined = (proc.stdout + proc.stderr)
        print(combined[-2000:])
        if proc.returncode != 0:
            print("INGEST DRY-RUN FAILED")
            return 1
        assert "dry-run" in combined or "Imported" in combined, combined[-2000:]
        print("INGEST --ingest-from-manifest --dry-run OK")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
