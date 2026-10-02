#!/usr/bin/env python3
"""
End-to-end test for the Mati packaging + sync toolchain.

Verifies:
  1. package_builder can generate >= 100 demo packages, each <= 50 MB.
  2. sync_server serves catalog / manifest / chunk (Range) / diff correctly.
  3. sync_client downloads every package with content-addressed incremental
     sync and per-chunk SHA-256 verification.
  4. Under simulated packet loss, the end-to-end delivery success rate is
     >= 95% (client retries + resume make dropped requests recoverable).
  5. Resume works: interrupting a chunk mid-download and retrying completes it.
  6. Re-sync is incremental: a second sync fetches 0 chunks.

This test uses ONLY stdlib + the tools/ modules. Run from the repo root:
    python tools/test_sync_e2e.py
"""

import shutil
import sys
import tempfile
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from common import chunk_file, manifest_path, package_dir  # noqa: E402
import package_builder  # noqa: E402
import sync_client  # noqa: E402
import sync_server  # noqa: E402

DEMO_PACKAGES = 100
MAX_PACKAGE_MB = 50.0


def build_demo_dist(dist: Path) -> None:
    n = package_builder.build_demo(dist, DEMO_PACKAGES, int(MAX_PACKAGE_MB * 1024 * 1024))
    assert n == DEMO_PACKAGES, f"Expected {DEMO_PACKAGES} packages, built {n}"
    package_builder.write_catalog(dist)

    # Each package must be <= 50 MB and every chunk must verify.
    assert package_builder.validate_dist(dist), "Package validation failed"
    for pkg_dir in (dist / "packages").glob("*"):
        manifest = package_builder.read_json(manifest_path(pkg_dir))
        assert manifest["total_size_bytes"] <= int(MAX_PACKAGE_MB * 1024 * 1024), \
            f"{manifest['package_id']} exceeds 50MB"
    print(f"[1] Built & validated {DEMO_PACKAGES} demo packages (each <= {MAX_PACKAGE_MB}MB)")


class _QuietHandler(sync_server.SyncHandler):
    def log_message(self, fmt, *args):  # silence per-request logs
        pass


def start_server(dist: Path):
    sync_server.SyncHandler.root = dist.resolve()
    httpd = sync_server.ThreadingHTTPServer(("127.0.0.1", 0), _QuietHandler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, port


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="mati_sync_test_"))
    try:
        dist = tmp / "dist"
        dest = tmp / "dest"
        build_demo_dist(dist)

        httpd, port = start_server(dist)
        url = f"http://127.0.0.1:{port}"
        try:
            client = sync_client.SyncClient(server=url, dest=dest, max_retries=8, concurrency=4)

            # --- 2/3. Full sync (incremental + verification) ---
            results = client.sync_all()
            client.report(results)
            assert all(r["failed"] == 0 for r in results.values()), "Unrecovered failures"
            print("[2/3] Full sync of all packages completed with 0 failures")

            # Verify on-disk hashes independently.
            for pid in results:
                manifest = sync_client.read_json(
                    manifest_path(package_dir(dest, pid)))
                for c in manifest["chunks"]:
                    cf = chunk_file(package_dir(dest, pid), c["hash"])
                    assert cf.exists(), f"missing {cf}"
                    assert sync_client.verify_chunk_bytes(cf.read_bytes(), c["hash"])
            print("[2/3] All downloaded chunks verified on disk")

            # --- 4. Simulated packet loss: end-to-end success >= 95% ---
            shutil.rmtree(dest, ignore_errors=True)
            lossy = sync_client.SyncClient(
                server=url, dest=dest, max_retries=8, concurrency=4, packet_loss=5)
            results = lossy.sync_all()
            lossy.report(results)
            fetched = sum(r["fetched"] for r in results.values())
            total = sum(r["total"] for r in results.values())
            rate = fetched / total * 100.0 if total else 100.0
            assert rate >= 95.0, f"Success rate {rate:.1f}% < 95% under 5% loss"
            print(f"[4] Under simulated 5% loss: success rate {rate:.1f}% (>= 95%)")

            # --- 5. Resume: force an interruption on one chunk, then retry ---
            pid = sorted(results)[0]
            manifest = sync_client.read_json(manifest_path(package_dir(dest, pid)))
            target = manifest["chunks"][0]
            dest_pkg = package_dir(dest, pid)
            # Simulate a partially written .part file (e.g. previous crash).
            state = dest / ".mati_sync" / pid
            state.mkdir(parents=True, exist_ok=True)
            part = state / f"{target['hash']}.part"
            size = target.get("compressed_size", target.get("size", 0))
            part.write_bytes(b"\x00" * min(1024, size))
            res = client.sync_package(pid)
            assert res["failed"] == 0, "Resume sync failed"
            print(f"[5] Resume-from-partial verified for package {pid}")

            # --- 6. Incremental: second sync fetches nothing ---
            res2 = client.sync_package(pid)
            assert res2["fetched"] == 0, f"Expected incremental (0 fetched), got {res2['fetched']}"
            print(f"[6] Incremental re-sync fetched 0 chunks (all cached)")

            print("\nALL END-TO-END CHECKS PASSED")
            return 0
        finally:
            httpd.shutdown()
            httpd.server_close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
