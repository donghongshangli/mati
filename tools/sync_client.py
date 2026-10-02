#!/usr/bin/env python3
"""
Mati sync client.

Downloads content packages from a tools/sync_server.py endpoint with:
  * Content-addressed incremental sync (only missing chunks are fetched)
  * Resumable transfers (per-chunk HTTP Range continuation via .part files)
  * Per-chunk SHA-256 verification
  * Exponential-backoff retries (robust under packet loss)
  * Optional concurrent downloads

Local state is kept under <dest>/.mati_sync/<package_id>/ and never
overwrites existing files until a chunk has been fully verified.

Usage:
    python tools/sync_client.py --server http://127.0.0.1:8000 \
        --dest mati_data/content --package demo_science_grade_10_unit_001

    # Sync ALL packages in the catalog:
    python tools/sync_client.py --server http://127.0.0.1:8000 --dest mati_data/content

    # Test helper (NOT for production): simulate network packet loss:
    python tools/sync_client.py --server http://127.0.0.1:8000 --dest tmp \
        --simulate-packet-loss 5
"""

import argparse
import json
import logging
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CHUNK_EXT,
    NETWORK_BLOCK_SIZE,
    STATE_DIR,
    chunk_file,
    decompress_bytes,
    manifest_path,
    package_dir,
    read_json,
    verify_chunk_bytes,
    write_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sync_client")

DEFAULT_MAX_RETRIES = 5
DEFAULT_CONCURRENCY = 4


class SyncError(Exception):
    pass


class SyncClient:
    def __init__(
        self,
        server: str,
        dest: Path,
        max_retries: int = DEFAULT_MAX_RETRIES,
        concurrency: int = DEFAULT_CONCURRENCY,
        packet_loss: int = 0,  # % simulated loss for testing only
    ):
        self.server = server.rstrip("/")
        self.dest = Path(dest)
        self.max_retries = max_retries
        self.concurrency = max(1, concurrency)
        self.packet_loss = packet_loss
        self._rng = random.Random(20260801)  # deterministic for tests

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    def _maybe_drop(self) -> None:
        """Simulate a dropped request (test only)."""
        if self.packet_loss > 0 and self._rng.randint(1, 100) <= self.packet_loss:
            raise urllib.error.URLError("Simulated packet loss")

    def _get(self, path: str, headers: Optional[Dict[str, str]] = None) -> bytes:
        url = f"{self.server}/{path.lstrip('/')}"
        req = urllib.request.Request(url, headers=headers or {})
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self._maybe_drop()
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return resp.read()
            except Exception as exc:  # retry on any transient failure
                last_exc = exc
                if attempt == self.max_retries:
                    break
                delay = min(0.5 * (2 ** (attempt - 1)), 8.0)
                logger.warning("Request %s failed (attempt %d/%d): %s; retrying in %.1fs",
                               path, attempt, self.max_retries, exc, delay)
                time.sleep(delay)
        raise SyncError(f"Request failed after {self.max_retries} attempts: {path}: {last_exc}")

    def _get_json(self, path: str) -> dict:
        return json.loads(self._get(path).decode("utf-8"))

    # ------------------------------------------------------------------
    # Fetch metadata
    # ------------------------------------------------------------------
    def fetch_catalog(self) -> dict:
        return self._get_json("catalog")

    def fetch_manifest(self, package_id: str) -> dict:
        return self._get_json(f"manifest/{urllib.parse.quote(package_id)}")

    # ------------------------------------------------------------------
    # Chunk download (resumable)
    # ------------------------------------------------------------------
    def _state_dir(self, package_id: str) -> Path:
        return self.dest / STATE_DIR / package_id

    def _existing_hashes(self, package_id: str) -> set:
        """Hashes already fully downloaded and verified (from .done markers)."""
        state = self._state_dir(package_id)
        done = set()
        if state.exists():
            for marker in state.glob("*.done"):
                done.add(marker.stem)
        return done

    def _download_chunk(self, package_id: str, chunk_hash: str, expected_size: int) -> bool:
        """
        Download one chunk with resume support. Returns True on success.

        `expected_size` is the COMPRESSED size (bytes on the wire / disk).
        Writes to <dest>/.mati_sync/<pkg>/<hash>.part, then decompresses and
        verifies the content-address SHA-256 (of the uncompressed bytes) before
        promoting to the final location.
        """
        state = self._state_dir(package_id)
        state.mkdir(parents=True, exist_ok=True)
        part = state / f"{chunk_hash}.part"
        done = state / f"{chunk_hash}.done"
        final_dir = package_dir(self.dest, package_id) / "chunks"
        final = final_dir / f"{chunk_hash}{CHUNK_EXT}"

        # Fast path: already completed and verified.
        if done.exists():
            return True
        if final.exists():
            try:
                if verify_chunk_bytes(final.read_bytes(), chunk_hash):
                    done.touch()
                    return True
            except Exception:
                pass

        # Resume from existing partial bytes (compressed payload length).
        offset = part.stat().st_size if part.exists() else 0
        if offset > expected_size:
            offset = 0  # corrupt partial; restart this chunk

        url_path = f"chunk/{urllib.parse.quote(package_id)}/{chunk_hash}"
        with open(part, "ab") as fh:
            while offset < expected_size:
                end = min(offset + NETWORK_BLOCK_SIZE - 1, expected_size - 1)
                headers = {"Range": f"bytes={offset}-{end}"}
                try:
                    data = self._get(url_path, headers)
                except SyncError as exc:
                    logger.error("Chunk %s aborted at offset %d: %s", chunk_hash, offset, exc)
                    return False
                fh.write(data)
                offset += len(data)

        # Verify integrity (decompress, then hash-compare) before promoting.
        if not verify_chunk_bytes(part.read_bytes(), chunk_hash):
            logger.error("Hash mismatch for chunk %s; discarding partial.", chunk_hash)
            part.unlink(missing_ok=True)
            return False

        final_dir.mkdir(parents=True, exist_ok=True)
        part.replace(final)
        done.touch()
        logger.info("  OK chunk %s (%d bytes compressed)", chunk_hash, expected_size)
        return True

    # ------------------------------------------------------------------
    # Sync a package
    # ------------------------------------------------------------------
    def sync_package(self, package_id: str) -> Dict[str, int]:
        manifest = self.fetch_manifest(package_id)
        chunks = manifest["chunks"]
        total = len(chunks)

        # Persist the manifest locally so the destination mirrors the server
        # layout (packages/<id>/manifest.json + chunks/), ready for ingest.
        local_manifest = manifest_path(package_dir(self.dest, package_id))
        write_json(local_manifest, manifest)

        # Incremental: skip hashes we already have.
        have = self._existing_hashes(package_id)
        wanted = [c for c in chunks if c["hash"] not in have]
        logger.info("Package %s: %d chunks total, %d already present, %d to fetch",
                    package_id, total, total - len(wanted), len(wanted))

        ok = 0
        failed: List[str] = []

        def work(chunk: dict) -> bool:
            expected = chunk.get("compressed_size", chunk.get("size", 0))
            return self._download_chunk(package_id, chunk["hash"], expected)

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(work, c): c for c in wanted}
            for fut in as_completed(futures):
                c = futures[fut]
                try:
                    if fut.result():
                        ok += 1
                    else:
                        failed.append(c["hash"])
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error("Chunk %s raised: %s", c["hash"], exc)
                    failed.append(c["hash"])

        result = {"package_id": package_id, "total": total,
                  "fetched": ok, "failed": len(failed)}
        logger.info("Package %s synced: %d/%d", package_id, ok, total)
        return result

    def sync_all(self) -> Dict[str, Dict[str, int]]:
        catalog = self.fetch_catalog()
        results = {}
        for pid in sorted(catalog.get("packages", {})):
            results[pid] = self.sync_package(pid)
        return results

    def report(self, results: Dict[str, Dict[str, int]]) -> None:
        fetched = sum(r["fetched"] for r in results.values())
        failed = sum(r["failed"] for r in results.values())
        total = sum(r["total"] for r in results.values())
        success_rate = (fetched / total * 100.0) if total else 100.0
        print("\n=== Sync report ===")
        print(f"Packages: {len(results)}")
        print(f"Chunks: {fetched}/{total} delivered, {failed} failed")
        print(f"End-to-end success rate: {success_rate:.1f}%")
        return success_rate


def main() -> int:
    parser = argparse.ArgumentParser(description="Mati sync client")
    parser.add_argument("--server", required=True, help="Server base URL")
    parser.add_argument("--dest", type=Path, default=Path("mati_data/content"),
                        help="Destination directory")
    parser.add_argument("--package", help="Sync a single package id (default: all)")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--simulate-packet-loss", type=int, default=0, metavar="PCT",
                        help="Simulate P% dropped requests (TEST ONLY)")
    args = parser.parse_args()

    client = SyncClient(
        server=args.server,
        dest=args.dest,
        max_retries=args.max_retries,
        concurrency=args.concurrency,
        packet_loss=args.simulate_packet_loss,
    )

    if args.package:
        results = {args.package: client.sync_package(args.package)}
    else:
        results = client.sync_all()

    client.report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
