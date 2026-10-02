#!/usr/bin/env python3
"""
Mati content package builder.

Builds lightweight content packages (each <= max-package-mb) with a
manifest.json describing every chunk (id, hash, size, collection, metadata).
Each chunk is stored as its own file (chunks/<sha256>.chunk) so sync clients
can perform content-addressed incremental sync and resumable (Range-based)
downloads.

Output layout (out_dir):
    catalog.json
    packages/<package_id>/manifest.json
    packages/<package_id>/chunks/<sha256>.chunk

Usage:
    # Build packages from real content files (txt/md/jsonl/json)
    python tools/package_builder.py --src scripts/data_collection/data/content --out dist

    # Generate N sample packages (for testing / demo only)
    python tools/package_builder.py --demo 100 --out dist

    # Validate an existing dist tree
    python tools/package_builder.py --validate dist
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CATALOG_NAME,
    COMPRESSION_ALGO,
    FORMAT_VERSION,
    MAX_CHUNK_BYTES,
    chunk_file,
    compress_bytes,
    decompress_bytes,
    hash_bytes,
    manifest_path,
    package_dir,
    read_json,
    simple_chunk,
    write_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("package_builder")

SUPPORTED_SUFFIXES = {".txt", ".md", ".jsonl", ".json"}


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------
def extract_json_content(path: Path) -> str:
    """Flatten Mati's content JSON (subject/grade/units/QA) into text."""
    try:
        data = read_json(path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Cannot read JSON %s: %s", path, exc)
        return ""

    lines: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in ("subject", "grade", "name", "summary", "question",
                        "answer", "acceptable_answers", "hint", "hints",
                        "description", "example", "formula", "explanation"):
                if key in node and isinstance(node[key], (str, int, float)):
                    lines.append(f"{key}: {node[key]}")
            for key in ("acceptable_answers", "hints"):
                if key in node and isinstance(node[key], list):
                    for item in node[key]:
                        if isinstance(item, str):
                            lines.append(f"{key[:-1]}: {item}")
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)
    return "\n".join(lines)


def extract_file_content(path: Path) -> str:
    """Extract plain text from a supported content file."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        return extract_json_content(path)
    try:
        if suffix == ".jsonl":
            lines: List[str] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        text = obj.get("text") or obj.get("content") or obj.get("question")
                        if text:
                            lines.append(str(text))
                    except json.JSONDecodeError:
                        lines.append(line)
            return "\n".join(lines)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Cannot read %s: %s", path, exc)
        return ""


def infer_metadata(file_path: Path) -> Dict[str, str]:
    """Best-effort grade/subject inference from path (mirrors ingest_content.py)."""
    grade = "unknown"
    for part in file_path.parts:
        low = part.lower()
        for token in ("grade_", "class_", "grade", "class"):
            if token in low:
                candidate = low.replace(token, "").strip("_")
                if candidate.isdigit():
                    grade = candidate
                    break
    stem = file_path.stem.lower()
    subject = stem.split("_grade_")[0].split("_")[0]
    collection = f"neb_{subject}_grade_{grade}" if subject and subject != "unknown" else f"neb_{stem}"
    return {"grade": grade, "subject": subject, "collection_name": collection}


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------
def build_package(
    out_dir: Path,
    package_id: str,
    chunks: List[Dict[str, Any]],
    version: str = "1.0.0",
    description: str = "",
    max_package_bytes: int = 50 * 1024 * 1024,
) -> Dict[str, Any]:
    """Write one package (manifest + chunk files) under out_dir."""
    pkg_dir = package_dir(out_dir, package_id)
    pkg_dir.mkdir(parents=True, exist_ok=True)

    total_size = 0        # compressed bytes (what is stored/transferred)
    total_source = 0      # uncompressed source bytes
    written: List[Dict[str, Any]] = []
    collections = set()

    for idx, item in enumerate(chunks):
        text = (item.get("text") or "").strip()
        if not text:
            continue
        cid = item.get("id") or f"{package_id}_{idx}"
        coll = item.get("collection") or "default"
        meta = item.get("metadata") or {}
        raw = text.encode("utf-8")
        h = hash_bytes(raw)                  # hash of UNCOMPRESSED content
        payload = compress_bytes(raw)        # stored/transferred bytes
        size = len(raw)
        compressed_size = len(payload)

        if total_size + compressed_size > max_package_bytes:
            logger.warning(
                "Package %s would exceed %d bytes; %d chunks written so far. "
                "Use a smaller max-chunk or split sources.",
                package_id, max_package_bytes, len(written),
            )
            break

        cf = chunk_file(pkg_dir, h)
        if not cf.exists():
            cf.parent.mkdir(parents=True, exist_ok=True)
            # Byte-exact write of the compressed payload (Path.write_text would
            # translate \n -> \r\n on Windows and corrupt the payload).
            cf.write_bytes(payload)

        written.append({
            "id": cid,
            "hash": h,
            "size": size,                   # uncompressed size
            "compressed_size": compressed_size,
            "compression": COMPRESSION_ALGO,
            "collection": coll,
            "metadata": meta,
        })
        collections.add(coll)
        total_size += compressed_size
        total_source += size

    manifest = {
        "format_version": FORMAT_VERSION,
        "package_id": package_id,
        "version": version,
        "description": description,
        "compression": COMPRESSION_ALGO,
        "total_chunks": len(written),
        "total_size_bytes": total_size,       # on-disk / transfer size
        "total_source_bytes": total_source,   # uncompressed size
        "collections": sorted(collections),
        "chunks": written,
    }
    write_json(manifest_path(pkg_dir), manifest)
    return manifest


def build_from_sources(src_dir: Path, out_dir: Path, max_package_bytes: int) -> int:
    """Build one package per content file found under src_dir."""
    files = [
        p for p in src_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not files:
        logger.warning("No supported content files found under %s", src_dir)
        return 0

    count = 0
    for path in sorted(files):
        text = extract_file_content(path)
        if not text.strip():
            continue
        meta = infer_metadata(path)
        pid = f"{meta['collection_name']}_{path.stem}".replace(" ", "_")
        pid = "".join(c if c.isalnum() or c in "-_" else "_" for c in pid).lower()
        pieces = simple_chunk(text, MAX_CHUNK_BYTES)
        chunks = [{"text": piece, "metadata": meta} for piece in pieces]
        build_package(out_dir, pid, chunks, max_package_bytes=max_package_bytes)
        count += 1
        logger.info("Built package %s (%d chunks)", pid, len(chunks))
    return count


def build_demo(out_dir: Path, count: int, max_package_bytes: int) -> int:
    """
    Generate `count` sample packages with synthetic content.

    NOTE: This produces DEMO data for testing the toolchain only. It does NOT
    represent real curriculum content and must not be shipped as educational
    material.
    """
    subjects = ["science", "mathematics", "english", "computer_science"]
    for i in range(1, count + 1):
        subject = subjects[i % len(subjects)]
        grade = 8 + (i % 5)
        pid = f"demo_{subject}_grade_{grade}_unit_{i:03d}"
        coll = f"neb_{subject}_grade_{grade}"
        chunks = []
        for j in range(3):
            text = (
                f"Sample lesson content for {subject} grade {grade} unit {i} "
                f"section {j}. This synthetic paragraph is used to exercise the "
                f"packaging and sync toolchain; it carries no curriculum meaning.\n\n"
                f"Question {j + 1}: What is the core idea of this section?\n"
                f"Answer {j + 1}: A demonstration answer for tooling validation only."
            )
            chunks.append({
                "text": text,
                "collection": coll,
                "metadata": {"subject": subject, "grade": str(grade),
                             "source": "demo", "demo": True},
            })
        build_package(out_dir, pid, chunks, max_package_bytes=max_package_bytes)
        logger.info("Built demo package %s (%d chunks)", pid, len(chunks))
    return count


def write_catalog(out_dir: Path) -> None:
    """Aggregate all packages into a single catalog.json."""
    packages_dir = out_dir / "packages"
    entries: Dict[str, Any] = {}
    if packages_dir.exists():
        for mp in packages_dir.glob("*/manifest.json"):
            try:
                manifest = read_json(mp)
                rel = mp.relative_to(out_dir).as_posix()
                entries[manifest["package_id"]] = {
                    "version": manifest["version"],
                    "total_chunks": manifest["total_chunks"],
                    "total_size_bytes": manifest["total_size_bytes"],
                    "collections": manifest["collections"],
                    "manifest": rel,
                }
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Skipping invalid manifest %s: %s", mp, exc)
    catalog = {
        "format_version": FORMAT_VERSION,
        "generated_at": os.urandom(8).hex(),
        "package_count": len(entries),
        "packages": entries,
    }
    write_json(out_dir / CATALOG_NAME, catalog)
    logger.info("Catalog written: %d packages", len(entries))


def validate_dist(out_dir: Path) -> bool:
    """Verify every manifest's chunk files exist and match their hash."""
    ok = True
    catalog = read_json(out_dir / CATALOG_NAME)
    packages = catalog.get("packages", {})
    for pid, info in packages.items():
        pkg_dir = package_dir(out_dir, pid)
        manifest = read_json(manifest_path(pkg_dir))
        for chunk in manifest["chunks"]:
            cf = chunk_file(pkg_dir, chunk["hash"])
            if not cf.exists():
                logger.error("Missing chunk %s in %s", chunk["hash"], pid)
                ok = False
                continue
            # Chunks are stored compressed; hash is over the uncompressed bytes.
            try:
                payload = cf.read_bytes()
                raw = decompress_bytes(payload)
            except Exception:
                raw = cf.read_bytes()  # tolerate legacy uncompressed chunks
            if hash_bytes(raw) != chunk["hash"]:
                logger.error("Hash mismatch for chunk %s in %s", chunk["hash"], pid)
                ok = False
    logger.info("Validation %s (%d packages)", "OK" if ok else "FAILED", len(packages))
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Mati content package builder")
    parser.add_argument("--src", type=Path, help="Directory of content files to package")
    parser.add_argument("--out", type=Path, default=Path("dist"), help="Output directory")
    parser.add_argument("--demo", type=int, default=0,
                        help="Generate N synthetic demo packages (testing only)")
    parser.add_argument("--max-package-mb", type=float, default=50.0,
                        help="Maximum package size in MB (default 50)")
    parser.add_argument("--validate", action="store_true", help="Validate an existing dist tree")
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(args.max_package_mb * 1024 * 1024)

    if args.validate:
        return 0 if validate_dist(out_dir) else 1

    if args.demo and args.demo > 0:
        n = build_demo(out_dir, args.demo, max_bytes)
    elif args.src and args.src.exists():
        n = build_from_sources(args.src, out_dir, max_bytes)
    else:
        parser.error("Provide --src <dir> or --demo <N>")
        return 2

    write_catalog(out_dir)

    # Verify each package is within the size budget.
    over = 0
    for pkg_dir_ in (out_dir / "packages").glob("*"):
        manifest = read_json(manifest_path(pkg_dir_))
        if manifest["total_size_bytes"] > max_bytes:
            over += 1
            logger.warning("Package %s exceeds budget (%d bytes)",
                           manifest["package_id"], manifest["total_size_bytes"])
    logger.info("Built %d packages; %d over %d MB budget",
                n, over, int(args.max_package_mb))
    return 0


if __name__ == "__main__":
    sys.exit(main())
