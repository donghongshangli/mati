#!/usr/bin/env python3
"""
Shared helpers for Mati resource packaging & sync tools.

This module intentionally uses ONLY the Python standard library so that the
packaging / sync toolchain can run on the same low-resource machines as the
rest of Mati, without pulling in heavy third-party dependencies.
"""

import hashlib
import json
import re
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Format / layout constants
# ---------------------------------------------------------------------------
FORMAT_VERSION = 2
HASH_ALGO = "sha256"
CHUNK_EXT = ".chunk"
MANIFEST_NAME = "manifest.json"
CATALOG_NAME = "catalog.json"

# Network transfer uses 64 KB logical blocks. A "chunk" in Mati terms is a
# single content unit (a paragraph/QA block); the sync client downloads each
# chunk with HTTP Range requests in these block-sized pieces so a dropped
# connection can resume from the last completed block.
NETWORK_BLOCK_SIZE = 64 * 1024

# Default maximum size (bytes) of a single content chunk before hard-splitting.
MAX_CHUNK_BYTES = 256 * 1024

# Local sync state directory (created under the destination directory).
STATE_DIR = ".mati_sync"

PACKAGES_DIR = "packages"


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------
def hash_bytes(data: bytes) -> str:
    """Return the content-address hash of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def hash_text(text: str) -> str:
    """Return the content-address hash of UTF-8 encoded text."""
    return hash_bytes(text.encode("utf-8"))


def hash_file(path: Path) -> str:
    """Return the content-address hash of a file's contents."""
    return hash_bytes(path.read_bytes())


# ---------------------------------------------------------------------------
# Compression (real, dependency-free: zlib from the standard library)
# ---------------------------------------------------------------------------
# Content is hashed on the UNCOMPRESSED bytes (so identical content dedupes
# regardless of compression), but stored/transferred as compressed payload.
COMPRESSION_ALGO = "zlib"


def compress_bytes(data: bytes) -> bytes:
    """Compress raw bytes with zlib (level 6, good ratio/speed balance)."""
    return zlib.compress(data, level=6)


def decompress_bytes(data: bytes) -> bytes:
    """Decompress bytes produced by compress_bytes."""
    return zlib.decompress(data)


def verify_chunk_bytes(data: bytes, chunk_hash: str) -> bool:
    """Check that decompressing `data` yields bytes matching `chunk_hash`."""
    try:
        return hash_bytes(decompress_bytes(data)) == chunk_hash
    except Exception:
        return False


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# On-disk package layout
# ---------------------------------------------------------------------------
def chunks_dir(pkg_dir: Path) -> Path:
    return pkg_dir / "chunks"


def chunk_file(pkg_dir: Path, chunk_hash: str) -> Path:
    return chunks_dir(pkg_dir) / f"{chunk_hash}{CHUNK_EXT}"


def manifest_path(pkg_dir: Path) -> Path:
    return pkg_dir / MANIFEST_NAME


def catalog_path(out_dir: Path) -> Path:
    return out_dir / CATALOG_NAME


def package_dir(out_dir: Path, package_id: str) -> Path:
    return out_dir / PACKAGES_DIR / package_id


# ---------------------------------------------------------------------------
# Text chunking (lightweight, dependency-free)
# ---------------------------------------------------------------------------
def simple_chunk(text: str, max_bytes: int = MAX_CHUNK_BYTES) -> List[str]:
    """
    Split text into chunks no larger than max_bytes (measured in UTF-8 bytes).

    Prefers paragraph boundaries, then sentence boundaries, then hard
    character cuts. This is intentionally independent of Mati's
    EnhancedChunker so packaging can run without sentence-transformers.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text.encode("utf-8")) <= max_bytes:
        return [text]

    parts: List[str] = []
    current: List[str] = []
    current_len = 0

    paragraphs = re.split(r"\n\s*\n", text)
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        pbytes = len(para.encode("utf-8"))
        if current_len + pbytes + 1 <= max_bytes:
            current.append(para)
            current_len += pbytes + 1
            continue

        if current:
            parts.append("\n\n".join(current))
            current, current_len = [], 0

        if pbytes > max_bytes:
            # Hard-split an oversized paragraph on sentence boundaries.
            sentences = re.split(r"(?<=[.!?。！？])\s+", para)
            buf = ""
            for sent in sentences:
                candidate = (buf + " " + sent).strip()
                if len(candidate.encode("utf-8")) <= max_bytes:
                    buf = candidate
                else:
                    if buf:
                        parts.append(buf)
                    # Final fallback: hard character cut.
                    while len(sent.encode("utf-8")) > max_bytes:
                        hard = sent.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
                        parts.append(hard)
                        sent = sent[len(hard):]
                    buf = sent
            if buf:
                parts.append(buf)
        else:
            current = [para]
            current_len = pbytes

    if current:
        parts.append("\n\n".join(current))

    return [p for p in parts if p.strip()]
