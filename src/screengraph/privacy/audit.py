"""Zero-pixel persistence audit: disk magic-byte scan, SQLite blob bounds, RAM-wipe hash chain."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path

from screengraph.db.schema import MAX_VECTOR_BLOB_BYTES

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".mp4", ".mov",
                    ".mkv", ".avi", ".webm", ".raw", ".npy", ".npz"}
MAGIC = {
    "png": b"\x89PNG\r\n\x1a\n",
    "jpeg": b"\xff\xd8\xff",
    "gif": b"GIF8",
    "tiff_le": b"II*\x00",
    "tiff_be": b"MM\x00*",
    "npy": b"\x93NUMPY",
}  # BMP ('BM') is too short a signature to trust alone; it is covered by the extension check


def _sniff(head: bytes) -> str | None:
    for name, sig in MAGIC.items():
        if head.startswith(sig):
            return name
    if head[4:8] == b"ftyp":
        return "mp4/mov"
    if head[:4] == b"RIFF" and head[8:12] in (b"WEBP", b"AVI "):
        return "webp/avi"
    return None


def scan_directory(root: str | Path) -> dict:
    root = Path(root)
    files = hits_magic = hits_ext = 0
    total_bytes = 0
    offenders: list[str] = []
    digest = hashlib.sha256()
    if root.exists():
        for dirpath, _, names in os.walk(root):
            for name in sorted(names):
                p = Path(dirpath) / name
                try:
                    size = p.stat().st_size
                    with open(p, "rb") as fh:
                        head = fh.read(16)
                except OSError:
                    continue
                files += 1
                total_bytes += size
                digest.update(f"{p.relative_to(root)}:{size}".encode())
                kind = _sniff(head)
                if kind:
                    hits_magic += 1
                    offenders.append(f"{p.name} ({kind})")
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    hits_ext += 1
                    offenders.append(f"{p.name} (extension)")
    return {"root": str(root), "files_scanned": files, "bytes_scanned": total_bytes,
            "image_magic_hits": hits_magic, "image_extension_hits": hits_ext, "offenders": offenders[:20],
            "manifest_sha256": digest.hexdigest()}


def scan_sqlite_blobs(conn: sqlite3.Connection) -> dict:
    """Every BLOB column must be a bounded vector; scan blobs for image magic bytes too."""
    max_blob, suspects, n = 0, 0, 0
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '%fts%'")]
    for t in tables:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({t})")]
        for c in cols:
            for (v,) in conn.execute(f"SELECT {c} FROM {t} WHERE typeof({c}) = 'blob'"):
                n += 1
                max_blob = max(max_blob, len(v))
                if len(v) > MAX_VECTOR_BLOB_BYTES or _sniff(bytes(v[:16])):
                    suspects += 1
    return {"blobs_checked": n, "max_blob_bytes": max_blob, "blob_limit_bytes": MAX_VECTOR_BLOB_BYTES,
            "pixel_blob_suspects": suspects}
