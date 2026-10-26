"""One shared SHA-256 implementation for export, profiling and benchmark identity checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

from breakpoint_eval.file_digest import sha256_file


def test_sha256_file_matches_hashlib_across_chunk_boundary(tmp_path: Path) -> None:
    payload = b"breakpoint" * 200_000  # 2 MB, larger than one read chunk
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_of_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty"
    path.write_bytes(b"")
    assert sha256_file(path) == hashlib.sha256(b"").hexdigest()
