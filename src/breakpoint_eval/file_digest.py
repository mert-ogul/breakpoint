"""SHA-256 of a file's bytes; stdlib only so profiling and export code can share it."""

from __future__ import annotations

import hashlib
from pathlib import Path

_READ_CHUNK_BYTES = 1 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()
