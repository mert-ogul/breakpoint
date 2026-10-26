"""Canonical artifact filenames and the readers that still accept legacy filenames.

Evidence saved before the rename (packaged data, older export directories) uses the legacy
names below. New outputs are always written under the canonical names. Legacy names are
read here and nowhere else.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BENCHMARK_CONFIG_NAME = "benchmark_config.json"
PROTOCOL_LOCK_NAME = "protocol_lock.json"
EXPORT_VALIDATION_NAME = "export_validation.json"
EXPORT_VALIDATION_INT8_NAME = "export_validation_int8.json"

LEGACY_NAMES = {
    BENCHMARK_CONFIG_NAME: "v1_config.json",
    PROTOCOL_LOCK_NAME: "v1_protocol_lock.json",
    EXPORT_VALIDATION_NAME: "export_validation_v2.json",
    EXPORT_VALIDATION_INT8_NAME: "export_validation_v2_int8.json",
}


def canonical_name(name: str) -> str:
    """Map a legacy filename to its canonical name; other names are returned unchanged."""
    for canonical, legacy in LEGACY_NAMES.items():
        if name == legacy:
            return canonical
    return name


def resolve_artifact(directory: Path, canonical: str) -> Path | None:
    """Locate a JSON artifact by canonical or legacy name.

    Raises ValueError when both files exist with different JSON content.
    """
    directory = Path(directory)
    current = directory / canonical
    legacy = directory / LEGACY_NAMES[canonical]
    if current.is_file() and legacy.is_file():
        if _read_json(current) != _read_json(legacy):
            raise ValueError(
                f"{current.name} and {legacy.name} both exist in {directory} and differ; "
                "remove the one that is out of date"
            )
        return current
    if current.is_file():
        return current
    if legacy.is_file():
        return legacy
    return None


def require_artifact(directory: Path, canonical: str) -> Path:
    path = resolve_artifact(directory, canonical)
    if path is None:
        raise FileNotFoundError(
            f"missing {canonical} (or legacy {LEGACY_NAMES[canonical]}) in {directory}"
        )
    return path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
