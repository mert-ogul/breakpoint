"""Canonical artifact names, legacy readers, and protocol identity compatibility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from breakpoint_eval.artifact_names import (
    EXPORT_VALIDATION_INT8_NAME,
    EXPORT_VALIDATION_NAME,
    LEGACY_NAMES,
    PROTOCOL_LOCK_NAME,
    canonical_name,
    require_artifact,
    resolve_artifact,
)
from breakpoint_eval.benchmark_protocol import (
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    assert_lock_matches,
    load_benchmark_config,
    protocol_identity,
)
from breakpoint_eval.export_validation import find_export_validation_v2

ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_canonical_names_have_no_chronology() -> None:
    for name in LEGACY_NAMES:
        assert "v1" not in name and "v2" not in name
    assert canonical_name("v1_protocol_lock.json") == PROTOCOL_LOCK_NAME
    assert canonical_name("export_validation_v2_int8.json") == EXPORT_VALIDATION_INT8_NAME
    assert canonical_name("episodes.jsonl") == "episodes.jsonl"


def test_resolve_artifact_reads_either_name(tmp_path: Path) -> None:
    assert resolve_artifact(tmp_path, PROTOCOL_LOCK_NAME) is None
    with pytest.raises(FileNotFoundError, match="v1_protocol_lock.json"):
        require_artifact(tmp_path, PROTOCOL_LOCK_NAME)
    _write(tmp_path / "v1_protocol_lock.json", {"a": 1})
    assert resolve_artifact(tmp_path, PROTOCOL_LOCK_NAME).name == "v1_protocol_lock.json"
    _write(tmp_path / PROTOCOL_LOCK_NAME, {"a": 1})
    assert resolve_artifact(tmp_path, PROTOCOL_LOCK_NAME).name == PROTOCOL_LOCK_NAME


def test_resolve_artifact_fails_when_names_disagree(tmp_path: Path) -> None:
    _write(tmp_path / EXPORT_VALIDATION_NAME, {"passed": True})
    _write(tmp_path / "export_validation_v2.json", {"passed": False})
    with pytest.raises(ValueError, match="both exist"):
        resolve_artifact(tmp_path, EXPORT_VALIDATION_NAME)
    with pytest.raises(ValueError, match="both exist"):
        find_export_validation_v2(tmp_path / "act_chunk_fp32.onnx")


def test_find_export_validation_prefers_matching_variant(tmp_path: Path) -> None:
    _write(tmp_path / "export_validation_v2_int8.json", {})
    assert find_export_validation_v2(tmp_path / "m.onnx") is None
    found = find_export_validation_v2(tmp_path / "m.onnx", variant_label="int8")
    assert found.name == "export_validation_v2_int8.json"


def test_active_config_has_named_protocol() -> None:
    cfg = load_benchmark_config(ROOT / "configs" / "act-transfer.json")
    assert protocol_identity(cfg) == (PROTOCOL_NAME, PROTOCOL_VERSION)
    assert cfg["schema_version"] == 1


def test_legacy_protocol_label_maps_to_named_identity(tmp_path: Path) -> None:
    legacy = {"schema_version": 1, "protocol": "v0.1"}
    assert protocol_identity(legacy) == (PROTOCOL_NAME, PROTOCOL_VERSION)
    _write(tmp_path / "old.json", legacy)
    assert load_benchmark_config(tmp_path / "old.json")["protocol"] == "v0.1"
    fresh = {"protocol_name": PROTOCOL_NAME, "protocol_version": PROTOCOL_VERSION}
    assert_lock_matches(legacy, fresh)
    with pytest.raises(ValueError, match="protocol identity changed"):
        assert_lock_matches(legacy, {**fresh, "protocol_version": "0.2"})
    with pytest.raises(ValueError, match="unknown legacy protocol"):
        protocol_identity({"protocol": "v9"})
