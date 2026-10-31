"""Compact episode records produced by scripts/package_benchmark.py."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location(
        "package_benchmark", ROOT / "scripts" / "package_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode(**overrides):
    base = {
        "episode_index": 3,
        "seed": 5003,
        "policy_rng_seed": 15003,
        "outcome": "success",
        "success": True,
        "runtime_error": None,
        "n_control_steps": 284,
        "n_inference_calls": 3,
        "n_fresh_chunks": 3,
        "n_underflows": 0,
        "n_missed_deadlines": 0,
        "sum_reward": 249.0,
        "max_reward": 4.0,
        "control_period_s": 0.02,
        "latency_mode": "constant",
        "latency_s": 0.0,
        "image_mode": "native",
        "image_degrade_summary": {"capture_hw": [480, 640], "jpeg_quality": None},
        "latency_trace_provenance": None,
        "steps": [
            {"control_step_index": 282, "sim_time_s": 5.66, "is_success": False},
            {"control_step_index": 283, "sim_time_s": 5.68, "is_success": True},
        ],
    }
    base.update(overrides)
    return base


def test_compact_row_records_success_step_and_simulated_duration() -> None:
    module = load_script()
    row = module.compact_episode_row(
        axis="latency_ms",
        plan_id="p",
        condition_id="c",
        referenced=False,
        source="outputs/x/run_report.json",
        episode=episode(),
    )
    assert row["success_step_index"] == 283
    assert row["sim_time_s"] == 5.68 and row["final_step_sim_time_s"] == 5.68
    assert row["consumed_trace_indices"] is None
    assert row["source_report"] == "outputs/x/run_report.json"
    assert "steps" not in row


def test_compact_row_keeps_replay_window_and_consumed_delays() -> None:
    module = load_script()
    trace = {
        "window_start": 33,
        "window_length": 32,
        "consumed_trace_indices": [33, 34],
        "consumed_delays_s": [0.7, 0.8],
    }
    row = module.compact_episode_row(
        axis="device_profile",
        plan_id="p",
        condition_id="c",
        referenced=True,
        source="s",
        episode=episode(
            latency_trace_provenance=trace, outcome="time_limit_truncation", success=False
        ),
    )
    assert row["trace_window_start"] == 33
    assert row["consumed_delays_s"] == [0.7, 0.8]
    assert row["referenced_baseline"] is True


def test_package_protocol_copies_int4_unsupported(tmp_path: Path) -> None:
    module = load_script()
    full = tmp_path / "run"
    full.mkdir()
    (full / "benchmark_config.json").write_text("{}\n", encoding="utf-8")
    (full / "protocol_lock.json").write_text("{}\n", encoding="utf-8")
    (full / "run_records.json").write_text("[]\n", encoding="utf-8")
    (full / "plans").mkdir()
    payload = {
        "status": "unsupported",
        "detail": "missing graph",
        "measurements_fabricated": False,
        "path": None,
        "sha256": None,
    }
    (full / "int4_unsupported.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (full / "int8_unsupported.json").write_text(
        json.dumps({**payload, "detail": "int8 missing"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "pkg"
    module.package_protocol(full, out)
    assert (out / "protocol" / "int4_unsupported.json").is_file()
    assert (out / "protocol" / "int8_unsupported.json").is_file()
    assert json.loads((out / "protocol" / "int4_unsupported.json").read_text()) == payload


def test_package_validation_and_identities_include_int4_when_ok(tmp_path: Path) -> None:
    module = load_script()
    export = tmp_path / "export"
    export.mkdir()
    fp32 = export / "act_chunk_fp32.onnx"
    int4 = export / "act_chunk_int4_matmul_nbits.onnx"
    fp32.write_bytes(b"fp32")
    int4.write_bytes(b"int4")
    for name in (
        "export_validation.json",
        "export_manifest.json",
        "corpus_eval.json",
        "export_validation_int4.json",
    ):
        (export / name).write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
    (export / "corpus_eval.npz").write_bytes(b"npz")
    lock = {
        "artifacts": {
            "fp32_path": str(fp32),
            "fp32_sha256": _sha(fp32),
            "fp32_recipe_id": "r",
            "int8_path": None,
            "int8_sha256": None,
            "int8_status": "missing",
            "int8_detail": None,
            "int4_path": str(int4),
            "int4_sha256": _sha(int4),
            "int4_status": "ok",
            "int4_detail": None,
            "profile_identities": {},
        }
    }
    out = tmp_path / "pkg"
    large: list[dict] = []
    module.package_validation(export, lock, out, large)
    assert (out / "validation" / "export_validation_int4.json").is_file()
    assert any(e["path"].endswith("act_chunk_int4_matmul_nbits.onnx") for e in large)
    module.package_identities(lock, [], out)
    identities = json.loads((out / "identities" / "identities.json").read_text())
    assert identities["model_int4_weight_only"] == {
        "sha256": _sha(int4),
        "status": "ok",
        "detail": None,
    }


def test_package_identities_omit_int4_for_old_lock(tmp_path: Path) -> None:
    module = load_script()
    lock = {
        "artifacts": {
            "fp32_sha256": "a",
            "fp32_recipe_id": "r",
            "int8_sha256": None,
            "int8_status": "missing",
            "int8_detail": None,
            "profile_identities": {},
        }
    }
    out = tmp_path / "pkg"
    module.package_identities(lock, [], out)
    identities = json.loads((out / "identities" / "identities.json").read_text())
    assert "model_int4_weight_only" not in identities
