"""Focused v0.1 protocol tests (no full hardware/simulator runs)."""

from __future__ import annotations

import importlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from breakpoint_eval.experiment_plan import ExperimentAxis
from breakpoint_eval.latency_trace import (
    EndOfTraceBehavior,
    LatencyTraceReplayer,
    TraceExhaustedError,
    contiguous_window_starts,
    extract_contiguous_window,
)
from breakpoint_eval.source_revision import read_source_revision
from breakpoint_eval.v1_protocol import (
    assert_lock_matches,
    build_full_axis_plans,
    build_smoke_plan,
    condition_is_shared_baseline,
    freeze_protocol_lock,
    host_settings_from_config,
    load_v1_config,
    resolve_profile_by_identity,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "v1.json"


def test_config_loads() -> None:
    cfg = load_v1_config(CONFIG_PATH)
    assert cfg["protocol"] == "v0.1"
    assert cfg["statistics"]["breakpoint_ratio"] == 0.8
    assert cfg["statistics"]["baseline_success_floor"] == 0.5
    assert cfg["host"]["compute_device"] == "cpu"
    assert cfg["trace_window"]["length"] == 32
    assert cfg["full"]["jpeg_qualities"][0] == "bypass"
    assert 100 not in cfg["full"]["jpeg_qualities"]


def test_contiguous_window_starts_even_spread() -> None:
    n_trace, window, n_ep = 100, 32, 5
    starts = contiguous_window_starts(n_trace, window_length=window, n_episodes=n_ep)
    assert starts[0] == 0
    assert starts[-1] == n_trace - window
    for i, s in enumerate(starts):
        expected = int(math.floor(i * (n_trace - window) / (n_ep - 1)))
        assert s == expected
        assert 0 <= s <= n_trace - window


def test_contiguous_window_bounds_and_consumed_indices() -> None:
    values = [float(i) * 0.01 for i in range(40)]
    starts = contiguous_window_starts(40, window_length=32, n_episodes=3)
    assert starts == [0, 4, 8]
    window = extract_contiguous_window(values, start=starts[1], length=32)
    assert window == values[4:36]
    replayer = LatencyTraceReplayer(
        values,
        end_of_trace=EndOfTraceBehavior.FAIL,
        window_start=starts[1],
        window_length=32,
    )
    got = [replayer.next_latency_s() for _ in range(4)]
    assert got == values[4:8]
    prov = replayer.provenance()
    assert prov["sampling_design"] == "contiguous_window"
    assert prov["window_start"] == 4
    assert prov["window_end_exclusive"] == 36
    assert prov["consumed_trace_indices"] == [4, 5, 6, 7]
    assert prov["consumed_delays_s"] == values[4:8]
    assert prov["used_sample_stats"]["count"] == 4
    assert prov["full_profile_stats"]["count"] == 40
    assert prov["block_resample_interchangeable_with_contiguous_window"] is False
    # Exhaustion fails (does not silently wrap).
    for _ in range(28):
        replayer.next_latency_s()
    with pytest.raises(TraceExhaustedError):
        replayer.next_latency_s()


def test_window_rejects_block_resample_mix() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        LatencyTraceReplayer(
            [0.1] * 40,
            window_start=0,
            window_length=32,
            block_length=8,
            block_resample_seed=1,
        )


def test_smoke_and_full_isolation_and_variant_selection(tmp_path: Path) -> None:
    cfg = load_v1_config(CONFIG_PATH)
    # Minimal fake artifacts object via building conditions without resolve.
    from breakpoint_eval.v1_protocol import ResolvedArtifacts

    fake = ResolvedArtifacts(
        export_dir=tmp_path,
        fp32_path=tmp_path / "act_chunk_fp32.onnx",
        fp32_sha256="a" * 64,
        fp32_recipe_id="torch_onnx_export_fp32_opset17",
        int8_path=tmp_path / "act_chunk_int8_qdq.onnx",
        int8_sha256="b" * 64,
        int8_status="ok",
        int8_detail=None,
        profiles={
            "laptop-m3": tmp_path / "laptop.json",
            "rpi5": tmp_path / "rpi.json",
        },
        profile_identities={},
    )
    # Device-profile conditions need loadable profiles — stub via monkeypatch path.
    # Build plans that do not touch device profiles when profiles are unloadable:
    # use empty device_profiles for this unit test of other axes.
    cfg_smoke = json.loads(json.dumps(cfg))
    cfg_smoke["smoke"]["device_profiles"] = []
    smoke = build_smoke_plan(cfg_smoke, fake)
    assert smoke.phase == "smoke"
    assert [s.env_seed for s in smoke.seed_pairs] == [3000, 3001, 3002, 3003, 3004]
    assert [s.policy_rng_seed for s in smoke.seed_pairs] == [13000, 13001, 13002, 13003, 13004]
    ids = [c.condition_id for c in smoke.conditions]
    assert "latency_ms_0" in ids
    assert "latency_ms_640" in ids
    assert any("60x80" in c or c.endswith("60x80") for c in ids)
    assert "jpeg_q10" in ids
    assert "precision_onnx_fp32" in ids
    assert "precision_int8" in ids

    cfg_full = json.loads(json.dumps(cfg))
    cfg_full["full"]["device_profiles"] = []
    plans = build_full_axis_plans(cfg_full, fake)
    axes = [p.axis for p in plans]
    assert axes == [
        ExperimentAxis.LATENCY_MS.value,
        ExperimentAxis.RESOLUTION.value,
        ExperimentAxis.JPEG.value,
        ExperimentAxis.PRECISION.value,
    ]
    lat = plans[0]
    assert [c.params["latency_ms"] for c in lat.conditions] == [0, 40, 80, 160, 320, 640]
    res = plans[1]
    assert [c.params["capture_hw"] for c in res.conditions] == [
        [480, 640],
        [240, 320],
        [120, 160],
        [60, 80],
    ]
    jpeg = plans[2]
    assert jpeg.conditions[0].condition_id == "jpeg_bypass"
    assert all(c.params.get("jpeg_quality") != 100 for c in jpeg.conditions)
    assert condition_is_shared_baseline(lat.conditions[0])
    assert condition_is_shared_baseline(res.conditions[0])
    assert condition_is_shared_baseline(jpeg.conditions[0])

    # INT8 not runnable → omitted from precision conditions, status recorded elsewhere.
    fake_bad = ResolvedArtifacts(
        export_dir=tmp_path,
        fp32_path=tmp_path / "act_chunk_fp32.onnx",
        fp32_sha256="a" * 64,
        fp32_recipe_id="torch_onnx_export_fp32_opset17",
        int8_path=None,
        int8_sha256=None,
        int8_status="unsupported",
        int8_detail="ORT QDQ unavailable",
        profiles={},
        profile_identities={},
    )
    smoke_no_int8 = build_smoke_plan(cfg_smoke, fake_bad)
    assert "precision_int8" not in [c.condition_id for c in smoke_no_int8.conditions]


def test_identity_profile_resolution_strict(tmp_path: Path) -> None:
    from breakpoint_eval.latency_trace import PROFILE_SCHEMA_VERSION, TIMED_BOUNDARY_ORT_RUN

    def write_profile(name: str, sha: str, device: str = "laptop-m3") -> Path:
        latency = [0.01] * 40
        payload = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "device_label": device,
            "candidate": "act_aloha_transfer",
            "timed_boundary": TIMED_BOUNDARY_ORT_RUN,
            "chunk_horizon": 100,
            "model": {
                "sha256": sha,
                "precision_recipe": "torch_onnx_export_fp32_opset17",
                "variant_label": "onnx_fp32",
            },
            "input_contract": {"shapes": {}, "corpus_sha256": "c"},
            "runtime": {
                "session_providers": ["CPUExecutionProvider"],
                "intra_op_num_threads": 4,
                "inter_op_num_threads": 1,
            },
            "output_validation": {"passed": True},
            "measurements": {
                "n_measured": len(latency),
                "inference_latency_s": latency,
                "summary": {
                    "count": len(latency),
                    "median_s": 0.01,
                    "p95_s": 0.01,
                    "p99_s": 0.01,
                },
            },
        }
        path = tmp_path / name
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return path

    write_profile("a.json", "sha_a")
    write_profile("b.json", "sha_b")
    with pytest.raises(FileNotFoundError, match="Ambiguous|ambiguous"):
        resolve_profile_by_identity(
            tmp_path,
            device_label="laptop-m3",
            candidate="act_aloha_transfer",
            variant_label="onnx_fp32",
        )
    matched = resolve_profile_by_identity(
        tmp_path,
        device_label="laptop-m3",
        candidate="act_aloha_transfer",
        variant_label="onnx_fp32",
        expected_model_sha256="sha_b",
    )
    assert matched.name == "b.json"


def test_resume_refuses_identity_change() -> None:
    cfg = load_v1_config(CONFIG_PATH)
    host = host_settings_from_config(cfg)
    from breakpoint_eval.v1_protocol import ResolvedArtifacts

    arts = ResolvedArtifacts(
        export_dir=Path("/tmp/export"),
        fp32_path=Path("/tmp/export/a.onnx"),
        fp32_sha256="aaa",
        fp32_recipe_id="r",
        int8_path=None,
        int8_sha256=None,
        int8_status="missing",
        int8_detail=None,
        profiles={},
        profile_identities={},
    )
    lock = freeze_protocol_lock(
        config=cfg,
        artifacts=arts,
        source_revision={"kind": "git", "revision": "abc"},
        host=host,
        phase="smoke",
    )
    changed = json.loads(json.dumps(lock))
    changed["artifacts"]["fp32_sha256"] = "bbb"
    with pytest.raises(ValueError, match="resume refused"):
        assert_lock_matches(lock, changed)


def test_source_revision_archive_context(tmp_path: Path) -> None:
    (tmp_path / "source_revision.txt").write_text("deadbeef\n", encoding="utf-8")
    info = read_source_revision(tmp_path)
    assert info["kind"] == "source_revision.txt"
    assert info["revision"] == "deadbeef"
    assert info["archive_context"] is True
    assert info["dirty"] is None


def test_v1_profile_module_avoids_heavy_imports() -> None:
    # Ensure importing v1_profile does not pull torch/sim.
    banned = {"torch", "mujoco", "dm_control", "lerobot", "gymnasium"}
    before = set(sys.modules) & banned
    mod = importlib.import_module("breakpoint_eval.v1_profile")
    after = set(sys.modules) & banned
    assert after == before
    assert hasattr(mod, "resolve_validated_fp32_bundle")
    assert hasattr(mod, "run_v1_device_profile")


def test_scripts_help_and_dry_run() -> None:
    py = str(ROOT / ".venv" / "bin" / "python")
    if not Path(py).is_file():
        py = sys.executable
    run_v1 = ROOT / "scripts" / "run_v1.py"
    profile_v1 = ROOT / "scripts" / "profile_v1.py"
    help_run = subprocess.run(
        [py, str(run_v1), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_run.returncode == 0
    assert "--phase" in help_run.stdout
    assert "--dry-run" in help_run.stdout
    assert "--resume" in help_run.stdout

    help_prof = subprocess.run(
        [py, str(profile_v1), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_prof.returncode == 0
    assert "--device-label" in help_prof.stdout
    assert "--export-dir" in help_prof.stdout
    assert "--n-warmup" in help_prof.stdout

    # Dry-run without real artifacts should fail validation (not create dirs).
    out = ROOT / "outputs" / "_v1_dry_run_should_not_exist"
    if out.exists():
        # Clean leftover from prior failed attempts.
        import shutil

        shutil.rmtree(out)
    dry = subprocess.run(
        [
            py,
            str(run_v1),
            "--phase",
            "smoke",
            "--out",
            str(out),
            "--dry-run",
            "--export-dir",
            str(ROOT / "artifacts" / "export" / "act-v1-missing"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert dry.returncode != 0
    assert not out.exists()
