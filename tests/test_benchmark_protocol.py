"""Focused protocol tests (no full hardware/simulator runs)."""

from __future__ import annotations

import importlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from breakpoint_eval.benchmark_protocol import (
    assert_lock_matches,
    build_full_axis_plans,
    build_smoke_plan,
    condition_is_shared_baseline,
    condition_report_problem,
    freeze_protocol_lock,
    host_settings_from_config,
    load_benchmark_config,
    resolve_h264_support,
    resolve_profile_by_identity,
)
from breakpoint_eval.experiment_plan import (
    ExperimentAxis,
    ExperimentPlan,
    build_seed_pairs,
    h264_conditions,
)
from breakpoint_eval.latency_trace import (
    EndOfTraceBehavior,
    LatencyTraceReplayer,
    TraceExhaustedError,
    contiguous_window_starts,
    extract_contiguous_window,
)
from breakpoint_eval.source_revision import read_source_revision

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "act-transfer.json"


def test_config_loads() -> None:
    cfg = load_benchmark_config(CONFIG_PATH)
    assert cfg["protocol_name"] == "act_deployment_stress"
    assert cfg["protocol_version"] == "0.1"
    assert cfg["statistics"]["breakpoint_ratio"] == 0.8
    assert cfg["statistics"]["baseline_success_floor"] == 0.5
    assert cfg["host"]["compute_device"] == "cpu"
    assert cfg["trace_window"]["length"] == 32
    assert cfg["full"]["jpeg_qualities"][0] == "bypass"
    assert 100 not in cfg["full"]["jpeg_qualities"]
    assert cfg["h264"] == {
        "gop": 30,
        "preset": "veryfast",
        "latency_mode": "quality_only",
        "frame_rate_hz": 50,
    }
    assert cfg["smoke"]["h264_bitrates_kbps"] == [100]
    assert cfg["full"]["h264_bitrates_kbps"][0] == "bypass"


def test_load_config_refuses_h264_deployment_mode(tmp_path: Path) -> None:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["h264"]["latency_mode"] = "deployment"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="quality_only"):
        load_benchmark_config(path)


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


def _fake_h264_probe(*, available: bool):
    from breakpoint_eval.h264_ffmpeg import FFmpegProbe

    return FFmpegProbe(
        available=available,
        ffmpeg_path="/usr/bin/ffmpeg" if available else None,
        version_line="ffmpeg version test",
        version="test",
        has_libx264=available,
        has_h264_decoder=available,
        pyav_available=available,
        pyav_version="12.0.0" if available else None,
        libav_versions={"libavcodec": (60, 0, 0)} if available else {},
        validated_flags={},
        unsupported_reason=None if available else "libx264 missing in test double",
    )


def test_smoke_and_full_isolation_and_variant_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_benchmark_config(CONFIG_PATH)
    # Minimal fake artifacts object via building conditions without resolve.
    from breakpoint_eval import h264_ffmpeg
    from breakpoint_eval.benchmark_protocol import ResolvedArtifacts

    monkeypatch.setattr(h264_ffmpeg, "probe_ffmpeg_h264", lambda: _fake_h264_probe(available=True))

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
        int4_path=tmp_path / "act_chunk_int4_matmul_nbits.onnx",
        int4_sha256="c" * 64,
        int4_status="ok",
        int4_detail=None,
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
    assert "h264_100kbps" in ids
    assert "h264_bypass" not in ids
    assert "precision_onnx_fp32" in ids
    assert "precision_int8" in ids
    assert "precision_int4_weight_only" in ids
    assert any("INT4 status at plan build: ok" in n for n in smoke.notes)
    assert any("H.264 status at plan build: ok" in n for n in smoke.notes)

    cfg_full = json.loads(json.dumps(cfg))
    cfg_full["full"]["device_profiles"] = []
    plans = build_full_axis_plans(cfg_full, fake)
    axes = [p.axis for p in plans]
    assert axes == [
        ExperimentAxis.LATENCY_MS.value,
        ExperimentAxis.RESOLUTION.value,
        ExperimentAxis.JPEG.value,
        ExperimentAxis.H264.value,
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
    h264 = plans[3]
    assert h264.plan_id == "v0.1_full_h264"
    assert h264.conditions[0].condition_id == "h264_bypass"
    assert condition_is_shared_baseline(h264.conditions[0])
    assert [c.condition_id for c in h264.conditions] == [
        "h264_bypass",
        "h264_2000kbps",
        "h264_1000kbps",
        "h264_500kbps",
        "h264_250kbps",
        "h264_100kbps",
    ]
    assert condition_is_shared_baseline(lat.conditions[0])
    assert condition_is_shared_baseline(res.conditions[0])
    assert condition_is_shared_baseline(jpeg.conditions[0])
    prec = plans[4]
    assert [c.condition_id for c in prec.conditions] == [
        "precision_onnx_fp32",
        "precision_int8",
        "precision_int4_weight_only",
    ]
    assert any("INT4 status at plan build: ok" in n for n in prec.notes)

    # INT8/INT4 not runnable → omitted from precision conditions, status recorded elsewhere.
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
        int4_status="unsupported",
        int4_detail="MatMulNBits unavailable",
    )
    smoke_no_int8 = build_smoke_plan(cfg_smoke, fake_bad)
    smoke_ids = [c.condition_id for c in smoke_no_int8.conditions]
    assert "precision_int8" not in smoke_ids
    assert "precision_int4_weight_only" not in smoke_ids
    assert "h264_100kbps" in smoke_ids
    full_no = build_full_axis_plans(cfg_full, fake_bad)
    prec_no_ids = [c.condition_id for c in full_no[4].conditions]
    assert prec_no_ids == ["precision_onnx_fp32"]
    assert any("INT4 status at plan build: unsupported" in n for n in full_no[4].notes)

    monkeypatch.setattr(h264_ffmpeg, "probe_ffmpeg_h264", lambda: _fake_h264_probe(available=False))
    smoke_no_h264 = build_smoke_plan(cfg_smoke, fake)
    assert "h264_100kbps" not in [c.condition_id for c in smoke_no_h264.conditions]
    assert any("H.264 status at plan build: unsupported" in n for n in smoke_no_h264.notes)
    full_no_h264 = build_full_axis_plans(cfg_full, fake)
    assert [p.axis for p in full_no_h264] == [
        ExperimentAxis.LATENCY_MS.value,
        ExperimentAxis.RESOLUTION.value,
        ExperimentAxis.JPEG.value,
        ExperimentAxis.PRECISION.value,
    ]
    assert full_no_h264[2].notes == ["Isolated JPEG axis; bypass ≠ quality 100."]
    assert not any("H.264 status" in n for n in full_no_h264[2].notes)


def test_freeze_protocol_lock_includes_int4_fields(tmp_path: Path) -> None:
    cfg = load_benchmark_config(CONFIG_PATH)
    host = host_settings_from_config(cfg)
    from breakpoint_eval.benchmark_protocol import ResolvedArtifacts

    arts = ResolvedArtifacts(
        export_dir=tmp_path,
        fp32_path=tmp_path / "a.onnx",
        fp32_sha256="aaa",
        fp32_recipe_id="r",
        int8_path=None,
        int8_sha256=None,
        int8_status="missing",
        int8_detail=None,
        profiles={},
        profile_identities={},
        int4_path=tmp_path / "int4.onnx",
        int4_sha256="d" * 64,
        int4_status="ok",
        int4_detail=None,
    )
    lock = freeze_protocol_lock(
        config=cfg,
        artifacts=arts,
        source_revision={"kind": "git", "revision": "abc"},
        host=host,
        phase="smoke",
    )
    art = lock["artifacts"]
    assert art["int4_path"] == str(tmp_path / "int4.onnx")
    assert art["int4_sha256"] == "d" * 64
    assert art["int4_status"] == "ok"
    assert art["int4_detail"] is None


def test_resume_refuses_int4_sha256_change() -> None:
    cfg = load_benchmark_config(CONFIG_PATH)
    host = host_settings_from_config(cfg)
    from breakpoint_eval.benchmark_protocol import ResolvedArtifacts

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
        int4_sha256="e" * 64,
        int4_status="ok",
    )
    lock = freeze_protocol_lock(
        config=cfg,
        artifacts=arts,
        source_revision={"kind": "git", "revision": "abc"},
        host=host,
        phase="smoke",
    )
    changed = json.loads(json.dumps(lock))
    changed["artifacts"]["int4_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="resume refused"):
        assert_lock_matches(lock, changed)


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
    cfg = load_benchmark_config(CONFIG_PATH)
    host = host_settings_from_config(cfg)
    from breakpoint_eval.benchmark_protocol import ResolvedArtifacts

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


def test_benchmark_profile_module_avoids_heavy_imports() -> None:
    # Ensure importing benchmark_profile does not pull torch/sim.
    banned = {"torch", "mujoco", "dm_control", "lerobot", "gymnasium"}
    before = set(sys.modules) & banned
    mod = importlib.import_module("breakpoint_eval.benchmark_profile")
    after = set(sys.modules) & banned
    assert after == before
    assert hasattr(mod, "resolve_validated_fp32_bundle")
    assert hasattr(mod, "run_benchmark_profile")


def test_resolve_h264_support_ok_and_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    from breakpoint_eval import h264_ffmpeg

    cfg = load_benchmark_config(CONFIG_PATH)
    monkeypatch.setattr(h264_ffmpeg, "probe_ffmpeg_h264", lambda: _fake_h264_probe(available=True))
    ok = resolve_h264_support(cfg)
    assert ok["status"] == "ok"
    assert ok["detail"] is None
    assert ok["gop"] == 30
    assert ok["preset"] == "veryfast"
    assert ok["latency_mode"] == "quality_only"
    assert ok["frame_rate_hz"] == 50.0
    assert ok["tune"] == "zerolatency"
    assert ok["pixel_format"] == "yuv420p"
    assert ok["max_b_frames"] == 0
    assert ok["rc_lookahead"] == 0
    assert ok["pyav_version"] == "12.0.0"
    assert ok["libav_versions"]["libavcodec"] == (60, 0, 0)
    assert ok["ffmpeg_version_line"] == "ffmpeg version test"
    assert ok["measurements_fabricated"] is False

    monkeypatch.setattr(h264_ffmpeg, "probe_ffmpeg_h264", lambda: _fake_h264_probe(available=False))
    bad = resolve_h264_support(cfg)
    assert bad["status"] == "unsupported"
    assert "libx264 missing" in (bad["detail"] or "")


def test_freeze_lock_stores_h264_and_resume_refuses_libav_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from breakpoint_eval import h264_ffmpeg
    from breakpoint_eval.benchmark_protocol import ResolvedArtifacts

    cfg = load_benchmark_config(CONFIG_PATH)
    host = host_settings_from_config(cfg)
    monkeypatch.setattr(h264_ffmpeg, "probe_ffmpeg_h264", lambda: _fake_h264_probe(available=True))
    support = resolve_h264_support(cfg)
    arts = ResolvedArtifacts(
        export_dir=tmp_path,
        fp32_path=tmp_path / "a.onnx",
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
        h264_support=support,
    )
    assert lock["h264"]["status"] == "ok"
    assert lock["h264"]["libav_versions"]["libavcodec"] == (60, 0, 0)
    changed = json.loads(json.dumps(lock))
    changed["h264"]["libav_versions"] = {"libavcodec": (61, 0, 0)}
    with pytest.raises(ValueError, match="resume refused"):
        assert_lock_matches(lock, changed)


def _h264_plan() -> ExperimentPlan:
    return ExperimentPlan(
        plan_id="h264_fixture",
        candidate="act_aloha_transfer",
        phase="exploratory",
        n_episodes=1,
        axis=ExperimentAxis.H264.value,
        seed_pairs=build_seed_pairs(1, env_seed_start=5000, policy_seed_offset=10000),
        conditions=h264_conditions((500,), frame_rate_hz=50.0, gop=30, preset="veryfast"),
        baseline_condition_id=None,
    )


def _good_h264_camera() -> dict:
    return {
        "n_frames_in": 10,
        "n_frames_decoded": 10,
        "n_startup_holds": 0,
        "n_last_frame_holds": 0,
        "first_frame_keyframe": True,
        "encoded_bytes_total": 1200,
        "n_keyframes": 1,
    }


def _h264_report(plan: ExperimentPlan, *, camera_overrides: dict | None = None) -> dict:
    cam = _good_h264_camera()
    if camera_overrides:
        cam.update(camera_overrides)
    pair = plan.seed_pairs[0]
    cond = plan.conditions[0]
    return {
        "provenance": {
            "candidate": plan.candidate,
            "env_created": True,
            "env_seeds": [pair.env_seed],
            "policy_rng_seeds": [pair.policy_rng_seed],
            "latency_mode": "constant",
            "onnx_runtime": {"variant_label": "onnx_fp32"},
        },
        "summary": {"n_episodes_total": 1, "n_runtime_error": 0},
        "episodes": [
            {
                "episode_index": 0,
                "seed": pair.env_seed,
                "policy_rng_seed": pair.policy_rng_seed,
                "outcome": "success",
                "success": True,
                "runtime_error": None,
                "n_control_steps": 10,
                "latency_mode": "constant",
                "latency_s": 0.0,
                "image_mode": "h264",
                "h264_summary": {
                    "latency_mode": "quality_only",
                    "requested": {
                        "bitrate_kbps": cond.params["h264_bitrate_kbps"],
                        "gop": cond.params["h264_gop"],
                        "preset": cond.params["h264_preset"],
                        "frame_rate_hz": cond.params["frame_rate_hz"],
                    },
                    "cameras": {"top": cam},
                },
            }
        ],
    }


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"latency_mode": "deployment"}, "latency_mode is not quality_only"),
        ({"requested_bitrate": 999.0}, "requested bitrate differs"),
        ({"requested_gop": 15}, "requested gop differs"),
        ({"requested_preset": "slow"}, "requested preset differs"),
        ({"requested_fps": 25.0}, "requested frame_rate_hz differs"),
        ({"n_frames_in": 9}, "n_frames_in"),
        ({"n_frames_decoded": 9}, "n_frames_decoded"),
        ({"n_startup_holds": 1}, "n_startup_holds"),
        ({"n_last_frame_holds": 1}, "n_last_frame_holds"),
        ({"first_frame_keyframe": False}, "first_frame_keyframe"),
        ({"encoded_bytes_total": 0}, "encoded_bytes_total"),
    ],
)
def test_condition_report_problem_h264_summary_fields(
    tmp_path: Path, override: dict, match: str
) -> None:
    plan = _h264_plan()
    report = _h264_report(plan)
    ep = report["episodes"][0]
    summary = ep["h264_summary"]
    if "latency_mode" in override:
        summary["latency_mode"] = override["latency_mode"]
    if "requested_bitrate" in override:
        summary["requested"]["bitrate_kbps"] = override["requested_bitrate"]
    if "requested_gop" in override:
        summary["requested"]["gop"] = override["requested_gop"]
    if "requested_preset" in override:
        summary["requested"]["preset"] = override["requested_preset"]
    if "requested_fps" in override:
        summary["requested"]["frame_rate_hz"] = override["requested_fps"]
    for key in (
        "n_frames_in",
        "n_frames_decoded",
        "n_startup_holds",
        "n_last_frame_holds",
        "first_frame_keyframe",
        "encoded_bytes_total",
    ):
        if key in override:
            summary["cameras"]["top"][key] = override[key]
    path = tmp_path / "run_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    problem = condition_report_problem(path, plan, plan.conditions[0])
    assert problem is not None
    assert match in problem


def test_condition_report_problem_h264_ok(tmp_path: Path) -> None:
    plan = _h264_plan()
    path = tmp_path / "run_report.json"
    path.write_text(json.dumps(_h264_report(plan)), encoding="utf-8")
    assert condition_report_problem(path, plan, plan.conditions[0]) is None


def test_scripts_help_and_dry_run() -> None:
    py = str(ROOT / ".venv" / "bin" / "python")
    if not Path(py).is_file():
        py = sys.executable
    run_benchmark = ROOT / "scripts" / "run_benchmark.py"
    profile_device = ROOT / "scripts" / "profile_device.py"
    help_run = subprocess.run(
        [py, str(run_benchmark), "--help"],
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
        [py, str(profile_device), "--help"],
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
    out = ROOT / "outputs" / "_benchmark_dry_run_should_not_exist"
    if out.exists():
        # Clean leftover from prior failed attempts.
        import shutil

        shutil.rmtree(out)
    dry = subprocess.run(
        [
            py,
            str(run_benchmark),
            "--phase",
            "smoke",
            "--out",
            str(out),
            "--dry-run",
            "--export-dir",
            str(ROOT / "artifacts" / "export" / "act-transfer-missing"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert dry.returncode != 0
    assert not out.exists()
