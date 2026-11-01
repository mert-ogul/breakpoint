"""Compact episode records produced by scripts/package_benchmark.py."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

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
    assert row["h264_encoded_bytes"] is None
    assert row["h264_frames"] is None


def test_compact_row_h264_fields_sum_cameras() -> None:
    from breakpoint_eval.benchmark_report_data import measured_h264_kbps

    module = load_script()
    row = module.compact_episode_row(
        axis="h264",
        plan_id="v0.1_full_h264",
        condition_id="h264_500kbps",
        referenced=False,
        source="s",
        episode=episode(
            image_mode="h264",
            n_control_steps=10,
            h264_summary={
                "requested": {"bitrate_kbps": 500.0, "frame_rate_hz": 50.0},
                "cameras": {
                    "top": {
                        "n_frames_in": 10,
                        "encoded_bytes_total": 1000,
                        "n_startup_holds": 0,
                        "n_last_frame_holds": 0,
                        "n_keyframes": 1,
                        "first_frame_keyframe": True,
                    },
                    "wrist": {
                        "n_frames_in": 10,
                        "encoded_bytes_total": 500,
                        "n_startup_holds": 0,
                        "n_last_frame_holds": 1,
                        "n_keyframes": 2,
                        "first_frame_keyframe": True,
                    },
                },
            },
        ),
    )
    assert row["h264_requested_kbps"] == 500.0
    assert row["h264_encoded_bytes"] == 1500
    assert row["h264_frames"] == 20
    assert row["h264_holds"] == 1
    assert row["h264_keyframes"] == 3
    assert row["h264_first_frame_keyframe"] is True
    assert row["h264_actual_kbps"] == pytest.approx(
        measured_h264_kbps(1500, 20, frame_rate_hz=50.0)
    )

    one_false = module.compact_episode_row(
        axis="h264",
        plan_id="v0.1_full_h264",
        condition_id="h264_500kbps",
        referenced=False,
        source="s",
        episode=episode(
            image_mode="h264",
            n_control_steps=10,
            h264_summary={
                "requested": {"bitrate_kbps": 500.0, "frame_rate_hz": 50.0},
                "cameras": {
                    "top": {
                        "n_frames_in": 10,
                        "encoded_bytes_total": 1000,
                        "n_startup_holds": 0,
                        "n_last_frame_holds": 0,
                        "n_keyframes": 1,
                        "first_frame_keyframe": True,
                    },
                    "wrist": {
                        "n_frames_in": 10,
                        "encoded_bytes_total": 500,
                        "n_startup_holds": 0,
                        "n_last_frame_holds": 0,
                        "n_keyframes": 1,
                        "first_frame_keyframe": False,
                    },
                },
            },
        ),
    )
    assert one_false["h264_first_frame_keyframe"] is False


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
    h264_payload = {**payload, "detail": "libx264 missing"}
    (full / "h264_unsupported.json").write_text(
        json.dumps(h264_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    out = tmp_path / "pkg"
    module.package_protocol(full, out)
    assert (out / "protocol" / "int4_unsupported.json").is_file()
    assert (out / "protocol" / "int8_unsupported.json").is_file()
    assert json.loads((out / "protocol" / "int4_unsupported.json").read_text()) == payload
    assert json.loads((out / "protocol" / "h264_unsupported.json").read_text()) == h264_payload


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


def _pytorch_episode(seed: int, policy_seed: int, *, latency_s: float = 0.0) -> dict:
    return {
        "episode_index": 0,
        "seed": seed,
        "policy_rng_seed": policy_seed,
        "outcome": "success",
        "success": True,
        "runtime_error": None,
        "n_control_steps": 8,
        "n_inference_calls": 1,
        "n_fresh_chunks": 1,
        "n_underflows": 0,
        "n_missed_deadlines": 0,
        "sum_reward": 1.0,
        "max_reward": 1.0,
        "control_period_s": 0.1,
        "latency_mode": "constant",
        "latency_s": latency_s,
        "image_mode": "native",
        "image_degrade_summary": {"capture_hw": [96, 96], "jpeg_quality": None},
        "latency_trace_provenance": None,
        "steps": [{"control_step_index": 7, "sim_time_s": 0.8, "is_success": True}],
    }


def _pytorch_report(seed: int = 6000, policy_seed: int = 16000, *, latency_s: float = 0.0) -> dict:
    return {
        "provenance": {
            "candidate": "diffusion_pusht",
            "env_created": True,
            "env_seeds": [seed],
            "policy_rng_seeds": [policy_seed],
            "latency_mode": "constant",
            "policy_backend": "pytorch",
            "onnx_runtime": None,
        },
        "summary": {"n_episodes_total": 1, "n_runtime_error": 0},
        "episodes": [_pytorch_episode(seed, policy_seed, latency_s=latency_s)],
    }


def test_package_pytorch_run_root_with_referenced_baseline(tmp_path: Path) -> None:
    """A synthetic Diffusion PushT run packages when the lock carries pytorch + pytorch_fp32."""
    from breakpoint_eval.benchmark_protocol import (
        BACKEND_ONNXRUNTIME,
        condition_incomplete_reason,
    )
    from breakpoint_eval.experiment_plan import (
        ConditionSpec,
        ExperimentAxis,
        ExperimentPlan,
        build_seed_pairs,
        latency_conditions,
        write_experiment_plan,
    )

    module = load_script()
    full = tmp_path / "full"
    export = tmp_path / "export"
    export.mkdir()
    checkpoint_files = {"model.safetensors": "a" * 64, "config.json": "b" * 64}
    (export / "export_manifest.json").write_text(
        json.dumps(
            {
                "candidate": "diffusion_pusht",
                "checkpoint_files": checkpoint_files,
                "control_period_s": 0.1,
                "hub_revision": "rev",
                "inference": {"num_inference_steps": 100},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    agreement = export / "upstream_agreement.json"
    agreement.write_text(
        json.dumps({"passed": True, "checkpoint_files": checkpoint_files}, indent=2) + "\n",
        encoding="utf-8",
    )
    timing = tmp_path / "timing.json"
    timing.write_text(
        json.dumps(
            {
                "device_label": "laptop-m3",
                "model": {"checkpoint_files": checkpoint_files},
                "measurements": {
                    "inference_latency_s": [0.5, 0.6],
                    "n_measured": 2,
                    "summary": {"count": 2, "median_s": 0.55, "p95_s": 0.6, "p99_s": 0.6},
                },
                "runtime": {
                    "device": "cpu",
                    "torch_threads": 4,
                    "torch_version": "2.6.0",
                    "timed_boundary": "pytorch_generate_actions_full_denoise",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "timing.npz").write_bytes(b"npz")

    seeds = build_seed_pairs(1, env_seed_start=6000, policy_seed_offset=10000)
    plan = ExperimentPlan(
        plan_id="v0.1_full_latency",
        candidate="diffusion_pusht",
        phase="exploratory",
        n_episodes=1,
        axis=ExperimentAxis.LATENCY_MS.value,
        seed_pairs=seeds,
        conditions=latency_conditions((0, 100)),
        baseline_condition_id="latency_ms_0",
    )
    ref_plan = ExperimentPlan(
        plan_id="v0.1_full_resolution",
        candidate="diffusion_pusht",
        phase="exploratory",
        n_episodes=1,
        axis=ExperimentAxis.RESOLUTION.value,
        seed_pairs=seeds,
        conditions=[
            ConditionSpec(
                condition_id="resolution_native",
                axis=ExperimentAxis.RESOLUTION.value,
                params={
                    "mode": "native",
                    "bypass": True,
                    "latency_s": 0.0,
                    "native_hw": [96, 96],
                    "capture_hw": [96, 96],
                },
                label="native",
            )
        ],
        baseline_condition_id="resolution_native",
    )
    plans_dir = full / "plans"
    plans_dir.mkdir(parents=True)
    write_experiment_plan(plan, plans_dir / f"{plan.plan_id}.json")
    write_experiment_plan(ref_plan, plans_dir / f"{ref_plan.plan_id}.json")

    shared = full / "shared_baselines" / "default_fp32_native_L0"
    shared.mkdir(parents=True)
    baseline = _pytorch_report()
    (shared / "run_report.json").write_text(json.dumps(baseline) + "\n", encoding="utf-8")
    (shared / "baseline_fingerprint.json").write_text(
        json.dumps({"condition_id": "latency_ms_0"}, indent=2) + "\n", encoding="utf-8"
    )

    axis = full / "axes" / plan.plan_id
    for condition_id, latency_s in (("latency_ms_0", 0.0), ("latency_ms_100", 0.1)):
        cond_dir = axis / "conditions" / condition_id
        cond_dir.mkdir(parents=True)
        (cond_dir / "run_report.json").write_text(
            json.dumps(_pytorch_report(latency_s=latency_s)) + "\n", encoding="utf-8"
        )
        (cond_dir / "provenance.json").write_text("{}\n", encoding="utf-8")
        (cond_dir / "planned_settings.json").write_text("{}\n", encoding="utf-8")
    (axis / "experiment_summary.json").write_text("{}\n", encoding="utf-8")

    ref_dir = full / "axes" / ref_plan.plan_id / "conditions" / "resolution_native"
    ref_dir.mkdir(parents=True)
    (ref_dir / "run_report.json").write_text(json.dumps(baseline) + "\n", encoding="utf-8")
    (ref_dir / "baseline_reference.json").write_text(
        json.dumps(
            {
                "status": "referenced_baseline",
                "reference_report": str(shared / "run_report.json"),
                "reference_dir": str(shared),
                "condition_id": "resolution_native",
                "by_explicit_reference_only": True,
                "measurements_fabricated": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (ref_dir / "provenance.json").write_text("{}\n", encoding="utf-8")
    (full / "axes" / ref_plan.plan_id / "experiment_summary.json").write_text(
        "{}\n", encoding="utf-8"
    )

    lock = {
        "schema_version": 1,
        "backend": "pytorch",
        "candidate": "diffusion_pusht",
        "baseline_reuse": {
            "enabled": True,
            "fingerprint": {
                "latency_s": 0.0,
                "image_mode": "native",
                "variant_label": "pytorch_fp32",
                "no_device_profile": True,
            },
        },
        "artifacts": {
            "export_dir": str(export),
            "checkpoint_files": checkpoint_files,
            "upstream_agreement_path": str(agreement),
            "upstream_agreement_sha256": _sha(agreement),
            "timing_profile_path": str(timing),
            "timing_profile_sha256": _sha(timing),
        },
    }
    (full / "protocol_lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (full / "benchmark_config.json").write_text("{}\n", encoding="utf-8")
    (full / "run_records.json").write_text(
        json.dumps(
            [
                {
                    "plan_id": plan.plan_id,
                    "records": [
                        {"condition_id": "latency_ms_0", "status": "complete"},
                        {"condition_id": "latency_ms_100", "status": "complete"},
                    ],
                },
                {
                    "plan_id": ref_plan.plan_id,
                    "records": [
                        {"condition_id": "resolution_native", "status": "referenced_baseline"},
                    ],
                },
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "pkg"
    n = module.build_package(full, out, lock, export, {})
    assert n == 3
    assert (out / "episodes" / "episodes.jsonl").is_file()
    assert (out / "validation" / "upstream_agreement.json").is_file()
    assert (out / "timing" / "laptop-m3.json").is_file()

    reason = condition_incomplete_reason(
        axis / "conditions" / "latency_ms_0",
        plan=plan,
        condition=plan.conditions[0],
        shared_baseline_dir=shared,
        backend=BACKEND_ONNXRUNTIME,
        variant_label="onnx_fp32",
    )
    assert reason is not None
    assert "onnx_fp32" in reason


def test_package_pytorch_validation_and_identities(tmp_path: Path) -> None:
    module = load_script()
    export = tmp_path / "export"
    export.mkdir()
    checkpoint_files = {"model.safetensors": "a" * 64, "config.json": "b" * 64}
    (export / "export_manifest.json").write_text(
        json.dumps(
            {
                "candidate": "diffusion_pusht",
                "checkpoint_files": checkpoint_files,
                "control_period_s": 0.1,
                "hub_revision": "rev",
                "inference": {"num_inference_steps": 100},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    agreement = tmp_path / "upstream_agreement.json"
    agreement.write_text(
        json.dumps({"passed": True, "checkpoint_files": checkpoint_files}, indent=2) + "\n",
        encoding="utf-8",
    )
    timing = tmp_path / "timing.json"
    timing.write_text(
        json.dumps(
            {
                "device_label": "laptop-m3",
                "model": {"checkpoint_files": checkpoint_files},
                "measurements": {
                    "inference_latency_s": [0.5, 0.6, 0.7],
                    "n_measured": 3,
                    "summary": {"count": 3, "median_s": 0.6, "p95_s": 0.7, "p99_s": 0.7},
                },
                "runtime": {
                    "device": "cpu",
                    "torch_threads": 4,
                    "torch_version": "2.6.0",
                    "timed_boundary": "pytorch_generate_actions_full_denoise",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "timing.npz").write_bytes(b"npz")
    lock = {
        "artifacts": {
            "checkpoint_files": checkpoint_files,
            "upstream_agreement_path": str(agreement),
            "upstream_agreement_sha256": _sha(agreement),
            "timing_profile_path": str(timing),
            "timing_profile_sha256": _sha(timing),
        }
    }
    out = tmp_path / "pkg"
    large: list[dict] = []
    module.package_pytorch_validation(export, lock, out, large)
    assert (out / "validation" / "export_manifest.json").is_file()
    assert (out / "validation" / "upstream_agreement.json").is_file()
    assert (out / "timing" / "laptop-m3.json").is_file()
    assert any(e["path"].endswith("timing.npz") for e in large)
    assert not (out / "validation" / "export_validation.json").exists()
    assert not (out / "profiles").exists()
    module.package_pytorch_identities(lock, [], out)
    identities = json.loads((out / "identities" / "identities.json").read_text())
    assert identities["checkpoint_files"] == checkpoint_files
    assert "model_fp32" not in identities
