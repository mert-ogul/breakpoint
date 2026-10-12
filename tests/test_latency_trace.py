"""Latency profile validation and ordered trace replay."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from breakpoint_eval.latency_scheduler import (
    LatencyConfig,
    LatencyMode,
    LatencyScheduler,
)
from breakpoint_eval.latency_trace import (
    PROFILE_SCHEMA_VERSION,
    TIMED_BOUNDARY_ORT_RUN,
    EndOfTraceBehavior,
    LatencyTraceReplayer,
    TraceExhaustedError,
    assert_profile_compatible,
    load_latency_profile,
    profiling_command_hint,
    require_unique_device_profile,
    summarize_latency_s,
    validate_latency_sequence,
)


def _valid_profile_payload(latency_s: list[float], **overrides: object) -> dict:
    stats = summarize_latency_s(latency_s)
    payload = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "device_label": "rpi5",
        "candidate": "act_aloha_transfer",
        "timed_boundary": TIMED_BOUNDARY_ORT_RUN,
        "chunk_horizon": 100,
        "model": {
            "sha256": "abc123",
            "precision_recipe": "torch_onnx_export_fp32_opset17",
            "variant_label": "onnx_fp32",
        },
        "input_contract": {
            "corpus_sha256": "corp456",
            "shapes": {
                "observation_images_top": [1, 3, 480, 640],
                "observation_state": [1, 14],
            },
        },
        "runtime": {
            "session_providers": ["CPUExecutionProvider"],
            "intra_op_num_threads": 4,
            "inter_op_num_threads": 1,
        },
        "output_validation": {"passed": True},
        "measurements": {
            "n_warmup": 50,
            "n_measured": len(latency_s),
            "inference_latency_s": latency_s,
            "summary": stats.to_dict(),
        },
    }
    payload.update(overrides)
    return payload


def test_reject_nan_negative_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_latency_sequence([])
    with pytest.raises(ValueError, match="NaN"):
        validate_latency_sequence([0.1, float("nan")])
    with pytest.raises(ValueError, match="negative"):
        validate_latency_sequence([0.1, -0.01])


def test_reject_unvalidated_or_mismatched_profile(tmp_path: Path) -> None:
    bad = _valid_profile_payload([0.01, 0.02])
    bad["output_validation"] = {"passed": False}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="output_validation"):
        load_latency_profile(path)

    mismatch = _valid_profile_payload([0.01, 0.02])
    mismatch["measurements"]["summary"]["median_s"] = 9.9
    path2 = tmp_path / "mismatch.json"
    path2.write_text(json.dumps(mismatch), encoding="utf-8")
    with pytest.raises(ValueError, match="summary"):
        load_latency_profile(path2)


def test_assert_profile_compatible_rejects_label_only_match(tmp_path: Path) -> None:
    path_payload = _valid_profile_payload([0.01, 0.02, 0.03])
    p = tmp_path / "p.json"
    p.write_text(json.dumps(path_payload), encoding="utf-8")
    prof = load_latency_profile(p)
    with pytest.raises(ValueError, match="model_sha256"):
        assert_profile_compatible(
            prof,
            candidate="act_aloha_transfer",
            model_sha256="other",
        )
    assert_profile_compatible(
        prof,
        candidate="act_aloha_transfer",
        model_sha256="abc123",
        precision_recipe="torch_onnx_export_fp32_opset17",
        chunk_horizon=100,
    )


def test_trace_exhaustion_default_and_cycle() -> None:
    seq = [0.01, 0.02, 0.03]
    failer = LatencyTraceReplayer(seq, end_of_trace=EndOfTraceBehavior.FAIL)
    assert [failer.next_latency_s() for _ in range(3)] == seq
    with pytest.raises(TraceExhaustedError):
        failer.next_latency_s()

    cycler = LatencyTraceReplayer(seq, end_of_trace=EndOfTraceBehavior.CYCLE)
    got = [cycler.next_latency_s() for _ in range(5)]
    assert got == [0.01, 0.02, 0.03, 0.01, 0.02]


def test_block_resample_requires_seed_and_is_reproducible() -> None:
    seq = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    with pytest.raises(ValueError, match="seed"):
        LatencyTraceReplayer(seq, block_length=2)
    a = LatencyTraceReplayer(seq, block_length=2, block_resample_seed=7)
    b = LatencyTraceReplayer(seq, block_length=2, block_resample_seed=7)
    assert [a.next_latency_s() for _ in range(6)] == [b.next_latency_s() for _ in range(6)]
    seq_a = [
        LatencyTraceReplayer(seq, block_length=2, block_resample_seed=7).next_latency_s()
        for _ in range(6)
    ]
    seq_c = [
        LatencyTraceReplayer(seq, block_length=2, block_resample_seed=8).next_latency_s()
        for _ in range(6)
    ]
    assert seq_a != seq_c


def test_temporal_ordering_preserved_in_scheduler() -> None:
    seq = [0.05, 0.01, 0.04]  # jitter burst pattern
    replayer = LatencyTraceReplayer(seq, end_of_trace=EndOfTraceBehavior.FAIL)
    sched = LatencyScheduler(
        LatencyConfig(
            latency_s=0.0,
            control_period_s=0.02,
            n_action_steps=2,
            mode=LatencyMode.TRACE_REPLAY,
        ),
        initial_command=np.zeros(2, dtype=np.float32),
        latency_replayer=replayer,
    )
    chunk = np.zeros((2, 2), dtype=np.float32)
    observed: list[float] = []
    for expected in seq:
        t_req = sched.sim_time_s
        sched.submit_inference(request_time_s=t_req, chunk=chunk)
        assert sched._pending is not None  # noqa: SLF001
        observed.append(sched._pending.ready_time_s - t_req)  # noqa: SLF001
        assert observed[-1] == pytest.approx(expected)
        # Clear in-flight so the next request can use the next trace sample.
        sched._pending = None  # noqa: SLF001
    assert observed == seq


def test_missing_rpi5_profile_prints_command(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles" / "rpi5"
    profiles.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="breakpoint profile") as exc:
        require_unique_device_profile(
            profiles,
            device_label="rpi5",
            candidate="act_aloha_transfer",
            variant_label="onnx_fp32",
        )
    assert "breakpoint profile" in str(exc.value)
    hint = profiling_command_hint(device_label="rpi5")
    assert "--device-label rpi5" in hint


def test_ambiguous_profiles_rejected(tmp_path: Path) -> None:
    profiles = tmp_path / "rpi5"
    profiles.mkdir()
    for name in ("a.json", "b.json"):
        payload = _valid_profile_payload([0.01, 0.02])
        (profiles / name).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Ambiguous"):
        require_unique_device_profile(
            profiles,
            device_label="rpi5",
            candidate="act_aloha_transfer",
            variant_label="onnx_fp32",
        )
