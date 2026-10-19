"""Compact episode records produced by scripts/package_v1.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location("package_v1", ROOT / "scripts" / "package_v1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def episode(**overrides):
    base = {
        "episode_index": 3, "seed": 5003, "policy_rng_seed": 15003,
        "outcome": "success", "success": True, "runtime_error": None, "n_control_steps": 284,
        "n_inference_calls": 3, "n_fresh_chunks": 3, "n_underflows": 0, "n_missed_deadlines": 0,
        "sum_reward": 249.0, "max_reward": 4.0, "control_period_s": 0.02,
        "latency_mode": "constant", "latency_s": 0.0, "image_mode": "native",
        "image_degrade_summary": {"capture_hw": [480, 640], "jpeg_quality": None},
        "latency_trace_provenance": None,
        "steps": [{"control_step_index": 282, "sim_time_s": 5.66, "is_success": False},
                  {"control_step_index": 283, "sim_time_s": 5.68, "is_success": True}],
    }
    base.update(overrides)
    return base


def test_compact_row_records_success_step_and_simulated_duration() -> None:
    module = load_script()
    row = module.compact_episode_row(
        axis="latency_ms", plan_id="p", condition_id="c", referenced=False,
        source="outputs/x/run_report.json", episode=episode())
    assert row["success_step_index"] == 283
    assert row["sim_time_s"] == 5.68 and row["final_step_sim_time_s"] == 5.68
    assert row["consumed_trace_indices"] is None
    assert row["source_report"] == "outputs/x/run_report.json"
    assert "steps" not in row


def test_compact_row_keeps_replay_window_and_consumed_delays() -> None:
    module = load_script()
    trace = {"window_start": 33, "window_length": 32, "consumed_trace_indices": [33, 34],
             "consumed_delays_s": [0.7, 0.8]}
    row = module.compact_episode_row(
        axis="device_profile", plan_id="p", condition_id="c", referenced=True, source="s",
        episode=episode(latency_trace_provenance=trace, outcome="time_limit_truncation",
                        success=False))
    assert row["trace_window_start"] == 33
    assert row["consumed_delays_s"] == [0.7, 0.8]
    assert row["referenced_baseline"] is True
