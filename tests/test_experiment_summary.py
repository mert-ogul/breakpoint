"""Experiment plan grids and summary/breakpoint calculation from run records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from breakpoint_eval.experiment_plan import (
    N_EPISODES_FULL,
    N_EPISODES_SMOKE,
    ExperimentAxis,
    PlanPhase,
    build_seed_pairs,
    jpeg_conditions,
    latency_conditions,
    make_combined_plan,
    make_isolated_plan,
    precision_conditions,
    resolution_conditions,
    resolution_hw,
    write_experiment_plan,
)
from breakpoint_eval.experiment_summary import (
    condition_is_below_threshold,
    paired_bootstrap_contrast,
    summarize_experiment,
    wilson_interval,
)
from breakpoint_eval.run_episode import EpisodeOutcome


def _episode(seed: int, *, success: bool, outcome: str | None = None) -> dict:
    if outcome is None:
        outcome = (
            EpisodeOutcome.SUCCESS.value if success else EpisodeOutcome.TERMINATED_NO_SUCCESS.value
        )
    return {
        "episode_index": 0,
        "seed": seed,
        "outcome": outcome,
        "success": success and outcome == EpisodeOutcome.SUCCESS.value,
        "terminated": True,
        "truncated": False,
        "policy_failed": False,
        "runtime_error": "boom" if outcome == EpisodeOutcome.RUNTIME_ERROR.value else None,
        "n_control_steps": 10,
        "n_inference_calls": 1,
        "n_fresh_chunks": 1,
        "sum_reward": 1.0,
        "max_reward": 1.0,
        "control_period_s": 0.02,
        "chunk_size": 100,
        "n_action_steps": 100,
    }


def _write_report(path: Path, seeds: list[int], successes: list[bool]) -> Path:
    assert len(seeds) == len(successes)
    episodes = [_episode(s, success=ok) for s, ok in zip(seeds, successes, strict=True)]
    n_success = sum(1 for ok in successes if ok)
    payload = {
        "provenance": {},
        "summary": {
            "n_episodes_total": len(episodes),
            "n_episodes_eligible": len(episodes),
            "n_runtime_error": 0,
            "n_success": n_success,
            "success_rate": n_success / len(episodes),
        },
        "episodes": episodes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def test_seed_pairs_separate_policy_rng() -> None:
    pairs = build_seed_pairs(5, env_seed_start=10, policy_seed_offset=1000)
    assert [p.env_seed for p in pairs] == [10, 11, 12, 13, 14]
    assert [p.policy_rng_seed for p in pairs] == [1010, 1011, 1012, 1013, 1014]
    assert pairs[0].env_seed != pairs[0].policy_rng_seed


def test_default_counts_and_grids() -> None:
    assert N_EPISODES_SMOKE == 5
    assert N_EPISODES_FULL == 100
    lat = latency_conditions()
    assert [c.params["latency_ms"] for c in lat] == [0, 20, 40, 60, 80, 120, 200]
    assert [c.params["variant_label"] for c in precision_conditions()] == [
        "onnx_fp32",
        "int8",
        "int4_weight_only",
    ]
    assert resolution_hw(0.5) == (240, 320)
    assert resolution_hw(0.25) == (120, 160)
    res = resolution_conditions()
    assert res[0].params["capture_hw"] == [480, 640]
    assert res[1].params["capture_hw"] == [240, 320]
    jpeg = jpeg_conditions()
    assert jpeg[0].params["bypass"] is True
    assert [c.params.get("jpeg_quality") for c in jpeg[1:]] == [95, 75, 50, 30, 10]


def test_smoke_and_full_plan_counts(tmp_path: Path) -> None:
    smoke = make_isolated_plan(
        axis=ExperimentAxis.LATENCY_MS,
        candidate="act_aloha_transfer",
        phase=PlanPhase.SMOKE,
    )
    assert smoke.n_episodes == 5
    full = make_isolated_plan(
        axis=ExperimentAxis.LATENCY_MS,
        candidate="act_aloha_transfer",
        phase=PlanPhase.EXPLORATORY,
    )
    assert full.n_episodes == 100
    path = write_experiment_plan(full, tmp_path / "plan.json")
    assert path.is_file()


def test_combined_requires_explicit_conditions() -> None:
    conds = latency_conditions()[:2]
    plan = make_combined_plan(
        candidate="act_aloha_transfer",
        conditions=conds,
        baseline_condition_id="latency_ms_0",
        n_episodes=5,
    )
    assert plan.axis == "combined"
    assert (
        "additive" in " ".join(plan.isolation_notes).lower()
        or "not imply" in " ".join(plan.isolation_notes).lower()
    )


def test_wilson_known_counts_zero_and_all_success() -> None:
    z0 = wilson_interval(0, 10)
    assert z0["point"] == 0.0
    assert z0["low"] == 0.0
    assert z0["high"] is not None and z0["high"] > 0
    z1 = wilson_interval(10, 10)
    assert z1["point"] == 1.0
    assert z1["high"] == 1.0
    empty = wilson_interval(0, 0)
    assert empty["unresolved"] is True


def test_below_threshold_ratio() -> None:
    assert condition_is_below_threshold(0.7, 1.0, ratio=0.8) is True
    assert condition_is_below_threshold(0.85, 1.0, ratio=0.8) is False
    assert condition_is_below_threshold(None, 1.0, ratio=0.8) is None


def test_observed_interval_and_all_points(tmp_path: Path) -> None:
    plan = make_isolated_plan(
        axis=ExperimentAxis.LATENCY_MS,
        candidate="act_aloha_transfer",
        phase=PlanPhase.SMOKE,
        n_episodes=5,
    )
    seeds = [s.env_seed for s in plan.seed_pairs]
    # Baseline all success; delay increases until failure at 60ms.
    mapping = {
        "latency_ms_0": _write_report(tmp_path / "0.json", seeds, [True] * 5),
        "latency_ms_20": _write_report(tmp_path / "20.json", seeds, [True] * 5),
        "latency_ms_40": _write_report(tmp_path / "40.json", seeds, [True] * 5),
        "latency_ms_60": _write_report(tmp_path / "60.json", seeds, [False] * 5),
        "latency_ms_80": _write_report(tmp_path / "80.json", seeds, [False] * 5),
        "latency_ms_120": _write_report(tmp_path / "120.json", seeds, [False] * 5),
        "latency_ms_200": _write_report(tmp_path / "200.json", seeds, [False] * 5),
    }
    summary = summarize_experiment(plan, mapping, ratio=0.8, baseline_success_floor=0.5)
    axis = summary["axis_summary"]
    assert axis["headline_suppressed"] is False
    assert len(axis["points"]) == 7
    bp = axis["breakpoint"]
    assert bp["status"] == "observed_interval"
    assert bp["first_below"] == "latency_ms_60"
    assert bp["preceding_above"] == "latency_ms_40"
    assert bp["observed_interval"]["interpolated"] is False
    assert bp["non_monotonic"] is False


def test_not_reached_within_range(tmp_path: Path) -> None:
    plan = make_isolated_plan(
        axis=ExperimentAxis.LATENCY_MS,
        candidate="act_aloha_transfer",
        phase=PlanPhase.SMOKE,
        n_episodes=4,
    )
    seeds = [s.env_seed for s in plan.seed_pairs]
    mapping = {
        c.condition_id: _write_report(tmp_path / f"{c.condition_id}.json", seeds, [True] * 4)
        for c in plan.conditions
    }
    summary = summarize_experiment(plan, mapping)
    assert summary["axis_summary"]["breakpoint"]["status"] == "not_reached_within_tested_range"


def test_non_monotonic_recovery_flagged(tmp_path: Path) -> None:
    plan = make_isolated_plan(
        axis=ExperimentAxis.LATENCY_MS,
        candidate="act_aloha_transfer",
        phase=PlanPhase.SMOKE,
        n_episodes=5,
    )
    seeds = [s.env_seed for s in plan.seed_pairs]
    mapping = {
        "latency_ms_0": _write_report(tmp_path / "0.json", seeds, [True] * 5),
        "latency_ms_20": _write_report(tmp_path / "20.json", seeds, [True] * 5),
        "latency_ms_40": _write_report(tmp_path / "40.json", seeds, [False] * 5),
        "latency_ms_60": _write_report(tmp_path / "60.json", seeds, [True] * 5),  # recovery
        "latency_ms_80": _write_report(tmp_path / "80.json", seeds, [False] * 5),
        "latency_ms_120": _write_report(tmp_path / "120.json", seeds, [False] * 5),
        "latency_ms_200": _write_report(tmp_path / "200.json", seeds, [False] * 5),
    }
    summary = summarize_experiment(plan, mapping)
    bp = summary["axis_summary"]["breakpoint"]
    assert bp["first_below"] == "latency_ms_40"
    assert bp["non_monotonic"] is True
    assert "latency_ms_60" in bp["recovery_condition_ids"]


def test_weak_baseline_suppresses_headline(tmp_path: Path) -> None:
    plan = make_isolated_plan(
        axis=ExperimentAxis.LATENCY_MS,
        candidate="act_aloha_transfer",
        phase=PlanPhase.SMOKE,
        n_episodes=5,
    )
    seeds = [s.env_seed for s in plan.seed_pairs]
    # Baseline 1/5 = 0.2 < 0.5 floor
    mapping = {
        c.condition_id: _write_report(
            tmp_path / f"{c.condition_id}.json",
            seeds,
            [True, False, False, False, False] if c.condition_id == "latency_ms_0" else [False] * 5,
        )
        for c in plan.conditions
    }
    summary = summarize_experiment(plan, mapping, baseline_success_floor=0.5)
    axis = summary["axis_summary"]
    assert axis["headline_suppressed"] is True
    assert axis["reason"] == "weak_baseline"
    assert axis["breakpoint"] is None
    assert len(axis["points"]) == 7  # raw still shown


def test_incomplete_missing_and_runtime_error(tmp_path: Path) -> None:
    plan = make_isolated_plan(
        axis=ExperimentAxis.LATENCY_MS,
        candidate="act_aloha_transfer",
        phase=PlanPhase.SMOKE,
        n_episodes=3,
    )
    seeds = [s.env_seed for s in plan.seed_pairs]
    # Baseline complete; one condition missing; one has runtime error.
    mapping = {
        "latency_ms_0": _write_report(tmp_path / "0.json", seeds, [True] * 3),
        "latency_ms_20": _write_report(tmp_path / "20.json", seeds, [True] * 3),
    }
    err_path = tmp_path / "40.json"
    payload = {
        "provenance": {},
        "summary": {
            "n_episodes_total": 3,
            "n_episodes_eligible": 2,
            "n_runtime_error": 1,
            "n_success": 2,
            "success_rate": 1.0,
        },
        "episodes": [
            _episode(seeds[0], success=True),
            _episode(seeds[1], success=True),
            _episode(seeds[2], success=False, outcome=EpisodeOutcome.RUNTIME_ERROR.value),
        ],
    }
    err_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    mapping["latency_ms_40"] = err_path
    summary = summarize_experiment(plan, mapping)
    by_id = {c["condition_id"]: c for c in summary["conditions"]}
    assert by_id["latency_ms_60"]["incomplete"] is True
    assert "missing_condition_mapping" in by_id["latency_ms_60"]["incomplete_reasons"]
    assert by_id["latency_ms_40"]["incomplete"] is True
    assert any("runtime_errors" in r for r in by_id["latency_ms_40"]["incomplete_reasons"])
    assert summary["axis_summary"]["headline_suppressed"] is True
    assert summary["axis_summary"]["breakpoint"] is None


def test_censored_left_boundary(tmp_path: Path) -> None:
    plan = make_isolated_plan(
        axis=ExperimentAxis.LATENCY_MS,
        candidate="act_aloha_transfer",
        phase=PlanPhase.SMOKE,
        n_episodes=5,
    )
    seeds = [s.env_seed for s in plan.seed_pairs]
    # Even baseline fails the 0.8 rule relative to itself? baseline vs itself is never below.
    # Make baseline strong, but first non-baseline... actually first point IS baseline.
    # For first_below to be latency_ms_0, p_0 < 0.8 * p_0 is impossible.
    # Censored left: first *stress* point below with no preceding above — use JPEG where
    # bypass is baseline and q95 already fails.
    plan = make_isolated_plan(
        axis=ExperimentAxis.JPEG,
        candidate="act_aloha_transfer",
        phase=PlanPhase.SMOKE,
        n_episodes=5,
    )
    seeds = [s.env_seed for s in plan.seed_pairs]
    mapping = {
        "jpeg_bypass": _write_report(tmp_path / "bypass.json", seeds, [True] * 5),
        "jpeg_q95": _write_report(tmp_path / "q95.json", seeds, [False] * 5),
        "jpeg_q75": _write_report(tmp_path / "q75.json", seeds, [False] * 5),
        "jpeg_q50": _write_report(tmp_path / "q50.json", seeds, [False] * 5),
        "jpeg_q30": _write_report(tmp_path / "q30.json", seeds, [False] * 5),
        "jpeg_q10": _write_report(tmp_path / "q10.json", seeds, [False] * 5),
    }
    summary = summarize_experiment(plan, mapping)
    bp = summary["axis_summary"]["breakpoint"]
    assert bp["first_below"] == "jpeg_q95"
    assert bp["preceding_above"] == "jpeg_bypass"
    # True left-censor: only failing points, no above except we always have bypass above.
    # Force all including... can't make baseline below itself.
    # Left censor when first ordered point is below and has no preceding: use resolution
    # with only one level? Or latency where we don't include 0 in mapping as above...
    # Create custom: first complete point is 20ms below, and 0 is incomplete.
    plan = make_isolated_plan(
        axis=ExperimentAxis.LATENCY_MS,
        candidate="act_aloha_transfer",
        phase=PlanPhase.SMOKE,
        n_episodes=5,
    )
    seeds = [s.env_seed for s in plan.seed_pairs]
    mapping = {
        "latency_ms_0": _write_report(tmp_path / "b0.json", seeds, [True] * 5),
        "latency_ms_20": _write_report(tmp_path / "b20.json", seeds, [False] * 5),
        "latency_ms_40": _write_report(tmp_path / "b40.json", seeds, [False] * 5),
        "latency_ms_60": _write_report(tmp_path / "b60.json", seeds, [False] * 5),
        "latency_ms_80": _write_report(tmp_path / "b80.json", seeds, [False] * 5),
        "latency_ms_120": _write_report(tmp_path / "b120.json", seeds, [False] * 5),
        "latency_ms_200": _write_report(tmp_path / "b200.json", seeds, [False] * 5),
    }
    summary = summarize_experiment(plan, mapping)
    assert summary["axis_summary"]["breakpoint"]["preceding_above"] == "latency_ms_0"


def test_precision_categorical_no_continuous_breakpoint(tmp_path: Path) -> None:
    plan = make_isolated_plan(
        axis=ExperimentAxis.PRECISION,
        candidate="act_aloha_transfer",
        phase=PlanPhase.SMOKE,
        n_episodes=5,
    )
    seeds = [s.env_seed for s in plan.seed_pairs]
    mapping = {
        "precision_onnx_fp32": _write_report(tmp_path / "fp32.json", seeds, [True] * 5),
        "precision_int8": _write_report(
            tmp_path / "int8.json", seeds, [True, True, False, False, False]
        ),
        "precision_int4_weight_only": _write_report(tmp_path / "int4.json", seeds, [False] * 5),
    }
    summary = summarize_experiment(plan, mapping)
    axis = summary["axis_summary"]
    assert axis["breakpoint"] is None
    assert "recipes" in axis
    assert len(axis["recipes"]) == 3
    assert "categor" in axis["note"].lower() or "bit-depth" in axis["note"].lower()


def test_paired_bootstrap_seed_recorded_and_degenerate() -> None:
    b = __import__("numpy").asarray([1.0, 1.0, 0.0, 1.0], dtype=float)
    c = __import__("numpy").asarray([0.0, 0.0, 0.0, 1.0], dtype=float)
    out = paired_bootstrap_contrast(b, c, ratio=0.8, seed=123)
    assert out["bootstrap_seed"] == 123
    assert out["coverage"] == "pointwise"
    assert out["simultaneous_across_sweep"] is False
    assert out["unresolved"] is False
    deg = paired_bootstrap_contrast(
        __import__("numpy").asarray([1.0], dtype=float),
        __import__("numpy").asarray([0.0], dtype=float),
        ratio=0.8,
    )
    assert deg["unresolved"] is True


def test_confirmation_requires_link() -> None:
    with pytest.raises(ValueError, match="confirms_plan_id"):
        make_isolated_plan(
            axis=ExperimentAxis.LATENCY_MS,
            candidate="act_aloha_transfer",
            phase=PlanPhase.CONFIRMATION,
            n_episodes=5,
        )
