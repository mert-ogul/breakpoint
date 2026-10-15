"""Offline tests for closed-loop diagnosis helpers (no simulator)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from breakpoint_eval.export_closed_loop_diagnosis import (
    ClosedLoopDiagnosisRequest,
    summarize_chunk_error_trace,
)
from breakpoint_eval.export_parity import (
    FAILURE_KIND_LENGTH_MISMATCH,
    FAILURE_KIND_TRAJECTORY_DIVERGENCE,
    compare_action_traces_with_tails,
)
from breakpoint_eval.run_export import build_export_validation


def test_summarize_chunk_error_trace_reports_first_over_atol() -> None:
    summary = summarize_chunk_error_trace(
        [1e-7, 2e-7, 5e-3, 1e-6],
        fresh_steps=[0, 2],
        atol=1e-4,
        rtol=1e-4,
    )
    assert summary["n_steps"] == 4
    assert summary["first_over_atol_step"] == 2
    assert summary["passed_atol"] is False
    assert summary["fresh_step_errors"][1]["step"] == 2
    assert summary["max_abs_error"] == pytest.approx(5e-3)


def test_summarize_chunk_error_trace_empty() -> None:
    summary = summarize_chunk_error_trace([], fresh_steps=[], atol=1e-4, rtol=1e-4)
    assert summary["n_steps"] == 0
    assert summary["passed_atol"] is False


def test_independent_trace_keeps_unmatched_tails_and_all_seeds() -> None:
    """Regression: unequal lengths fail; divergence reports first step."""
    a = [np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)]
    short = compare_action_traces_with_tails(a, a[:1])
    assert short["failure_kind"] == FAILURE_KIND_LENGTH_MISMATCH
    assert short["unmatched_tail_steps"] == 1
    assert short["passed"] is False

    diverged = compare_action_traces_with_tails(
        a, [np.zeros(2, dtype=np.float32), np.full(2, 0.5, dtype=np.float32)]
    )
    assert diverged["failure_kind"] == FAILURE_KIND_TRAJECTORY_DIVERGENCE
    assert diverged["first_divergence_step"] == 1


def test_validation_gate_still_fails_independent_divergence() -> None:
    """Do not loosen the deployment gate when independent rollouts diverge."""
    validation = build_export_validation(
        open_loop_passed=True,
        closed_loop={
            "status": "ok",
            "passed": False,
            "completed": True,
            "comparison_kind": "independent_rollout_action_trace",
            "failure_kind": "independent_trajectory_divergence",
        },
        closed_loop_requested=True,
        corpus_suitable_as_deployment_evidence=True,
    )
    assert validation["passed"] is False
    assert validation["diagnostic_parity_ok"] is True
    assert "closed_loop_not_passed" in validation["not_passed_reasons"]
    assert any("independent_trajectory_divergence" in r for r in validation["not_passed_reasons"])


def test_diagnosis_refuses_overwrite_export_dir(tmp_path: Path) -> None:
    from breakpoint_eval.export_closed_loop_diagnosis import run_closed_loop_diagnosis

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    req = ClosedLoopDiagnosisRequest(
        export_dir=export_dir,
        output_dir=export_dir,
        seeds=(2000,),
    )
    with pytest.raises(ValueError, match="differ from export_dir"):
        run_closed_loop_diagnosis(req)
