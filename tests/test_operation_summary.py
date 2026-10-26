"""Console summaries and exit codes derived from saved payloads (no optional dependencies)."""

from __future__ import annotations

from pathlib import Path

from breakpoint_eval.operation_summary import (
    closed_loop_diagnosis_summary,
    export_exit_code,
    export_summary,
    export_validation_summary,
    profile_summary,
)

PATH = Path("/tmp/example.json")


def _export_payload(**validation: object) -> dict:
    return {
        "comparisons": {"onnx_fp32_vs_pytorch_fp32": {"passed": True}},
        "export_validation": validation,
        "variants": [{"label": "onnx_fp32", "status": "ok", "recipe_id": "r"}],
    }


def test_export_exit_code_requires_open_loop_parity() -> None:
    assert export_exit_code(_export_payload()) == 0
    failed = _export_payload()
    failed["comparisons"]["onnx_fp32_vs_pytorch_fp32"]["passed"] = False
    assert export_exit_code(failed) == 1
    assert export_exit_code({}) == 1
    assert export_exit_code(_export_payload(open_loop_fp32_parity_passed=False)) == 1


def test_export_exit_code_enforces_only_requested_closed_loop() -> None:
    assert export_exit_code(_export_payload(passed=False)) == 0
    requested = {"closed_loop_requested": True}
    assert export_exit_code(_export_payload(**requested)) == 1
    assert export_exit_code(_export_payload(**requested, closed_loop_completed=True)) == 1
    assert (
        export_exit_code(
            _export_payload(**requested, closed_loop_completed=True, closed_loop_passed=True)
        )
        == 0
    )


def test_export_summary_prefers_validation_failure_kind() -> None:
    payload = _export_payload(closed_loop_failure_kind="divergence")
    payload["closed_loop_onnx_fp32_vs_pytorch"] = {"failure_kind": "other", "status": "done"}
    summary = export_summary(PATH, payload)
    assert summary["closed_loop_failure_kind"] == "divergence"
    assert summary["closed_loop_status"] == "done"
    assert summary["onnx_fp32_parity_passed"] is True
    assert summary["variants"] == [{"label": "onnx_fp32", "status": "ok", "recipe_id": "r"}]


def test_export_validation_summary_reads_binding_and_gates() -> None:
    payload = {
        "schema_version": 2,
        "passed": True,
        "numerical_parity": {"passed": True},
        "closed_loop_task_outcomes": {"passed": False},
        "independent_trajectory": {"similar": False, "gates_deployment_validation": False},
        "binding": {"model_sha256": "m", "corpus_eval_sha256": "c", "code_dirty": True},
    }
    summary = export_validation_summary(PATH, payload)
    assert summary["validation"] == str(PATH)
    assert summary["numerical_parity_passed"] is True
    assert summary["task_outcomes_passed"] is False
    assert summary["independent_trajectory_gates_deployment"] is False
    assert (summary["model_sha256"], summary["corpus_eval_sha256"]) == ("m", "c")
    assert summary["code_revision"] is None and summary["code_dirty"] is True
    assert export_validation_summary(PATH, {})["passed"] is None


def test_diagnosis_summary_takes_worst_shadow_error_and_all_repeatability() -> None:
    payload = {
        "independent_rollouts": {
            "passed": False,
            "failure_kind": "length",
            "seeds": [{"seed": 1, "pytorch_outcome": "success", "onnx_outcome": "timeout"}],
        },
        "same_observation_shadow": [{"max_abs_error": 0.1}, {"max_abs_error": None}],
        "pytorch_repeatability": [{"passed": True}, {"passed": False}],
    }
    summary = closed_loop_diagnosis_summary(PATH, payload)
    assert summary["shadow_max_abs_error"] == 0.1
    assert summary["repeatability_all_passed"] is False
    assert summary["independent_seeds"][0]["outcomes"] == ["success", "timeout"]
    empty = closed_loop_diagnosis_summary(PATH, {})
    assert empty["shadow_max_abs_error"] is None
    assert empty["repeatability_all_passed"] is True


def test_profile_summary_reports_sample_count_and_percentiles_in_seconds() -> None:
    payload = {
        "device_label": "laptop",
        "measurements": {"n_warmup": 5, "summary": {"count": 10, "median_s": 0.01, "p95_s": 0.02}},
        "output_validation": {"passed": True},
        "timed_boundary": "ort_session_run_action_chunk",
    }
    summary = profile_summary(PATH, payload)
    assert summary["n_measured"] == 10 and summary["n_warmup"] == 5
    assert (summary["median_s"], summary["p95_s"], summary["p99_s"]) == (0.01, 0.02, None)
    assert summary["validation_passed"] is True
