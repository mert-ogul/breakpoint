"""Acceptance/rejection tests for export_validation schema v2 (no Torch)."""

from __future__ import annotations

import pytest

from breakpoint_eval.export_validation import (
    SCHEMA_VERSION,
    ValidationBinding,
    assert_validation_v2_acceptable,
    build_export_validation_v2,
    evaluate_closed_loop_task_outcomes,
    evaluate_independent_trajectory_diagnostic,
    evaluate_numerical_parity,
)

RECIPE_ONNX_FP32 = "torch_onnx_export_fp32_opset17"


def _binding(**overrides: object) -> ValidationBinding:
    base = dict(
        model_sha256="abc",
        corpus_eval_sha256="corp",
        hub_revision="rev",
        candidate="act_aloha_transfer",
        recipe_id=RECIPE_ONNX_FP32,
        variant_label="onnx_fp32",
        code_revision="deadbeef",
        code_dirty=False,
        uncommitted_paths=(),
    )
    base.update(overrides)
    return ValidationBinding(**base)  # type: ignore[arg-type]


def _shadow(seed: int = 2000, **overrides: object) -> dict:
    payload = {
        "seed": seed,
        "passed_atol": True,
        "finite_ok": True,
        "shapes_ok": True,
        "compares_every_control_step": True,
        "covers_every_fresh_chunk_refresh": True,
        "covers_every_inference_call": True,
        "shadow_controls_simulation": False,
        "n_fresh_chunks": 4,
        "n_inference_calls": 4,
        "max_abs_error": 1e-6,
    }
    payload.update(overrides)
    return payload


def _seed_outcome(seed: int = 2000, **overrides: object) -> dict:
    payload = {
        "seed": seed,
        "pytorch_outcome": "success",
        "onnx_outcome": "success",
        "pytorch_success": True,
        "onnx_success": True,
        "pytorch_terminated": True,
        "onnx_terminated": True,
        "pytorch_truncated": False,
        "onnx_truncated": False,
        "pytorch_n_control_steps": 301,
        "onnx_n_control_steps": 302,
        "passed": False,
        "length_mismatch": True,
        "unmatched_tail_steps": 1,
        "first_divergence_step": 100,
        "max_abs_error": 0.02,
        "failure_kind": "episode_length_mismatch",
    }
    payload.update(overrides)
    return payload


def test_numerical_parity_requires_fresh_chunk_coverage() -> None:
    open_loop = {"passed": True, "finite_ok": True, "shapes_ok": True}
    ok = evaluate_numerical_parity(open_loop_report=open_loop, shadow_reports=[_shadow()])
    assert ok["passed"] is True

    bad = evaluate_numerical_parity(
        open_loop_report=open_loop,
        shadow_reports=[_shadow(covers_every_fresh_chunk_refresh=False)],
    )
    assert bad["passed"] is False
    assert any("fresh_chunk" in r for r in bad["not_passed_reasons"])


def test_numerical_parity_rejects_nonfinite_or_failed_open_loop() -> None:
    shadow = [_shadow()]
    bad = evaluate_numerical_parity(
        open_loop_report={"passed": False, "finite_ok": False, "shapes_ok": True},
        shadow_reports=shadow,
    )
    assert bad["passed"] is False
    assert "open_loop_nonfinite_or_missing" in bad["not_passed_reasons"]


def test_numerical_parity_rejects_missing_shape_or_finite_fields() -> None:
    shadow = [_shadow()]
    missing_shape = evaluate_numerical_parity(
        open_loop_report={"passed": True, "finite_ok": True},
        shadow_reports=shadow,
    )
    assert missing_shape["passed"] is False
    assert "open_loop_shape_mismatch_or_missing" in missing_shape["not_passed_reasons"]

    missing_shadow_finite = evaluate_numerical_parity(
        open_loop_report={"passed": True, "finite_ok": True, "shapes_ok": True},
        shadow_reports=[_shadow(finite_ok=None)],  # type: ignore[arg-type]
    )
    assert missing_shadow_finite["passed"] is False
    assert any("nonfinite_or_missing" in r for r in missing_shadow_finite["not_passed_reasons"])


def test_numerical_parity_rejects_missing_evidence() -> None:
    missing_open = evaluate_numerical_parity(open_loop_report=None, shadow_reports=[_shadow()])
    assert missing_open["passed"] is False
    assert "open_loop_report_missing" in missing_open["not_passed_reasons"]

    missing_shadow = evaluate_numerical_parity(
        open_loop_report={"passed": True, "finite_ok": True, "shapes_ok": True},
        shadow_reports=[],
    )
    assert missing_shadow["passed"] is False
    assert "shadow_reports_missing" in missing_shadow["not_passed_reasons"]


def test_assert_rejects_incomplete_and_corpus_mismatch() -> None:
    result = build_export_validation_v2(
        binding=_binding(),
        open_loop_report={"passed": True, "finite_ok": True, "shapes_ok": True},
        shadow_reports=[_shadow()],
        independent_seed_reports=[_seed_outcome()],
        corpus_suitable_as_deployment_evidence=True,
    )
    with pytest.raises(ValueError, match="corpus_eval_sha256 mismatch"):
        assert_validation_v2_acceptable(result, model_sha256="abc", corpus_eval_sha256="other")

    incomplete = dict(result)
    incomplete["closed_loop_task_outcomes"] = {
        **result["closed_loop_task_outcomes"],
        "passed": False,
        "completed": False,
    }
    incomplete["passed"] = True  # forged top-level pass
    with pytest.raises(ValueError, match="closed_loop_task_outcomes"):
        assert_validation_v2_acceptable(incomplete, model_sha256="abc", corpus_eval_sha256="corp")


def test_task_outcomes_require_recorded_fields() -> None:
    ok = evaluate_closed_loop_task_outcomes([_seed_outcome(2000), _seed_outcome(2001)])
    assert ok["passed"] is True
    assert ok["completed"] is True
    assert ok["smoke_check"] is True
    assert ok["statistical_equivalence_claim"] is False
    assert ok["task_success_match_required"] is False

    missing = evaluate_closed_loop_task_outcomes(
        [{"seed": 1, "pytorch_outcome": "success", "onnx_outcome": "success"}]
    )
    assert missing["passed"] is False
    assert any("missing_outcome_fields" in r for r in missing["not_passed_reasons"])


def test_task_outcomes_reject_unfinished_or_arbitrary_outcomes() -> None:
    unfinished = evaluate_closed_loop_task_outcomes(
        [
            _seed_outcome(
                2100,
                pytorch_terminated=False,
                pytorch_truncated=False,
                pytorch_outcome="success",
            )
        ]
    )
    assert unfinished["passed"] is False
    assert any("incomplete_or_invalid" in r for r in unfinished["not_passed_reasons"])

    arbitrary = evaluate_closed_loop_task_outcomes(
        [_seed_outcome(2100, pytorch_outcome="policy_failed", pytorch_success=False)]
    )
    assert arbitrary["passed"] is False

    # Mismatched task success still counts as completed smoke (not a numerical gate).
    mismatch = evaluate_closed_loop_task_outcomes(
        [
            _seed_outcome(
                2100,
                pytorch_success=True,
                onnx_success=False,
                onnx_outcome="time_limit_truncation",
                onnx_terminated=False,
                onnx_truncated=True,
            )
        ]
    )
    assert mismatch["passed"] is True
    assert mismatch["seeds"][0]["task_success_match"] is False


def test_independent_trajectory_does_not_gate() -> None:
    diag = evaluate_independent_trajectory_diagnostic([_seed_outcome()])
    assert diag["similar"] is False
    assert diag["gates_deployment_validation"] is False

    result = build_export_validation_v2(
        binding=_binding(),
        open_loop_report={"passed": True, "finite_ok": True, "shapes_ok": True},
        shadow_reports=[_shadow(2000), _shadow(2001)],
        independent_seed_reports=[_seed_outcome(2000), _seed_outcome(2001)],
        corpus_suitable_as_deployment_evidence=True,
    )
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["passed"] is True
    assert result["independent_trajectory"]["similar"] is False
    assert result["independent_trajectory"]["gates_deployment_validation"] is False


def test_v2_rejects_unsuitable_corpus_and_hash_mismatch() -> None:
    result = build_export_validation_v2(
        binding=_binding(),
        open_loop_report={"passed": True, "finite_ok": True, "shapes_ok": True},
        shadow_reports=[_shadow()],
        independent_seed_reports=[_seed_outcome()],
        corpus_suitable_as_deployment_evidence=False,
    )
    assert result["passed"] is False
    with pytest.raises(ValueError, match="passed is not true"):
        assert_validation_v2_acceptable(result, model_sha256="abc", corpus_eval_sha256="corp")

    good = build_export_validation_v2(
        binding=_binding(),
        open_loop_report={"passed": True, "finite_ok": True, "shapes_ok": True},
        shadow_reports=[_shadow()],
        independent_seed_reports=[_seed_outcome()],
        corpus_suitable_as_deployment_evidence=True,
    )
    with pytest.raises(ValueError, match="model_sha256 mismatch"):
        assert_validation_v2_acceptable(good, model_sha256="other", corpus_eval_sha256="corp")
