"""Versioned ACT export validation (schema v2).

Separates:
  1. numerical_parity: same-input open-loop corpus + fresh-chunk shadow checks
  2. closed_loop_task_outcomes: completed baseline rollouts with recorded outcomes
  3. independent_trajectory: action/length similarity diagnostics (not a pass gate)

Independent trajectory equality is recorded but does not fail deployment validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from breakpoint_eval.artifact_names import (
    EXPORT_VALIDATION_INT4_NAME,
    EXPORT_VALIDATION_INT8_NAME,
    EXPORT_VALIDATION_NAME,
    resolve_artifact,
)
from breakpoint_eval.export_tolerances import (
    ONNX_FP32_CLOSED_LOOP_ATOL,
    ONNX_FP32_CLOSED_LOOP_RTOL,
    ONNX_FP32_PARITY_ACTION_UNIT_MEAN_ABS,
    ONNX_FP32_PARITY_ATOL,
    ONNX_FP32_PARITY_MAX_ABS,
    ONNX_FP32_PARITY_MEAN_ABS,
    ONNX_FP32_PARITY_RTOL,
)
from breakpoint_eval.file_digest import sha256_file
from breakpoint_eval.source_revision import read_source_revision

SCHEMA_VERSION = 2
VALIDATION_FILENAME = EXPORT_VALIDATION_NAME
INT8_VALIDATION_FILENAME = EXPORT_VALIDATION_INT8_NAME
INT4_VALIDATION_FILENAME = EXPORT_VALIDATION_INT4_NAME

# Terminal outcomes from a finished episode (not policy_failed / runtime_error).
_COMPLETED_TASK_OUTCOMES = frozenset(
    {
        "success",
        "terminated_no_success",
        "time_limit_truncation",
    }
)

_REQUIRED_BINDING_FIELDS = (
    "model_sha256",
    "corpus_eval_sha256",
    "hub_revision",
    "candidate",
    "recipe_id",
    "variant_label",
)


@dataclass(frozen=True)
class ValidationBinding:
    model_sha256: str
    corpus_eval_sha256: str
    hub_revision: str
    candidate: str
    recipe_id: str
    variant_label: str
    code_revision: str | None
    code_dirty: bool | None
    uncommitted_paths: tuple[str, ...] | None


def collect_code_revision(repo_root: Path) -> dict[str, Any]:
    """Record HEAD and uncommitted paths; dirty state and paths are None when unknown."""
    info = read_source_revision(Path(repo_root).resolve())
    porcelain = info["status_porcelain"]
    uncommitted: list[str] | None = None
    if porcelain is not None:
        uncommitted = [
            (line[3:] if len(line) >= 4 else line).strip()
            for line in porcelain.splitlines()
            if line.strip()
        ]
    return {
        "code_revision": info["revision"],
        "code_dirty": info["dirty"],
        "uncommitted_paths": uncommitted,
    }


def _validation_settings() -> dict[str, Any]:
    return {
        "parity_atol": ONNX_FP32_PARITY_ATOL,
        "parity_rtol": ONNX_FP32_PARITY_RTOL,
        "parity_max_abs": ONNX_FP32_PARITY_MAX_ABS,
        "parity_mean_abs": ONNX_FP32_PARITY_MEAN_ABS,
        "parity_action_unit_mean_abs": ONNX_FP32_PARITY_ACTION_UNIT_MEAN_ABS,
        "closed_loop_atol": ONNX_FP32_CLOSED_LOOP_ATOL,
        "closed_loop_rtol": ONNX_FP32_CLOSED_LOOP_RTOL,
        "fresh_chunk_same_input_required": True,
        "cached_action_trace_not_numerical_parity": True,
        "independent_trajectory_is_diagnostic_only": True,
        "two_seed_task_outcomes_are_smoke_not_statistical": True,
    }


def evaluate_numerical_parity(
    *,
    open_loop_report: dict[str, Any] | None,
    shadow_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Same-input numerical parity only (corpus open-loop + fresh-chunk shadow).

    Missing ``finite_ok`` / ``shapes_ok`` fields fail closed (no default-True).
    """
    reasons: list[str] = []
    if not open_loop_report:
        reasons.append("open_loop_report_missing")
        open_passed = False
    else:
        if open_loop_report.get("passed") is not True:
            reasons.append("open_loop_parity_failed")
        if open_loop_report.get("finite_ok") is not True:
            reasons.append("open_loop_nonfinite_or_missing")
        if open_loop_report.get("shapes_ok") is not True:
            reasons.append("open_loop_shape_mismatch_or_missing")
        open_passed = (
            open_loop_report.get("passed") is True
            and open_loop_report.get("finite_ok") is True
            and open_loop_report.get("shapes_ok") is True
        )

    if not shadow_reports:
        reasons.append("shadow_reports_missing")
        shadow_passed = False
        fresh_ok = False
    else:
        shadow_passed = True
        fresh_ok = True
        for report in shadow_reports:
            seed = report.get("seed")
            if report.get("passed_atol") is not True:
                shadow_passed = False
                reasons.append(f"shadow_seed_{seed}_atol_failed")
            if report.get("finite_ok") is not True:
                shadow_passed = False
                reasons.append(f"shadow_seed_{seed}_nonfinite_or_missing")
            if report.get("shapes_ok") is not True:
                shadow_passed = False
                reasons.append(f"shadow_seed_{seed}_shape_mismatch_or_missing")
            if report.get("compares_every_control_step") is not True:
                fresh_ok = False
                reasons.append(f"shadow_seed_{seed}_incomplete_steps")
            if report.get("covers_every_fresh_chunk_refresh") is not True:
                fresh_ok = False
                reasons.append(f"shadow_seed_{seed}_missing_fresh_chunk_coverage")
            if report.get("covers_every_inference_call") is not True:
                fresh_ok = False
                reasons.append(f"shadow_seed_{seed}_missing_inference_coverage")
            if report.get("shadow_controls_simulation") is True:
                reasons.append(f"shadow_seed_{seed}_controlled_sim")
                shadow_passed = False
            n_fresh = report.get("n_fresh_chunks")
            n_infer = report.get("n_inference_calls")
            if n_fresh is None or n_infer is None or int(n_fresh) < 1:
                fresh_ok = False
                reasons.append(f"shadow_seed_{seed}_no_fresh_chunks")
            elif int(n_fresh) != int(n_infer):
                # ACT should request fresh chunk iff inference call.
                reasons.append(f"shadow_seed_{seed}_fresh_infer_mismatch")
                fresh_ok = False

    passed = open_passed and shadow_passed and fresh_ok
    return {
        "kind": "same_input_numerical_parity",
        "passed": passed,
        "open_loop_passed": open_passed,
        "shadow_fresh_chunk_passed": bool(shadow_passed and fresh_ok),
        "not_passed_reasons": [] if passed else reasons,
        "note": (
            "Requires held-out corpus open-loop full-chunk parity with explicit "
            "shapes_ok/finite_ok and same-observation shadow compares that cover "
            "every fresh action-chunk inference. Cached queued-action traces are "
            "not numerical export evidence. Independent trajectory equality is "
            "diagnostic only."
        ),
    }


def _side_completed(report: dict[str, Any], *, prefix: str) -> bool:
    """True when one side finished with a known terminal outcome and end flags."""
    outcome = report.get(f"{prefix}_outcome")
    if outcome not in _COMPLETED_TASK_OUTCOMES:
        return False
    terminated = report.get(f"{prefix}_terminated")
    truncated = report.get(f"{prefix}_truncated")
    if terminated is not True and truncated is not True:
        return False
    n_steps = report.get(f"{prefix}_n_control_steps")
    if n_steps is None or int(n_steps) < 1:
        return False
    if report.get(f"{prefix}_success") is None:
        return False
    return True


def evaluate_closed_loop_task_outcomes(
    seed_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Require completed paired rollouts with recorded outcomes (smoke when n=2).

    Matching task success across backends is not required: a completed-task
    failure is not a numerical-export error. Unfinished / arbitrary outcomes fail.
    """
    reasons: list[str] = []
    if not seed_reports:
        return {
            "kind": "closed_loop_task_outcomes",
            "passed": False,
            "completed": False,
            "n_seeds": 0,
            "seeds": [],
            "smoke_check": True,
            "statistical_equivalence_claim": False,
            "not_passed_reasons": ["seed_reports_missing"],
            "note": "Two-seed closed-loop outcomes are a smoke check, not performance equivalence.",
        }

    seed_summaries: list[dict[str, Any]] = []
    for report in seed_reports:
        seed = report.get("seed")
        pt_outcome = report.get("pytorch_outcome")
        ort_outcome = report.get("onnx_outcome")
        pt_success = report.get("pytorch_success")
        ort_success = report.get("onnx_success")
        missing = []
        for key in (
            "pytorch_outcome",
            "onnx_outcome",
            "pytorch_success",
            "onnx_success",
            "pytorch_terminated",
            "onnx_terminated",
            "pytorch_truncated",
            "onnx_truncated",
            "pytorch_n_control_steps",
            "onnx_n_control_steps",
        ):
            if key not in report or report.get(key) is None:
                missing.append(key)
        if missing:
            reasons.append(f"seed_{seed}_missing_outcome_fields:{','.join(missing)}")
        if not _side_completed(report, prefix="pytorch"):
            reasons.append(f"seed_{seed}_pytorch_episode_incomplete_or_invalid_outcome")
        if not _side_completed(report, prefix="onnx"):
            reasons.append(f"seed_{seed}_onnx_episode_incomplete_or_invalid_outcome")
        seed_summaries.append(
            {
                "seed": seed,
                "pytorch_outcome": pt_outcome,
                "onnx_outcome": ort_outcome,
                "pytorch_success": pt_success,
                "onnx_success": ort_success,
                "pytorch_terminated": report.get("pytorch_terminated"),
                "onnx_terminated": report.get("onnx_terminated"),
                "pytorch_truncated": report.get("pytorch_truncated"),
                "onnx_truncated": report.get("onnx_truncated"),
                "pytorch_n_control_steps": report.get("pytorch_n_control_steps"),
                "onnx_n_control_steps": report.get("onnx_n_control_steps"),
                "task_success_match": pt_success == ort_success,
            }
        )

    completed = all(
        _side_completed(report, prefix="pytorch") and _side_completed(report, prefix="onnx")
        for report in seed_reports
    )
    n_seeds = len(seed_summaries)
    passed = completed and not reasons
    return {
        "kind": "closed_loop_task_outcomes",
        "passed": passed,
        "completed": completed,
        "n_seeds": n_seeds,
        "seeds": seed_summaries,
        "smoke_check": n_seeds <= 2,
        "statistical_equivalence_claim": False,
        "task_success_match_required": False,
        "not_passed_reasons": reasons,
        "note": (
            "Requires completed baseline rollouts with recorded success/terminated/"
            "truncated fields and a known terminal outcome. Two-seed results are a "
            "smoke check, not statistical evidence of performance equivalence. "
            "Mismatched task success is recorded but does not fail this gate."
        ),
    }


def evaluate_independent_trajectory_diagnostic(
    seed_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Action/length similarity across independent resets; diagnostic only."""
    seeds = []
    all_similar = True
    for report in seed_reports:
        similar = report.get("passed") is True and report.get("length_mismatch") is not True
        if not similar:
            all_similar = False
        seeds.append(
            {
                "seed": report.get("seed"),
                "passed_similarity": similar,
                "length_mismatch": report.get("length_mismatch"),
                "unmatched_tail_steps": report.get("unmatched_tail_steps"),
                "first_divergence_step": report.get("first_divergence_step"),
                "max_abs_error": report.get("max_abs_error"),
                "failure_kind": report.get("failure_kind"),
            }
        )
    return {
        "kind": "independent_trajectory_similarity",
        "similar": all_similar,
        "gates_deployment_validation": False,
        "seeds": seeds,
        "note": (
            "Independent-rollout action equality and identical episode lengths are "
            "diagnostics only; they are not numerical export requirements."
        ),
    }


def build_export_validation_v2(
    *,
    binding: ValidationBinding,
    open_loop_report: dict[str, Any] | None,
    shadow_reports: list[dict[str, Any]],
    independent_seed_reports: list[dict[str, Any]],
    corpus_suitable_as_deployment_evidence: bool,
    diagnosis_path: str | None = None,
    source_export_manifest_path: str | None = None,
) -> dict[str, Any]:
    numerical = evaluate_numerical_parity(
        open_loop_report=open_loop_report,
        shadow_reports=shadow_reports,
    )
    task_outcomes = evaluate_closed_loop_task_outcomes(independent_seed_reports)
    trajectory = evaluate_independent_trajectory_diagnostic(independent_seed_reports)

    reasons: list[str] = []
    if not corpus_suitable_as_deployment_evidence:
        reasons.append("corpus_not_suitable_as_deployment_evidence")
    if not numerical["passed"]:
        reasons.append("numerical_parity_failed")
        reasons.extend(numerical.get("not_passed_reasons") or [])
    if not task_outcomes["passed"]:
        reasons.append("closed_loop_task_outcomes_failed")
        reasons.extend(task_outcomes.get("not_passed_reasons") or [])

    passed = (
        corpus_suitable_as_deployment_evidence
        and numerical["passed"] is True
        and task_outcomes["passed"] is True
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "passed": passed,
        "not_passed_reasons": [] if passed else reasons,
        "binding": {
            "model_sha256": binding.model_sha256,
            "corpus_eval_sha256": binding.corpus_eval_sha256,
            "hub_revision": binding.hub_revision,
            "candidate": binding.candidate,
            "recipe_id": binding.recipe_id,
            "variant_label": binding.variant_label,
            "code_revision": binding.code_revision,
            "code_dirty": binding.code_dirty,
            "uncommitted_paths": (
                None if binding.uncommitted_paths is None else list(binding.uncommitted_paths)
            ),
        },
        "validation_settings": _validation_settings(),
        "numerical_parity": numerical,
        "closed_loop_task_outcomes": task_outcomes,
        "independent_trajectory": trajectory,
        "corpus_suitable_as_deployment_evidence": corpus_suitable_as_deployment_evidence,
        "diagnosis_path": diagnosis_path,
        "source_export_manifest_path": source_export_manifest_path,
        "note": (
            "schema_version=2: deployment passed requires same-input numerical parity "
            "and completed closed-loop task outcomes with recorded fields. Independent "
            "trajectory similarity is diagnostic only."
        ),
    }


def _require_binding_identity(binding: dict[str, Any], *, label: str) -> None:
    for key in _REQUIRED_BINDING_FIELDS:
        value = binding.get(key)
        if value is None or value == "":
            raise ValueError(f"{label} binding.{key} missing; refuse unbound validation")


def assert_validation_v2_acceptable(
    result: dict[str, Any],
    *,
    model_sha256: str,
    corpus_eval_sha256: str | None,
    hub_revision: str | None = None,
    candidate: str | None = None,
    recipe_id: str | None = None,
    variant_label: str | None = None,
) -> None:
    """Fail-closed acceptance for FP32 deployment runtime binding.

    Identity arguments left as None are not compared.
    """
    if not isinstance(result, dict):
        raise ValueError("export_validation must be a JSON object")
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"export_validation schema_version must be {SCHEMA_VERSION}; "
            f"got {result.get('schema_version')!r}"
        )
    if (
        result.get("variant_label") == "int8"
        or (result.get("binding") or {}).get("variant_label") == "int8"
    ):
        raise ValueError(
            "FP32 deployment resolve refused INT8 validation sidecar; "
            "use export_validation_int8.json for INT8 graphs"
        )
    if (
        result.get("variant_label") == "int4_weight_only"
        or (result.get("binding") or {}).get("variant_label") == "int4_weight_only"
    ):
        raise ValueError(
            "FP32 deployment resolve refused INT4 validation sidecar; "
            "use export_validation_int4.json for INT4 graphs"
        )
    if result.get("passed") is not True:
        raise ValueError(
            "export_validation.passed is not true; refuse unvalidated deployment runtime "
            f"(reasons={result.get('not_passed_reasons')})"
        )
    binding = result.get("binding")
    if not isinstance(binding, dict):
        raise ValueError("export_validation binding missing or not an object")
    _require_binding_identity(binding, label="export_validation")
    if not result.get("validation_settings"):
        raise ValueError("export_validation validation_settings missing")
    if binding.get("model_sha256") != model_sha256:
        raise ValueError(
            "export_validation model_sha256 mismatch: "
            f"validation={binding.get('model_sha256')!r} model={model_sha256!r}"
        )
    if corpus_eval_sha256 is None or corpus_eval_sha256 == "":
        raise ValueError("corpus_eval_sha256 required for deployment runtime binding")
    if binding.get("corpus_eval_sha256") != corpus_eval_sha256:
        raise ValueError(
            "export_validation corpus_eval_sha256 mismatch: "
            f"validation={binding.get('corpus_eval_sha256')!r} "
            f"manifest={corpus_eval_sha256!r}"
        )
    if hub_revision is not None and hub_revision != "":
        if binding.get("hub_revision") != hub_revision:
            raise ValueError(
                "export_validation hub_revision mismatch: "
                f"validation={binding.get('hub_revision')!r} manifest={hub_revision!r}"
            )
    for key, expected in (
        ("candidate", candidate),
        ("recipe_id", recipe_id),
        ("variant_label", variant_label),
    ):
        if expected is not None and binding.get(key) != expected:
            raise ValueError(
                f"export_validation {key} mismatch: "
                f"validation={binding.get(key)!r} expected={expected!r}"
            )
    numerical = result.get("numerical_parity") or {}
    if numerical.get("passed") is not True:
        raise ValueError("export_validation numerical_parity.passed is not true")
    tasks = result.get("closed_loop_task_outcomes") or {}
    if tasks.get("passed") is not True or tasks.get("completed") is not True:
        raise ValueError("export_validation closed_loop_task_outcomes incomplete or failed")
    if result.get("corpus_suitable_as_deployment_evidence") is not True:
        raise ValueError("export_validation corpus is not suitable as deployment evidence")


def _evaluate_quant_runtime_checks(
    parity_report: dict[str, Any] | None, *, reason_prefix: str
) -> dict[str, Any]:
    """Finite outputs and matching shapes for a quantized variant (no FP32 allclose)."""
    reasons: list[str] = []
    if not parity_report:
        return {
            "passed": False,
            "shapes_ok": False,
            "finite_ok": False,
            "fp32_allclose_required": False,
            "not_passed_reasons": [f"{reason_prefix}_parity_report_missing"],
        }
    shapes_ok = parity_report.get("shapes_ok") is True
    finite_ok = parity_report.get("finite_ok") is True
    if not shapes_ok:
        reasons.append(f"{reason_prefix}_shape_mismatch_or_missing")
    if not finite_ok:
        reasons.append(f"{reason_prefix}_nonfinite_or_missing")
    return {
        "passed": shapes_ok and finite_ok,
        "shapes_ok": shapes_ok,
        "finite_ok": finite_ok,
        "max_abs_error": parity_report.get("max_abs_error"),
        "mean_abs_error": parity_report.get("mean_abs_error"),
        "action_unit_mean_abs_error": parity_report.get("action_unit_mean_abs_error"),
        "fp32_allclose_required": False,
        "task_success_required": False,
        "not_passed_reasons": reasons,
    }


def evaluate_int8_runtime_checks(parity_report: dict[str, Any] | None) -> dict[str, Any]:
    """INT8 requires finite outputs and matching shapes; not FP32 allclose."""
    return _evaluate_quant_runtime_checks(parity_report, reason_prefix="int8")


def build_int8_validation_v2(
    *,
    binding: dict[str, Any],
    parity_vs_fp32: dict[str, Any] | None,
    quantization_coverage: dict[str, Any] | None,
    parent_fp32_sha256: str,
    calibration_sha256: str,
    variant_status: str,
    source_export_manifest_path: str | None = None,
) -> dict[str, Any]:
    """INT8 sidecar: own identity, finite/shape checks, coverage; no task/allclose gate."""
    runtime = evaluate_int8_runtime_checks(parity_vs_fp32)
    reasons: list[str] = []
    if variant_status != "ok":
        reasons.append(f"int8_variant_status={variant_status}")
    if not parent_fp32_sha256:
        reasons.append("parent_fp32_sha256_missing")
    if not calibration_sha256:
        reasons.append("calibration_sha256_missing")
    if binding.get("model_sha256") in {None, ""}:
        reasons.append("int8_model_sha256_missing")
    if binding.get("recipe_id") in {None, ""}:
        reasons.append("int8_recipe_id_missing")
    if binding.get("candidate") in {None, ""}:
        reasons.append("candidate_missing")
    if binding.get("hub_revision") in {None, ""}:
        reasons.append("hub_revision_missing")
    if not quantization_coverage:
        reasons.append("quantization_coverage_missing")
    if not runtime["passed"]:
        reasons.extend(runtime.get("not_passed_reasons") or [])
    if binding.get("parent_fp32_sha256") not in {None, "", parent_fp32_sha256}:
        reasons.append("parent_fp32_sha256_binding_mismatch")
    if binding.get("calibration_sha256") not in {None, "", calibration_sha256}:
        reasons.append("calibration_sha256_binding_mismatch")

    passed = variant_status == "ok" and runtime["passed"] and not reasons
    return {
        "schema_version": SCHEMA_VERSION,
        "variant_label": "int8",
        "passed": passed,
        "not_passed_reasons": [] if passed else reasons,
        "binding": {
            **binding,
            "variant_label": "int8",
            "parent_fp32_sha256": parent_fp32_sha256,
            "calibration_sha256": calibration_sha256,
        },
        "runtime_checks": runtime,
        "quantization_coverage": quantization_coverage or {},
        "numeric_errors": {"vs_onnx_fp32": _parity_numeric_block(parity_vs_fp32)},
        "task_success_required": False,
        "fp32_allclose_required": False,
        "fp32_validation_sidecar_authorizes_int8": False,
        "source_export_manifest_path": source_export_manifest_path,
        "note": (
            "INT8 validation is separate from FP32. Requires runnable QDQ graph with "
            "finite matching shapes vs the FP32 parent, recorded calibration corpus, "
            "and quantization coverage. Does not require FP32-level allclose or task "
            "success; reduced task performance is measured later."
        ),
    }


def assert_int8_validation_acceptable(
    result: dict[str, Any],
    *,
    model_sha256: str,
    parent_fp32_sha256: str | None = None,
    calibration_sha256: str | None = None,
) -> None:
    if not isinstance(result, dict):
        raise ValueError("export_validation_int8 must be a JSON object")
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"export_validation_int8 schema_version must be {SCHEMA_VERSION}; "
            f"got {result.get('schema_version')!r}"
        )
    if result.get("variant_label") != "int8":
        raise ValueError("INT8 resolve requires variant_label=int8 validation sidecar")
    if result.get("passed") is not True:
        raise ValueError(
            "export_validation_int8.passed is not true; refuse unvalidated INT8 runtime "
            f"(reasons={result.get('not_passed_reasons')})"
        )
    binding = result.get("binding") or {}
    if binding.get("model_sha256") != model_sha256:
        raise ValueError(
            "INT8 validation model_sha256 mismatch: "
            f"validation={binding.get('model_sha256')!r} model={model_sha256!r}"
        )
    if not binding.get("recipe_id"):
        raise ValueError("INT8 validation binding.recipe_id missing")
    if not binding.get("parent_fp32_sha256"):
        raise ValueError("INT8 validation binding.parent_fp32_sha256 missing")
    if not binding.get("calibration_sha256"):
        raise ValueError("INT8 validation binding.calibration_sha256 missing")
    if parent_fp32_sha256 is not None and binding.get("parent_fp32_sha256") != parent_fp32_sha256:
        raise ValueError("INT8 parent_fp32_sha256 mismatch")
    if calibration_sha256 is not None and binding.get("calibration_sha256") != calibration_sha256:
        raise ValueError("INT8 calibration_sha256 mismatch")
    runtime = result.get("runtime_checks") or {}
    if runtime.get("passed") is not True:
        raise ValueError("INT8 runtime_checks.passed is not true")
    if result.get("fp32_validation_sidecar_authorizes_int8") is True:
        raise ValueError("INT8 must not be authorized by the FP32 validation sidecar")


def evaluate_int4_runtime_checks(parity_report: dict[str, Any] | None) -> dict[str, Any]:
    """INT4 requires finite outputs and matching shapes; not FP32 allclose."""
    return _evaluate_quant_runtime_checks(parity_report, reason_prefix="int4")


def _parity_numeric_block(parity: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "max_abs_error": (parity or {}).get("max_abs_error"),
        "mean_abs_error": (parity or {}).get("mean_abs_error"),
        "action_unit_mean_abs_error": (parity or {}).get("action_unit_mean_abs_error"),
        "shapes_ok": (parity or {}).get("shapes_ok"),
        "finite_ok": (parity or {}).get("finite_ok"),
        "fp32_allclose_enforced": False,
    }


def _resolve_parent_fp32_sha(export_dir: Path, fp32: dict[str, Any], *, for_label: str) -> str:
    parent_sha = str((fp32.get("graph") or {}).get("sha256") or "")
    if not parent_sha and fp32.get("path"):
        parent_path = Path(fp32["path"])
        if not parent_path.is_file():
            parent_path = export_dir / Path(fp32["path"]).name
        if parent_path.is_file():
            parent_sha = sha256_file(parent_path)
    if not parent_sha:
        raise ValueError(f"cannot resolve parent FP32 sha256 for {for_label} validation")
    return parent_sha


def _variant_file_sha(
    export_dir: Path,
    variant: dict[str, Any],
    *,
    label: str,
    display_name: str,
) -> tuple[str, str]:
    """Return (status, model_sha256), verifying the on-disk hash when status is ok."""
    status = str(variant.get("status") or "errored")
    model_sha = str((variant.get("graph") or {}).get("sha256") or "")
    if status != "ok":
        return status, model_sha
    model_path_value = variant.get("path")
    if not model_path_value:
        raise ValueError(f"{label} status=ok but path missing")
    model_path = Path(model_path_value)
    if not model_path.is_file():
        model_path = export_dir / Path(model_path_value).name
    if not model_path.is_file():
        raise FileNotFoundError(f"{display_name} ONNX not found: {model_path_value}")
    file_sha = sha256_file(model_path)
    if model_sha and model_sha != file_sha:
        raise ValueError(
            f"{display_name} manifest graph sha256 {model_sha} does not match file {file_sha}"
        )
    return status, file_sha


def _load_quant_export_context(
    export_dir: Path,
    *,
    variant_label: str,
    display_name: str,
) -> tuple[Path, dict[str, Any], dict[str, Any], str, str, str]:
    """Load manifest, parent FP32 sha, and variant model-file hash check.

    Returns (manifest_path, manifest, variant, parent_sha, status, variant_sha).
    """
    export_dir = Path(export_dir).resolve()
    manifest_path = export_dir / "export_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing export_manifest.json in {export_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    variants = manifest.get("variants") or []
    fp32 = next((v for v in variants if v.get("label") == "onnx_fp32"), None)
    variant = next((v for v in variants if v.get("label") == variant_label), None)
    if fp32 is None:
        raise ValueError("export manifest missing onnx_fp32 variant")
    if variant is None:
        raise ValueError(f"export manifest missing {variant_label} variant")
    parent_sha = _resolve_parent_fp32_sha(export_dir, fp32, for_label=display_name)
    status, variant_sha = _variant_file_sha(
        export_dir, variant, label=variant_label, display_name=display_name
    )
    return manifest_path, manifest, variant, parent_sha, status, variant_sha


def _write_validation_json(path: Path, result: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return path


def _quant_binding_base(
    *,
    manifest: dict[str, Any],
    variant: dict[str, Any],
    variant_sha: str,
    parent_sha: str,
    variant_label: str,
) -> dict[str, Any]:
    return {
        "model_sha256": variant_sha,
        "corpus_eval_sha256": str(
            ((manifest.get("corpus") or {}).get("evaluation") or {}).get("sha256") or ""
        ),
        "hub_revision": str(manifest.get("hub_revision") or ""),
        "candidate": str(manifest.get("candidate") or ""),
        "recipe_id": str(variant.get("recipe_id") or ""),
        "variant_label": variant_label,
        "parent_fp32_sha256": parent_sha,
    }


def build_int4_validation_v2(
    *,
    binding: dict[str, Any],
    parity_vs_onnx_fp32: dict[str, Any] | None,
    parity_vs_pytorch_fp32: dict[str, Any] | None,
    quantization_coverage: dict[str, Any] | None,
    parent_fp32_sha256: str,
    variant_status: str,
    source_export_manifest_path: str | None = None,
) -> dict[str, Any]:
    """INT4 sidecar: own identity, finite/shape checks, coverage; no task/allclose gate."""
    runtime = evaluate_int4_runtime_checks(parity_vs_onnx_fp32)
    reasons: list[str] = []
    if variant_status != "ok":
        reasons.append(f"int4_variant_status={variant_status}")
    if not parent_fp32_sha256:
        reasons.append("parent_fp32_sha256_missing")
    if binding.get("model_sha256") in {None, ""}:
        reasons.append("int4_model_sha256_missing")
    if binding.get("recipe_id") in {None, ""}:
        reasons.append("int4_recipe_id_missing")
    if binding.get("candidate") in {None, ""}:
        reasons.append("candidate_missing")
    if binding.get("hub_revision") in {None, ""}:
        reasons.append("hub_revision_missing")
    if not quantization_coverage:
        reasons.append("quantization_coverage_missing")
    nbits = int((quantization_coverage or {}).get("matmul_nbits_nodes") or 0)
    if nbits < 1:
        reasons.append("matmul_nbits_nodes_missing_or_zero")
    if not runtime["passed"]:
        reasons.extend(runtime.get("not_passed_reasons") or [])
    if binding.get("parent_fp32_sha256") not in {None, "", parent_fp32_sha256}:
        reasons.append("parent_fp32_sha256_binding_mismatch")

    passed = variant_status == "ok" and runtime["passed"] and not reasons
    return {
        "schema_version": SCHEMA_VERSION,
        "variant_label": "int4_weight_only",
        "passed": passed,
        "not_passed_reasons": [] if passed else reasons,
        "binding": {
            **binding,
            "variant_label": "int4_weight_only",
            "parent_fp32_sha256": parent_fp32_sha256,
        },
        "runtime_checks": runtime,
        "quantization_coverage": quantization_coverage or {},
        "numeric_errors": {
            "vs_onnx_fp32": _parity_numeric_block(parity_vs_onnx_fp32),
            "vs_pytorch_fp32": _parity_numeric_block(parity_vs_pytorch_fp32),
        },
        "task_success_required": False,
        "fp32_allclose_required": False,
        "fp32_validation_sidecar_authorizes_int4": False,
        "full_model_int4_claim": False,
        "native_int4_execution_claim": False,
        "source_export_manifest_path": source_export_manifest_path,
        "note": (
            "INT4 validation is separate from FP32. Requires runnable MatMulNBits "
            "weight-only graph with finite matching shapes vs the FP32 parent and "
            "recorded quantization coverage. Does not require FP32-level allclose or "
            "task success; not a native INT4 activation execution claim."
        ),
    }


def assert_int4_validation_acceptable(
    result: dict[str, Any],
    *,
    model_sha256: str,
    parent_fp32_sha256: str | None = None,
) -> None:
    if not isinstance(result, dict):
        raise ValueError("export_validation_int4 must be a JSON object")
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"export_validation_int4 schema_version must be {SCHEMA_VERSION}; "
            f"got {result.get('schema_version')!r}"
        )
    if result.get("variant_label") != "int4_weight_only":
        raise ValueError("INT4 resolve requires variant_label=int4_weight_only validation sidecar")
    if result.get("passed") is not True:
        raise ValueError(
            "export_validation_int4.passed is not true; refuse unvalidated INT4 runtime "
            f"(reasons={result.get('not_passed_reasons')})"
        )
    binding = result.get("binding") or {}
    if binding.get("model_sha256") != model_sha256:
        raise ValueError(
            "INT4 validation model_sha256 mismatch: "
            f"validation={binding.get('model_sha256')!r} model={model_sha256!r}"
        )
    if not binding.get("recipe_id"):
        raise ValueError("INT4 validation binding.recipe_id missing")
    if not binding.get("parent_fp32_sha256"):
        raise ValueError("INT4 validation binding.parent_fp32_sha256 missing")
    if parent_fp32_sha256 is not None and binding.get("parent_fp32_sha256") != parent_fp32_sha256:
        raise ValueError("INT4 parent_fp32_sha256 mismatch")
    runtime = result.get("runtime_checks") or {}
    if runtime.get("passed") is not True:
        raise ValueError("INT4 runtime_checks.passed is not true")
    if result.get("fp32_validation_sidecar_authorizes_int4") is True:
        raise ValueError("INT4 must not be authorized by the FP32 validation sidecar")
    coverage = result.get("quantization_coverage") or {}
    if int(coverage.get("matmul_nbits_nodes") or 0) < 1:
        raise ValueError("INT4 validation requires matmul_nbits_nodes > 0")
    if result.get("full_model_int4_claim") is True:
        raise ValueError("INT4 full_model_int4_claim must be false")
    if result.get("native_int4_execution_claim") is True:
        raise ValueError("INT4 native_int4_execution_claim must be false")


def write_int4_validation_from_export(
    export_dir: Path,
    *,
    output_path: Path | None = None,
) -> Path:
    """Write INT4 sidecar from export manifest comparisons."""
    export_dir = Path(export_dir).resolve()
    manifest_path, manifest, int4, parent_sha, status, int4_sha = _load_quant_export_context(
        export_dir, variant_label="int4_weight_only", display_name="INT4"
    )
    comparisons = manifest.get("comparisons") or {}
    parity_onnx = comparisons.get("int4_weight_only_vs_onnx_fp32")
    parity_pt = comparisons.get("int4_weight_only_vs_pytorch_fp32")
    coverage = int4.get("quantization_coverage")
    result = build_int4_validation_v2(
        binding=_quant_binding_base(
            manifest=manifest,
            variant=int4,
            variant_sha=int4_sha,
            parent_sha=parent_sha,
            variant_label="int4_weight_only",
        ),
        parity_vs_onnx_fp32=parity_onnx if isinstance(parity_onnx, dict) else None,
        parity_vs_pytorch_fp32=parity_pt if isinstance(parity_pt, dict) else None,
        quantization_coverage=coverage if isinstance(coverage, dict) else None,
        parent_fp32_sha256=parent_sha,
        variant_status=status,
        source_export_manifest_path=str(manifest_path),
    )
    out = (
        Path(output_path).resolve()
        if output_path is not None
        else (export_dir / INT4_VALIDATION_FILENAME)
    )
    return _write_validation_json(out, result)


def build_validation_v2_from_artifacts(
    *,
    export_dir: Path,
    diagnosis_path: Path,
    repo_root: Path,
    output_path: Path | None = None,
) -> Path:
    """Assemble the schema 2 validation from an existing export manifest + closed-loop diagnosis JSON."""
    export_dir = Path(export_dir).resolve()
    diagnosis_path = Path(diagnosis_path).resolve()
    manifest_path = export_dir / "export_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing export_manifest.json in {export_dir}")
    if not diagnosis_path.is_file():
        raise FileNotFoundError(f"missing diagnosis: {diagnosis_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))

    variants = manifest.get("variants") or []
    fp32 = next((v for v in variants if v.get("label") == "onnx_fp32"), None)
    if fp32 is None or fp32.get("status") != "ok":
        raise ValueError("export manifest missing ok onnx_fp32 variant")
    model_name = Path(fp32.get("path") or "act_chunk_fp32.onnx").name
    model_path = export_dir / model_name
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    model_sha = sha256_file(model_path)
    graph_sha = (fp32.get("graph") or {}).get("sha256")
    if graph_sha and graph_sha != model_sha:
        raise ValueError(f"manifest graph sha256 {graph_sha} does not match file {model_sha}")

    candidate = str(manifest.get("candidate") or "")
    hub_revision = str(manifest.get("hub_revision") or "")
    recipe_id = str(fp32.get("recipe_id") or "")
    if not candidate or not hub_revision or not recipe_id:
        raise ValueError(
            "export manifest missing required identity "
            "(candidate/hub_revision/recipe_id); refuse to invent fields"
        )

    diag_binding = diagnosis.get("artifact_binding") or {}
    diag_sha = diag_binding.get("model_sha256")
    if not diag_sha:
        raise ValueError(
            "diagnosis missing artifact_binding.model_sha256; refuse rebound/stale diagnosis"
        )
    if str(diag_sha) != model_sha:
        raise ValueError(
            "diagnosis artifact_binding.model_sha256 does not match ONNX on disk: "
            f"diagnosis={diag_sha} file={model_sha}"
        )
    if diag_binding.get("candidate") and str(diag_binding["candidate"]) != candidate:
        raise ValueError(
            "diagnosis candidate does not match export manifest: "
            f"diagnosis={diag_binding.get('candidate')!r} manifest={candidate!r}"
        )
    if diag_binding.get("hub_revision") and str(diag_binding["hub_revision"]) != hub_revision:
        raise ValueError(
            "diagnosis hub_revision does not match export manifest: "
            f"diagnosis={diag_binding.get('hub_revision')!r} manifest={hub_revision!r}"
        )
    if diag_binding.get("recipe_id") and str(diag_binding["recipe_id"]) != recipe_id:
        raise ValueError(
            "diagnosis recipe_id does not match export manifest: "
            f"diagnosis={diag_binding.get('recipe_id')!r} manifest={recipe_id!r}"
        )

    corpus = (manifest.get("corpus") or {}).get("evaluation") or {}
    corpus_sha = corpus.get("sha256")
    if not corpus_sha:
        raise ValueError("export manifest missing corpus.evaluation.sha256")
    if diag_binding.get("corpus_eval_sha256") and str(diag_binding["corpus_eval_sha256"]) != str(
        corpus_sha
    ):
        raise ValueError(
            "diagnosis corpus_eval_sha256 does not match export manifest evaluation corpus"
        )
    corpus_meta = (manifest.get("corpus") or {}).get("meta") or {}
    suitable = bool(corpus_meta.get("suitable_as_deployment_evidence"))

    open_loop = (manifest.get("comparisons") or {}).get("onnx_fp32_vs_pytorch_fp32")
    shadow = list(diagnosis.get("same_observation_shadow") or [])
    independent = (diagnosis.get("independent_rollouts") or {}).get("seeds") or []

    code = collect_code_revision(repo_root)
    binding = ValidationBinding(
        model_sha256=model_sha,
        corpus_eval_sha256=str(corpus_sha),
        hub_revision=hub_revision,
        candidate=candidate,
        recipe_id=recipe_id,
        variant_label=str(fp32.get("label") or "onnx_fp32"),
        code_revision=code["code_revision"],
        code_dirty=code["code_dirty"],
        uncommitted_paths=(
            None if code["uncommitted_paths"] is None else tuple(code["uncommitted_paths"])
        ),
    )
    result = build_export_validation_v2(
        binding=binding,
        open_loop_report=open_loop if isinstance(open_loop, dict) else None,
        shadow_reports=shadow,
        independent_seed_reports=list(independent),
        corpus_suitable_as_deployment_evidence=suitable,
        diagnosis_path=str(diagnosis_path),
        source_export_manifest_path=str(manifest_path),
    )
    result["diagnosis_artifact_binding"] = diag_binding

    out = (
        Path(output_path).resolve()
        if output_path is not None
        else (export_dir / VALIDATION_FILENAME)
    )
    if out.resolve() == manifest_path.resolve():
        raise ValueError("refuse to overwrite export_manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return out


def load_export_validation_v2(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"export_validation must be a JSON object: {path}")
    return payload


def find_export_validation_v2(model_path: Path, *, variant_label: str | None = None) -> Path | None:
    parent = Path(model_path).resolve().parent
    label = (variant_label or "").lower()
    name = Path(model_path).name.lower()
    if label == "int8" or "int8" in name:
        return resolve_artifact(parent, INT8_VALIDATION_FILENAME)
    if label == "int4_weight_only" or "int4" in name:
        return resolve_artifact(parent, INT4_VALIDATION_FILENAME)
    return resolve_artifact(parent, VALIDATION_FILENAME)


def write_int8_validation_from_export(
    export_dir: Path,
    *,
    output_path: Path | None = None,
) -> Path:
    """Write INT8 sidecar from export manifest comparisons."""
    export_dir = Path(export_dir).resolve()
    manifest_path, manifest, int8, parent_sha, status, int8_sha = _load_quant_export_context(
        export_dir, variant_label="int8", display_name="INT8"
    )
    calib = (manifest.get("corpus") or {}).get("calibration") or {}
    calib_sha = str(calib.get("sha256") or "")
    parity = (manifest.get("comparisons") or {}).get("int8_vs_onnx_fp32")
    coverage = int8.get("quantization_coverage")
    binding = _quant_binding_base(
        manifest=manifest,
        variant=int8,
        variant_sha=int8_sha,
        parent_sha=parent_sha,
        variant_label="int8",
    )
    binding["calibration_sha256"] = calib_sha
    result = build_int8_validation_v2(
        binding=binding,
        parity_vs_fp32=parity if isinstance(parity, dict) else None,
        quantization_coverage=coverage if isinstance(coverage, dict) else None,
        parent_fp32_sha256=parent_sha,
        calibration_sha256=calib_sha,
        variant_status=status,
        source_export_manifest_path=str(manifest_path),
    )
    out = (
        Path(output_path).resolve()
        if output_path is not None
        else (export_dir / INT8_VALIDATION_FILENAME)
    )
    return _write_validation_json(out, result)


__all__ = [
    "INT4_VALIDATION_FILENAME",
    "INT8_VALIDATION_FILENAME",
    "SCHEMA_VERSION",
    "VALIDATION_FILENAME",
    "ValidationBinding",
    "assert_int4_validation_acceptable",
    "assert_int8_validation_acceptable",
    "assert_validation_v2_acceptable",
    "build_export_validation_v2",
    "build_int4_validation_v2",
    "build_int8_validation_v2",
    "build_validation_v2_from_artifacts",
    "collect_code_revision",
    "evaluate_closed_loop_task_outcomes",
    "evaluate_independent_trajectory_diagnostic",
    "evaluate_int4_runtime_checks",
    "evaluate_int8_runtime_checks",
    "evaluate_numerical_parity",
    "find_export_validation_v2",
    "load_export_validation_v2",
    "write_int4_validation_from_export",
    "write_int8_validation_from_export",
]
