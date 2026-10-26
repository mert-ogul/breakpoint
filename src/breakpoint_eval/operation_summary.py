"""Condense saved export, validation, diagnosis and profile JSON into the CLI console summary.

Pure functions over already-saved payloads; the CLI prints the result and uses the exit-code
helper. No third-party imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def export_validation_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    binding = payload.get("binding") or {}
    return {
        "validation": str(path),
        "schema_version": payload.get("schema_version"),
        "passed": payload.get("passed"),
        "numerical_parity_passed": (payload.get("numerical_parity") or {}).get("passed"),
        "task_outcomes_passed": (payload.get("closed_loop_task_outcomes") or {}).get("passed"),
        "independent_trajectory_similar": (payload.get("independent_trajectory") or {}).get(
            "similar"
        ),
        "independent_trajectory_gates_deployment": (
            payload.get("independent_trajectory") or {}
        ).get("gates_deployment_validation"),
        "not_passed_reasons": payload.get("not_passed_reasons"),
        "model_sha256": binding.get("model_sha256"),
        "corpus_eval_sha256": binding.get("corpus_eval_sha256"),
        "code_revision": binding.get("code_revision"),
        "code_dirty": binding.get("code_dirty"),
    }


def closed_loop_diagnosis_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    independent = payload.get("independent_rollouts") or {}
    shadow = payload.get("same_observation_shadow") or []
    return {
        "diagnosis": str(path),
        "independent_passed": independent.get("passed"),
        "independent_failure_kind": independent.get("failure_kind"),
        "independent_seeds": [
            {
                "seed": s.get("seed"),
                "passed": s.get("passed"),
                "first_divergence_step": s.get("first_divergence_step"),
                "max_abs_error": s.get("max_abs_error"),
                "pytorch_n_steps": s.get("pytorch_n_steps"),
                "onnx_n_steps": s.get("onnx_n_steps"),
                "outcomes": [s.get("pytorch_outcome"), s.get("onnx_outcome")],
            }
            for s in (independent.get("seeds") or [])
        ],
        "shadow_max_abs_error": max((s.get("max_abs_error") or 0.0) for s in shadow)
        if shadow
        else None,
        "repeatability_all_passed": all(
            r.get("passed") for r in (payload.get("pytorch_repeatability") or [])
        ),
        "validation_gate_unchanged": (payload.get("interpretation") or {}).get(
            "validation_gate_unchanged"
        ),
    }


def export_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    validation = payload.get("export_validation") or {}
    closed = payload.get("closed_loop_onnx_fp32_vs_pytorch") or {}
    return {
        "manifest": str(path),
        "onnx_fp32_parity_passed": payload.get("comparisons", {})
        .get("onnx_fp32_vs_pytorch_fp32", {})
        .get("passed"),
        "export_validation_passed": validation.get("passed"),
        "diagnostic_parity_ok": validation.get("diagnostic_parity_ok"),
        "closed_loop_completed": validation.get("closed_loop_completed"),
        "closed_loop_passed": validation.get("closed_loop_passed"),
        "closed_loop_failure_kind": validation.get("closed_loop_failure_kind")
        or closed.get("failure_kind"),
        "corpus_suitable_as_deployment_evidence": validation.get(
            "corpus_suitable_as_deployment_evidence"
        ),
        "not_passed_reasons": validation.get("not_passed_reasons"),
        "variants": [
            {"label": v["label"], "status": v["status"], "recipe_id": v["recipe_id"]}
            for v in payload.get("variants", [])
        ],
        "closed_loop_status": closed.get("status"),
    }


def export_exit_code(payload: dict[str, Any]) -> int:
    """1 when open-loop parity failed or a requested closed-loop check did not pass.

    Synthetic or closed-loop-skipped diagnostic exports may leave ``passed`` false without
    failing the process.
    """
    validation = payload.get("export_validation") or {}
    open_loop_ok = bool(
        payload.get("comparisons", {}).get("onnx_fp32_vs_pytorch_fp32", {}).get("passed")
    )
    if validation.get("open_loop_fp32_parity_passed") is False or not open_loop_ok:
        return 1
    if validation.get("closed_loop_requested"):
        if validation.get("closed_loop_completed") is not True:
            return 1
        if validation.get("closed_loop_passed") is not True:
            return 1
    return 0


def profile_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    measurements = payload.get("measurements", {})
    summary = measurements.get("summary", {})
    return {
        "profile": str(path),
        "device_label": payload.get("device_label"),
        "n_warmup": measurements.get("n_warmup"),
        "n_measured": summary.get("count"),
        "median_s": summary.get("median_s"),
        "p95_s": summary.get("p95_s"),
        "p99_s": summary.get("p99_s"),
        "validation_passed": payload.get("output_validation", {}).get("passed"),
        "timed_boundary": payload.get("timed_boundary"),
    }
