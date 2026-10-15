"""Versioned ACT export validation (schema v2).

Separates:
  1. numerical_parity — same-input open-loop corpus + fresh-chunk shadow checks
  2. closed_loop_task_outcomes — completed baseline rollouts with recorded outcomes
  3. independent_trajectory — action/length similarity diagnostics (not a pass gate)

Independent trajectory equality is recorded but does not fail deployment validation.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from breakpoint_eval.act_export import sha256_file
from breakpoint_eval.export_tolerances import (
    ONNX_FP32_CLOSED_LOOP_ATOL,
    ONNX_FP32_CLOSED_LOOP_RTOL,
    ONNX_FP32_PARITY_ACTION_UNIT_MEAN_ABS,
    ONNX_FP32_PARITY_ATOL,
    ONNX_FP32_PARITY_MAX_ABS,
    ONNX_FP32_PARITY_MEAN_ABS,
    ONNX_FP32_PARITY_RTOL,
)

SCHEMA_VERSION = 2
VALIDATION_FILENAME = "export_validation_v2.json"


@dataclass(frozen=True)
class ValidationBinding:
    model_sha256: str
    corpus_eval_sha256: str
    hub_revision: str
    candidate: str
    recipe_id: str
    variant_label: str
    code_revision: str
    code_dirty: bool
    uncommitted_paths: tuple[str, ...]


def collect_code_revision(repo_root: Path) -> dict[str, Any]:
    """Record HEAD and uncommitted paths (fail-open to empty if not a git repo)."""
    root = Path(repo_root).resolve()
    revision = None
    dirty_paths: list[str] = []
    try:
        rev = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        revision = rev.stdout.strip() or None
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        for line in status.stdout.splitlines():
            path = line[3:].strip() if len(line) >= 4 else line.strip()
            if path:
                dirty_paths.append(path)
    except (OSError, subprocess.CalledProcessError):
        revision = None
        dirty_paths = []
    return {
        "code_revision": revision,
        "code_dirty": bool(dirty_paths),
        "uncommitted_paths": dirty_paths,
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
    """Same-input numerical parity only (corpus open-loop + fresh-chunk shadow)."""
    reasons: list[str] = []
    if not open_loop_report:
        reasons.append("open_loop_report_missing")
        open_passed = False
    else:
        if open_loop_report.get("passed") is not True:
            reasons.append("open_loop_parity_failed")
        if open_loop_report.get("finite_ok") is False:
            reasons.append("open_loop_nonfinite")
        if open_loop_report.get("shapes_ok") is False:
            reasons.append("open_loop_shape_mismatch")
        open_passed = open_loop_report.get("passed") is True and open_loop_report.get(
            "finite_ok", True
        )

    if not shadow_reports:
        reasons.append("shadow_reports_missing")
        shadow_passed = False
        fresh_ok = False
    else:
        shadow_passed = True
        fresh_ok = True
        for report in shadow_reports:
            if report.get("passed_atol") is not True:
                shadow_passed = False
                reasons.append(f"shadow_seed_{report.get('seed')}_atol_failed")
            if report.get("compares_every_control_step") is not True:
                fresh_ok = False
                reasons.append(f"shadow_seed_{report.get('seed')}_incomplete_steps")
            if report.get("covers_every_fresh_chunk_refresh") is not True:
                fresh_ok = False
                reasons.append(f"shadow_seed_{report.get('seed')}_missing_fresh_chunk_coverage")
            if report.get("covers_every_inference_call") is not True:
                fresh_ok = False
                reasons.append(f"shadow_seed_{report.get('seed')}_missing_inference_coverage")
            if report.get("shadow_controls_simulation") is True:
                reasons.append(f"shadow_seed_{report.get('seed')}_controlled_sim")
                shadow_passed = False
            n_fresh = report.get("n_fresh_chunks")
            n_infer = report.get("n_inference_calls")
            if n_fresh is None or n_infer is None or int(n_fresh) < 1:
                fresh_ok = False
                reasons.append(f"shadow_seed_{report.get('seed')}_no_fresh_chunks")
            elif int(n_fresh) != int(n_infer):
                # ACT should request fresh chunk iff inference call.
                reasons.append(f"shadow_seed_{report.get('seed')}_fresh_infer_mismatch")
                fresh_ok = False

    passed = open_passed and shadow_passed and fresh_ok
    return {
        "kind": "same_input_numerical_parity",
        "passed": passed,
        "open_loop_passed": open_passed,
        "shadow_fresh_chunk_passed": bool(shadow_passed and fresh_ok),
        "not_passed_reasons": [] if passed else reasons,
        "note": (
            "Requires held-out corpus open-loop chunk parity and same-observation "
            "shadow compares that cover every fresh action-chunk inference. "
            "Cached queued-action traces are not numerical export evidence."
        ),
    }


def evaluate_closed_loop_task_outcomes(
    seed_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Require completed paired rollouts with recorded outcomes (smoke when n=2)."""
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
        ):
            if key not in report or report.get(key) is None:
                missing.append(key)
        if missing:
            reasons.append(f"seed_{seed}_missing_outcome_fields:{','.join(missing)}")
        if pt_outcome in {None, "runtime_error"} or ort_outcome in {None, "runtime_error"}:
            reasons.append(f"seed_{seed}_runtime_or_missing_outcome")
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
            }
        )

    completed = all(
        s.get("pytorch_outcome") not in {None, "runtime_error"}
        and s.get("onnx_outcome") not in {None, "runtime_error"}
        and s.get("pytorch_success") is not None
        and s.get("onnx_success") is not None
        for s in seed_summaries
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
        "not_passed_reasons": reasons,
        "note": (
            "Requires completed baseline rollouts with recorded success/terminated/"
            "truncated fields. Two-seed results are a smoke check, not statistical "
            "evidence of performance equivalence."
        ),
    }


def evaluate_independent_trajectory_diagnostic(
    seed_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Action/length similarity across independent resets — diagnostic only."""
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
            "uncommitted_paths": list(binding.uncommitted_paths),
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


def assert_validation_v2_acceptable(
    result: dict[str, Any],
    *,
    model_sha256: str,
    corpus_eval_sha256: str | None,
    hub_revision: str | None = None,
) -> None:
    """Fail-closed acceptance for deployment runtime binding."""
    if not isinstance(result, dict):
        raise ValueError("export_validation_v2 must be a JSON object")
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"export_validation_v2 schema_version must be {SCHEMA_VERSION}; "
            f"got {result.get('schema_version')!r}"
        )
    if result.get("passed") is not True:
        raise ValueError(
            "export_validation_v2.passed is not true; refuse unvalidated deployment runtime "
            f"(reasons={result.get('not_passed_reasons')})"
        )
    binding = result.get("binding") or {}
    if binding.get("model_sha256") != model_sha256:
        raise ValueError(
            "export_validation_v2 model_sha256 mismatch: "
            f"validation={binding.get('model_sha256')!r} model={model_sha256!r}"
        )
    if corpus_eval_sha256 is not None and binding.get("corpus_eval_sha256") != corpus_eval_sha256:
        raise ValueError(
            "export_validation_v2 corpus_eval_sha256 mismatch: "
            f"validation={binding.get('corpus_eval_sha256')!r} "
            f"manifest={corpus_eval_sha256!r}"
        )
    if (
        hub_revision is not None
        and binding.get("hub_revision")
        and binding.get("hub_revision") != hub_revision
    ):
        raise ValueError(
            "export_validation_v2 hub_revision mismatch: "
            f"validation={binding.get('hub_revision')!r} manifest={hub_revision!r}"
        )
    numerical = result.get("numerical_parity") or {}
    if numerical.get("passed") is not True:
        raise ValueError("export_validation_v2 numerical_parity.passed is not true")
    tasks = result.get("closed_loop_task_outcomes") or {}
    if tasks.get("passed") is not True or tasks.get("completed") is not True:
        raise ValueError("export_validation_v2 closed_loop_task_outcomes incomplete or failed")
    if result.get("corpus_suitable_as_deployment_evidence") is False:
        raise ValueError("export_validation_v2 corpus is not suitable as deployment evidence")


def build_validation_v2_from_artifacts(
    *,
    export_dir: Path,
    diagnosis_path: Path,
    repo_root: Path,
    output_path: Path | None = None,
) -> Path:
    """Assemble v2 from an existing export manifest + closed-loop diagnosis JSON."""
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

    corpus = (manifest.get("corpus") or {}).get("evaluation") or {}
    corpus_sha = corpus.get("sha256")
    if not corpus_sha:
        raise ValueError("export manifest missing corpus.evaluation.sha256")
    corpus_meta = (manifest.get("corpus") or {}).get("meta") or {}
    suitable = bool(corpus_meta.get("suitable_as_deployment_evidence"))

    open_loop = (manifest.get("comparisons") or {}).get("onnx_fp32_vs_pytorch_fp32")
    shadow = list(diagnosis.get("same_observation_shadow") or [])
    independent = (diagnosis.get("independent_rollouts") or {}).get("seeds") or []

    code = collect_code_revision(repo_root)
    binding = ValidationBinding(
        model_sha256=model_sha,
        corpus_eval_sha256=str(corpus_sha),
        hub_revision=str(manifest.get("hub_revision") or ""),
        candidate=str(manifest.get("candidate") or ""),
        recipe_id=str(fp32.get("recipe_id") or ""),
        variant_label=str(fp32.get("label") or ""),
        code_revision=str(code.get("code_revision") or ""),
        code_dirty=bool(code.get("code_dirty")),
        uncommitted_paths=tuple(code.get("uncommitted_paths") or ()),
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
        raise ValueError(f"export_validation_v2 must be a JSON object: {path}")
    return payload


def find_export_validation_v2(model_path: Path) -> Path | None:
    sibling = Path(model_path).resolve().parent / VALIDATION_FILENAME
    return sibling if sibling.is_file() else None


__all__ = [
    "SCHEMA_VERSION",
    "VALIDATION_FILENAME",
    "ValidationBinding",
    "assert_validation_v2_acceptable",
    "build_export_validation_v2",
    "build_validation_v2_from_artifacts",
    "collect_code_revision",
    "evaluate_closed_loop_task_outcomes",
    "evaluate_independent_trajectory_diagnostic",
    "evaluate_numerical_parity",
    "find_export_validation_v2",
    "load_export_validation_v2",
]
