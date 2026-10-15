"""Parity checks with tolerances declared in export_tolerances (not post-hoc)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from breakpoint_eval.act_export import (
    ACTChunkModule,
    OrtActChunkSession,
    OrtPolicyAdapter,
    pytorch_chunk,
)
from breakpoint_eval.export_corpus import CorpusSample
from breakpoint_eval.export_tolerances import (
    ONNX_FP32_CLOSED_LOOP_ATOL,
    ONNX_FP32_CLOSED_LOOP_RTOL,
    ONNX_FP32_PARITY_ACTION_UNIT_MEAN_ABS,
    ONNX_FP32_PARITY_ATOL,
    ONNX_FP32_PARITY_MAX_ABS,
    ONNX_FP32_PARITY_MEAN_ABS,
    ONNX_FP32_PARITY_RTOL,
)
from breakpoint_eval.policy_adapter import PolicyAdapter

# Independent episodes (separate resets) compared by applied actions. This is not
# same-observation numerical export parity; tiny early diffs can compound.
COMPARISON_KIND_INDEPENDENT_ROLLOUT = "independent_rollout_action_trace"
FAILURE_KIND_TRAJECTORY_DIVERGENCE = "independent_trajectory_divergence"
FAILURE_KIND_LENGTH_MISMATCH = "episode_length_mismatch"
FAILURE_KIND_RUNTIME = "runtime_error"
FAILURE_KIND_EMPTY = "no_paired_actions"
FAILURE_KIND_SHAPE = "action_shape_mismatch"


@dataclass
class ParityTolerances:
    atol: float = ONNX_FP32_PARITY_ATOL
    rtol: float = ONNX_FP32_PARITY_RTOL
    max_abs: float = ONNX_FP32_PARITY_MAX_ABS
    mean_abs: float = ONNX_FP32_PARITY_MEAN_ABS
    action_unit_mean_abs: float = ONNX_FP32_PARITY_ACTION_UNIT_MEAN_ABS

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class SampleParity:
    seed: int
    sample_id: str
    shape_ok: bool
    finite_ok: bool
    max_abs_error: float
    mean_abs_error: float
    action_unit_mean_abs_error: float
    allclose: bool


@dataclass
class CorpusParityReport:
    reference: str
    candidate: str
    tolerances: dict[str, float]
    n_samples: int
    shapes_ok: bool
    finite_ok: bool
    max_abs_error: float
    mean_abs_error: float
    action_unit_mean_abs_error: float
    passed: bool
    samples: list[SampleParity] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def _action_unit_mean_abs(err: np.ndarray) -> float:
    # err: (B, T, A) or (T, A)
    if err.ndim == 3:
        err = err[0]
    # mean over time, then mean over action dims
    return float(np.mean(np.mean(np.abs(err), axis=0)))


def compare_chunks(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
    tolerances: ParityTolerances,
    sample_id: str | None = None,
) -> SampleParity:
    shape_ok = reference.shape == candidate.shape
    finite_ok = bool(np.isfinite(reference).all() and np.isfinite(candidate).all())
    if shape_ok:
        err = candidate.astype(np.float64) - reference.astype(np.float64)
        max_abs = float(np.max(np.abs(err)))
        mean_abs = float(np.mean(np.abs(err)))
        action_unit = _action_unit_mean_abs(err)
        allclose = bool(
            np.allclose(candidate, reference, atol=tolerances.atol, rtol=tolerances.rtol)
        )
    else:
        max_abs = float("inf")
        mean_abs = float("inf")
        action_unit = float("inf")
        allclose = False
    return SampleParity(
        seed=seed,
        sample_id=sample_id if sample_id is not None else f"seed{seed}_t0",
        shape_ok=shape_ok,
        finite_ok=finite_ok,
        max_abs_error=max_abs,
        mean_abs_error=mean_abs,
        action_unit_mean_abs_error=action_unit,
        allclose=allclose,
    )


def evaluate_corpus_parity(
    *,
    reference_chunks: list[np.ndarray],
    candidate_chunks: list[np.ndarray],
    seeds: list[int],
    reference_label: str,
    candidate_label: str,
    tolerances: ParityTolerances | None = None,
    enforce_fp32_gates: bool = True,
    sample_ids: list[str] | None = None,
) -> CorpusParityReport:
    """Compare chunk lists. When enforce_fp32_gates=False, still report errors."""
    tol = tolerances or ParityTolerances()
    if not (len(reference_chunks) == len(candidate_chunks) == len(seeds)):
        raise ValueError("reference, candidate, and seeds lengths must match")
    if sample_ids is None:
        sample_ids = [f"seed{seed}_t0" for seed in seeds]
    if len(sample_ids) != len(seeds):
        raise ValueError("sample_ids length must match seeds")
    samples = [
        compare_chunks(ref, cand, seed=seed, sample_id=sample_id, tolerances=tol)
        for ref, cand, seed, sample_id in zip(
            reference_chunks, candidate_chunks, seeds, sample_ids, strict=True
        )
    ]
    shapes_ok = all(s.shape_ok for s in samples)
    finite_ok = all(s.finite_ok for s in samples)
    max_abs = max(s.max_abs_error for s in samples) if samples else float("nan")
    mean_abs = float(np.mean([s.mean_abs_error for s in samples])) if samples else float("nan")
    action_unit = (
        float(np.mean([s.action_unit_mean_abs_error for s in samples])) if samples else float("nan")
    )
    if enforce_fp32_gates:
        passed = (
            shapes_ok
            and finite_ok
            and max_abs <= tol.max_abs
            and mean_abs <= tol.mean_abs
            and action_unit <= tol.action_unit_mean_abs
            and all(s.allclose for s in samples)
        )
    else:
        passed = shapes_ok and finite_ok
    return CorpusParityReport(
        reference=reference_label,
        candidate=candidate_label,
        tolerances=tol.to_dict(),
        n_samples=len(samples),
        shapes_ok=shapes_ok,
        finite_ok=finite_ok,
        max_abs_error=max_abs,
        mean_abs_error=mean_abs,
        action_unit_mean_abs_error=action_unit,
        passed=passed,
        samples=samples,
        note=(
            "FP32 gates enforced"
            if enforce_fp32_gates
            else "Reporting only (quantized variants are not required to meet FP32 gates)"
        ),
    )


def run_pytorch_chunks(
    module: ACTChunkModule,
    samples: list[CorpusSample],
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for sample in samples:
        images = torch.from_numpy(sample.images_top)
        state = torch.from_numpy(sample.state)
        out.append(pytorch_chunk(module, images, state))
    return out


def run_ort_chunks(
    session: OrtActChunkSession,
    samples: list[CorpusSample],
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for sample in samples:
        result = session.run(sample.images_top, sample.state)
        out.append(result.action_chunk)
    return out


def _closed_loop_tolerances() -> dict[str, float]:
    return {
        "atol": ONNX_FP32_CLOSED_LOOP_ATOL,
        "rtol": ONNX_FP32_CLOSED_LOOP_RTOL,
    }


@dataclass
class ClosedLoopSeedReport:
    """One seed's independent PyTorch vs ONNX episode comparison."""

    seed: int
    pytorch_outcome: str
    onnx_outcome: str
    pytorch_n_steps: int
    onnx_n_steps: int
    length_mismatch: bool
    unmatched_tail_steps: int
    n_steps_compared: int
    max_abs_error: float | None
    first_divergence_step: int | None
    first_divergence_max_abs_error: float | None
    passed: bool
    failure_kind: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClosedLoopCompareReport:
    """Aggregate closed-loop report across seeds (independent rollouts)."""

    status: str
    detail: str | None
    n_steps_compared: int
    max_abs_error: float | None
    passed: bool | None
    tolerances: dict[str, float]
    comparison_kind: str = COMPARISON_KIND_INDEPENDENT_ROLLOUT
    failure_kind: str | None = None
    seeds: list[ClosedLoopSeedReport] = field(default_factory=list)
    length_mismatch_any: bool = False
    # Retained for older consumers; prefer per-seed fields.
    pytorch_outcome: str | None = None
    onnx_outcome: str | None = None
    note: str = (
        "Compares independently controlled episodes (separate resets) with the same "
        "seeds. This is not same-observation numerical export parity; open-loop corpus "
        "parity covers identical inputs."
    )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def compare_action_traces_with_tails(
    pytorch_actions: list[np.ndarray],
    onnx_actions: list[np.ndarray],
    *,
    atol: float | None = None,
    rtol: float | None = None,
) -> dict[str, Any]:
    """Compare paired action lists; never silently ignore unmatched tails."""
    atol = ONNX_FP32_CLOSED_LOOP_ATOL if atol is None else atol
    rtol = ONNX_FP32_CLOSED_LOOP_RTOL if rtol is None else rtol
    n_pt = len(pytorch_actions)
    n_ort = len(onnx_actions)
    n = min(n_pt, n_ort)
    unmatched = abs(n_pt - n_ort)
    length_mismatch = unmatched > 0
    if n == 0:
        return {
            "n_steps_compared": 0,
            "max_abs_error": None,
            "first_divergence_step": None,
            "first_divergence_max_abs_error": None,
            "length_mismatch": length_mismatch,
            "unmatched_tail_steps": unmatched,
            "passed": False,
            "failure_kind": FAILURE_KIND_EMPTY
            if n_pt == 0 and n_ort == 0
            else FAILURE_KIND_LENGTH_MISMATCH,
            "detail": "no paired actions" if n == 0 else None,
        }
    max_abs = 0.0
    first_div: int | None = None
    first_div_err: float | None = None
    for i, (a, b) in enumerate(zip(pytorch_actions[:n], onnx_actions[:n], strict=True)):
        if a.shape != b.shape:
            return {
                "n_steps_compared": i,
                "max_abs_error": max_abs if i else None,
                "first_divergence_step": i,
                "first_divergence_max_abs_error": None,
                "length_mismatch": length_mismatch,
                "unmatched_tail_steps": unmatched,
                "passed": False,
                "failure_kind": FAILURE_KIND_SHAPE,
                "detail": f"shape mismatch at step {i}: {a.shape} vs {b.shape}",
            }
        err = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        max_abs = max(max_abs, err)
        if first_div is None and not np.allclose(a, b, atol=atol, rtol=rtol):
            first_div = i
            first_div_err = err
    numeric_ok = first_div is None
    passed = numeric_ok and not length_mismatch
    failure_kind = None
    detail = None
    if length_mismatch:
        failure_kind = FAILURE_KIND_LENGTH_MISMATCH
        detail = (
            f"unmatched trajectory tail: pytorch_n={n_pt} onnx_n={n_ort} "
            f"(compared overlapping prefix of {n}; tail not discarded from the verdict)"
        )
    elif not numeric_ok:
        failure_kind = FAILURE_KIND_TRAJECTORY_DIVERGENCE
        detail = (
            f"independent-rollout actions diverged at step {first_div} "
            f"(max_abs={first_div_err}); not a same-observation export parity check"
        )
    return {
        "n_steps_compared": n,
        "max_abs_error": max_abs,
        "first_divergence_step": first_div,
        "first_divergence_max_abs_error": first_div_err,
        "length_mismatch": length_mismatch,
        "unmatched_tail_steps": unmatched,
        "passed": passed,
        "failure_kind": failure_kind,
        "detail": detail,
    }


def compare_paired_action_traces(
    pytorch_actions: list[np.ndarray],
    onnx_actions: list[np.ndarray],
) -> ClosedLoopCompareReport:
    """Backward-compatible wrapper; prefer per-seed reports from rollouts."""
    tol = _closed_loop_tolerances()
    result = compare_action_traces_with_tails(pytorch_actions, onnx_actions)
    status = (
        "ok"
        if result["failure_kind"] not in {FAILURE_KIND_EMPTY, FAILURE_KIND_SHAPE}
        else "errored"
    )
    if result["failure_kind"] == FAILURE_KIND_EMPTY:
        status = "errored"
    return ClosedLoopCompareReport(
        status=status,
        detail=result["detail"],
        n_steps_compared=int(result["n_steps_compared"]),
        max_abs_error=result["max_abs_error"],
        passed=bool(result["passed"]),
        tolerances=tol,
        failure_kind=result["failure_kind"],
        length_mismatch_any=bool(result["length_mismatch"]),
        seeds=[
            ClosedLoopSeedReport(
                seed=-1,
                pytorch_outcome="unknown",
                onnx_outcome="unknown",
                pytorch_n_steps=len(pytorch_actions),
                onnx_n_steps=len(onnx_actions),
                length_mismatch=bool(result["length_mismatch"]),
                unmatched_tail_steps=int(result["unmatched_tail_steps"]),
                n_steps_compared=int(result["n_steps_compared"]),
                max_abs_error=result["max_abs_error"],
                first_divergence_step=result["first_divergence_step"],
                first_divergence_max_abs_error=result["first_divergence_max_abs_error"],
                passed=bool(result["passed"]),
                failure_kind=result["failure_kind"],
                detail=result["detail"],
            )
        ],
    )


def paired_closed_loop_rollouts(
    *,
    pytorch_adapter: PolicyAdapter,
    onnx_adapter: OrtPolicyAdapter,
    env: Any,
    seeds: list[int],
    max_steps: int | None = None,
) -> ClosedLoopCompareReport:
    """Paired episodes: same seeds; per-seed independent action-trace comparison."""
    from breakpoint_eval.run_episode import EpisodeOutcome, run_episode

    tol = _closed_loop_tolerances()
    seed_reports: list[ClosedLoopSeedReport] = []

    for seed in seeds:
        pt = run_episode(
            env=env,
            adapter=pytorch_adapter,
            seed=seed,
            episode_index=0,
            latency_s=0.0,
        )
        ort = run_episode(
            env=env,
            adapter=onnx_adapter,  # type: ignore[arg-type]
            seed=seed,
            episode_index=0,
            latency_s=0.0,
        )
        pt_outcome = pt.outcome.value
        ort_outcome = ort.outcome.value
        if (
            pt.outcome is EpisodeOutcome.RUNTIME_ERROR
            or ort.outcome is EpisodeOutcome.RUNTIME_ERROR
        ):
            seed_reports.append(
                ClosedLoopSeedReport(
                    seed=seed,
                    pytorch_outcome=pt_outcome,
                    onnx_outcome=ort_outcome,
                    pytorch_n_steps=len(pt.steps),
                    onnx_n_steps=len(ort.steps),
                    length_mismatch=len(pt.steps) != len(ort.steps),
                    unmatched_tail_steps=abs(len(pt.steps) - len(ort.steps)),
                    n_steps_compared=0,
                    max_abs_error=None,
                    first_divergence_step=None,
                    first_divergence_max_abs_error=None,
                    passed=False,
                    failure_kind=FAILURE_KIND_RUNTIME,
                    detail=f"pytorch={pt.runtime_error}; onnx={ort.runtime_error}",
                )
            )
            continue

        pt_actions: list[np.ndarray] = []
        ort_actions: list[np.ndarray] = []
        n_pt = len(pt.steps)
        n_ort = len(ort.steps)
        if max_steps is not None:
            n_pt = min(n_pt, max_steps)
            n_ort = min(n_ort, max_steps)
        for i in range(n_pt):
            if pt.steps[i].action is None:
                continue
            pt_actions.append(np.asarray(pt.steps[i].action, dtype=np.float32))
        for i in range(n_ort):
            if ort.steps[i].action is None:
                continue
            ort_actions.append(np.asarray(ort.steps[i].action, dtype=np.float32))

        result = compare_action_traces_with_tails(pt_actions, ort_actions)
        seed_reports.append(
            ClosedLoopSeedReport(
                seed=seed,
                pytorch_outcome=pt_outcome,
                onnx_outcome=ort_outcome,
                pytorch_n_steps=len(pt_actions),
                onnx_n_steps=len(ort_actions),
                length_mismatch=bool(result["length_mismatch"]),
                unmatched_tail_steps=int(result["unmatched_tail_steps"]),
                n_steps_compared=int(result["n_steps_compared"]),
                max_abs_error=result["max_abs_error"],
                first_divergence_step=result["first_divergence_step"],
                first_divergence_max_abs_error=result["first_divergence_max_abs_error"],
                passed=bool(result["passed"]),
                failure_kind=result["failure_kind"],
                detail=result["detail"],
            )
        )

    if not seed_reports:
        return ClosedLoopCompareReport(
            status="errored",
            detail="no closed-loop seeds",
            n_steps_compared=0,
            max_abs_error=None,
            passed=False,
            tolerances=tol,
            failure_kind=FAILURE_KIND_EMPTY,
        )

    any_runtime = any(s.failure_kind == FAILURE_KIND_RUNTIME for s in seed_reports)
    length_mismatch_any = any(s.length_mismatch for s in seed_reports)
    all_passed = all(s.passed for s in seed_reports)
    compared = [s for s in seed_reports if s.max_abs_error is not None]
    max_abs = max((s.max_abs_error for s in compared), default=None)
    n_compared = sum(s.n_steps_compared for s in seed_reports)

    failure_kind = None
    detail_parts: list[str] = []
    if any_runtime:
        failure_kind = FAILURE_KIND_RUNTIME
        status = "errored"
    elif not all_passed:
        status = "ok"
        # Prefer the earliest divergence across seeds for the aggregate label.
        div_seeds = [s for s in seed_reports if s.first_divergence_step is not None]
        if any(s.failure_kind == FAILURE_KIND_LENGTH_MISMATCH for s in seed_reports):
            failure_kind = FAILURE_KIND_LENGTH_MISMATCH
        elif div_seeds:
            failure_kind = FAILURE_KIND_TRAJECTORY_DIVERGENCE
        else:
            failure_kind = next((s.failure_kind for s in seed_reports if s.failure_kind), None)
        for s in seed_reports:
            if not s.passed:
                detail_parts.append(
                    f"seed={s.seed} failure_kind={s.failure_kind} "
                    f"first_div={s.first_divergence_step} "
                    f"max_abs={s.max_abs_error} "
                    f"len_pt={s.pytorch_n_steps} len_onnx={s.onnx_n_steps} "
                    f"outcomes=({s.pytorch_outcome},{s.onnx_outcome})"
                )
    else:
        status = "ok"

    return ClosedLoopCompareReport(
        status=status,
        detail="; ".join(detail_parts) if detail_parts else None,
        n_steps_compared=n_compared,
        max_abs_error=max_abs,
        passed=all_passed,
        tolerances=tol,
        failure_kind=failure_kind,
        seeds=seed_reports,
        length_mismatch_any=length_mismatch_any,
        pytorch_outcome="see_seeds",
        onnx_outcome="see_seeds",
    )


__all__ = [
    "COMPARISON_KIND_INDEPENDENT_ROLLOUT",
    "ClosedLoopCompareReport",
    "ClosedLoopSeedReport",
    "CorpusParityReport",
    "FAILURE_KIND_LENGTH_MISMATCH",
    "FAILURE_KIND_TRAJECTORY_DIVERGENCE",
    "ParityTolerances",
    "compare_action_traces_with_tails",
    "compare_paired_action_traces",
    "evaluate_corpus_parity",
    "paired_closed_loop_rollouts",
    "run_ort_chunks",
    "run_pytorch_chunks",
]
