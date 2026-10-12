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
) -> CorpusParityReport:
    """Compare chunk lists. When enforce_fp32_gates=False, still report errors."""
    tol = tolerances or ParityTolerances()
    if not (len(reference_chunks) == len(candidate_chunks) == len(seeds)):
        raise ValueError("reference, candidate, and seeds lengths must match")
    samples = [
        compare_chunks(ref, cand, seed=seed, tolerances=tol)
        for ref, cand, seed in zip(reference_chunks, candidate_chunks, seeds, strict=True)
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


@dataclass
class ClosedLoopCompareReport:
    status: str
    detail: str | None
    n_steps_compared: int
    max_abs_error: float | None
    passed: bool | None
    tolerances: dict[str, float]
    pytorch_outcome: str | None = None
    onnx_outcome: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_paired_action_traces(
    pytorch_actions: list[np.ndarray],
    onnx_actions: list[np.ndarray],
) -> ClosedLoopCompareReport:
    tol = {
        "atol": ONNX_FP32_CLOSED_LOOP_ATOL,
        "rtol": ONNX_FP32_CLOSED_LOOP_RTOL,
    }
    n = min(len(pytorch_actions), len(onnx_actions))
    if n == 0:
        return ClosedLoopCompareReport(
            status="errored",
            detail="no paired actions",
            n_steps_compared=0,
            max_abs_error=None,
            passed=False,
            tolerances=tol,
        )
    max_abs = 0.0
    for a, b in zip(pytorch_actions[:n], onnx_actions[:n], strict=True):
        if a.shape != b.shape:
            return ClosedLoopCompareReport(
                status="errored",
                detail=f"shape mismatch {a.shape} vs {b.shape}",
                n_steps_compared=n,
                max_abs_error=None,
                passed=False,
                tolerances=tol,
            )
        max_abs = max(max_abs, float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
    passed = all(
        np.allclose(a, b, atol=tol["atol"], rtol=tol["rtol"])
        for a, b in zip(pytorch_actions[:n], onnx_actions[:n], strict=True)
    )
    return ClosedLoopCompareReport(
        status="ok",
        detail=None,
        n_steps_compared=n,
        max_abs_error=max_abs,
        passed=passed,
        tolerances=tol,
    )


def paired_closed_loop_rollouts(
    *,
    pytorch_adapter: PolicyAdapter,
    onnx_adapter: OrtPolicyAdapter,
    env: Any,
    seeds: list[int],
    max_steps: int | None = None,
) -> ClosedLoopCompareReport:
    """Paired episodes: same seeds; compare applied actions while both succeed."""
    from breakpoint_eval.run_episode import EpisodeOutcome, run_episode

    all_pt: list[np.ndarray] = []
    all_ort: list[np.ndarray] = []
    pt_outcome = None
    ort_outcome = None
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
            return ClosedLoopCompareReport(
                status="errored",
                detail=f"pytorch={pt.runtime_error}; onnx={ort.runtime_error}",
                n_steps_compared=0,
                max_abs_error=None,
                passed=False,
                tolerances={
                    "atol": ONNX_FP32_CLOSED_LOOP_ATOL,
                    "rtol": ONNX_FP32_CLOSED_LOOP_RTOL,
                },
                pytorch_outcome=pt_outcome,
                onnx_outcome=ort_outcome,
            )
        n = min(len(pt.steps), len(ort.steps))
        if max_steps is not None:
            n = min(n, max_steps)
        for i in range(n):
            if pt.steps[i].action is None or ort.steps[i].action is None:
                continue
            all_pt.append(np.asarray(pt.steps[i].action, dtype=np.float32))
            all_ort.append(np.asarray(ort.steps[i].action, dtype=np.float32))
    report = compare_paired_action_traces(all_pt, all_ort)
    report.pytorch_outcome = pt_outcome
    report.onnx_outcome = ort_outcome
    return report


__all__ = [
    "ClosedLoopCompareReport",
    "CorpusParityReport",
    "ParityTolerances",
    "compare_paired_action_traces",
    "evaluate_corpus_parity",
    "paired_closed_loop_rollouts",
    "run_ort_chunks",
    "run_pytorch_chunks",
]
