"""Pre-stress validation gates shared across supported policies.

Every candidate must pass native load, documented success semantics, state reset,
camera-degrade boundary wiring, and zero-latency equivalence before a stress sweep.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from breakpoint_eval.camera_degrade import ImageDegradeConfig, ImageMode, degrade_observation
from breakpoint_eval.candidate_map import CandidateSpec
from breakpoint_eval.diffusion_adapter import (
    assert_queues_cleared_after_reset,
    is_diffusion_policy,
    snapshot_diffusion_state,
)
from breakpoint_eval.policy_adapter import PolicyAdapter
from breakpoint_eval.support_status import support_row


@dataclass(frozen=True)
class GateResult:
    gate: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_success_semantics(spec: CandidateSpec) -> GateResult:
    notes = (spec.notes or "").lower()
    if "success" not in notes:
        return GateResult(
            gate="environment_success_semantics",
            passed=False,
            detail="CandidateSpec.notes must document success semantics",
        )
    if spec.env_task.endswith("TransferCube-v0") and "reward==4" not in notes.replace(" ", ""):
        # Allow "reward == 4" with spaces.
        if "reward == 4" not in spec.notes and "reward==4" not in spec.notes:
            return GateResult(
                gate="environment_success_semantics",
                passed=False,
                detail="ALOHA transfer success must cite reward==4",
            )
    if spec.env_task.endswith("Insertion-v0") and "peg" not in notes:
        return GateResult(
            gate="environment_success_semantics",
            passed=False,
            detail="ALOHA insertion success must describe peg/pin contact",
        )
    if spec.env_task == "PushT-v0" and "0.95" not in notes and "coverage" not in notes:
        return GateResult(
            gate="environment_success_semantics",
            passed=False,
            detail="PushT success must document coverage threshold",
        )
    return GateResult(
        gate="environment_success_semantics",
        passed=True,
        detail=f"Documented for {spec.env_task}: {spec.notes.split('.')[0]}",
    )


def validate_native_baseline_load(
    adapter: PolicyAdapter, observation: dict[str, Any]
) -> GateResult:
    try:
        adapter.reset()
        step = adapter.select_action(observation)
        if step.action.ndim != 1 or not np.isfinite(step.action).all():
            return GateResult(
                gate="native_baseline_load",
                passed=False,
                detail=f"non-finite or bad action shape {step.action.shape}",
            )
        if not step.inference_call:
            return GateResult(
                gate="native_baseline_load",
                passed=False,
                detail="first select_action after reset must be a fresh inference",
            )
        return GateResult(
            gate="native_baseline_load",
            passed=True,
            detail=(
                f"policy_type={adapter.chunk_info.policy_type} "
                f"action_dim={step.action.shape[0]} "
                f"n_action_steps={adapter.chunk_info.n_action_steps}"
            ),
        )
    except Exception as exc:  # noqa: BLE001 — gate boundary
        return GateResult(
            gate="native_baseline_load",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def validate_state_reset(adapter: PolicyAdapter, observation: dict[str, Any]) -> GateResult:
    try:
        adapter.reset()
        adapter.select_action(observation)
        adapter.reset()
        policy = adapter.policy
        if is_diffusion_policy(policy):
            assert_queues_cleared_after_reset(policy)
            snap = snapshot_diffusion_state(policy)
            detail = f"diffusion queues cleared: {snap}"
        elif hasattr(policy, "_action_queue"):
            if len(policy._action_queue) != 0:
                return GateResult(
                    gate="state_reset",
                    passed=False,
                    detail=f"ACT action queue not empty after reset: {len(policy._action_queue)}",
                )
            detail = "ACT action queue cleared"
        else:
            detail = "reset() invoked"
        # Fresh inference required again after reset.
        step = adapter.select_action(observation)
        if not step.inference_call:
            return GateResult(
                gate="state_reset",
                passed=False,
                detail="select_action after reset did not request a fresh chunk",
            )
        return GateResult(gate="state_reset", passed=True, detail=detail)
    except Exception as exc:  # noqa: BLE001
        return GateResult(gate="state_reset", passed=False, detail=f"{type(exc).__name__}: {exc}")


def validate_camera_degrade_boundary(raw_observation: dict[str, Any]) -> GateResult:
    try:
        if "pixels" not in raw_observation:
            return GateResult(
                gate="camera_degrade",
                passed=False,
                detail="raw observation missing pixels key",
            )
        cfg = ImageDegradeConfig(mode=ImageMode.NATIVE)
        result = degrade_observation(raw_observation, cfg)
        if "pixels" not in result.observation:
            return GateResult(
                gate="camera_degrade",
                passed=False,
                detail="degrade_observation dropped pixels",
            )
        return GateResult(
            gate="camera_degrade",
            passed=True,
            detail=f"native degrade ok; n_frames={len(result.frames)}",
        )
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            gate="camera_degrade",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _clone_obs_stream(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {k: (v.clone() if torch.is_tensor(v) else v) for k, v in obs.items()}
        for obs in observations
    ]


def validate_zero_latency_equivalence(
    adapter: PolicyAdapter,
    observations: list[dict[str, Any]],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    rng_seed: int = 0,
) -> GateResult:
    """Scheduler-owned chunks at L=0 semantics vs direct select_action on the same obs stream.

    Stochastic policies (Diffusion) reseed the torch RNG before each path so the
    comparison isolates adapter/scheduling differences rather than noise draws.
    """
    if len(observations) < 2:
        return GateResult(
            gate="zero_latency_equivalence",
            passed=False,
            detail="need at least 2 observations",
        )
    try:
        # Direct upstream-style path.
        torch.manual_seed(rng_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rng_seed)
        adapter.reset()
        direct: list[np.ndarray] = []
        for obs in _clone_obs_stream(observations):
            direct.append(adapter.select_action(obs).action.copy())

        # Scheduler-style: compute full chunks when queue would be empty, else observe.
        torch.manual_seed(rng_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rng_seed)
        adapter.reset()
        scheduled: list[np.ndarray] = []
        buffer: list[np.ndarray] = []
        n = adapter.chunk_info.n_action_steps
        for obs in _clone_obs_stream(observations):
            if not buffer:
                chunk = adapter.compute_action_chunk(obs)
                if chunk.shape[0] != n:
                    return GateResult(
                        gate="zero_latency_equivalence",
                        passed=False,
                        detail=f"chunk length {chunk.shape[0]} != n_action_steps {n}",
                    )
                buffer = [chunk[i] for i in range(n)]
            else:
                adapter.observe(obs)
            scheduled.append(buffer.pop(0).astype(np.float32, copy=False))

        for i, (a, b) in enumerate(zip(direct, scheduled, strict=True)):
            if not np.allclose(a, b, atol=atol, rtol=rtol):
                max_abs = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
                return GateResult(
                    gate="zero_latency_equivalence",
                    passed=False,
                    detail=f"mismatch at step {i}: max_abs={max_abs}",
                )
        return GateResult(
            gate="zero_latency_equivalence",
            passed=True,
            detail=f"matched {len(observations)} steps (atol={atol}, rtol={rtol})",
        )
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            gate="zero_latency_equivalence",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def run_policy_validation_gates(
    *,
    spec: CandidateSpec,
    adapter: PolicyAdapter,
    policy_observation: dict[str, Any],
    raw_observation: dict[str, Any] | None = None,
    zero_latency_observations: list[dict[str, Any]] | None = None,
) -> list[GateResult]:
    """Run the required pre-stress gates. Does not start a stress sweep."""
    results = [
        validate_success_semantics(spec),
        validate_native_baseline_load(adapter, policy_observation),
        validate_state_reset(adapter, policy_observation),
    ]
    if raw_observation is not None:
        results.append(validate_camera_degrade_boundary(raw_observation))
    else:
        results.append(
            GateResult(
                gate="camera_degrade",
                passed=False,
                detail="raw_observation not provided",
            )
        )
    if zero_latency_observations is not None:
        obs_stream = zero_latency_observations
    else:
        # Distinct tensors per step so history policies see a stream, not one alias.
        n_steps = max(2, adapter.chunk_info.n_action_steps + 1)
        obs_stream = [
            {k: (v.clone() if torch.is_tensor(v) else v) for k, v in policy_observation.items()}
            for _ in range(n_steps)
        ]
    results.append(validate_zero_latency_equivalence(adapter, obs_stream))
    return results


def assert_gates_passed(results: list[GateResult]) -> None:
    failed = [r for r in results if not r.passed]
    if failed:
        detail = "; ".join(f"{r.gate}: {r.detail}" for r in failed)
        raise AssertionError(f"policy validation gates failed: {detail}")


def gate_summary_for_candidate(candidate: str) -> dict[str, Any]:
    row = support_row(candidate)
    return {
        "candidate": candidate,
        "support": row.to_dict(),
        "required_before_stress": [
            "native_baseline_load",
            "environment_success_semantics",
            "state_reset",
            "camera_degrade",
            "zero_latency_equivalence",
        ],
        "no_cross_task_pooling": True,
    }
