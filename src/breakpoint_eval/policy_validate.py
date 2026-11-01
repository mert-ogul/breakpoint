"""Pre-stress validation gates shared across supported policies.

Every candidate must pass native load, documented success semantics, state reset,
camera-degrade boundary wiring, and zero-latency equivalence before a stress sweep.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from breakpoint_eval.camera_degrade import ImageDegradeConfig, ImageMode, degrade_observation
from breakpoint_eval.candidate_map import CandidateSpec
from breakpoint_eval.diffusion_adapter import (
    assert_queues_cleared_after_reset,
    diffusion_inference_provenance,
    is_diffusion_policy,
    snapshot_diffusion_state,
)
from breakpoint_eval.episode_outcome import EpisodeOutcome
from breakpoint_eval.policy_adapter import PolicyAdapter
from breakpoint_eval.run_episode import EpisodeResult
from breakpoint_eval.support_status import support_row

LOGGER = logging.getLogger("breakpoint_eval.policy_validate")

_COMPLETED_OUTCOMES = frozenset(
    {
        EpisodeOutcome.SUCCESS,
        EpisodeOutcome.TERMINATED_NO_SUCCESS,
        EpisodeOutcome.TIME_LIMIT_TRUNCATION,
    }
)


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
    except Exception as exc:  # noqa: BLE001  gate boundary
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


def _actions_from_episode(result: EpisodeResult) -> list[np.ndarray]:
    actions: list[np.ndarray] = []
    for step in result.steps:
        if step.action is None:
            raise ValueError("episode step missing action")
        actions.append(np.asarray(step.action, dtype=np.float32))
    return actions


def _episode_completed(result: EpisodeResult) -> bool:
    """True when the episode finished under normal control (not error/empty)."""
    return (
        result.outcome in _COMPLETED_OUTCOMES
        and result.runtime_error is None
        and not result.policy_failed
        and result.n_control_steps > 0
    )


def compare_action_traces_bitwise(
    left: list[np.ndarray], right: list[np.ndarray]
) -> dict[str, Any]:
    """Compare action sequences for length and bitwise equality."""
    n = min(len(left), len(right))
    max_abs = 0.0
    first_mismatch: int | None = None
    for i in range(n):
        a = np.asarray(left[i], dtype=np.float32)
        b = np.asarray(right[i], dtype=np.float32)
        if a.shape != b.shape:
            diff = float("inf")
        else:
            diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        if diff > max_abs:
            max_abs = diff
        if first_mismatch is None and not np.array_equal(a, b):
            first_mismatch = i
    if len(left) != len(right) and first_mismatch is None:
        first_mismatch = n
        max_abs = float("inf")
    return {
        "steps_left": len(left),
        "steps_right": len(right),
        "max_abs_action_difference": max_abs,
        "first_mismatch_step": first_mismatch,
        "passed": first_mismatch is None and len(left) == len(right),
    }


def compare_seed_agreement(
    *,
    seed: int,
    benchmark: EpisodeResult,
    upstream: EpisodeResult,
) -> dict[str, Any]:
    """Closed-loop agreement for one env seed (benchmark path vs upstream queue)."""
    bench_actions = _actions_from_episode(benchmark)
    up_actions = _actions_from_episode(upstream)
    action_cmp = compare_action_traces_bitwise(bench_actions, up_actions)
    both_completed = _episode_completed(benchmark) and _episode_completed(upstream)
    same_outcome = benchmark.outcome == upstream.outcome
    same_success = bool(benchmark.success) == bool(upstream.success)
    same_reward = benchmark.sum_reward == upstream.sum_reward
    same_steps = benchmark.n_control_steps == upstream.n_control_steps
    passed = bool(
        both_completed
        and action_cmp["passed"]
        and same_outcome
        and same_success
        and same_reward
        and same_steps
    )
    return {
        "seed": seed,
        "passed": passed,
        "steps": benchmark.n_control_steps,
        "upstream_steps": upstream.n_control_steps,
        "outcome": benchmark.outcome.value,
        "upstream_outcome": upstream.outcome.value,
        "runtime_error": benchmark.runtime_error,
        "upstream_runtime_error": upstream.runtime_error,
        "success": bool(benchmark.success),
        "upstream_success": bool(upstream.success),
        "sum_reward": benchmark.sum_reward,
        "upstream_sum_reward": upstream.sum_reward,
        "max_abs_action_difference": action_cmp["max_abs_action_difference"],
        "first_mismatch_step": action_cmp["first_mismatch_step"],
    }


def compare_determinism_rerun(
    *,
    seed: int,
    first: EpisodeResult,
    second: EpisodeResult,
) -> dict[str, Any]:
    """Require bitwise-equal actions when the benchmark path is run twice."""
    both_completed = _episode_completed(first) and _episode_completed(second)
    cmp = compare_action_traces_bitwise(_actions_from_episode(first), _actions_from_episode(second))
    return {
        "seed": seed,
        "passed": bool(both_completed and cmp["passed"]),
        "steps": first.n_control_steps,
        "rerun_steps": second.n_control_steps,
        "outcome": first.outcome.value,
        "rerun_outcome": second.outcome.value,
        "runtime_error": first.runtime_error,
        "rerun_runtime_error": second.runtime_error,
        "max_abs_action_difference": cmp["max_abs_action_difference"],
        "first_mismatch_step": cmp["first_mismatch_step"],
    }


def evaluate_seed_agreement(
    *,
    env: Any,
    adapter: PolicyAdapter,
    seed: int,
    policy_rng_seed: int,
    check_determinism: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run benchmark and upstream paths for one seed; optionally rerun for determinism."""
    from breakpoint_eval.run_episode import run_episode, run_episode_baseline_queue

    benchmark = run_episode(
        env=env,
        adapter=adapter,
        seed=seed,
        episode_index=0,
        latency_s=0.0,
        policy_rng_seed=policy_rng_seed,
    )
    upstream = run_episode_baseline_queue(
        env=env,
        adapter=adapter,
        seed=seed,
        episode_index=0,
        policy_rng_seed=policy_rng_seed,
    )
    seed_result = compare_seed_agreement(seed=seed, benchmark=benchmark, upstream=upstream)
    determinism = None
    if check_determinism:
        rerun = run_episode(
            env=env,
            adapter=adapter,
            seed=seed,
            episode_index=0,
            latency_s=0.0,
            policy_rng_seed=policy_rng_seed,
        )
        determinism = compare_determinism_rerun(seed=seed, first=benchmark, second=rerun)
    return seed_result, determinism


def build_upstream_agreement_payload(
    *,
    candidate: str,
    hub_revision: str,
    lerobot_commit: str,
    checkpoint_files: dict[str, str],
    torch_version: str,
    torch_threads: int,
    device: str,
    diffusion_settings: dict[str, Any],
    per_seed: list[dict[str, Any]],
    determinism: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the verify-upstream JSON body."""
    passed = all(row.get("passed") is True for row in per_seed)
    if determinism is not None:
        passed = passed and determinism.get("passed") is True
    return {
        "passed": passed,
        "candidate": candidate,
        "hub_revision": hub_revision,
        "lerobot_commit": lerobot_commit,
        "checkpoint_files": checkpoint_files,
        "torch_version": torch_version,
        "torch_threads": torch_threads,
        "device": device,
        "diffusion": diffusion_settings,
        "per_seed": per_seed,
        "determinism": determinism,
    }


def verify_upstream_agreement(
    candidate: str,
    seeds: list[int],
    device: str,
    torch_threads: int,
    output_path: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Closed-loop agreement: L=0 scheduler path vs upstream select_action queue.

    Writes one JSON report. Raises on setup/runtime errors (CLI maps to exit 2).
    """
    from breakpoint_eval.diffusion_export import checkpoint_file_hashes
    from breakpoint_eval.sim_env import control_period_s, load_policy, make_vector_env
    from breakpoint_eval.stack import (
        LEROBOT_COMMIT,
        download_candidate_checkpoint,
        ensure_lerobot_on_path,
        repo_root_from_package,
        resolve_candidate,
    )

    if not seeds:
        raise ValueError("seeds must be non-empty")
    if torch_threads < 1:
        raise ValueError("torch_threads must be >= 1")

    spec = resolve_candidate(candidate)
    if spec.policy_type != "diffusion":
        raise ValueError(f"verify-upstream expects a diffusion candidate; got {candidate!r}")

    root = (repo_root or repo_root_from_package()).resolve()
    ensure_lerobot_on_path(root)
    torch.set_num_threads(int(torch_threads))

    checkpoint_dir = download_candidate_checkpoint(spec, root / ".cache" / "huggingface")
    ckpt_files = checkpoint_file_hashes(checkpoint_dir)
    period_s = control_period_s(spec)
    policy = load_policy(spec, str(checkpoint_dir), device)
    adapter = PolicyAdapter(policy, control_period_s=period_s, device=device)
    diffusion_settings = diffusion_inference_provenance(policy)

    env = make_vector_env(spec, n_envs=1)
    per_seed: list[dict[str, Any]] = []
    determinism: dict[str, Any] | None = None
    try:
        for index, seed in enumerate(seeds):
            policy_rng_seed = int(seed) + 10_000
            seed_result, det = evaluate_seed_agreement(
                env=env,
                adapter=adapter,
                seed=int(seed),
                policy_rng_seed=policy_rng_seed,
                check_determinism=index == 0,
            )
            per_seed.append(seed_result)
            if det is not None:
                determinism = det
            if seed_result["passed"]:
                LOGGER.info(
                    "verify-upstream seed=%s passed=True max_abs=%s first_mismatch=%s",
                    seed,
                    seed_result["max_abs_action_difference"],
                    seed_result["first_mismatch_step"],
                )
            else:
                LOGGER.info(
                    "verify-upstream seed=%s passed=False outcome=%s upstream_outcome=%s "
                    "runtime_error=%s upstream_runtime_error=%s max_abs=%s first_mismatch=%s",
                    seed,
                    seed_result["outcome"],
                    seed_result["upstream_outcome"],
                    seed_result["runtime_error"],
                    seed_result["upstream_runtime_error"],
                    seed_result["max_abs_action_difference"],
                    seed_result["first_mismatch_step"],
                )
            if det is not None and not det["passed"]:
                LOGGER.info(
                    "verify-upstream determinism seed=%s passed=False outcome=%s "
                    "rerun_outcome=%s runtime_error=%s rerun_runtime_error=%s "
                    "max_abs=%s first_mismatch=%s",
                    seed,
                    det["outcome"],
                    det["rerun_outcome"],
                    det["runtime_error"],
                    det["rerun_runtime_error"],
                    det["max_abs_action_difference"],
                    det["first_mismatch_step"],
                )
    finally:
        env.close()

    payload = build_upstream_agreement_payload(
        candidate=candidate,
        hub_revision=spec.hub_revision,
        lerobot_commit=LEROBOT_COMMIT,
        checkpoint_files=ckpt_files,
        torch_version=torch.__version__,
        torch_threads=torch.get_num_threads(),
        device=device,
        diffusion_settings=diffusion_settings,
        per_seed=per_seed,
        determinism=determinism,
    )
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("Wrote upstream agreement report: %s (passed=%s)", out, payload["passed"])
    return payload
