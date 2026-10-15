"""Diagnose closed-loop export failures without changing the validation gate.

Closed-loop validation compares independently controlled episodes (separate
resets). Same-input numerical parity is a different check (open-loop corpus and
shadow chunk compares on a driver trajectory).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from breakpoint_eval.act_export import (
    ACTChunkModule,
    OrtActChunkSession,
    OrtPolicyAdapter,
    build_act_chunk_module,
    chunk_info_from_policy,
    pytorch_chunk,
)
from breakpoint_eval.export_parity import (
    COMPARISON_KIND_INDEPENDENT_ROLLOUT,
    compare_action_traces_with_tails,
    paired_closed_loop_rollouts,
)
from breakpoint_eval.export_tolerances import (
    ONNX_FP32_CLOSED_LOOP_ATOL,
    ONNX_FP32_CLOSED_LOOP_RTOL,
    ONNX_FP32_PARITY_ATOL,
    ONNX_FP32_PARITY_RTOL,
)
from breakpoint_eval.policy_adapter import PolicyAdapter
from breakpoint_eval.sim_env import (
    control_period_s,
    load_policy,
    make_vector_env,
    preprocess_observation,
)
from breakpoint_eval.stack import (
    download_candidate_checkpoint,
    ensure_lerobot_on_path,
    repo_root_from_package,
    resolve_candidate,
)

LOGGER = logging.getLogger("breakpoint_eval.export_closed_loop_diagnosis")


@dataclass(frozen=True)
class ClosedLoopDiagnosisRequest:
    export_dir: Path
    output_dir: Path
    candidate: str = "act_aloha_transfer"
    seeds: tuple[int, ...] = (2000, 2001)
    onnx_name: str = "act_chunk_fp32.onnx"
    device: str = "cpu"
    repo_root: Path | None = None


def actions_from_episode(episode: Any) -> list[np.ndarray]:
    """Applied actions only; skips steps with no action."""
    out: list[np.ndarray] = []
    for step in episode.steps:
        if step.action is None:
            continue
        out.append(np.asarray(step.action, dtype=np.float32))
    return out


def summarize_chunk_error_trace(
    errors: list[float],
    *,
    fresh_steps: list[int],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Summarize per-step same-obs chunk errors (offline helper)."""
    if not errors:
        return {
            "n_steps": 0,
            "max_abs_error": None,
            "first_over_atol_step": None,
            "fresh_step_errors": [],
            "passed_atol": False,
        }
    first_over = next((i for i, e in enumerate(errors) if e > atol), None)
    fresh_errs = [{"step": s, "max_abs_error": errors[s]} for s in fresh_steps if s < len(errors)]
    return {
        "n_steps": len(errors),
        "max_abs_error": float(max(errors)),
        "first_over_atol_step": first_over,
        "fresh_step_errors": fresh_errs,
        "atol": atol,
        "rtol": rtol,
        "passed_atol": first_over is None,
    }


def run_pytorch_repeatability(
    *,
    env: Any,
    adapter: PolicyAdapter,
    seed: int,
) -> dict[str, Any]:
    """Two PyTorch episodes with the same seed; compare applied actions."""
    from breakpoint_eval.export_parity import episode_end_fields
    from breakpoint_eval.run_episode import run_episode

    first = run_episode(env=env, adapter=adapter, seed=seed, episode_index=0, latency_s=0.0)
    second = run_episode(env=env, adapter=adapter, seed=seed, episode_index=1, latency_s=0.0)
    a = actions_from_episode(first)
    b = actions_from_episode(second)
    cmp = compare_action_traces_with_tails(
        a,
        b,
        atol=ONNX_FP32_CLOSED_LOOP_ATOL,
        rtol=ONNX_FP32_CLOSED_LOOP_RTOL,
    )
    return {
        "seed": seed,
        "comparison_kind": "pytorch_vs_pytorch_repeatability",
        "episodes": [
            episode_end_fields(first, prefix="episode"),
            episode_end_fields(second, prefix="episode"),
        ],
        "outcomes": [first.outcome.value, second.outcome.value],
        "n_steps": [len(a), len(b)],
        **cmp,
    }


def run_shadow_chunk_parity(
    *,
    env: Any,
    driver: PolicyAdapter,
    module: ACTChunkModule,
    session: OrtActChunkSession,
    seed: int,
) -> dict[str, Any]:
    """Driver controls the sim; ORT is shadow-only on identical observations."""
    from breakpoint_eval.run_episode import _classify_outcome, _extract_is_success

    # Driver selects actions; both graphs evaluated on the same observation each step.
    driver.reset()
    obs, _info = env.reset(seed=[seed])
    errors: list[float] = []
    fresh_steps: list[int] = []
    inference_steps: list[int] = []
    step_index = 0
    terminated = False
    truncated = False
    is_success = False
    done = False
    while not done:
        proc = preprocess_observation(obs)
        images = proc["observation.images.top"].cpu()
        state = proc["observation.state"].cpu()
        pt_chunk = pytorch_chunk(module, images, state)
        ort_chunk = session.run(images.numpy(), state.numpy()).action_chunk
        err = float(np.max(np.abs(pt_chunk.astype(np.float64) - ort_chunk.astype(np.float64))))
        errors.append(err)
        action_step = driver.select_action(proc)
        if action_step.requested_fresh_chunk:
            fresh_steps.append(step_index)
        if action_step.inference_call:
            inference_steps.append(step_index)
        action = np.asarray(action_step.action, dtype=np.float32).reshape(1, -1)
        obs, _reward, terminated_arr, truncated_arr, info = env.step(action)
        terminated = bool(np.any(terminated_arr))
        truncated = bool(np.any(truncated_arr))
        is_success = _extract_is_success(info)
        done = terminated or truncated
        step_index += 1
        if step_index > 10_000:
            raise RuntimeError("shadow rollout exceeded 10000 steps")

    outcome = _classify_outcome(
        policy_failed=False,
        is_success=is_success,
        terminated=terminated,
        truncated=truncated,
    )
    summary = summarize_chunk_error_trace(
        errors,
        fresh_steps=fresh_steps,
        atol=ONNX_FP32_PARITY_ATOL,
        rtol=ONNX_FP32_PARITY_RTOL,
    )
    return {
        "seed": seed,
        "comparison_kind": "same_observation_shadow_chunks",
        "driver": "pytorch",
        "shadow": "onnx",
        "shadow_controls_simulation": False,
        "driver_outcome": outcome.value,
        "driver_success": bool(is_success and outcome.value == "success"),
        "driver_terminated": terminated,
        "driver_truncated": truncated,
        "n_control_steps": step_index,
        "n_fresh_chunks": len(fresh_steps),
        "n_inference_calls": len(inference_steps),
        "fresh_chunk_steps": fresh_steps,
        "inference_call_steps": inference_steps,
        "compares_every_control_step": True,
        "covers_every_inference_call": bool(inference_steps)
        and all(0 <= s < len(errors) for s in inference_steps),
        "covers_every_fresh_chunk_refresh": bool(fresh_steps)
        and all(0 <= s < len(errors) for s in fresh_steps),
        **summary,
        "note": (
            "PyTorch drives the environment through termination/truncation. At each "
            "control step both graphs see the same observation (including every "
            "action-chunk refresh); ORT never applies actions."
        ),
    }


def _resolve_onnx_path(export_dir: Path, onnx_name: str) -> Path:
    path = (export_dir / onnx_name).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {path}")
    return path


def run_closed_loop_diagnosis(req: ClosedLoopDiagnosisRequest) -> Path:
    """Write diagnosis JSON under ``req.output_dir``; does not edit export manifests."""
    repo_root = (req.repo_root or repo_root_from_package()).resolve()
    export_dir = Path(req.export_dir).resolve()
    output_dir = Path(req.output_dir).resolve()
    if output_dir.resolve() == export_dir.resolve():
        raise ValueError(
            "output_dir must differ from export_dir; refuse to overwrite export artifacts"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_lerobot_on_path(repo_root)
    spec = resolve_candidate(req.candidate)
    checkpoint = download_candidate_checkpoint(spec, repo_root / ".cache" / "huggingface")
    period_s = control_period_s(spec)
    policy = load_policy(spec, str(checkpoint), req.device)
    module = build_act_chunk_module(policy).cpu().eval()
    chunk_info = chunk_info_from_policy(policy, period_s)
    onnx_path = _resolve_onnx_path(export_dir, req.onnx_name)
    session = OrtActChunkSession(onnx_path, providers=["CPUExecutionProvider"])

    manifest_path = export_dir / "export_manifest.json"
    source_validation: dict[str, Any] | None = None
    source_closed_loop: dict[str, Any] | None = None
    if manifest_path.is_file():
        source = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_validation = dict(source.get("export_validation") or {})
        source_closed_loop = dict(source.get("closed_loop_onnx_fp32_vs_pytorch") or {})

    seeds = list(req.seeds)
    env = make_vector_env(spec, n_envs=1)
    try:
        pt_adapter = PolicyAdapter(policy, control_period_s=period_s, device="cpu")
        ort_adapter = OrtPolicyAdapter(session, chunk_info=chunk_info)

        repeatability = [
            run_pytorch_repeatability(env=env, adapter=pt_adapter, seed=seed) for seed in seeds
        ]
        shadow = [
            run_shadow_chunk_parity(
                env=env,
                driver=pt_adapter,
                module=module,
                session=session,
                seed=seed,
            )
            for seed in seeds
        ]
        independent = paired_closed_loop_rollouts(
            pytorch_adapter=pt_adapter,
            onnx_adapter=ort_adapter,
            env=env,
            seeds=seeds,
        )
    finally:
        env.close()

    payload = {
        "candidate": req.candidate,
        "export_dir": str(export_dir),
        "onnx_path": str(onnx_path),
        "seeds": seeds,
        "source_export_validation": source_validation,
        "source_closed_loop": source_closed_loop,
        "interpretation": {
            "closed_loop_gate_compares": COMPARISON_KIND_INDEPENDENT_ROLLOUT,
            "same_input_numerical_parity": (
                "open-loop corpus parity and shadow chunk compares on a driver trajectory"
            ),
            "validation_gate_unchanged": True,
            "root_cause_if_independent_fails_and_shadow_passes": (
                "Independent trajectory divergence under separate resets, not a "
                "same-observation ONNX numerical export failure."
            ),
        },
        "pytorch_repeatability": repeatability,
        "same_observation_shadow": shadow,
        "independent_rollouts": independent.to_dict(),
        "tolerances": {
            "closed_loop_atol": ONNX_FP32_CLOSED_LOOP_ATOL,
            "closed_loop_rtol": ONNX_FP32_CLOSED_LOOP_RTOL,
            "parity_atol": ONNX_FP32_PARITY_ATOL,
            "parity_rtol": ONNX_FP32_PARITY_RTOL,
        },
    }
    out_path = output_dir / "closed_loop_diagnosis.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote closed-loop diagnosis %s", out_path)
    return out_path


__all__ = [
    "ClosedLoopDiagnosisRequest",
    "actions_from_episode",
    "run_closed_loop_diagnosis",
    "run_pytorch_repeatability",
    "run_shadow_chunk_parity",
    "summarize_chunk_error_trace",
]
