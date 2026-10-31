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
    RECIPE_ONNX_FP32,
    ACTChunkModule,
    OrtActChunkSession,
    OrtPolicyAdapter,
    build_act_chunk_module,
    chunk_info_from_policy,
    pytorch_chunk,
    sha256_file,
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


# Held-out closed-loop diagnosis seeds (disjoint from calib 9000–9007 / eval 2000–2007).
DEFAULT_DIAGNOSIS_SEEDS: tuple[int, ...] = (2100, 2101)


@dataclass(frozen=True)
class ClosedLoopDiagnosisRequest:
    export_dir: Path
    output_dir: Path
    candidate: str = "act_aloha_transfer"
    seeds: tuple[int, ...] = DEFAULT_DIAGNOSIS_SEEDS
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
    finite_ok: bool = True,
    shapes_ok: bool = True,
) -> dict[str, Any]:
    """Summarize per-step same-obs chunk errors (offline helper).

    NaN/Inf errors must fail: ``e > atol`` is false for NaN, so use finite checks
    and ``e <= atol`` (NaN comparisons fail) instead of only ``e > atol``.
    """
    if not errors:
        return {
            "n_steps": 0,
            "max_abs_error": None,
            "first_over_atol_step": None,
            "fresh_step_errors": [],
            "finite_ok": False,
            "shapes_ok": shapes_ok,
            "passed_atol": False,
        }
    errs = [float(e) for e in errors]
    finite_errors = bool(finite_ok) and all(np.isfinite(e) for e in errs)
    first_over = next(
        (i for i, e in enumerate(errs) if (not np.isfinite(e)) or e > atol),
        None,
    )
    fresh_errs = [{"step": s, "max_abs_error": errs[s]} for s in fresh_steps if s < len(errs)]
    max_abs = float(np.nanmax(np.asarray(errs, dtype=np.float64))) if errs else None
    passed = (
        bool(shapes_ok)
        and finite_errors
        and first_over is None
        and all(np.isfinite(e) and e <= atol for e in errs)
    )
    return {
        "n_steps": len(errs),
        "max_abs_error": max_abs,
        "first_over_atol_step": first_over,
        "fresh_step_errors": fresh_errs,
        "atol": atol,
        "rtol": rtol,
        "finite_ok": finite_errors,
        "shapes_ok": bool(shapes_ok),
        "passed_atol": passed,
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
    shapes_ok = True
    finite_ok = True
    while not done:
        proc = preprocess_observation(obs)
        images = proc["observation.images.top"].cpu()
        state = proc["observation.state"].cpu()
        pt_chunk = pytorch_chunk(module, images, state)
        ort_chunk = session.run(images.numpy(), state.numpy()).action_chunk
        if pt_chunk.shape != ort_chunk.shape:
            shapes_ok = False
            errors.append(float("inf"))
        elif not (np.isfinite(pt_chunk).all() and np.isfinite(ort_chunk).all()):
            finite_ok = False
            errors.append(float("nan"))
        else:
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
        finite_ok=finite_ok,
        shapes_ok=shapes_ok,
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
    model_sha256 = sha256_file(onnx_path)
    session = OrtActChunkSession(onnx_path, providers=["CPUExecutionProvider"])

    manifest_path = export_dir / "export_manifest.json"
    source_validation: dict[str, Any] | None = None
    source_closed_loop: dict[str, Any] | None = None
    source_manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        source = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_manifest = source
        source_validation = dict(source.get("export_validation") or {})
        source_closed_loop = dict(source.get("closed_loop_onnx_fp32_vs_pytorch") or {})

    # Bind diagnosis to on-disk identity only; missing hub/recipe fields stay missing.
    artifact_binding: dict[str, Any] = {
        "model_sha256": model_sha256,
        "onnx_path": str(onnx_path),
        "onnx_name": req.onnx_name,
        "candidate": req.candidate,
    }
    if source_manifest is not None:
        if source_manifest.get("hub_revision"):
            artifact_binding["hub_revision"] = str(source_manifest["hub_revision"])
        if source_manifest.get("candidate"):
            artifact_binding["manifest_candidate"] = str(source_manifest["candidate"])
        variants = source_manifest.get("variants") or []
        fp32 = next((v for v in variants if v.get("label") == "onnx_fp32"), None)
        if isinstance(fp32, dict):
            if fp32.get("recipe_id"):
                artifact_binding["recipe_id"] = str(fp32["recipe_id"])
            graph_sha = (fp32.get("graph") or {}).get("sha256")
            if graph_sha:
                artifact_binding["manifest_graph_sha256"] = str(graph_sha)
                if str(graph_sha) != model_sha256:
                    raise ValueError(
                        "export manifest onnx_fp32 graph sha256 does not match file on disk: "
                        f"manifest={graph_sha} file={model_sha256}"
                    )
        corpus_eval = (source_manifest.get("corpus") or {}).get("evaluation") or {}
        if corpus_eval.get("sha256"):
            artifact_binding["corpus_eval_sha256"] = str(corpus_eval["sha256"])
    if "recipe_id" not in artifact_binding:
        artifact_binding["recipe_id"] = RECIPE_ONNX_FP32

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
        "artifact_binding": artifact_binding,
        "source_export_validation": source_validation,
        "source_closed_loop": source_closed_loop,
        "interpretation": {
            "closed_loop_gate_compares": COMPARISON_KIND_INDEPENDENT_ROLLOUT,
            "same_input_numerical_parity": (
                "open-loop corpus parity and shadow chunk compares on a driver trajectory"
            ),
            "independent_trajectory_is_diagnostic_only": True,
            "completed_task_failure_is_not_numerical_export_error": True,
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
    "DEFAULT_DIAGNOSIS_SEEDS",
    "ClosedLoopDiagnosisRequest",
    "actions_from_episode",
    "run_closed_loop_diagnosis",
    "run_pytorch_repeatability",
    "run_shadow_chunk_parity",
    "summarize_chunk_error_trace",
]
