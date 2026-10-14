"""Multi-episode baseline run using the verified LeRobot stack."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from breakpoint_eval.camera_degrade import ImageDegradeConfig, ImageMode, config_provenance
from breakpoint_eval.latency_scheduler import CHUNK_ARRIVAL_POLICY
from breakpoint_eval.latency_trace import LatencyProfile, LatencyTraceReplayer
from breakpoint_eval.policy_adapter import PolicyAdapter
from breakpoint_eval.run_episode import EpisodeOutcome, EpisodeResult, run_episode
from breakpoint_eval.seed_list import build_seeds, parse_seed_list
from breakpoint_eval.sim_env import (
    control_period_s,
    load_policy,
    make_env_config,
    make_vector_env,
)
from breakpoint_eval.stack import (
    LEROBOT_COMMIT,
    download_candidate_checkpoint,
    ensure_lerobot_on_path,
    repo_root_from_package,
    resolve_candidate,
)
from breakpoint_eval.write_report import write_run_report, write_video

LOGGER = logging.getLogger("breakpoint_eval.run_baseline")

# Re-export for callers that historically imported seeds from this module.
__all__ = [
    "RunRequest",
    "agreement_action_trace",
    "build_seeds",
    "parse_seed_list",
    "run_baseline",
]


@dataclass(frozen=True)
class RunRequest:
    candidate: str
    seeds: list[int]
    output_dir: Path
    device: str
    save_video: bool
    latency_s: float = 0.0
    latency_profile: LatencyProfile | None = None
    latency_end_of_trace: str = "fail"
    latency_block_length: int | None = None
    latency_block_seed: int | None = None
    image_cfg: ImageDegradeConfig | None = None
    repo_root: Path | None = None
    onnx_model: Path | None = None
    policy_rng_seeds: list[int] | None = None


def run_baseline(req: RunRequest) -> Path:
    if req.latency_profile is not None and req.latency_s != 0.0:
        raise ValueError(
            "constant --latency-ms and measured --latency-profile are mutually exclusive"
        )
    if req.policy_rng_seeds is not None and len(req.policy_rng_seeds) != len(req.seeds):
        raise ValueError("policy_rng_seeds length must match env seeds length")
    if req.latency_profile is not None and req.onnx_model is None:
        raise ValueError(
            "measured latency profile requires --onnx-model matching the profiled graph; "
            "do not replay ONNX timings onto an unlabeled PyTorch FP32 rollout"
        )

    repo_root = (req.repo_root or repo_root_from_package()).resolve()
    ensure_lerobot_on_path(repo_root)
    spec = resolve_candidate(req.candidate)
    checkpoint_dir = download_candidate_checkpoint(spec, repo_root / ".cache" / "huggingface")

    period_s = control_period_s(spec)
    env_cfg = make_env_config(spec)
    onnx_runtime = None
    policy = None
    if req.onnx_model is not None:
        from breakpoint_eval.act_export import (
            OrtActChunkSession,
            OrtPolicyAdapter,
            chunk_info_from_policy,
        )
        from breakpoint_eval.export_runtime import resolve_onnx_runtime

        if spec.policy_type != "act":
            raise ValueError(
                f"--onnx-model is only supported for ACT candidates; got {spec.policy_type}"
            )
        onnx_runtime = resolve_onnx_runtime(
            Path(req.onnx_model),
            expected_candidate=req.candidate,
            require_export_validation=True,
        )
        # Load PyTorch policy only for chunk_info metadata / diffusion-forbidden checks;
        # actions come from ORT.
        policy = load_policy(spec, str(checkpoint_dir), req.device)
        chunk_info = chunk_info_from_policy(policy, period_s)
        if chunk_info.n_action_steps != onnx_runtime.chunk_horizon:
            raise ValueError(
                f"ONNX chunk horizon {onnx_runtime.chunk_horizon} != policy "
                f"n_action_steps {chunk_info.n_action_steps}"
            )
        session = OrtActChunkSession(onnx_runtime.model_path, providers=["CPUExecutionProvider"])
        adapter: Any = OrtPolicyAdapter(
            session,
            chunk_info=chunk_info,
            image_key="observation.images.top",
        )
        backend = "onnxruntime"
    else:
        policy = load_policy(spec, str(checkpoint_dir), req.device)
        adapter = PolicyAdapter(policy, control_period_s=period_s, device=req.device)
        backend = "pytorch"

    image_cfg = req.image_cfg or ImageDegradeConfig(mode=ImageMode.NATIVE)
    if image_cfg.frame_rate_hz is None or image_cfg.mode is ImageMode.H264:
        # H.264 stream rate must match the environment control / camera rate.
        image_cfg = replace(image_cfg, frame_rate_hz=float(env_cfg.fps))

    latency_replayer: LatencyTraceReplayer | None = None
    if req.latency_profile is not None:
        from breakpoint_eval.latency_trace import EndOfTraceBehavior, assert_profile_compatible

        assert onnx_runtime is not None  # enforced above
        assert_profile_compatible(
            req.latency_profile,
            candidate=req.candidate,
            model_sha256=onnx_runtime.model_sha256,
            precision_recipe=onnx_runtime.recipe_id,
            variant_label=onnx_runtime.variant_label,
            chunk_horizon=onnx_runtime.chunk_horizon,
            hub_revision=onnx_runtime.hub_revision or None,
            input_shapes=onnx_runtime.input_shapes,
        )
        latency_replayer = LatencyTraceReplayer.from_profile(
            req.latency_profile,
            end_of_trace=EndOfTraceBehavior(req.latency_end_of_trace),
            block_length=req.latency_block_length,
            block_resample_seed=req.latency_block_seed,
        )

    policy_rng_seeds = (
        list(req.policy_rng_seeds)
        if req.policy_rng_seeds is not None
        else [int(s) + 10_000 for s in req.seeds]
    )

    provenance: dict[str, Any] = {
        "candidate": req.candidate,
        "hub_repo": spec.hub_repo,
        "hub_revision": spec.hub_revision,
        "lerobot_commit": LEROBOT_COMMIT,
        "policy_type": spec.policy_type,
        "env_type": spec.env_type,
        "env_task": spec.env_task,
        "device": req.device,
        "python": sys.version,
        "checkpoint_dir": str(checkpoint_dir),
        "seeds": list(req.seeds),
        "env_seeds": list(req.seeds),
        "policy_rng_seeds": policy_rng_seeds,
        "control_period_s": period_s,
        "fps": env_cfg.fps,
        "chunk_size": adapter.chunk_info.chunk_size,
        "n_action_steps": adapter.chunk_info.n_action_steps,
        "n_obs_steps": adapter.chunk_info.n_obs_steps,
        "temporal_ensemble_coeff": adapter.chunk_info.temporal_ensemble_coeff,
        "save_video": req.save_video,
        "policy_loaded": True,
        "policy_backend": backend,
        "onnx_runtime": onnx_runtime.to_dict() if onnx_runtime is not None else None,
        "latency_s": req.latency_s if latency_replayer is None else None,
        "latency_ms": (req.latency_s * 1000.0) if latency_replayer is None else None,
        "latency_mode": "trace_replay" if latency_replayer is not None else "constant",
        "latency_profile_identity": (
            req.latency_profile.identity() if req.latency_profile is not None else None
        ),
        "chunk_arrival_policy": CHUNK_ARRIVAL_POLICY,
        "host_compute_added_to_latency": False,
        "constant_delay_distinct_from_device_replay": True,
        "simulator_policy_on_host_latency_may_be_from_profile_device": True,
        "image_degrade": config_provenance(image_cfg),
        "image_degrade_separate_from_latency": True,
        "cross_task_success_pooling": False,
        "success_semantics": spec.notes,
    }
    if spec.policy_type == "diffusion":
        from breakpoint_eval.diffusion_adapter import diffusion_inference_provenance

        provenance["diffusion_inference"] = diffusion_inference_provenance(policy)
        provenance["device_trace_from_act_forbidden"] = True

    try:
        env = make_vector_env(spec, n_envs=1)
    except (OSError, ImportError, ModuleNotFoundError, RuntimeError) as exc:
        err = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("Environment construction failed after policy load")
        provenance["env_created"] = False
        provenance["env_error"] = err
        episodes = [
            EpisodeResult(
                episode_index=i,
                seed=seed,
                outcome=EpisodeOutcome.RUNTIME_ERROR,
                success=False,
                terminated=False,
                truncated=False,
                policy_failed=False,
                runtime_error=err,
                n_control_steps=0,
                n_inference_calls=0,
                n_fresh_chunks=0,
                sum_reward=0.0,
                max_reward=0.0,
                control_period_s=period_s,
                chunk_size=adapter.chunk_info.chunk_size,
                n_action_steps=adapter.chunk_info.n_action_steps,
                latency_s=req.latency_s,
                chunk_arrival_policy=CHUNK_ARRIVAL_POLICY,
                image_mode=image_cfg.mode.value,
            )
            for i, seed in enumerate(req.seeds)
        ]
        return write_run_report(req.output_dir, provenance=provenance, episodes=episodes)

    provenance["env_created"] = True
    episodes: list[EpisodeResult] = []
    try:
        for episode_index, seed in enumerate(req.seeds):
            frames: list[Any] | None = [] if req.save_video else None
            result = run_episode(
                env=env,
                adapter=adapter,
                seed=seed,
                episode_index=episode_index,
                render_frames=frames,
                latency_s=0.0 if latency_replayer is not None else req.latency_s,
                latency_replayer=latency_replayer,
                image_cfg=image_cfg,
                policy_rng_seed=policy_rng_seeds[episode_index],
            )
            if req.save_video and frames:
                video_path = (
                    req.output_dir / "videos" / f"episode_{episode_index:04d}_seed_{seed}.mp4"
                )
                write_video(video_path, frames, fps=int(env_cfg.fps))
                result.video_path = str(video_path)
            episodes.append(result)
            LOGGER.info(
                "episode=%s seed=%s policy_rng_seed=%s outcome=%s steps=%s inferences=%s",
                episode_index,
                seed,
                policy_rng_seeds[episode_index],
                result.outcome.value,
                result.n_control_steps,
                result.n_inference_calls,
            )
    finally:
        env.close()

    return write_run_report(req.output_dir, provenance=provenance, episodes=episodes)


def agreement_action_trace(
    *,
    adapter: PolicyAdapter,
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run adapter on a fixed observation sequence (for upstream agreement tests)."""
    adapter.reset()
    trace: list[dict[str, Any]] = []
    for obs in observations:
        step = adapter.select_action(obs)
        trace.append(
            {
                "control_step_index": step.control_step_index,
                "requested_fresh_chunk": step.requested_fresh_chunk,
                "inference_call": step.inference_call,
                "action": step.action.tolist(),
            }
        )
    return trace
