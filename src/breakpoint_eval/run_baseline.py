"""Multi-episode baseline run using the verified LeRobot stack."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from breakpoint_eval.policy_adapter import PolicyAdapter
from breakpoint_eval.run_episode import EpisodeOutcome, EpisodeResult, run_episode
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


@dataclass(frozen=True)
class RunRequest:
    candidate: str
    seeds: list[int]
    output_dir: Path
    device: str
    save_video: bool
    repo_root: Path | None = None


def parse_seed_list(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("seed list is empty")
    return [int(p) for p in parts]


def build_seeds(*, n_episodes: int | None, seeds: list[int] | None, start_seed: int) -> list[int]:
    if seeds is not None and n_episodes is not None and len(seeds) != n_episodes:
        raise ValueError(f"--seeds length ({len(seeds)}) must match --n-episodes ({n_episodes})")
    if seeds is not None:
        return seeds
    if n_episodes is None:
        raise ValueError("Provide --n-episodes and/or --seeds")
    if n_episodes < 1:
        raise ValueError("--n-episodes must be >= 1")
    return [start_seed + i for i in range(n_episodes)]


def run_baseline(req: RunRequest) -> Path:
    repo_root = (req.repo_root or repo_root_from_package()).resolve()
    ensure_lerobot_on_path(repo_root)
    spec = resolve_candidate(req.candidate)
    checkpoint_dir = download_candidate_checkpoint(spec, repo_root / ".cache" / "huggingface")

    period_s = control_period_s(spec)
    env_cfg = make_env_config(spec)
    policy = load_policy(spec, str(checkpoint_dir), req.device)
    adapter = PolicyAdapter(policy, control_period_s=period_s, device=req.device)

    provenance: dict[str, Any] = {
        "candidate": req.candidate,
        "hub_repo": spec.hub_repo,
        "hub_revision": spec.hub_revision,
        "lerobot_commit": LEROBOT_COMMIT,
        "env_type": spec.env_type,
        "env_task": spec.env_task,
        "device": req.device,
        "python": sys.version,
        "checkpoint_dir": str(checkpoint_dir),
        "seeds": list(req.seeds),
        "control_period_s": period_s,
        "fps": env_cfg.fps,
        "chunk_size": adapter.chunk_info.chunk_size,
        "n_action_steps": adapter.chunk_info.n_action_steps,
        "n_obs_steps": adapter.chunk_info.n_obs_steps,
        "temporal_ensemble_coeff": adapter.chunk_info.temporal_ensemble_coeff,
        "save_video": req.save_video,
        "policy_loaded": True,
    }

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
            )
            if req.save_video and frames:
                video_path = (
                    req.output_dir / "videos" / f"episode_{episode_index:04d}_seed_{seed}.mp4"
                )
                write_video(video_path, frames, fps=int(env_cfg.fps))
                result.video_path = str(video_path)
            episodes.append(result)
            LOGGER.info(
                "episode=%s seed=%s outcome=%s steps=%s inferences=%s",
                episode_index,
                seed,
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
