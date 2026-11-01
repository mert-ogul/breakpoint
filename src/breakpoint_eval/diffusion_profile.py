"""Profile one full Diffusion Policy action-generation call on real PushT observations."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from breakpoint_eval.camera_degrade import ImageDegradeConfig, ImageMode, degrade_observation
from breakpoint_eval.device_profile import detect_host_info, read_temperature_c
from breakpoint_eval.diffusion_adapter import (
    TIMED_BOUNDARY_DIFFUSION_GENERATE,
    compute_diffusion_action_chunk,
    diffusion_inference_provenance,
    pytorch_full_generation_provenance,
)
from breakpoint_eval.diffusion_export import checkpoint_file_hashes
from breakpoint_eval.latency_trace import summarize_latency_s, validate_latency_sequence
from breakpoint_eval.sim_env import load_policy, make_vector_env, preprocess_observation
from breakpoint_eval.stack import (
    download_candidate_checkpoint,
    ensure_lerobot_on_path,
    repo_root_from_package,
    resolve_candidate,
)

LOGGER = logging.getLogger("breakpoint_eval.diffusion_profile")

DEFAULT_WARMUPS = 5
DEFAULT_MEASURED = 50
PROFILE_OBS_SEEDS = tuple(range(2300, 2350))
KIND_ACTION_GENERATION_TIMING = "action_generation_timing"

_OUTSIDE_BOUNDARY = (
    "env.reset",
    "image_degrade_native",
    "preprocess_observation",
    "policy.reset",
    "torch_thread_setup",
    "observation_corpus_build",
)
_INSIDE_BOUNDARY = (
    "normalize_inputs",
    "observation_history_update",
    "all_denoising_iterations",
    "unnormalize_outputs",
)


@dataclass(frozen=True)
class DiffusionProfileRequest:
    candidate: str = "diffusion_pusht"
    output_dir: Path | None = None
    device: str = "cpu"
    device_label: str = "host"
    torch_threads: int | None = None
    n_warmup: int = DEFAULT_WARMUPS
    n_measured: int = DEFAULT_MEASURED
    repo_root: Path | None = None
    observation_seeds: tuple[int, ...] = PROFILE_OBS_SEEDS


def _collect_native_observations(
    *,
    spec: Any,
    seeds: tuple[int, ...] | list[int],
    device: str,
) -> list[dict[str, torch.Tensor]]:
    """env.reset → native degrade → preprocess, matching the episode runner."""
    env = make_vector_env(spec, n_envs=1)
    image_cfg = ImageDegradeConfig(mode=ImageMode.NATIVE)
    observations: list[dict[str, torch.Tensor]] = []
    try:
        for seed in seeds:
            raw, _info = env.reset(seed=[int(seed)])
            degraded = degrade_observation(raw, image_cfg)
            processed = preprocess_observation(degraded.observation)
            observations.append(
                {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in processed.items()
                }
            )
    finally:
        env.close()
    if len(observations) != len(seeds):
        raise RuntimeError(f"expected {len(seeds)} observations, collected {len(observations)}")
    return observations


def run_diffusion_pytorch_profile(req: DiffusionProfileRequest) -> Path:
    if req.n_warmup < 0:
        raise ValueError("n_warmup must be >= 0")
    if req.n_measured < 1:
        raise ValueError("n_measured must be >= 1")
    if len(req.observation_seeds) < req.n_measured:
        raise ValueError(
            f"need at least n_measured={req.n_measured} observation seeds; "
            f"got {len(req.observation_seeds)}"
        )

    spec = resolve_candidate(req.candidate)
    if spec.policy_type != "diffusion":
        raise ValueError(f"{req.candidate!r} is not a diffusion candidate")

    root = (req.repo_root or repo_root_from_package()).resolve()
    ensure_lerobot_on_path(root)
    out = (req.output_dir or (root / "profiles" / req.device_label)).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if req.torch_threads is not None:
        if req.torch_threads < 1:
            raise ValueError("torch_threads must be >= 1")
        torch.set_num_threads(int(req.torch_threads))

    ckpt = download_candidate_checkpoint(spec, root / ".cache" / "huggingface")
    ckpt_files = checkpoint_file_hashes(ckpt)
    policy = load_policy(spec, str(ckpt), req.device)
    policy.eval()
    runtime = pytorch_full_generation_provenance()
    inference = diffusion_inference_provenance(policy)

    corpus_seeds = tuple(req.observation_seeds[: req.n_measured])
    observations = _collect_native_observations(spec=spec, seeds=corpus_seeds, device=req.device)

    def _one(obs: dict[str, torch.Tensor]) -> tuple[float, np.ndarray]:
        policy.reset()
        t0 = time.perf_counter_ns()
        chunk = compute_diffusion_action_chunk(policy, obs)
        t1 = time.perf_counter_ns()
        return (t1 - t0) / 1e9, chunk

    for i in range(req.n_warmup):
        _one(observations[i % len(observations)])

    measured_s: list[float] = []
    for i in range(req.n_measured):
        dt_s, chunk = _one(observations[i])
        if not np.isfinite(chunk).all():
            raise RuntimeError(f"non-finite diffusion chunk during measured call {i}")
        measured_s.append(dt_s)

    measured_s = validate_latency_sequence(measured_s)
    stats = summarize_latency_s(measured_s)
    payload: dict[str, Any] = {
        "kind": KIND_ACTION_GENERATION_TIMING,
        "device_label": req.device_label,
        "candidate": req.candidate,
        "hub_revision": spec.hub_revision,
        "host": detect_host_info(),
        "runtime": {
            "library": "pytorch",
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": req.device,
            **runtime.to_dict(),
        },
        "model": {
            "precision_recipe": "pytorch_fp32",
            "variant_label": "pytorch_fp32",
            "checkpoint_dir": str(ckpt),
            "checkpoint_files": ckpt_files,
        },
        "inference": inference,
        "chunk_horizon": int(policy.config.n_action_steps),
        "timed_boundary": TIMED_BOUNDARY_DIFFUSION_GENERATE,
        "timed_boundary_contents": {
            "inside": list(_INSIDE_BOUNDARY),
            "outside": list(_OUTSIDE_BOUNDARY),
        },
        "stages": {
            "warmup_calls": req.n_warmup,
            "measured_calls": req.n_measured,
            "observation_seeds": list(corpus_seeds),
            "note": runtime.note,
        },
        "measurements": {
            "n_warmup": req.n_warmup,
            "n_measured": req.n_measured,
            "clock": "time.perf_counter_ns",
            "unit": "seconds",
            "inference_latency_s": measured_s,
            "summary": stats.to_dict(),
        },
        "environment_notes": {
            "temperature": read_temperature_c(),
        },
        "semantics": {
            "times_full_action_generation": True,
            "times_single_unet_pass_only": False,
            "includes_all_denoising_iterations": True,
            "act_timings_reused": False,
            "corpus": "pusht_env_reset_native_degrade_preprocess",
            "observation_seeds": list(corpus_seeds),
        },
    }
    path = out / f"{req.candidate}__pytorch_fp32__{req.device_label}__full_denoise.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        path.with_suffix(".npz"),
        inference_latency_s=np.asarray(measured_s, dtype=np.float64),
    )
    LOGGER.info(
        "Wrote diffusion PyTorch profile %s (n=%s median_s=%s)",
        path,
        stats.count,
        stats.median_s,
    )
    return path
