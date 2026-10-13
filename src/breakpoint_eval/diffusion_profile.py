"""Profile a full Diffusion Policy action-generation call (all denoising steps).

This is distinct from ACT ONNX Runtime ``InferenceSession.run`` profiling.
Timings must not be copied across candidates or treated as a single UNet pass.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from breakpoint_eval.device_profile import detect_host_info, read_temperature_c
from breakpoint_eval.diffusion_adapter import (
    TIMED_BOUNDARY_DIFFUSION_GENERATE,
    compute_diffusion_action_chunk,
    diffusion_inference_provenance,
    pytorch_full_generation_provenance,
)
from breakpoint_eval.latency_trace import (
    PROFILE_SCHEMA_VERSION,
    summarize_latency_s,
    validate_latency_sequence,
)
from breakpoint_eval.sim_env import load_policy
from breakpoint_eval.stack import (
    download_candidate_checkpoint,
    ensure_lerobot_on_path,
    repo_root_from_package,
    resolve_candidate,
)

LOGGER = logging.getLogger("breakpoint_eval.diffusion_profile")

DEFAULT_WARMUPS = 5
DEFAULT_MEASURED = 50


@dataclass(frozen=True)
class DiffusionProfileRequest:
    candidate: str = "diffusion_pusht"
    output_dir: Path | None = None
    device: str = "cpu"
    device_label: str = "host"
    n_warmup: int = DEFAULT_WARMUPS
    n_measured: int = DEFAULT_MEASURED
    repo_root: Path | None = None
    image_shape: tuple[int, int, int, int] = (1, 3, 96, 96)
    state_dim: int = 2


def _synthetic_observation(
    *,
    seed: int,
    image_shape: tuple[int, int, int, int],
    state_dim: int,
    device: str,
) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    images = torch.from_numpy(rng.random(image_shape, dtype=np.float32)).to(device)
    state = torch.from_numpy(
        rng.standard_normal((1, state_dim), dtype=np.float32).astype(np.float32)
    ).to(device)
    return {
        "observation.image": images,
        "observation.state": state,
    }


def run_diffusion_pytorch_profile(req: DiffusionProfileRequest) -> Path:
    if req.n_warmup < 0:
        raise ValueError("n_warmup must be >= 0")
    if req.n_measured < 1:
        raise ValueError("n_measured must be >= 1")

    spec = resolve_candidate(req.candidate)
    if spec.policy_type != "diffusion":
        raise ValueError(f"{req.candidate!r} is not a diffusion candidate")

    root = (req.repo_root or repo_root_from_package()).resolve()
    ensure_lerobot_on_path(root)
    out = (req.output_dir or (root / "profiles" / req.device_label)).resolve()
    out.mkdir(parents=True, exist_ok=True)

    ckpt = download_candidate_checkpoint(spec, root / ".cache" / "huggingface")
    policy = load_policy(spec, str(ckpt), req.device)
    policy.eval()
    runtime = pytorch_full_generation_provenance()
    inference = diffusion_inference_provenance(policy)

    def _one(seed: int) -> tuple[float, np.ndarray]:
        policy.reset()
        obs = _synthetic_observation(
            seed=seed,
            image_shape=req.image_shape,
            state_dim=req.state_dim,
            device=req.device,
        )
        # Feed construction and reset are outside the timed boundary.
        t0 = time.perf_counter_ns()
        chunk = compute_diffusion_action_chunk(policy, obs)
        t1 = time.perf_counter_ns()
        return (t1 - t0) / 1e9, chunk

    for i in range(req.n_warmup):
        _one(seed=10_000 + i)

    measured_s: list[float] = []
    for i in range(req.n_measured):
        dt_s, chunk = _one(seed=20_000 + i)
        if not np.isfinite(chunk).all():
            raise RuntimeError(f"non-finite diffusion chunk during measured call {i}")
        measured_s.append(dt_s)

    measured_s = validate_latency_sequence(measured_s)
    stats = summarize_latency_s(measured_s)
    payload: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "device_label": req.device_label,
        "candidate": req.candidate,
        "hub_revision": spec.hub_revision,
        "host": detect_host_info(),
        "runtime": {
            "library": "pytorch",
            "torch_version": torch.__version__,
            "device": req.device,
            **runtime.to_dict(),
        },
        "model": {
            "precision_recipe": "pytorch_fp32",
            "variant_label": "pytorch_fp32",
            "checkpoint_dir": str(ckpt),
        },
        "inference": inference,
        "chunk_horizon": int(policy.config.n_action_steps),
        "timed_boundary": TIMED_BOUNDARY_DIFFUSION_GENERATE,
        "stages": {
            "warmup_calls": req.n_warmup,
            "measured_calls": req.n_measured,
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
            "corpus": "synthetic_shape_matched",
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
