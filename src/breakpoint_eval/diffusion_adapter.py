"""Diffusion Policy adapter pieces that diverge from ACT queue semantics.

Preserves observation history, noise-scheduler configuration, denoising step
count, action horizon slicing, and preprocessing. Fresh action generation is the
full ``generate_actions`` call (all denoising iterations), not one UNet pass.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

TIMED_BOUNDARY_DIFFUSION_GENERATE = "pytorch_generate_actions_full_denoise"
RUNTIME_LABEL_PYTORCH_FULL = "pytorch_full_action_generation"
RUNTIME_LABEL_MIXED_UNET_ONNX = "mixed_onnx_unet_plus_python_scheduler"


@dataclass(frozen=True)
class DiffusionRuntimeProvenance:
    """Labels how policy latency was measured for Diffusion Policy."""

    timed_boundary: str
    runtime_label: str
    includes_all_denoising_iterations: bool
    includes_observation_encode: bool
    includes_noise_scheduler_loop: bool
    includes_unnormalize: bool
    single_unet_pass_only: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timed_boundary": self.timed_boundary,
            "runtime_label": self.runtime_label,
            "includes_all_denoising_iterations": self.includes_all_denoising_iterations,
            "includes_observation_encode": self.includes_observation_encode,
            "includes_noise_scheduler_loop": self.includes_noise_scheduler_loop,
            "includes_unnormalize": self.includes_unnormalize,
            "single_unet_pass_only": self.single_unet_pass_only,
            "note": self.note,
        }


def is_diffusion_policy(policy: Any) -> bool:
    name = getattr(getattr(policy, "config", None), "type", None) or getattr(policy, "name", None)
    return name == "diffusion" or hasattr(policy, "diffusion")


def diffusion_inference_provenance(policy: Any) -> dict[str, Any]:
    cfg = policy.config
    n_steps = int(policy.diffusion.num_inference_steps)
    return {
        "policy_type": "diffusion",
        "n_obs_steps": int(cfg.n_obs_steps),
        "horizon": int(cfg.horizon),
        "n_action_steps": int(cfg.n_action_steps),
        "noise_scheduler_type": cfg.noise_scheduler_type,
        "num_train_timesteps": int(cfg.num_train_timesteps),
        "num_inference_steps": n_steps,
        "prediction_type": cfg.prediction_type,
        "image_features": list(cfg.image_features) if cfg.image_features else [],
        "timed_boundary_default": TIMED_BOUNDARY_DIFFUSION_GENERATE,
        "note": (
            "Policy inference latency is one fresh action-generation call including "
            "all denoising iterations. Timing a single denoising network pass alone "
            "is not policy inference latency."
        ),
    }


def _prepare_normalized_batch(policy: Any, observation: dict[str, Tensor]) -> dict[str, Tensor]:
    batch = policy.normalize_inputs(observation)
    if policy.config.image_features:
        batch = dict(batch)
        batch["observation.images"] = torch.stack(
            [batch[key] for key in policy.config.image_features], dim=-4
        )
    return batch


def observe_diffusion(policy: Any, observation: dict[str, Tensor]) -> None:
    """Update observation history without generating or consuming actions."""
    from lerobot.common.policies.utils import populate_queues

    if policy._queues is None:
        policy.reset()
    batch = _prepare_normalized_batch(policy, observation)
    # Populate only observation queues; leave the action queue untouched.
    obs_only = {k: v for k, v in batch.items() if k != "action"}
    policy._queues = populate_queues(policy._queues, obs_only)


def compute_diffusion_action_chunk(
    policy: Any,
    observation: dict[str, Tensor],
    *,
    generator: torch.Generator | None = None,
) -> np.ndarray:
    """Full fresh action generation for the latency scheduler.

    Updates observation history like upstream ``select_action``, runs all
    denoising iterations via ``diffusion.generate_actions``, returns the
    ``n_action_steps`` chunk, and leaves the policy action queue empty so the
    scheduler owns buffering.
    """
    from lerobot.common.policies.utils import populate_queues

    if policy._queues is None:
        policy.reset()

    saved_action = deque(list(policy._queues["action"]), maxlen=policy._queues["action"].maxlen)
    policy._queues["action"].clear()

    batch = _prepare_normalized_batch(policy, observation)
    policy._queues = populate_queues(policy._queues, batch)

    stacked = {k: torch.stack(list(policy._queues[k]), dim=1) for k in batch if k in policy._queues}
    with torch.inference_mode():
        if generator is not None:
            # Upstream generate_actions does not thread a generator today; keep the
            # hook for profiling/tests that pin RNG without changing default path.
            actions = _generate_actions_with_generator(policy, stacked, generator)
        else:
            actions = policy.diffusion.generate_actions(stacked)
        actions = policy.unnormalize_outputs({"action": actions})["action"]

    chunk = actions[0].detach().to("cpu").numpy().astype(np.float32, copy=False)
    if chunk.ndim != 2:
        raise ValueError(f"diffusion chunk must be 2-D (T,A), got {chunk.shape}")
    if chunk.shape[0] != int(policy.config.n_action_steps):
        raise ValueError(
            f"expected n_action_steps={policy.config.n_action_steps}, got {chunk.shape[0]}"
        )
    if not np.isfinite(chunk).all():
        raise ValueError("Diffusion Policy produced non-finite action chunk")

    # Scheduler owns the chunk; restore any pre-existing unused actions (normally empty).
    policy._queues["action"] = saved_action
    return chunk


def _generate_actions_with_generator(
    policy: Any, batch: dict[str, Tensor], generator: torch.Generator
) -> Tensor:
    """Same slicing as DiffusionModel.generate_actions with an explicit RNG."""
    model = policy.diffusion
    batch_size, n_obs_steps = batch["observation.state"].shape[:2]
    if n_obs_steps != model.config.n_obs_steps:
        raise ValueError(
            f"n_obs_steps mismatch: batch={n_obs_steps} config={model.config.n_obs_steps}"
        )
    global_cond = model._prepare_global_conditioning(batch)
    actions = model.conditional_sample(batch_size, global_cond=global_cond, generator=generator)
    start = n_obs_steps - 1
    end = start + model.config.n_action_steps
    return actions[:, start:end]


def pytorch_full_generation_provenance() -> DiffusionRuntimeProvenance:
    return DiffusionRuntimeProvenance(
        timed_boundary=TIMED_BOUNDARY_DIFFUSION_GENERATE,
        runtime_label=RUNTIME_LABEL_PYTORCH_FULL,
        includes_all_denoising_iterations=True,
        includes_observation_encode=True,
        includes_noise_scheduler_loop=True,
        includes_unnormalize=True,
        single_unet_pass_only=False,
        note=(
            "Timed boundary is one fresh Diffusion Policy action-generation call "
            "in PyTorch, including observation encoding, the full noise-scheduler "
            "loop, and unnormalize. Not a single UNet forward."
        ),
    )


def mixed_unet_onnx_provenance() -> DiffusionRuntimeProvenance:
    return DiffusionRuntimeProvenance(
        timed_boundary="mixed_onnx_unet_forward_plus_python_scheduler",
        runtime_label=RUNTIME_LABEL_MIXED_UNET_ONNX,
        includes_all_denoising_iterations=True,
        includes_observation_encode=True,
        includes_noise_scheduler_loop=True,
        includes_unnormalize=True,
        single_unet_pass_only=False,
        note=(
            "Denoising network may run in ONNX Runtime while the Python noise "
            "scheduler loop and remaining computation stay outside the graph. "
            "Labeled as mixed runtime; not equivalent to a pure ORT action-chunk export."
        ),
    )


def snapshot_diffusion_state(policy: Any) -> dict[str, Any]:
    """Capture queue lengths and scheduler config for reset/validation checks."""
    queues = policy._queues or {}
    return {
        "action_queue_len": len(queues.get("action", [])),
        "obs_queue_lens": {k: len(v) for k, v in queues.items() if k != "action"},
        "n_obs_steps": int(policy.config.n_obs_steps),
        "horizon": int(policy.config.horizon),
        "n_action_steps": int(policy.config.n_action_steps),
        "num_inference_steps": int(policy.diffusion.num_inference_steps),
        "noise_scheduler_type": policy.config.noise_scheduler_type,
    }


def assert_queues_cleared_after_reset(policy: Any) -> None:
    snap = snapshot_diffusion_state(policy)
    if snap["action_queue_len"] != 0:
        raise AssertionError(f"action queue not cleared after reset: {snap}")
    for key, length in snap["obs_queue_lens"].items():
        if length != 0:
            raise AssertionError(f"observation queue {key!r} not cleared after reset: {snap}")
