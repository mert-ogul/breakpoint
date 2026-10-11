"""Thin adapter over a LeRobot PreTrainedPolicy for the episode runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class ChunkInfo:
    chunk_size: int
    n_action_steps: int
    n_obs_steps: int
    temporal_ensemble_coeff: float | None
    control_period_s: float


@dataclass(frozen=True)
class ActionStep:
    """One control-step action plus whether a fresh policy compute ran."""

    action: np.ndarray  # (action_dim,) float32, env units
    requested_fresh_chunk: bool
    inference_call: bool
    control_step_index: int


def _queue_len_before_select(policy: Any) -> int | None:
    """Return remaining cached actions, or None if every step computes (temporal ensemble)."""
    temporal = getattr(getattr(policy, "config", None), "temporal_ensemble_coeff", None)
    if temporal is not None:
        return None
    if hasattr(policy, "_action_queue"):
        return len(policy._action_queue)
    queues = getattr(policy, "_queues", None)
    if isinstance(queues, dict) and "action" in queues:
        return len(queues["action"])
    return 0


class PolicyAdapter:
    """Expose reset / select_action / chunk metadata without reimplementing norms."""

    def __init__(self, policy: Any, *, control_period_s: float, device: str) -> None:
        self._policy = policy
        self._device = device
        self._control_step_index = 0
        cfg = policy.config
        self.chunk_info = ChunkInfo(
            chunk_size=int(cfg.chunk_size) if hasattr(cfg, "chunk_size") else int(cfg.horizon),
            n_action_steps=int(cfg.n_action_steps),
            n_obs_steps=int(cfg.n_obs_steps),
            temporal_ensemble_coeff=getattr(cfg, "temporal_ensemble_coeff", None),
            control_period_s=control_period_s,
        )
        self._policy.eval()

    @property
    def policy(self) -> Any:
        return self._policy

    def reset(self) -> None:
        """Clear action queues, observation history, and temporal aggregation."""
        self._policy.reset()
        self._control_step_index = 0

    def _to_device(self, observation: dict[str, Tensor]) -> dict[str, Tensor]:
        return {
            key: value.to(self._device, non_blocking=True) for key, value in observation.items()
        }

    def compute_action_chunk(self, observation: dict[str, Tensor]) -> np.ndarray:
        """Compute a full action chunk from a captured observation (inference_mode).

        Used by the latency scheduler. Does not consume the policy's internal queue
        across control steps; queue policies are reset around the one-shot drain so
        the scheduler owns buffering under delay.
        """
        if hasattr(self._policy, "compute_action_chunk"):
            chunk = self._policy.compute_action_chunk(observation)
            arr = np.asarray(chunk, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"compute_action_chunk must return 2-D, got {arr.shape}")
            return arr

        temporal = self.chunk_info.temporal_ensemble_coeff
        if temporal is not None:
            raise NotImplementedError(
                "latency chunk compute requires action-queue policies "
                f"(temporal_ensemble_coeff={temporal})"
            )

        batch = self._to_device(observation)
        n = self.chunk_info.n_action_steps
        saved_queue = None
        if hasattr(self._policy, "_action_queue"):
            saved_queue = list(self._policy._action_queue)
            self._policy._action_queue.clear()

        # Drain via policy.select_action without advancing adapter control index.
        actions: list[np.ndarray] = []
        with torch.inference_mode():
            for _ in range(n):
                action_t = self._policy.select_action(batch)
                action = action_t.to("cpu").numpy()
                if action.ndim == 2:
                    action = action[0]
                actions.append(action.astype(np.float32, copy=False))

        if saved_queue is not None:
            from collections import deque

            self._policy._action_queue = deque(saved_queue, maxlen=n)
        elif hasattr(self._policy, "_action_queue"):
            self._policy._action_queue.clear()

        chunk = np.stack(actions, axis=0)
        if not np.isfinite(chunk).all():
            raise ValueError("Policy produced non-finite action chunk")
        return chunk

    def select_action(self, observation: dict[str, Tensor]) -> ActionStep:
        """Return one env action; count inference only when a fresh chunk is computed."""
        queued = _queue_len_before_select(self._policy)
        if queued is None:
            requested_fresh_chunk = True
            inference_call = True
        else:
            requested_fresh_chunk = queued == 0
            inference_call = queued == 0

        batch = self._to_device(observation)
        with torch.inference_mode():
            action_t = self._policy.select_action(batch)

        action = action_t.to("cpu").numpy()
        if action.ndim == 2:
            if action.shape[0] != 1:
                raise ValueError(f"Expected batch size 1 action, got shape {action.shape}")
            action = action[0]
        if action.ndim != 1:
            raise ValueError(f"Expected 1-D action, got shape {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("Policy produced non-finite action")

        step = ActionStep(
            action=action.astype(np.float32, copy=False),
            requested_fresh_chunk=requested_fresh_chunk,
            inference_call=inference_call,
            control_step_index=self._control_step_index,
        )
        self._control_step_index += 1
        return step
