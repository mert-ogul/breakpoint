"""Unit tests for multi-candidate support, diffusion adapter, and validation gates."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from breakpoint_eval.candidate_map import CANDIDATES
from breakpoint_eval.diffusion_adapter import (
    compute_diffusion_action_chunk,
    is_diffusion_policy,
    observe_diffusion,
)
from breakpoint_eval.model_resolve import resolve_model_to_candidate
from breakpoint_eval.policy_adapter import PolicyAdapter
from breakpoint_eval.policy_validate import (
    assert_gates_passed,
    run_policy_validation_gates,
    validate_success_semantics,
    validate_zero_latency_equivalence,
)
from breakpoint_eval.stack import ensure_lerobot_on_path, repo_root_from_package
from breakpoint_eval.support_status import SUPPORT_MATRIX, support_row


def _ensure_lerobot() -> None:
    ensure_lerobot_on_path(repo_root_from_package())


def _populate_queues(queues, batch):
    for key in batch:
        if key not in queues:
            continue
        if len(queues[key]) != queues[key].maxlen:
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            queues[key].append(batch[key])
    return queues


class _UpstreamLikePolicy:
    def __init__(self, n_action_steps: int = 3) -> None:
        self.config = SimpleNamespace(
            type="act",
            chunk_size=n_action_steps,
            n_action_steps=n_action_steps,
            n_obs_steps=1,
            temporal_ensemble_coeff=None,
        )
        self._action_queue: deque[torch.Tensor] = deque(maxlen=n_action_steps)
        self._chunk_id = 0

    def eval(self) -> _UpstreamLikePolicy:
        return self

    def reset(self) -> None:
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self._chunk_id = 0

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if len(self._action_queue) == 0:
            self._chunk_id += 1
            chunk = torch.arange(self.config.n_action_steps, dtype=torch.float32).view(-1, 1, 1)
            chunk = chunk + 10.0 * self._chunk_id
            chunk = chunk.expand(-1, 1, 2).contiguous()
            self._action_queue.extend(list(chunk))
        return self._action_queue.popleft()


class _TinyDiffusion:
    """Minimal Diffusion-like policy for offline tests (not a real UNet)."""

    name = "diffusion"

    def __init__(
        self, *, n_obs_steps: int = 2, n_action_steps: int = 3, action_dim: int = 2
    ) -> None:
        self.config = SimpleNamespace(
            type="diffusion",
            n_obs_steps=n_obs_steps,
            horizon=n_obs_steps + n_action_steps,
            n_action_steps=n_action_steps,
            noise_scheduler_type="DDPM",
            num_train_timesteps=10,
            prediction_type="epsilon",
            image_features=["observation.image"],
            temporal_ensemble_coeff=None,
        )
        self._chunk_id = 0
        self.diffusion = SimpleNamespace(num_inference_steps=4, generate_actions=self._generate)
        self._queues = None
        self.reset()

    def eval(self) -> _TinyDiffusion:
        return self

    def reset(self) -> None:
        self._chunk_id = 0
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "observation.images": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }

    def normalize_inputs(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return dict(batch)

    def unnormalize_outputs(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return batch

    def _generate(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self._chunk_id += 1
        b = batch["observation.state"].shape[0]
        t = self.config.n_action_steps
        a = 2
        base = float(self._chunk_id * 10)
        # Depend on obs history so observe() vs missing history would diverge.
        hist = batch["observation.state"].mean().item()
        actions = torch.arange(t, dtype=torch.float32).view(1, t, 1).expand(b, t, a).clone()
        actions = actions + base + hist
        return actions

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch = self.normalize_inputs(batch)
        batch = dict(batch)
        batch["observation.images"] = torch.stack(
            [batch[key] for key in self.config.image_features], dim=-4
        )
        self._queues = _populate_queues(self._queues, batch)
        if len(self._queues["action"]) == 0:
            stacked = {
                k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues
            }
            actions = self.diffusion.generate_actions(stacked)
            actions = self.unnormalize_outputs({"action": actions})["action"]
            self._queues["action"].extend(actions.transpose(0, 1))
        return self._queues["action"].popleft()


def test_candidates_and_hub_resolve() -> None:
    assert set(CANDIDATES) == {"act_aloha_transfer", "act_aloha_insertion", "diffusion_pusht"}
    key, spec = resolve_model_to_candidate("lerobot/act_aloha_sim_insertion_human")
    assert key == "act_aloha_insertion"
    assert spec.policy_type == "act"
    key, spec = resolve_model_to_candidate("lerobot/diffusion_pusht")
    assert key == "diffusion_pusht"
    assert spec.policy_type == "diffusion"


def test_support_matrix_no_cross_task_pooling_flags() -> None:
    for name, row in SUPPORT_MATRIX.items():
        assert row.candidate == name
        assert "pool" not in row.notes.lower() or "do not pool" in row.notes.lower()
    assert support_row("diffusion_pusht").onnx_fp32 == "unsupported"
    assert support_row("act_aloha_insertion").policy_type == "act"


def test_success_semantics_documented_for_all_candidates() -> None:
    for spec in CANDIDATES.values():
        gate = validate_success_semantics(spec)
        assert gate.passed, gate.detail


def test_tiny_diffusion_zero_latency_and_observe() -> None:
    _ensure_lerobot()
    policy = _TinyDiffusion()
    assert is_diffusion_policy(policy)
    adapter = PolicyAdapter(policy, control_period_s=0.1, device="cpu")
    assert adapter.chunk_info.policy_type == "diffusion"
    assert adapter.chunk_info.n_obs_steps == 2

    obs_stream = []
    for i in range(7):
        obs_stream.append(
            {
                "observation.state": torch.full((1, 2), float(i), dtype=torch.float32),
                "observation.image": torch.zeros(1, 3, 8, 8),
            }
        )
    gate = validate_zero_latency_equivalence(adapter, obs_stream, atol=0, rtol=0)
    assert gate.passed, gate.detail

    # observe advances history without filling the action queue
    adapter.reset()
    observe_diffusion(
        policy,
        {
            "observation.state": torch.ones(1, 2),
            "observation.image": torch.zeros(1, 3, 8, 8),
        },
    )
    assert len(policy._queues["action"]) == 0
    assert len(policy._queues["observation.state"]) == 2

    chunk = compute_diffusion_action_chunk(
        policy,
        {
            "observation.state": torch.full((1, 2), 3.0),
            "observation.image": torch.zeros(1, 3, 8, 8),
        },
    )
    assert chunk.shape == (3, 2)
    assert len(policy._queues["action"]) == 0


def test_act_like_adapter_still_zero_latency() -> None:
    policy = _UpstreamLikePolicy(n_action_steps=3)
    adapter = PolicyAdapter(policy, control_period_s=0.02, device="cpu")
    obs = [{"observation.state": torch.zeros(1, 2)} for _ in range(7)]
    gate = validate_zero_latency_equivalence(adapter, obs, atol=0, rtol=0)
    assert gate.passed, gate.detail


def test_run_policy_validation_gates_tiny_diffusion() -> None:
    _ensure_lerobot()
    policy = _TinyDiffusion()
    adapter = PolicyAdapter(policy, control_period_s=0.1, device="cpu")
    pol_obs = {
        "observation.state": torch.zeros(1, 2),
        "observation.image": torch.zeros(1, 3, 8, 8),
    }
    raw = {
        "pixels": np.zeros((96, 96, 3), dtype=np.uint8),
        "agent_pos": np.zeros((2,), dtype=np.float32),
    }
    results = run_policy_validation_gates(
        spec=CANDIDATES["diffusion_pusht"],
        adapter=adapter,
        policy_observation=pol_obs,
        raw_observation=raw,
        zero_latency_observations=[
            {
                "observation.state": torch.full((1, 2), float(i)),
                "observation.image": torch.zeros(1, 3, 8, 8),
            }
            for i in range(5)
        ],
    )
    assert_gates_passed(results)


def test_profile_rejects_diffusion_ort() -> None:
    from pathlib import Path

    from breakpoint_eval.run_profile import ProfileCliRequest, run_profile

    with pytest.raises(ValueError, match="Diffusion Policy"):
        run_profile(
            ProfileCliRequest(
                model_path=Path("/tmp/missing.onnx"),
                corpus_path=Path("/tmp/missing.npz"),
                output_dir=Path("/tmp/out"),
                device_label="rpi5",
                candidate="diffusion_pusht",
                recipe_id="torch_onnx_export_fp32_opset17",
            )
        )
