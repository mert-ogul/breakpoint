"""Adapter must match upstream select_action outputs and chunk cadence."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from breakpoint_eval.policy_adapter import PolicyAdapter
from breakpoint_eval.run_baseline import agreement_action_trace


class _UpstreamLikePolicy:
    def __init__(self, n_action_steps: int = 3) -> None:
        self.config = SimpleNamespace(
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

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if len(self._action_queue) == 0:
            self._chunk_id += 1
            chunk = torch.arange(self.config.n_action_steps, dtype=torch.float32).view(-1, 1, 1)
            chunk = chunk + 10.0 * self._chunk_id
            chunk = chunk.expand(-1, 1, 2).contiguous()
            self._action_queue.extend(list(chunk))
        return self._action_queue.popleft()


def test_adapter_actions_match_direct_upstream_select_action() -> None:
    policy_a = _UpstreamLikePolicy()
    policy_b = _UpstreamLikePolicy()
    adapter = PolicyAdapter(policy_a, control_period_s=0.02, device="cpu")

    observations = [{"observation.state": torch.zeros(1, 2)} for _ in range(7)]
    adapter_trace = agreement_action_trace(adapter=adapter, observations=observations)

    policy_b.reset()
    upstream_actions = []
    for obs in observations:
        with torch.inference_mode():
            action = policy_b.select_action(obs).numpy()[0]
        upstream_actions.append(action)

    for row, expected in zip(adapter_trace, upstream_actions, strict=True):
        np.testing.assert_allclose(row["action"], expected, rtol=0, atol=0)

    # Fresh chunk every n_action_steps, matching upstream queue depletion.
    assert [row["requested_fresh_chunk"] for row in adapter_trace] == [
        True,
        False,
        False,
        True,
        False,
        False,
        True,
    ]


@pytest.mark.integration
def test_real_act_adapter_matches_policy_factory_actions() -> None:
    """Optional: requires vendor LeRobot + downloaded ACT checkpoint."""
    from breakpoint_eval.sim_env import load_policy
    from breakpoint_eval.stack import (
        download_candidate_checkpoint,
        ensure_lerobot_on_path,
        repo_root_from_package,
        resolve_candidate,
    )

    repo_root = repo_root_from_package()
    try:
        ensure_lerobot_on_path(repo_root)
        spec = resolve_candidate("act_aloha_transfer")
        ckpt = download_candidate_checkpoint(spec, repo_root / ".cache" / "huggingface")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"stack unavailable: {exc}")

    device = "cpu"
    policy = load_policy(spec, str(ckpt), device)
    adapter = PolicyAdapter(policy, control_period_s=1.0 / 50.0, device=device)

    obs = {
        "observation.state": torch.zeros(1, 14),
        "observation.images.top": torch.rand(1, 3, 480, 640),
    }
    # Clone policy state by resetting both paths from a fresh reset.
    adapter.reset()
    a0 = adapter.select_action(obs)

    policy.reset()
    with torch.inference_mode():
        b0 = policy.select_action({k: v.clone() for k, v in obs.items()}).cpu().numpy()[0]
    np.testing.assert_allclose(a0.action, b0, rtol=1e-5, atol=1e-5)
    assert a0.inference_call is True
