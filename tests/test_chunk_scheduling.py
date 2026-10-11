"""Chunk scheduling and cross-episode reset for PolicyAdapter."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

from breakpoint_eval.policy_adapter import PolicyAdapter


class FakeChunkPolicy:
    """Mimics ACT queue cadence: infer every n_action_steps control steps."""

    def __init__(self, n_action_steps: int = 4, action_dim: int = 3) -> None:
        self.config = SimpleNamespace(
            chunk_size=n_action_steps,
            n_action_steps=n_action_steps,
            n_obs_steps=1,
            temporal_ensemble_coeff=None,
        )
        self._action_queue: deque[torch.Tensor] = deque(maxlen=n_action_steps)
        self._action_dim = action_dim
        self.infer_calls = 0
        self.reset_calls = 0

    def eval(self) -> FakeChunkPolicy:
        return self

    def reset(self) -> None:
        self.reset_calls += 1
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if len(self._action_queue) == 0:
            self.infer_calls += 1
            # Distinct chunk values so tests can see reuse vs refresh.
            base = float(self.infer_calls)
            actions = torch.full((self.config.n_action_steps, 1, self._action_dim), base)
            self._action_queue.extend(list(actions))
        return self._action_queue.popleft()


def test_cached_actions_are_not_inference_calls() -> None:
    policy = FakeChunkPolicy(n_action_steps=4)
    adapter = PolicyAdapter(policy, control_period_s=0.02, device="cpu")
    obs = {"observation.state": torch.zeros(1, 2)}

    flags = [adapter.select_action(obs) for _ in range(9)]
    assert [s.inference_call for s in flags] == [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]
    assert [s.requested_fresh_chunk for s in flags] == [s.inference_call for s in flags]
    assert policy.infer_calls == 3
    assert flags[0].action[0] == 1.0
    assert flags[3].action[0] == 1.0
    assert flags[4].action[0] == 2.0


def test_reset_clears_queue_between_episodes() -> None:
    policy = FakeChunkPolicy(n_action_steps=4)
    adapter = PolicyAdapter(policy, control_period_s=0.02, device="cpu")
    obs = {"observation.state": torch.zeros(1, 2)}

    first = adapter.select_action(obs)
    assert first.inference_call is True
    assert first.action[0] == 1.0

    adapter.reset()
    assert policy.reset_calls == 1
    # After reset, next step must request a fresh chunk again.
    second = adapter.select_action(obs)
    assert second.inference_call is True
    assert second.control_step_index == 0
    assert second.action[0] == 2.0


def test_temporal_ensemble_counts_every_step_as_inference() -> None:
    class EnsemblePolicy:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                chunk_size=8,
                n_action_steps=8,
                n_obs_steps=1,
                temporal_ensemble_coeff=0.01,
            )
            self.infer_calls = 0

        def eval(self) -> EnsemblePolicy:
            return self

        def reset(self) -> None:
            return None

        def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
            self.infer_calls += 1
            return torch.zeros(1, 2)

    policy = EnsemblePolicy()
    adapter = PolicyAdapter(policy, control_period_s=0.1, device="cpu")
    obs = {"observation.state": torch.zeros(1, 2)}
    steps = [adapter.select_action(obs) for _ in range(3)]
    assert all(s.inference_call and s.requested_fresh_chunk for s in steps)
    assert policy.infer_calls == 3
    assert np.isfinite(steps[0].action).all()
