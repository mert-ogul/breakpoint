"""Outcome classification and success-rate denominator integrity."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

from breakpoint_eval.policy_adapter import PolicyAdapter
from breakpoint_eval.run_episode import EpisodeOutcome, run_episode
from breakpoint_eval.write_report import summarize_episodes


class _QueuePolicy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            chunk_size=2,
            n_action_steps=2,
            n_obs_steps=1,
            temporal_ensemble_coeff=None,
        )
        self._action_queue: deque[torch.Tensor] = deque(maxlen=2)

    def eval(self) -> _QueuePolicy:
        return self

    def reset(self) -> None:
        self._action_queue = deque([], maxlen=2)

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if len(self._action_queue) == 0:
            self._action_queue.extend([torch.zeros(1, 2), torch.zeros(1, 2)])
        return self._action_queue.popleft()


class _ScriptedVectorEnv:
    """Minimal SyncVectorEnv-like stub with is_success in info."""

    def __init__(self, script: list[tuple[float, bool, bool, bool]]) -> None:
        # each entry: reward, terminated, truncated, is_success
        self._script = script
        self._i = 0
        self.num_envs = 1

    def reset(self, seed=None):  # noqa: ANN001
        self._i = 0
        obs = {
            "pixels": np.zeros((1, 16, 16, 3), dtype=np.uint8),
            "agent_pos": np.zeros((1, 2), dtype=np.float32),
        }
        return obs, {}

    def step(self, action):  # noqa: ANN001
        reward, terminated, truncated, is_success = self._script[self._i]
        self._i += 1
        obs = {
            "pixels": np.zeros((1, 16, 16, 3), dtype=np.uint8),
            "agent_pos": np.zeros((1, 2), dtype=np.float32),
        }
        info: dict = {"is_success": np.array([is_success])}
        if terminated or truncated:
            info["final_info"] = [{"is_success": is_success}]
        return (
            obs,
            np.array([reward], dtype=np.float32),
            np.array([terminated]),
            np.array([truncated]),
            info,
        )

    def call(self, name: str):  # noqa: ANN001
        if name == "render":
            return [None]
        raise AttributeError(name)

    def close(self) -> None:
        return None


def test_success_uses_env_predicate_not_positive_reward(monkeypatch) -> None:  # noqa: ANN001
    # Reward > 0 but not success until final step.
    env = _ScriptedVectorEnv(
        [
            (1.0, False, False, False),
            (2.0, False, False, False),
            (4.0, True, False, True),
        ]
    )
    adapter = PolicyAdapter(_QueuePolicy(), control_period_s=0.02, device="cpu")

    # Avoid importing real lerobot preprocess in this unit test.
    def fake_preprocess(observation):  # noqa: ANN001
        return {
            "observation.state": torch.from_numpy(observation["agent_pos"]).float(),
            "observation.images.top": torch.from_numpy(observation["pixels"])
            .permute(0, 3, 1, 2)
            .float()
            / 255.0,
        }

    monkeypatch.setattr("breakpoint_eval.run_episode.preprocess_observation", fake_preprocess)
    result = run_episode(env=env, adapter=adapter, seed=7, episode_index=0)
    assert result.outcome == EpisodeOutcome.SUCCESS
    assert result.success is True
    assert result.n_inference_calls == 2  # 3 steps, chunk size 2 → infer at 0 and 2
    assert result.n_control_steps == 3


def test_time_limit_truncation_outcome(monkeypatch) -> None:  # noqa: ANN001
    env = _ScriptedVectorEnv([(0.0, False, True, False)])
    adapter = PolicyAdapter(_QueuePolicy(), control_period_s=0.02, device="cpu")

    def fake_preprocess(observation):  # noqa: ANN001
        return {"observation.state": torch.from_numpy(observation["agent_pos"]).float()}

    monkeypatch.setattr("breakpoint_eval.run_episode.preprocess_observation", fake_preprocess)
    result = run_episode(env=env, adapter=adapter, seed=1, episode_index=0)
    assert result.outcome == EpisodeOutcome.TIME_LIMIT_TRUNCATION
    assert result.truncated is True
    assert result.success is False


def test_runtime_errors_excluded_from_success_denominator(monkeypatch) -> None:  # noqa: ANN001
    class BoomEnv(_ScriptedVectorEnv):
        def step(self, action):  # noqa: ANN001
            raise RuntimeError("sim exploded")

    adapter = PolicyAdapter(_QueuePolicy(), control_period_s=0.02, device="cpu")

    def fake_preprocess(observation):  # noqa: ANN001
        return {"observation.state": torch.from_numpy(observation["agent_pos"]).float()}

    monkeypatch.setattr("breakpoint_eval.run_episode.preprocess_observation", fake_preprocess)
    errored = run_episode(env=BoomEnv([]), adapter=adapter, seed=0, episode_index=0)
    assert errored.outcome == EpisodeOutcome.RUNTIME_ERROR

    ok_env = _ScriptedVectorEnv([(4.0, True, False, True)])
    ok = run_episode(env=ok_env, adapter=adapter, seed=1, episode_index=1)
    summary = summarize_episodes([errored, ok])
    assert summary["n_runtime_error"] == 1
    assert summary["n_episodes_eligible"] == 1
    assert summary["n_success"] == 1
    assert summary["success_rate"] == 1.0
