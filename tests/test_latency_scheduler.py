"""Discrete-event latency scheduler: timestamps, discard_stale, reset, equivalence."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from breakpoint_eval.latency_scheduler import (
    CHUNK_ARRIVAL_POLICY,
    LatencyConfig,
    LatencyScheduler,
    SchedulerEventType,
)
from breakpoint_eval.policy_adapter import PolicyAdapter
from breakpoint_eval.run_episode import run_episode, run_episode_baseline_queue

CONTROL_PERIOD_S = 0.02
TIMESTAMP_ATOL_S = 1e-12


class TinyDeterministicEnv:
    """VectorEnv-like stub: records applied actions; ends after max_steps."""

    def __init__(self, *, action_dim: int = 2, max_steps: int = 20) -> None:
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.num_envs = 1
        self._step = 0
        self.applied: list[np.ndarray] = []
        self.apply_times_s: list[float] = []

    def reset(self, seed=None):  # noqa: ANN001
        self._step = 0
        self.applied.clear()
        self.apply_times_s.clear()
        pos = np.full((1, self.action_dim), 0.5, dtype=np.float32)
        obs = {
            "pixels": np.zeros((1, 8, 8, 3), dtype=np.uint8),
            "agent_pos": pos.copy(),
        }
        return obs, {}

    def step(self, action):  # noqa: ANN001
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        self.applied.append(a.copy())
        self.apply_times_s.append(self._step * CONTROL_PERIOD_S)
        self._step += 1
        terminated = False
        truncated = self._step >= self.max_steps
        success = False
        obs = {
            "pixels": np.zeros((1, 8, 8, 3), dtype=np.uint8),
            "agent_pos": a.copy()[None, :],
        }
        info = {"is_success": np.array([success])}
        if truncated or terminated:
            info["final_info"] = [{"is_success": success}]
        return (
            obs,
            np.array([0.0], dtype=np.float32),
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


class TinyChunkPolicy:
    """Deterministic chunks: action[i] = [chunk_id, i]."""

    def __init__(self, n_action_steps: int = 4, action_dim: int = 2) -> None:
        self.config = SimpleNamespace(
            chunk_size=n_action_steps,
            n_action_steps=n_action_steps,
            n_obs_steps=1,
            temporal_ensemble_coeff=None,
        )
        self._action_queue: deque[torch.Tensor] = deque(maxlen=n_action_steps)
        self._chunk_id = 0
        self.action_dim = action_dim

    def eval(self) -> TinyChunkPolicy:
        return self

    def reset(self) -> None:
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        # Keep chunk_id so cross-episode tests can see continuity unless caller resets id.
        # Episode runner calls reset(); for equivalence we zero chunk_id in fixture setup.

    def compute_action_chunk(self, observation) -> np.ndarray:  # noqa: ANN001
        self._chunk_id += 1
        rows = []
        for i in range(self.config.n_action_steps):
            row = np.zeros(self.action_dim, dtype=np.float32)
            row[0] = float(self._chunk_id)
            row[1] = float(i)
            rows.append(row)
        return np.stack(rows, axis=0)

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if len(self._action_queue) == 0:
            chunk = self.compute_action_chunk(batch)
            for row in chunk:
                self._action_queue.append(torch.from_numpy(row).view(1, -1))
        return self._action_queue.popleft()


@pytest.fixture
def fake_preprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    def _preprocess(observation):  # noqa: ANN001
        return {"observation.state": torch.from_numpy(observation["agent_pos"]).float()}

    monkeypatch.setattr("breakpoint_eval.run_episode.preprocess_observation", _preprocess)


def test_chunk_arrival_policy_is_discard_stale() -> None:
    assert CHUNK_ARRIVAL_POLICY == "discard_stale"
    cfg = LatencyConfig(latency_s=0.025, control_period_s=0.02, n_action_steps=4)
    assert cfg.chunk_arrival_policy == "discard_stale"


def test_equal_time_complete_before_apply() -> None:
    sched = LatencyScheduler(
        LatencyConfig(latency_s=0.0, control_period_s=0.02, n_action_steps=2),
        initial_command=np.array([0.5, 0.5], dtype=np.float32),
    )
    chunk = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    sched.submit_inference(request_time_s=0.0, chunk=chunk)
    decision = sched.decide_apply(requested_this_step=True)
    types = [e.event_type for e in sched.events]
    # request, complete, apply — complete before apply at t=0
    assert types[:3] == [
        SchedulerEventType.INFERENCE_REQUEST,
        SchedulerEventType.INFERENCE_COMPLETE,
        SchedulerEventType.CONTROL_APPLY,
    ]
    assert decision.from_chunk is True
    np.testing.assert_allclose(decision.action, [1.0, 0.0])


def test_noninteger_delay_25ms_with_20ms_period() -> None:
    sched = LatencyScheduler(
        LatencyConfig(latency_s=0.025, control_period_s=0.02, n_action_steps=4),
        initial_command=np.array([9.0, 9.0], dtype=np.float32),
    )
    chunk = np.stack(
        [np.array([1.0, float(i)], dtype=np.float32) for i in range(4)],
        axis=0,
    )
    assert sched.needs_inference_request()
    sched.submit_inference(request_time_s=0.0, chunk=chunk)

    d0 = sched.decide_apply(requested_this_step=True)
    assert d0.buffer_underflow is True
    np.testing.assert_allclose(d0.action, [9.0, 9.0])
    assert d0.sim_time_s == pytest.approx(0.0, abs=TIMESTAMP_ATOL_S)

    d1 = sched.decide_apply(requested_this_step=False)
    assert d1.buffer_underflow is True  # ready at 0.025 > 0.02
    assert d1.sim_time_s == pytest.approx(0.02, abs=TIMESTAMP_ATOL_S)

    d2 = sched.decide_apply(requested_this_step=False)
    assert d2.from_chunk is True
    # discard i=0 (t=0) and i=1 (t=0.02); keep i=2 (t=0.04)
    np.testing.assert_allclose(d2.action, [1.0, 2.0])
    assert d2.chunk_index == 2
    assert d2.observation_age_s == pytest.approx(0.04, abs=TIMESTAMP_ATOL_S)
    assert d2.ready_time_s == pytest.approx(0.025, abs=TIMESTAMP_ATOL_S)
    complete = [e for e in sched.events if e.event_type == SchedulerEventType.INFERENCE_COMPLETE][0]
    assert complete.time_s == pytest.approx(0.025, abs=TIMESTAMP_ATOL_S)


def test_delay_exceeding_chunk_horizon_missed_deadline() -> None:
    n = 3
    dt = 0.02
    latency = n * dt + 0.001  # longer than last intended time
    sched = LatencyScheduler(
        LatencyConfig(latency_s=latency, control_period_s=dt, n_action_steps=n),
        initial_command=np.array([3.0, 3.0], dtype=np.float32),
    )
    chunk = np.stack([np.array([1.0, float(i)], dtype=np.float32) for i in range(n)], axis=0)
    sched.submit_inference(request_time_s=0.0, chunk=chunk)

    # Advance to first apply after ready
    steps_needed = int(np.ceil(latency / dt))
    last = None
    for i in range(steps_needed + 1):
        last = sched.decide_apply(requested_this_step=(i == 0))
    assert last is not None
    assert sched.n_missed_deadlines == 1
    assert last.buffer_underflow is True
    np.testing.assert_allclose(last.action, [3.0, 3.0])


def test_startup_hold_is_initial_agent_pos_not_zero(fake_preprocess: None) -> None:
    env = TinyDeterministicEnv(max_steps=3)
    policy = TinyChunkPolicy(n_action_steps=4)
    adapter = PolicyAdapter(policy, control_period_s=CONTROL_PERIOD_S, device="cpu")
    result = run_episode(env=env, adapter=adapter, seed=0, episode_index=0, latency_s=0.05)
    # First applies before ready use initial hold 0.5, not zeros.
    assert result.steps[0].buffer_underflow is True
    np.testing.assert_allclose(env.applied[0], [0.5, 0.5])
    assert not np.allclose(env.applied[0], 0.0)


def test_episode_reset_clears_pending_and_buffer() -> None:
    sched = LatencyScheduler(
        LatencyConfig(latency_s=0.1, control_period_s=0.02, n_action_steps=2),
        initial_command=np.array([1.0, 1.0], dtype=np.float32),
    )
    chunk = np.array([[2.0, 0.0], [2.0, 1.0]], dtype=np.float32)
    sched.submit_inference(request_time_s=0.0, chunk=chunk)
    sched.decide_apply(requested_this_step=True)
    sched.reset(initial_command=np.array([4.0, 4.0], dtype=np.float32))
    assert sched.needs_inference_request()
    assert sched.n_inference_calls == 0
    d = sched.decide_apply(requested_this_step=False)
    # No pending after reset → underflow to new initial
    np.testing.assert_allclose(d.action, [4.0, 4.0])


def test_no_apply_after_terminate() -> None:
    sched = LatencyScheduler(
        LatencyConfig(latency_s=0.0, control_period_s=0.02, n_action_steps=2),
        initial_command=np.array([1.0, 1.0], dtype=np.float32),
    )
    chunk = np.array([[5.0, 0.0], [5.0, 1.0]], dtype=np.float32)
    sched.submit_inference(request_time_s=0.0, chunk=chunk)
    sched.decide_apply(requested_this_step=True)
    sched.mark_terminated()
    assert sched.needs_inference_request() is False
    with pytest.raises(RuntimeError):
        sched.decide_apply(requested_this_step=False)


def test_zero_delay_matches_baseline_queue_trajectory(fake_preprocess: None) -> None:
    env_a = TinyDeterministicEnv(max_steps=10)
    env_b = TinyDeterministicEnv(max_steps=10)
    policy_a = TinyChunkPolicy(n_action_steps=4)
    policy_b = TinyChunkPolicy(n_action_steps=4)
    # Align chunk id sequences: both start at 0; compute_action_chunk increments.
    # baseline select_action also increments via compute inside select_action.
    adapter_a = PolicyAdapter(policy_a, control_period_s=CONTROL_PERIOD_S, device="cpu")
    adapter_b = PolicyAdapter(policy_b, control_period_s=CONTROL_PERIOD_S, device="cpu")

    sched_result = run_episode(env=env_a, adapter=adapter_a, seed=0, episode_index=0, latency_s=0.0)
    base_result = run_episode_baseline_queue(env=env_b, adapter=adapter_b, seed=0, episode_index=0)

    assert sched_result.n_control_steps == base_result.n_control_steps
    assert len(env_a.applied) == len(env_b.applied)
    for a, b in zip(env_a.applied, env_b.applied, strict=True):
        np.testing.assert_array_equal(a, b)
    assert sched_result.n_inference_calls == base_result.n_inference_calls
    assert sched_result.chunk_arrival_policy == CHUNK_ARRIVAL_POLICY
