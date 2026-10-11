"""Single-episode rollout with explicit outcome and chunk provenance."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from breakpoint_eval.policy_adapter import ActionStep, PolicyAdapter
from breakpoint_eval.sim_env import preprocess_observation

LOGGER = logging.getLogger("breakpoint_eval.run_episode")


class EpisodeOutcome(StrEnum):
    SUCCESS = "success"
    TERMINATED_NO_SUCCESS = "terminated_no_success"
    TIME_LIMIT_TRUNCATION = "time_limit_truncation"
    POLICY_FAILED = "policy_failed"
    RUNTIME_ERROR = "runtime_error"


@dataclass
class ControlStepRecord:
    control_step_index: int
    sim_time_s: float
    requested_fresh_chunk: bool
    inference_call: bool
    reward: float
    terminated: bool
    truncated: bool
    is_success: bool


@dataclass
class EpisodeResult:
    episode_index: int
    seed: int
    outcome: EpisodeOutcome
    success: bool
    terminated: bool
    truncated: bool
    policy_failed: bool
    runtime_error: str | None
    n_control_steps: int
    n_inference_calls: int
    n_fresh_chunks: int
    sum_reward: float
    max_reward: float
    control_period_s: float
    chunk_size: int
    n_action_steps: int
    steps: list[ControlStepRecord] = field(default_factory=list)
    video_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outcome"] = self.outcome.value
        return payload


def _extract_is_success(info: dict[str, Any], *, env_index: int = 0) -> bool:
    if "final_info" in info:
        final = info["final_info"]
        if final is None:
            return False
        entry = final[env_index]
        if entry is None:
            return False
        return bool(entry.get("is_success", False))
    if "is_success" in info:
        value = info["is_success"]
        if isinstance(value, (list, tuple, np.ndarray)):
            return bool(value[env_index])
        return bool(value)
    return False


def run_episode(
    *,
    env: Any,
    adapter: PolicyAdapter,
    seed: int,
    episode_index: int,
    render_frames: list[np.ndarray] | None = None,
) -> EpisodeResult:
    """Run one episode; uses env is_success, not reward > 0."""
    chunk = adapter.chunk_info
    steps: list[ControlStepRecord] = []
    n_inference = 0
    n_fresh = 0
    sum_reward = 0.0
    max_reward = float("-inf")
    terminated = False
    truncated = False
    is_success = False
    policy_failed = False
    runtime_error: str | None = None
    outcome = EpisodeOutcome.TERMINATED_NO_SUCCESS

    try:
        adapter.reset()
        observation, _info = env.reset(seed=[seed])
        if render_frames is not None:
            frame = env.call("render")[0]
            if frame is not None:
                render_frames.append(np.asarray(frame))

        done = False
        while not done:
            obs_t = preprocess_observation(observation)
            try:
                action_step: ActionStep = adapter.select_action(obs_t)
            except ValueError as exc:
                policy_failed = True
                runtime_error = None
                outcome = EpisodeOutcome.POLICY_FAILED
                LOGGER.warning("Policy failure at step %s: %s", len(steps), exc)
                break

            if action_step.inference_call:
                n_inference += 1
            if action_step.requested_fresh_chunk:
                n_fresh += 1

            action_batch = action_step.action[None, ...]
            observation, reward, terminated_arr, truncated_arr, info = env.step(action_batch)
            reward_f = float(np.asarray(reward).reshape(-1)[0])
            terminated = bool(np.asarray(terminated_arr).reshape(-1)[0])
            truncated = bool(np.asarray(truncated_arr).reshape(-1)[0])
            is_success = _extract_is_success(info)
            sum_reward += reward_f
            max_reward = max(max_reward, reward_f)
            done = terminated or truncated

            steps.append(
                ControlStepRecord(
                    control_step_index=action_step.control_step_index,
                    sim_time_s=(action_step.control_step_index + 1) * chunk.control_period_s,
                    requested_fresh_chunk=action_step.requested_fresh_chunk,
                    inference_call=action_step.inference_call,
                    reward=reward_f,
                    terminated=terminated,
                    truncated=truncated,
                    is_success=is_success,
                )
            )
            if render_frames is not None:
                frame = env.call("render")[0]
                if frame is not None:
                    render_frames.append(np.asarray(frame))

        if not policy_failed:
            if is_success:
                outcome = EpisodeOutcome.SUCCESS
            elif truncated and not terminated:
                outcome = EpisodeOutcome.TIME_LIMIT_TRUNCATION
            elif truncated and terminated:
                # TimeLimit often sets both; prefer success already handled.
                outcome = (
                    EpisodeOutcome.TIME_LIMIT_TRUNCATION
                    if not is_success
                    else EpisodeOutcome.SUCCESS
                )
            elif terminated:
                outcome = EpisodeOutcome.TERMINATED_NO_SUCCESS
            else:
                outcome = EpisodeOutcome.TERMINATED_NO_SUCCESS

    except Exception as exc:  # noqa: BLE001 — boundary: classify as runtime_error
        runtime_error = f"{type(exc).__name__}: {exc}"
        outcome = EpisodeOutcome.RUNTIME_ERROR
        LOGGER.exception("Runtime error in episode seed=%s", seed)

    if max_reward == float("-inf"):
        max_reward = 0.0

    return EpisodeResult(
        episode_index=episode_index,
        seed=seed,
        outcome=outcome,
        success=bool(is_success and outcome == EpisodeOutcome.SUCCESS),
        terminated=terminated,
        truncated=truncated,
        policy_failed=policy_failed,
        runtime_error=runtime_error,
        n_control_steps=len(steps),
        n_inference_calls=n_inference,
        n_fresh_chunks=n_fresh,
        sum_reward=sum_reward,
        max_reward=max_reward,
        control_period_s=chunk.control_period_s,
        chunk_size=chunk.chunk_size,
        n_action_steps=chunk.n_action_steps,
        steps=steps,
    )
