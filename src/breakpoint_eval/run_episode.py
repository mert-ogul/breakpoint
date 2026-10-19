"""Single-episode rollout with explicit outcome, chunk provenance, and optional latency."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from breakpoint_eval.camera_degrade import (
    FrameDegradeStats,
    ImageDegradeConfig,
    ImageMode,
    degrade_observation,
    summarize_frame_stats,
)
from breakpoint_eval.latency_scheduler import (
    CHUNK_ARRIVAL_POLICY,
    LatencyConfig,
    LatencyMode,
    LatencyScheduler,
    initial_hold_command_from_obs,
)
from breakpoint_eval.latency_trace import LatencyTraceReplayer, TraceExhaustedError
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
    action: list[float] | None = None
    observation_age_s: float | None = None
    buffer_underflow: bool = False
    missed_deadline: bool = False
    from_chunk: bool = True
    request_time_s: float | None = None
    ready_time_s: float | None = None
    chunk_index: int | None = None


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
    policy_rng_seed: int | None = None
    latency_s: float = 0.0
    latency_mode: str = LatencyMode.CONSTANT.value
    chunk_arrival_policy: str = CHUNK_ARRIVAL_POLICY
    latency_events: list[dict[str, Any]] = field(default_factory=list)
    latency_trace_provenance: dict[str, Any] = field(default_factory=dict)
    n_underflows: int = 0
    n_missed_deadlines: int = 0
    image_mode: str = ImageMode.NATIVE.value
    image_degrade_summary: dict[str, Any] = field(default_factory=dict)
    h264_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outcome"] = self.outcome.value
        return payload


def _policy_observation(
    observation: dict[str, Any],
    image_cfg: ImageDegradeConfig,
    frame_stats: list[FrameDegradeStats],
    *,
    h264_session: Any | None = None,
    capture_timestamp_s: float | None = None,
) -> Any:
    """Degrade raw pixels then apply LeRobot preprocess (norms happen inside policy)."""
    degraded = degrade_observation(
        observation,
        image_cfg,
        h264_session=h264_session,
        capture_timestamp_s=capture_timestamp_s,
    )
    frame_stats.extend(degraded.frames)
    return preprocess_observation(degraded.observation)


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


def _classify_outcome(
    *,
    policy_failed: bool,
    is_success: bool,
    terminated: bool,
    truncated: bool,
) -> EpisodeOutcome:
    if policy_failed:
        return EpisodeOutcome.POLICY_FAILED
    if is_success:
        return EpisodeOutcome.SUCCESS
    if truncated and not terminated:
        return EpisodeOutcome.TIME_LIMIT_TRUNCATION
    if truncated and terminated:
        return EpisodeOutcome.TIME_LIMIT_TRUNCATION if not is_success else EpisodeOutcome.SUCCESS
    if terminated:
        return EpisodeOutcome.TERMINATED_NO_SUCCESS
    return EpisodeOutcome.TERMINATED_NO_SUCCESS


def run_episode(
    *,
    env: Any,
    adapter: Any,
    seed: int,
    episode_index: int,
    render_frames: list[np.ndarray] | None = None,
    latency_s: float = 0.0,
    latency_replayer: LatencyTraceReplayer | None = None,
    image_cfg: ImageDegradeConfig | None = None,
    policy_rng_seed: int | None = None,
) -> EpisodeResult:
    """Run one episode; uses env is_success, not reward > 0.

    When ``latency_s == 0`` and the scheduler path is used, behavior matches the
    verified zero-latency control loop (see docs/methodology.md). Host compute
    time is never added to ``latency_s``.

    Image degradation runs on raw rendered pixels before policy normalization and
    is independent of latency injection. Trace replay is distinct from constant
    delay: pass ``latency_replayer`` (with ``latency_s=0``) to inject measured
    ordered latencies.

    ``policy_rng_seed`` is applied separately from the environment ``seed`` so
    Diffusion (and other stochastic policies) can be paired across conditions.
    """
    if latency_replayer is not None and latency_s != 0.0:
        raise ValueError("trace replay requires latency_s=0 (delays come from the profile)")
    if latency_s < 0:
        raise ValueError("latency_s must be >= 0")
    cfg = image_cfg or ImageDegradeConfig()
    # Always use the discrete-event scheduler so L=0 shares the same code path.
    return _run_episode_with_scheduler(
        env=env,
        adapter=adapter,
        seed=seed,
        episode_index=episode_index,
        render_frames=render_frames,
        latency_s=latency_s,
        latency_replayer=latency_replayer,
        image_cfg=cfg,
        policy_rng_seed=policy_rng_seed,
    )


def _run_episode_with_scheduler(
    *,
    env: Any,
    adapter: Any,
    seed: int,
    episode_index: int,
    render_frames: list[np.ndarray] | None,
    latency_s: float,
    image_cfg: ImageDegradeConfig,
    latency_replayer: LatencyTraceReplayer | None = None,
    policy_rng_seed: int | None = None,
) -> EpisodeResult:
    chunk_info = adapter.chunk_info
    steps: list[ControlStepRecord] = []
    sum_reward = 0.0
    max_reward = float("-inf")
    terminated = False
    truncated = False
    is_success = False
    policy_failed = False
    runtime_error: str | None = None
    outcome = EpisodeOutcome.TERMINATED_NO_SUCCESS
    scheduler: LatencyScheduler | None = None
    frame_stats: list[FrameDegradeStats] = []
    h264_session = None
    h264_summary: dict[str, Any] = {}

    try:
        if image_cfg.mode is ImageMode.H264:
            from breakpoint_eval.h264_stream import EpisodeH264Session

            # Fresh encoder/decoder state for this episode (not a prerecorded clip).
            h264_session = EpisodeH264Session(image_cfg.build_h264_stream_config())
        if policy_rng_seed is not None:
            from breakpoint_eval.export_runtime import apply_policy_rng_seed

            apply_policy_rng_seed(policy_rng_seed)
        adapter.reset()
        observation, _info = env.reset(seed=[seed])
        initial = initial_hold_command_from_obs(observation)
        if latency_replayer is not None:
            lat_cfg = LatencyConfig(
                latency_s=0.0,
                control_period_s=chunk_info.control_period_s,
                n_action_steps=chunk_info.n_action_steps,
                mode=LatencyMode.TRACE_REPLAY,
                chunk_arrival_policy=CHUNK_ARRIVAL_POLICY,
            )
        else:
            lat_cfg = LatencyConfig(
                latency_s=latency_s,
                control_period_s=chunk_info.control_period_s,
                n_action_steps=chunk_info.n_action_steps,
                mode=LatencyMode.CONSTANT,
                chunk_arrival_policy=CHUNK_ARRIVAL_POLICY,
            )
        scheduler = LatencyScheduler(
            lat_cfg,
            initial_command=initial,
            latency_replayer=latency_replayer,
        )
        if render_frames is not None:
            frame = env.call("render")[0]
            if frame is not None:
                render_frames.append(np.asarray(frame))

        done = False
        while not done:
            requested = False
            try:
                capture_t = scheduler.sim_time_s if scheduler is not None else 0.0
                obs_t = _policy_observation(
                    observation,
                    image_cfg,
                    frame_stats,
                    h264_session=h264_session,
                    capture_timestamp_s=capture_t,
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                runtime_error = f"{type(exc).__name__}: {exc}"
                outcome = EpisodeOutcome.RUNTIME_ERROR
                LOGGER.warning("Image degrade/preprocess failure at step %s: %s", len(steps), exc)
                break

            if scheduler.needs_inference_request():
                try:
                    action_chunk = adapter.compute_action_chunk(obs_t)
                except (ValueError, NotImplementedError) as exc:
                    policy_failed = True
                    outcome = EpisodeOutcome.POLICY_FAILED
                    LOGGER.warning("Policy failure at step %s: %s", len(steps), exc)
                    break
                try:
                    scheduler.submit_inference(
                        request_time_s=scheduler.sim_time_s,
                        chunk=action_chunk,
                    )
                except TraceExhaustedError as exc:
                    runtime_error = f"{type(exc).__name__}: {exc}"
                    outcome = EpisodeOutcome.RUNTIME_ERROR
                    LOGGER.warning("Latency trace exhausted at step %s: %s", len(steps), exc)
                    break
                requested = True
            else:
                # Diffusion (n_obs_steps>1) must see every control-step observation.
                adapter.observe(obs_t)

            decision = scheduler.decide_apply(requested_this_step=requested)
            action_batch = decision.action[None, ...]
            observation, reward, terminated_arr, truncated_arr, info = env.step(action_batch)
            if h264_session is not None:
                h264_session.advance_time(chunk_info.control_period_s)
            reward_f = float(np.asarray(reward).reshape(-1)[0])
            terminated = bool(np.asarray(terminated_arr).reshape(-1)[0])
            truncated = bool(np.asarray(truncated_arr).reshape(-1)[0])
            is_success = _extract_is_success(info)
            sum_reward += reward_f
            max_reward = max(max_reward, reward_f)
            done = terminated or truncated

            steps.append(
                ControlStepRecord(
                    control_step_index=decision.control_step_index,
                    sim_time_s=decision.sim_time_s + chunk_info.control_period_s,
                    requested_fresh_chunk=decision.requested_fresh_chunk,
                    inference_call=decision.inference_call,
                    reward=reward_f,
                    terminated=terminated,
                    truncated=truncated,
                    is_success=is_success,
                    action=decision.action.astype(float).tolist(),
                    observation_age_s=decision.observation_age_s,
                    buffer_underflow=decision.buffer_underflow,
                    missed_deadline=decision.missed_deadline,
                    from_chunk=decision.from_chunk,
                    request_time_s=decision.request_time_s,
                    ready_time_s=decision.ready_time_s,
                    chunk_index=decision.chunk_index,
                )
            )
            if render_frames is not None:
                frame = env.call("render")[0]
                if frame is not None:
                    render_frames.append(np.asarray(frame))

            if done:
                scheduler.mark_terminated()

        if outcome not in {EpisodeOutcome.POLICY_FAILED, EpisodeOutcome.RUNTIME_ERROR}:
            outcome = _classify_outcome(
                policy_failed=policy_failed,
                is_success=is_success,
                terminated=terminated,
                truncated=truncated,
            )

    except Exception as exc:  # noqa: BLE001 — boundary: classify as runtime_error
        runtime_error = f"{type(exc).__name__}: {exc}"
        outcome = EpisodeOutcome.RUNTIME_ERROR
        LOGGER.exception("Runtime error in episode seed=%s", seed)
    finally:
        if h264_session is not None:
            h264_summary = h264_session.summary()
            h264_session.close()

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
        n_inference_calls=scheduler.n_inference_calls if scheduler is not None else 0,
        n_fresh_chunks=scheduler.n_fresh_chunks if scheduler is not None else 0,
        sum_reward=sum_reward,
        max_reward=max_reward,
        control_period_s=chunk_info.control_period_s,
        chunk_size=chunk_info.chunk_size,
        n_action_steps=chunk_info.n_action_steps,
        steps=steps,
        policy_rng_seed=policy_rng_seed,
        latency_s=latency_s,
        latency_mode=(
            LatencyMode.TRACE_REPLAY.value
            if latency_replayer is not None
            else LatencyMode.CONSTANT.value
        ),
        chunk_arrival_policy=CHUNK_ARRIVAL_POLICY,
        latency_events=scheduler.events_as_dicts() if scheduler is not None else [],
        latency_trace_provenance=(
            scheduler.provenance().get("trace_replay", {}) if scheduler is not None else {}
        ),
        n_underflows=scheduler.n_underflows if scheduler is not None else 0,
        n_missed_deadlines=scheduler.n_missed_deadlines if scheduler is not None else 0,
        image_mode=image_cfg.mode.value,
        image_degrade_summary=summarize_frame_stats(frame_stats, image_cfg),
        h264_summary=h264_summary,
    )


def run_episode_baseline_queue(
    *,
    env: Any,
    adapter: PolicyAdapter,
    seed: int,
    episode_index: int,
    render_frames: list[np.ndarray] | None = None,
    image_cfg: ImageDegradeConfig | None = None,
) -> EpisodeResult:
    """Legacy direct policy-queue loop (no scheduler). Used only for equivalence tests."""
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
    cfg = image_cfg or ImageDegradeConfig()
    frame_stats: list[FrameDegradeStats] = []

    try:
        adapter.reset()
        observation, _info = env.reset(seed=[seed])
        if render_frames is not None:
            frame = env.call("render")[0]
            if frame is not None:
                render_frames.append(np.asarray(frame))

        done = False
        while not done:
            try:
                obs_t = _policy_observation(observation, cfg, frame_stats)
            except (TypeError, ValueError, RuntimeError) as exc:
                runtime_error = f"{type(exc).__name__}: {exc}"
                outcome = EpisodeOutcome.RUNTIME_ERROR
                LOGGER.warning("Image degrade/preprocess failure at step %s: %s", len(steps), exc)
                break
            try:
                action_step: ActionStep = adapter.select_action(obs_t)
            except (ValueError, NotImplementedError) as exc:
                policy_failed = True
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
                    action=action_step.action.astype(float).tolist(),
                    from_chunk=True,
                )
            )
            if render_frames is not None:
                frame = env.call("render")[0]
                if frame is not None:
                    render_frames.append(np.asarray(frame))

        if outcome not in {EpisodeOutcome.POLICY_FAILED, EpisodeOutcome.RUNTIME_ERROR}:
            outcome = _classify_outcome(
                policy_failed=policy_failed,
                is_success=is_success,
                terminated=terminated,
                truncated=truncated,
            )

    except Exception as exc:  # noqa: BLE001
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
        latency_s=0.0,
        chunk_arrival_policy=CHUNK_ARRIVAL_POLICY,
        image_mode=cfg.mode.value,
        image_degrade_summary=summarize_frame_stats(frame_stats, cfg),
    )
