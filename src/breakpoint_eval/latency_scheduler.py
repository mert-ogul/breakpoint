"""Discrete-event latency scheduler on simulation/control time.

Supports constant delay and ordered device-trace replay. Host wall time spent
computing a chunk is never added to the simulated delay.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from breakpoint_eval.latency_trace import LatencyTraceReplayer

# Fixed for ALOHA absolute joint-target chunks; recorded in every result.
CHUNK_ARRIVAL_POLICY = "discard_stale"


class SchedulerEventType(StrEnum):
    INFERENCE_COMPLETE = "inference_complete"
    CONTROL_APPLY = "control_apply"
    INFERENCE_REQUEST = "inference_request"


class LatencyMode(StrEnum):
    CONSTANT = "constant"
    TRACE_REPLAY = "trace_replay"


@dataclass(frozen=True)
class LatencyConfig:
    control_period_s: float
    n_action_steps: int
    latency_s: float = 0.0
    mode: LatencyMode = LatencyMode.CONSTANT
    chunk_arrival_policy: str = CHUNK_ARRIVAL_POLICY

    def __post_init__(self) -> None:
        if self.mode is LatencyMode.CONSTANT and self.latency_s < 0:
            raise ValueError("latency_s must be >= 0")
        if self.control_period_s <= 0:
            raise ValueError("control_period_s must be > 0")
        if self.n_action_steps < 1:
            raise ValueError("n_action_steps must be >= 1")
        if self.chunk_arrival_policy != CHUNK_ARRIVAL_POLICY:
            raise ValueError(
                f"Unsupported chunk_arrival_policy={self.chunk_arrival_policy!r}; "
                f"only {CHUNK_ARRIVAL_POLICY!r} is implemented"
            )
        if self.mode is LatencyMode.TRACE_REPLAY and self.latency_s != 0.0:
            # Constant field unused in trace mode; keep at 0 to avoid mixed semantics.
            raise ValueError("TRACE_REPLAY mode requires latency_s=0 (delay comes from the trace)")


@dataclass
class PendingInference:
    request_time_s: float
    ready_time_s: float
    chunk: np.ndarray  # (n_action_steps, action_dim), already computed


@dataclass
class BufferedAction:
    action: np.ndarray
    intended_time_s: float
    request_time_s: float
    ready_time_s: float
    chunk_index: int


@dataclass
class LatencyEvent:
    time_s: float
    event_type: SchedulerEventType
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": self.time_s,
            "event_type": self.event_type.value,
            "detail": self.detail,
        }


@dataclass
class ApplyDecision:
    action: np.ndarray
    sim_time_s: float
    control_step_index: int
    from_chunk: bool
    buffer_underflow: bool
    missed_deadline: bool
    observation_age_s: float | None
    requested_fresh_chunk: bool
    inference_call: bool
    request_time_s: float | None
    ready_time_s: float | None
    chunk_index: int | None


class LatencyScheduler:
    """One-in-flight inference, upstream request-when-empty cadence, discard_stale."""

    def __init__(
        self,
        config: LatencyConfig,
        *,
        initial_command: np.ndarray,
        latency_replayer: LatencyTraceReplayer | None = None,
    ) -> None:
        if config.mode is LatencyMode.TRACE_REPLAY and latency_replayer is None:
            raise ValueError("TRACE_REPLAY mode requires a LatencyTraceReplayer")
        if config.mode is LatencyMode.CONSTANT and latency_replayer is not None:
            raise ValueError("constant latency mode cannot take a latency_replayer")
        self.config = config
        self._replayer = latency_replayer
        self._initial_command = np.asarray(initial_command, dtype=np.float32).copy()
        self._last_applied = self._initial_command.copy()
        self._pending: PendingInference | None = None
        self._buffer: list[BufferedAction] = []
        self._control_step_index = 0
        self._terminated = False
        self.events: list[LatencyEvent] = []
        self.n_inference_calls = 0
        self.n_fresh_chunks = 0
        self.n_underflows = 0
        self.n_missed_deadlines = 0
        self._last_latency_s: float | None = None

    def reset(self, *, initial_command: np.ndarray) -> None:
        self._initial_command = np.asarray(initial_command, dtype=np.float32).copy()
        self._last_applied = self._initial_command.copy()
        self._pending = None
        self._buffer.clear()
        self._control_step_index = 0
        self._terminated = False
        self.events.clear()
        self.n_inference_calls = 0
        self.n_fresh_chunks = 0
        self.n_underflows = 0
        self.n_missed_deadlines = 0
        self._last_latency_s = None

    def mark_terminated(self) -> None:
        """Stop releasing or applying buffered/pending actions after episode end."""
        self._terminated = True
        self._pending = None
        self._buffer.clear()

    @property
    def sim_time_s(self) -> float:
        return self._control_step_index * self.config.control_period_s

    def needs_inference_request(self) -> bool:
        if self._terminated:
            return False
        return self._pending is None and len(self._buffer) == 0

    def submit_inference(
        self,
        *,
        request_time_s: float,
        chunk: np.ndarray,
    ) -> None:
        if self._terminated:
            return
        if self._pending is not None:
            raise RuntimeError("only one inference request may be in flight")
        chunk_arr = np.asarray(chunk, dtype=np.float32)
        if chunk_arr.ndim != 2:
            raise ValueError(f"chunk must be 2-D, got {chunk_arr.shape}")
        if chunk_arr.shape[0] != self.config.n_action_steps:
            raise ValueError(
                f"chunk length {chunk_arr.shape[0]} != n_action_steps {self.config.n_action_steps}"
            )
        if self._replayer is not None:
            latency_s = self._replayer.next_latency_s()
        else:
            latency_s = float(self.config.latency_s)
        self._last_latency_s = latency_s
        ready = request_time_s + latency_s
        self._pending = PendingInference(
            request_time_s=request_time_s,
            ready_time_s=ready,
            chunk=chunk_arr.copy(),
        )
        self.n_inference_calls += 1
        self.n_fresh_chunks += 1
        self.events.append(
            LatencyEvent(
                time_s=request_time_s,
                event_type=SchedulerEventType.INFERENCE_REQUEST,
                detail={
                    "ready_time_s": ready,
                    "latency_s": latency_s,
                    "latency_mode": self.config.mode.value,
                    "n_actions": int(chunk_arr.shape[0]),
                },
            )
        )

    def _release_pending_if_ready(self, apply_time_s: float) -> None:
        if self._pending is None or self._terminated:
            return
        if self._pending.ready_time_s > apply_time_s:
            return

        pending = self._pending
        self._pending = None
        self.events.append(
            LatencyEvent(
                time_s=pending.ready_time_s,
                event_type=SchedulerEventType.INFERENCE_COMPLETE,
                detail={
                    "request_time_s": pending.request_time_s,
                    "apply_time_s": apply_time_s,
                    "chunk_arrival_policy": self.config.chunk_arrival_policy,
                },
            )
        )

        kept = 0
        for i in range(pending.chunk.shape[0]):
            intended = pending.request_time_s + i * self.config.control_period_s
            if intended < apply_time_s:
                continue
            self._buffer.append(
                BufferedAction(
                    action=pending.chunk[i].copy(),
                    intended_time_s=intended,
                    request_time_s=pending.request_time_s,
                    ready_time_s=pending.ready_time_s,
                    chunk_index=i,
                )
            )
            kept += 1

        if kept == 0:
            self.n_missed_deadlines += 1
            self.events[-1].detail["missed_deadline"] = True
            self.events[-1].detail["discarded_all"] = True
        else:
            self.events[-1].detail["actions_kept"] = kept
            self.events[-1].detail["actions_discarded"] = int(pending.chunk.shape[0] - kept)

    def decide_apply(self, *, requested_this_step: bool) -> ApplyDecision:
        """Release ready chunks (equal-time: complete before apply), then choose action."""
        if self._terminated:
            raise RuntimeError("decide_apply called after episode termination")

        apply_time_s = self.sim_time_s
        self._release_pending_if_ready(apply_time_s)

        from_chunk = False
        underflow = False
        missed = False
        observation_age_s: float | None = None
        request_time_s: float | None = None
        ready_time_s: float | None = None
        chunk_index: int | None = None

        if self._buffer:
            item = self._buffer.pop(0)
            action = item.action
            from_chunk = True
            observation_age_s = apply_time_s - item.request_time_s
            request_time_s = item.request_time_s
            ready_time_s = item.ready_time_s
            chunk_index = item.chunk_index
            if item.ready_time_s > item.intended_time_s + 1e-15:
                # Result arrived after this action's intended time; counted at release.
                missed = False
        else:
            action = self._last_applied.copy()
            underflow = True
            self.n_underflows += 1

        self._last_applied = np.asarray(action, dtype=np.float32).copy()
        decision = ApplyDecision(
            action=self._last_applied.copy(),
            sim_time_s=apply_time_s,
            control_step_index=self._control_step_index,
            from_chunk=from_chunk,
            buffer_underflow=underflow,
            missed_deadline=missed,
            observation_age_s=observation_age_s,
            requested_fresh_chunk=requested_this_step,
            inference_call=requested_this_step,
            request_time_s=request_time_s,
            ready_time_s=ready_time_s,
            chunk_index=chunk_index,
        )
        self.events.append(
            LatencyEvent(
                time_s=apply_time_s,
                event_type=SchedulerEventType.CONTROL_APPLY,
                detail={
                    "control_step_index": decision.control_step_index,
                    "from_chunk": from_chunk,
                    "buffer_underflow": underflow,
                    "observation_age_s": observation_age_s,
                    "chunk_index": chunk_index,
                    "request_time_s": request_time_s,
                    "ready_time_s": ready_time_s,
                },
            )
        )
        self._control_step_index += 1
        return decision

    def provenance(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "latency_mode": self.config.mode.value,
            "latency_s": self.config.latency_s
            if self.config.mode is LatencyMode.CONSTANT
            else None,
            "last_latency_s": self._last_latency_s,
            "control_period_s": self.config.control_period_s,
            "n_action_steps": self.config.n_action_steps,
            "chunk_arrival_policy": self.config.chunk_arrival_policy,
            "one_inference_in_flight": True,
            "host_compute_added_to_latency": False,
            "n_inference_calls": self.n_inference_calls,
            "n_fresh_chunks": self.n_fresh_chunks,
            "n_underflows": self.n_underflows,
            "n_missed_deadlines": self.n_missed_deadlines,
        }
        if self._replayer is not None:
            payload["trace_replay"] = self._replayer.provenance()
        return payload

    def events_as_dicts(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.events]


def initial_hold_command_from_obs(raw_observation: dict[str, Any]) -> np.ndarray:
    """Environment-appropriate hold: current proprioceptive state, not zeros."""
    if "agent_pos" not in raw_observation:
        raise ValueError("expected agent_pos in observation for initial hold command")
    pos = np.asarray(raw_observation["agent_pos"], dtype=np.float32)
    if pos.ndim == 2:
        pos = pos[0]
    if pos.ndim != 1:
        raise ValueError(f"agent_pos must be 1-D after batch squeeze, got {pos.shape}")
    return pos.copy()


def config_dict(config: LatencyConfig) -> dict[str, Any]:
    return asdict(config)
