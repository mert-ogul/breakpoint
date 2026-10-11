# Controller timing methodology

This document describes the **verified** LeRobot ACT / ALOHA control loop that
Breakpoint’s latency interventions must preserve, and the discrete-event latency
model built on top of it.

## Verified zero-latency loop (LeRobot@3c0a209 + ACT ALOHA)

Environment control rate (ALOHA transfer / insertion):

| Quantity | Symbol | Value |
| --- | --- | --- |
| Control / sim rate | `fps` | 50 Hz |
| Control period | `Δt` | 20 ms = 0.02 s |
| Episode horizon | | 400 control steps (8.0 s) |
| Action chunk | `n_action_steps` | 100 (= `chunk_size` for the pinned ACT cards) |
| Observation history | `n_obs_steps` | 1 |
| Temporal ensembling | | off (`temporal_ensemble_coeff = null`) |

Upstream evaluation (`lerobot/scripts/eval.py` + `ACTPolicy.select_action`) does:

1. At control step `k`, read observation `o_k` (after previous step, or reset).
2. If the policy action queue is empty, run one inference on `o_k` and enqueue
   `n_action_steps` actions (open-loop chunk).
3. Pop the next queued action and apply it with `env.step`.
4. Simulation advances exactly one control period. **No wall-clock sleep** is part
   of the control model.

Consequences for timing:

- Inference request time and first application time are the **same** control
  instant when latency is zero.
- Cached pops are **not** new inference requests; the next request occurs when
  the queue drains (every 100 steps for the pinned ACT cards).
- Host time spent in `forward` is **not** part of simulated time.

PushT (diffusion) uses 10 Hz (`Δt = 0.1 s`), `n_obs_steps = 2`,
`n_action_steps = 8`. The same request-when-empty cadence applies.

## Action semantics (initial / fallback command)

ALOHA actions are **absolute joint targets** in the same 14-D space as
`observation.state` / `agent_pos` (not a proven “safe zero”). Therefore:

- **Initial command** before the first chunk is applied: hold the reset
  `agent_pos` (current joint configuration).
- **Fallback** while a result is unavailable: hold the **last applied** command;
  if none exists yet, hold the initial command.
- A zero vector is **not** used as a neutral action for ALOHA.

## Constant simulated latency (discrete-event)

Latency is injected only in **simulation/control time**. The scheduler:

1. On an inference request at `t_req`, **captures** the observation and computes
   the chunk immediately in host time (for determinism of the numeric result).
2. Marks the chunk **available** at `t_ready = t_req + L` (configured constant
   latency `L`). Host compute duration is **not** added to `L`.
3. Advances physics on the control grid while the request is in flight, applying
   the fallback command each control step.
4. Allows **one** inference in flight; new requests follow the upstream cadence
   (request when the local action buffer is empty and no request is pending).

Events are ordered on the real line. At equal timestamps,
`inference_complete` is processed **before** `control_apply` so that `L = 0`
can apply the first chunk action on the requesting step (baseline equivalence).

Sub-control-period latencies (e.g. `L = 25 ms` with `Δt = 20 ms`) keep their
true `t_ready`; the chunk becomes usable at the first control apply with
`t_apply >= t_ready`.

### Chunk arrival policy: `discard_stale`

Pinned ACT chunks are absolute joint-target sequences indexed from the
observation time:

`intended_time[i] = t_req + i · Δt` for `i = 0 … n_action_steps-1`.

When a chunk becomes available at apply time `t_apply`:

- **Discard** actions with `intended_time[i] < t_apply`.
- **Start** at the first remaining action (do not re-index the sequence as if it
  began at `t_apply`).

This matches absolute joint targets planned from `t_req`. The alternative
(“shift”: always start at action 0 when the chunk arrives) is **not** used for
ALOHA joint targets. The active policy name is recorded in every run’s
provenance as `chunk_arrival_policy` and must not change silently across tests.

If every action is stale (e.g. `L` longer than the chunk horizon), the buffer
underflows and the fallback hold continues; a missed-deadline / underflow event
is logged.

### Observation age

For an applied action originating from a chunk requested at `t_req`:

`observation_age_s = t_apply - t_req`.

Fallback holds that are not taken from a chunk log `observation_age_s` as null
and set `buffer_underflow=true` when an apply occurs with an empty buffer.

## Determinism tolerance

Zero-latency scheduler mode must match the baseline (no-scheduler) action
trajectory on a deterministic stub env within:

`action_atol = 0` (bit-identical float32 actions on the stub policy), and

`timestamp_atol_s = 1e-12` for logged sim times.

Real neural policies may differ at floating-point noise levels across devices;
device and seed are recorded in provenance. Stub/unit tests use exact equality.

## What is not a latency model

- `time.sleep` / wall-clock pauses
- Adding measured host inference duration on top of configured `L`
- Changing chunk arrival policy between runs without recording it
