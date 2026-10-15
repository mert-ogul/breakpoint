# Methodology note — ACT transfer Pi 5 smoke (preliminary)

This package documents a **small simulation smoke** of an exported ACT ONNX
FP32 policy under zero injected delay versus replay of measured Raspberry Pi 5
action-chunk latencies. It is **not** a real-robot task test and does **not**
establish a latency breakpoint.

## Policy and export binding

- Candidate: `act_aloha_transfer` (LeRobot ACT ALOHA transfer cube).
- Graph: ONNX FP32 action-chunk export (`ort_session_run_action_chunk`).
- Model SHA-256: `27bf9278030e4d2357f64aa55174f5718ac43c8202c07a22ceac2eef7ecc2eba`
- Eval corpus SHA-256: `f9559ac9037385d302a99c6ec16a5327407fedd4aa3643eba2eaa9ebe8d787ea`
- Deployment gate: `evidence/export/export_validation_v2.json` (schema_version=2).
  Same-input numerical parity and recorded closed-loop task outcomes gate
  deployment use. Independent trajectory equality is diagnostic only.

## Timing and scheduling

- Profiled boundary: `ort_session_run_action_chunk` (fresh chunk inference).
- ORT threads on the Pi profile: **4 intra-op**, **1 inter-op**.
- Replay injects recorded delays at **fresh action-chunk computation**, not on
  every queued action pop.
- Chunk arrival policy: `discard_stale`.
- Host wall time spent computing the chunk is **not** added again on top of the
  replayed delay.

## Conditions (paired seeds 3000–3004)

| Condition | Latency | Successes | Runtime errors |
| --- | --- | --- | --- |
| Baseline | constant `latency_s=0` | 4/5 | 0 |
| Pi 5 replay | ordered profile samples | 2/5 | 0 |

Outcomes are recorded per episode in `evidence/runs/*/run_report.json`.

## Pi environment (as recorded)

- Temperature: **84.25°C** (cpu-thermal).
- Throttling status: **unknown** (not recorded in the profile).
- Power supply: **unknown** (not recorded).
- Cooling: **unknown** (not recorded).
- Profile status: **preliminary**.

## Seed 3001 visualization

Baseline success and Pi-replay time-limit truncation were **re-recorded** with
video using the same ONNX graph, seeds, and profile. Exact emitted replay
latencies are listed in `media/seed3001_replay_latencies.json` (ordered profile
prefix; `policy_rng_seed=13001`).

## Limits

Five paired episodes are a smoke sample. Do not treat the 4/5 vs 2/5 gap as a
calibrated breakpoint or as evidence of physical-robot performance.
