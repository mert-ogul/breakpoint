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
- Sampling latency from a normal distribution or substituting only a mean for a
  measured device trace

## Device profiling and trace replay

`breakpoint profile` measures **ONNX Runtime** policy computation on the local
machine **without** simulator dependencies. It times only
`InferenceSession.run` for batch size 1 on representative recorded corpus
inputs. It does **not** time random matmuls or action-queue dequeue. Defaults:
50 warmup calls (excluded) then 1000 measured calls (ordered sequence retained).

Profiles record device label, host CPU/OS, ORT provider/threads, model hash,
precision recipe, input shapes/corpus hash, chunk horizon, timed boundary,
median/p95/p99/count, and the raw latency sequence. Temperature is recorded when
the OS exposes it; missing telemetry is not fabricated. Manual cooling/power/load
notes are optional. Output validation on the profiling device must pass before
timings are trusted.

Constant `--latency-ms` sweeps remain distinct from measured replay. Replay uses
`--latency-profile PATH` or `--device rpi5` (resolves an **unambiguous** match
under `profiles/rpi5/`). Missing profiles print the `breakpoint profile …`
command; they do not invent latency numbers or auto-connect to a device. Default
end-of-trace behavior is **fail**; explicit `cycle` or seeded block resampling
may be requested. Simulator policy outputs may still run on the host while
latency is replayed from another device’s profile.

## Camera degradation (visual quality only)

Image interventions run at the **raw rendered-pixel boundary**: uint8 HWC RGB
frames under `observation["pixels"]`, **before** `preprocess_observation` and
policy normalization. Checkpoint-expected tensor shape and camera key order are
preserved. Degradation is **independent** of latency / network delay; these
modes isolate visual quality and are **not** a full sensor-stream transport
simulation.

| Mode | Behavior |
| --- | --- |
| `native` | Uncompressed identity bypass (copy only; no codec, no resize) |
| `downsample` | Resize to requested capture `HxW` with OpenCV `INTER_AREA`, then back to policy-native resolution with `INTER_LINEAR` |
| `jpeg` | Real JPEG encode/decode at **native** resolution via OpenCV `imencode`/ `imdecode` (RGB↔BGR for the codec). Quality is in `[1, 100]`; quality `100` is still encoded and is **not** an uncompressed identity control |

JPEG quality is a codec parameter, **not** a bitrate. Optional bandwidth estimates
use measured JPEG payload bytes × frame rate and exclude container/RTP/TCP
overhead. Provenance records OpenCV/NumPy versions, quality, requested and
effective dimensions, compressed byte counts, and frame rate.

CLI: `--image-mode`, `--capture-hw`, `--jpeg-quality`, and isolated smoke sweeps
`--image-sweep resolution|jpeg` (mutually exclusive with a latency sweep).

## ACT ONNX export and quantization

`breakpoint export` exports the verified ACT **tensor** path only:

| Inside ONNX | Outside ONNX |
| --- | --- |
| `normalize_inputs`, camera stack, `ACT.model` (eval, zero latent), `n_action_steps` slice, `unnormalize_outputs` | `preprocess_observation`, action queue, temporal ensembler, camera degrade, latency scheduler |

Declared FP32 parity tolerances live in `export_tolerances.py` and are recorded in the
artifact manifest **before** measurement. ONNX FP32 must pass shapes, finite outputs,
and per-element / action-unit error gates on a held-out corpus against PyTorch FP32.
Export-induced failure is investigated first. Paired closed-loop rollouts compare
action traces when the simulator is available; GL/env errors are recorded, not faked.

Quantized variants use pinned recipe ids (no silent switch after failure):

- INT8: ORT static QDQ (`QUInt8` activations / `QInt8` weights, MinMax calibration) on
  calibration seeds **disjoint** from evaluation seeds. Partial coverage is labeled;
  it is not claimed as full-model INT8.
- INT4: ORT `MatMulNBits` weight-only when the EP can load/run the graph. Activations
  remain float. Unsupported INT4 is an explicit capability result — not fake-quant or
  dequantized-float labeled as native INT4 execution.
- FP16: optional capability check; not a mandatory CPU speedup.

Each variant runs with its actual weights and runtime. Compare quantized outputs to the
ONNX FP32 baseline and show PyTorch FP32 separately to expose export effects. Quantized
policies are not required to preserve baseline task success; degradation is measured.
Optional deps: `pip install 'breakpoint-eval[onnx]'` (`onnx`, `onnxruntime`).

## Experiment plan and summary (from run records)

This section defines how Breakpoint **plans** multi-condition evaluations and how
it **summarizes** existing `run_report.json` records. Defaults below are project
conventions for empirical reporting, **not** safety criteria or known failure
thresholds.

### Seeds and sample counts

- Use the **same paired environment seeds** across all conditions in a plan.
- Record a **separate policy RNG seed** per episode (may equal the env seed or a
  documented offset); do not silently reuse one RNG stream for both roles.
- Keep condition lists and sample counts **fixed** for a given plan revision.
- **Smoke:** 5 paired episodes per condition (explicitly labeled smoke).
- **Full sweep default:** 100 paired episodes per condition (count always written
  into the plan and summary). One hundred episodes does **not** guarantee a
  precise boundary.
- Before publication, confirm any candidate boundary on **fresh** paired seeds
  with a **preregistered** episode count and threshold. Keep exploratory sweep
  results separate from confirmation results. No optional stopping for a desired
  headline.

### Starting grids (configurable experimental choices)

Run **one axis at a time** first. Suggested starting grids:

| Axis | Suggested levels |
| --- | --- |
| Constant added inference delay | 0, 20, 40, 60, 80, 120, 200 ms |
| Precision | verified ONNX FP32; INT8 (static QDQ recipe); supported weight-only INT4 — **categories**, not a continuous bit-depth axis |
| Resolution | native; 1/2 linear size; 1/4 linear size (record actual capture `HxW`) |
| JPEG quality | bypass (native uncompressed); 95; 75; 50; 30; 10 |

Isolation rules:

- **Precision:** isolate numerical effects under the **same** timing schedule
  (typically constant latency 0 or one shared measured profile).
- **Synthetic latency:** hold precision fixed (typically verified FP32).
- **Measured per-variant hardware timing** is a separate deployment condition,
  not a substitute for the constant-delay axis.
- Add explicit **small combined grids** only after isolated sweeps. Never imply
  that effects are additive across axes.

### Empirical breakpoint rule (default summary)

Let \(p_b\) be the matched baseline success rate (eligible episodes only; runtime
errors excluded from the denominator, consistent with `summarize_episodes`).
Let \(p_c\) be the condition success rate on the same paired seeds.

Default rule: a condition is **below threshold** when

\[
p_c < r \cdot p_b
\]

with default ratio \(r = 0.8\) (configurable). A baseline success rate below a
configurable **eligibility floor** (default 50%) is **weak** for this headline
rule: still show raw successes/episodes and intervals, but **suppress** the
headline breakpoint decision.

For an **ordered numeric severity** axis (latency ms, resolution linear scale,
JPEG quality ordered from mild→severe):

1. Show **all** tested points.
2. Report the **first tested** below-threshold condition and its **adjacent
   preceding** above-threshold condition as an **observed interval**.
3. Do **not** interpolate an exact boundary.
4. Flag recovery / non-monotonicity if a later point returns above threshold.
5. If every tested point stays above threshold: **not reached within tested
   range**.
6. If baseline is weak, too few episodes complete relative to the plan, a
   condition is incomplete, or confirmation is inconclusive: say so — do not
   invent a cutoff.

Precision recipes are **categories**: list results per recipe; do not invent a
continuous “bit-depth breakpoint.”

### Uncertainty reporting

- Report `successes / eligible_episodes` (and planned totals) with **95% Wilson
  score intervals** for each condition’s success rate.
- For contrasts relative to baseline, use **paired bootstrap** resampling of
  episode pairs on the statistic \(p_c - r\,p_b\). Record the bootstrap seed.
  Identify these intervals as **pointwise**. Do not claim simultaneous coverage
  across the whole sweep.
- Detect degenerate resampling or very small samples and mark uncertainty
  **unresolved**.
- Do not turn a noisy empirical crossing into a confident universal cutoff.

### Incomplete conditions

A planned condition is **incomplete** when the run record is missing, episode
count does not match the plan, or runtime errors leave the condition without a
valid success-rate denominator under the plan’s rules. Summaries must show
completed counts and errors and **suppress** that condition’s breakpoint
decision rather than cherry-picking successful executions.

## Reporting from saved records

`breakpoint report RUN_DIR` renders `summary.md`, `report.html`, and static
Matplotlib plots **only** from artifacts already on disk. Reporting must not
download models, open a simulator, or rerun episodes. Missing series are labeled
missing — never replaced with zeros or smooth fitted curves.

The HTML overview is a compact printable page with links to raw artifacts. It
must separate isolated-axis vs combined-stress results, label **simulated task
outcomes** vs **hardware-measured timing**, and must not imply that simulation
ran on a physical robot or that every numeric value came from the profiled
device. Optional dependency: `pip install 'breakpoint-eval[report]'`.
