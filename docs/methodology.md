# Methodology

This page defines how Breakpoint injects stress, validates the model and turns episodes into numbers. The saved ACT example is described in its [benchmark README](../benchmarks/act-transfer/deployment-stress/README.md); Diffusion PushT in its [benchmark README](../benchmarks/diffusion-pusht/stress/README.md). The thresholds below are project conventions, not safety criteria.

## Control timing

The pinned LeRobot commit (`3c0a209`) runs ALOHA at 50 Hz: one control step is 20 ms and an episode is at most 400 steps (8.0 s). The ACT policy has a chunk size and `n_action_steps` of 100, one observation step and no temporal ensembling. Each control step:

1. Read the observation.
2. If the action queue is empty, run one inference and queue the chunk.
3. Pop one action and apply it with `env.step`.

Simulation time advances one control period per step; host wall time is never part of it. With zero latency, a request and the first use of its chunk happen in the same control step.

### Constant added delay

Delay is injected in simulation time only. For a request at time `t_req` with delay `L`:

- The chunk is computed at once on the host, then becomes available at `t_req + L` (host compute time is not added to `L`).
- One inference is in flight at a time; the next request starts when the queue is empty and no request is pending.
- While a request is in flight, the controller holds the last applied command. Before any chunk exists it holds the reset joint position. ALOHA actions are absolute joint targets, so a zero vector is not a neutral command.
- At equal timestamps, completion is processed before control application, so `L = 0` matches the no-scheduler baseline.

The arrival policy is `discard_stale`. Action `i` of a chunk is planned for time `t_req + i * 20 ms`. When the chunk arrives at `t_apply`, actions planned for earlier times are dropped and the first remaining action is applied. If every action is stale, the controller keeps holding and logs an underflow. The policy name is recorded in each run's provenance.

### Device latency replay

`scripts/profile_device.py` times only `InferenceSession.run` for batch size 1 on recorded corpus inputs: 50 warm-ups (discarded), then 1000 measured calls, kept in order. The profile records the device label, host and OS, ONNX Runtime provider and threads, model and corpus hashes, timed boundary, summary statistics, the raw sequence and telemetry when the OS exposes it. Output parity on the profiling device must pass first.

Replay uses a recorded profile in place of a constant `L`. The benchmark gives each episode one contiguous window of 32 samples. For episode `i` of `n`, with `N` samples in the profile, the window starts at `floor(i * (N - 32) / (n - 1))`. Order inside the window is preserved, and an episode that needs more samples than its window holds fails. Window bounds and the consumed indices are saved per episode. The simulation runs on the host while the latency comes from the profile's device. This is contiguous window replay of the timed `InferenceSession.run` sequence.

These are not latency models:

- `time.sleep`
- host-time added to `L`
- a fitted distribution
- block resampling
- continuous whole-trace replay
- a changed arrival policy

## Image interventions

Interventions act on the raw rendered pixels (uint8 RGB under `observation["pixels"]`), before preprocessing and normalization. The policy still receives its native tensor shape and camera key order. They are independent of latency and of transport simulation.

| Mode | Behavior |
| --- | --- |
| `native` | Copy, no resize and no codec |
| `downsample` | Resize to the requested height and width with OpenCV `INTER_AREA`, then back to native size with `INTER_LINEAR` |
| `jpeg` | JPEG encode and decode at native size with OpenCV. Quality 1 to 100 is a codec parameter, not a bitrate. Quality 100 is still encoded |
| `h264` | One persistent libx264 encoder and decoder pair per camera, created fresh for every episode. Every rendered frame is encoded in order at the control rate (50 fps) with zerolatency, no B-frames, no lookahead and a single encoder thread. Each frame is decoded in the same control step; a frame that is not decoded is an error, not a hold. Bitrate is ABR; measured encoded bytes per frame are recorded. Quality only: no codec, transport or jitter delay is added |

Provenance records the OpenCV and NumPy versions, quality, requested and effective size, compressed byte counts and frame rate.

## Export and validation

`breakpoint export` exports the ACT tensor path to ONNX: input normalization, the camera stack, the model in eval mode with a zero latent, the `n_action_steps` slice and output unnormalization. The preprocessing step, action queue, image interventions and latency scheduler stay outside the graph.

The FP32 graph is accepted for profiling and evaluation only through `export_validation.json` (schema version 2; older exports named it `export_validation_v2.json`, which is still read). It requires:

- Same-input numerical parity with PyTorch FP32: finite outputs, matching shapes, and error within the tolerances in `export_tolerances.py`. The tolerances are written to the manifest before any comparison. The checks use a held-out corpus and same-observation comparisons that cover every fresh chunk inference.
- Completed closed-loop task episodes with recorded success, terminated and truncated fields. Two seeds are a smoke check, not statistical equivalence.
- A binding to the model hash, corpus hash, Hub revision, candidate, recipe and variant.

A synthetic corpus cannot produce a passed validation. Independent closed-loop trajectories need not match PyTorch step for step. That comparison is a diagnostic.

INT8 is ONNX Runtime static QDQ (`QUInt8` activations, `QInt8` weights, MinMax calibration) on calibration seeds that do not overlap the evaluation seeds. It is validated for identity, finite outputs, shapes and calibration coverage, with its own validation file. Task degradation is the measurement. Each variant runs with its own weights, and a failed recipe is recorded, not replaced.

INT4 is ONNX Runtime MatMulNBits weight-only RTN: 4 bits, block size 128, symmetric, MatMul only. It is added to an existing export with `breakpoint quantize-int4`, so the FP32 graph and its profiles are unchanged. Only weights of constant 2D MatMuls are stored as 4-bit blocks; Conv, Gemm and activation MatMuls stay FP32. Weights are dequantized and computed in float (`accuracy_level=0`). Validation is its own `export_validation_int4.json`: identity, finite outputs, shapes and recorded coverage on the evaluation corpus. There is no INT4 device timing.

Neither INT8 nor INT4 is a full-model quantization result, and neither requires FP32 numerical parity.

## Outcomes and denominators

Success is the environment's own `is_success`. An episode ends as `success`, `terminated_no_success`, `time_limit_truncation`, `policy_failed` or `runtime_error`. Only success counts as a success.

- The denominator is the eligible episodes: all outcomes except `runtime_error`.
- In the benchmark, a condition with a runtime error, a missing or duplicate episode, or a changed seed is incomplete. Its threshold decision is suppressed, and nothing is dropped to improve a rate.
- A time-limit truncation is a completed non-success.
- The report shows counts for every condition and does not fill missing series with zeros or fitted curves.

## Seed pairing

All conditions of a plan use the same environment seeds, so each seed is a pair across conditions. A separate policy RNG seed is recorded per episode, at a documented offset from the environment seed. Condition lists and counts are fixed for a plan. The benchmark uses 30 pairs; `breakpoint plan` defaults to 100 for a full plan and 5 for smoke.

Export seed pools (calibration `9000` to `9007`, evaluation `2000` to `2007`, diagnosis `2100` and `2101`) do not overlap with benchmark seeds. Confirmation of a candidate boundary needs fresh seeds and a fixed episode count, kept separate from the exploratory run, without stopping early on a favorable result.

## Uncertainty

- Each condition's success rate is shown as successes over eligible episodes with a 95% Wilson interval (per condition).
- The contrast with the baseline is a paired bootstrap of `p_condition - 0.8 * p_baseline` over episode pairs (seed 20261012, 2000 resamples). The interval is descriptive and pointwise for that contrast, not a simultaneous sweep coverage.
- Very small samples or degenerate resamples are marked unresolved.

## Threshold interpretation

A condition is below threshold when `p_c < 0.8 * p_b`, where `p_b` is the baseline rate on the same seeds. A baseline below 50% is too weak: the report shows its counts and intervals and suppresses the threshold decision.

For ordered axes (delay, resolution, JPEG quality), the report lists every tested point and names the first tested point below threshold. It flags a later recovery, and says "not reached" if no point falls below; it does not interpolate an exact boundary. Precision recipes and device profiles are categories compared one by one against the baseline, with no ordering or crossing point.

Isolation: constant-delay runs hold precision at FP32 and a native image. Precision runs use zero delay. H.264 runs at FP32, native resolution and 0 ms delay. INT4 runs at a native image and 0 ms delay. Device replay is a separate condition from the delay axis. Effects measured on separate axes are not additive.

## Diffusion Policy on PushT

Separate candidate from ACT transfer. Configuration is `configs/diffusion-pusht.json` (`diffusion_pusht_stress` 0.1). It uses its own env seeds, package root and report; episode rows are never pooled with ACT, and an ACT device profile is refused as the timing input.

### Policy and task

Pretrained policy `lerobot/diffusion_pusht` at Hub revision `202c8695947a458d8ce08ce1c0219f2588d60b33`, on PushT-v0 through `gym-pusht`. The LeRobot PushT env config runs at 10 Hz (100 ms control period) with a 300-step episode limit. Success is the environment's `is_success` (goal coverage at or above 0.95).

### Inference

Checkpoint settings: `n_obs_steps` 2, `horizon` 16, `n_action_steps` 8, DDPM noise scheduler with 100 denoising steps (`num_inference_steps`). Evaluation runs PyTorch FP32 on CPU with a fixed thread count from the config (`host.torch_num_threads`, 4). One fresh chunk is one full `generate_actions` call (observation encode, the full noise-scheduler loop, and unnormalize), not a single UNet pass. When the action buffer still has actions, the observation history is still updated every control step via `observe`.

### Stress

Constant delay uses the same discrete-event scheduler and `discard_stale` arrival policy as ACT, in simulated time only; host generation time is not added to `L`. With an 8-step chunk and a 100 ms period, `L = 800` ms makes every action in the chunk stale on arrival. Resolution starts from native 96x96. JPEG encode and decode are at native 96x96. A separate policy RNG seed is applied per episode (`env_seed + policy_seed_offset`).

Axes in the config: constant delay, resolution and JPEG. Precision recipes, H.264 and device-profile conditions are not in the plan. The packaged full phase uses `n_episodes` 5 (env seeds 6000 to 6004): 80 episode rows across 16 condition rows, with resolution and JPEG reusing the delay-plan 0 ms baseline (70 fresh episodes, 15 distinct run reports).

### Upstream check

`breakpoint verify-upstream` runs closed-loop agreement for Diffusion: the L=0 scheduler path against the upstream `select_action` queue on the same seeds. Per seed it requires matching outcomes, success, step counts, sum of rewards, and bitwise-equal actions. The first seed is also re-run for determinism. The written `upstream_agreement.json` must have `passed: true` and checkpoint hashes that match the export record before a pytorch-backend run will resolve its artifacts.

### Timing

`scripts/profile_diffusion.py` times one full PyTorch action-generation call (`pytorch_generate_actions_full_denoise`) on real PushT observations (env reset, native image path, then preprocess). Defaults: 5 warm-ups (discarded), then 50 measured calls, each on a distinct reset seed from 2300 upward. The packaged laptop-m3 profile (4 torch threads, CPU) has median 13433.8 ms, p95 32554.3 ms, p99 48941.5 ms and max 49842.4 ms. The control period is 100 ms and the largest tested delay is 800 ms. The profile is shown in the report; those latencies are not replayed into episodes as device delay.

### Not supported

| Item | Reason |
| --- | --- |
| ONNX export | Fresh inference is a multi-step denoising loop; a single-shot ONNX action-chunk graph matching that boundary is not supported |
| INT8 | No ONNX graph to quantize; recorded as unsupported |
| INT4 | No ONNX graph to quantize; recorded as unsupported |
| FP16 | Recorded as unsupported for this candidate |
| Device latency replay | No ORT action-chunk model to profile or replay; ACT profiles are refused |
| H.264 | Not attempted for this candidate (not in the plan) |
| Comparison media | Not attempted for this candidate |

### Separation

Own Hub revision, env seeds (smoke 4000 to 4004, full from 6000), config, export record, upstream agreement, action-generation timing profile, package directory and report block (`results-pusht`). No pooling with ACT transfer or insertion. No ACT profile is accepted as Diffusion timing.

## Experimental

FP16 export (off unless requested), the H.264 deployment mode that models codec buffering delay, and the ACT insertion candidate remain outside the validated ACT transfer and Diffusion PushT protocols. Unsupported combinations are reported as unsupported.
