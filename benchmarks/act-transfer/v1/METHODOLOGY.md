# Methodology: ACT cube transfer, v0.1

Simulation only. This is an exploratory run, not a safety assessment and not a statement about a
physical robot.

## Measured task

- **Environment:** ALOHA `AlohaTransferCube-v0` (gym-aloha 0.1.1, MuJoCo 2.3.7). A bimanual
  arm picks a cube with one gripper and hands it to the other.
- **Success:** the environment's own `is_success` (reward 4: the left gripper holds the cube off
  the table). Episodes are capped at 400 control steps at 50 Hz (8.0 s).
- **Policy:** ACT, Hub `lerobot/act_aloha_sim_transfer_cube_human` at revision
  `9e93545744656e57d20f8b2d631d5d72544002ce`, LeRobot commit
  `3c0a209f9fac4d2a57617e686a7f2a2309144ba2`. Chunk size 100, no temporal ensembling.
- **Runtime:** ONNX Runtime on CPU (`CPUExecutionProvider`, 4 intra-op / 1 inter-op threads) for
  every condition. The host never used MPS. FP32 graph sha256 `27bf9278030e…`; static QDQ INT8
  graph sha256 `792e9adc992405…` (see `data/identities/identities.json`).

## Design

- **Paired seeds.** 30 episodes per condition: env seeds 5000 to 5029, policy RNG seeds 15000 to
  15029. Every condition uses the same seed pairs.
- **Isolated axes.** One axis changes at a time, everything else at baseline:
  constant added delay (0, 40, 80, 160, 320, 640 ms), camera resolution (480x640 native, 240x320,
  120x160, 60x80), JPEG quality (bypass, 95, 75, 50, 25, 10), precision (FP32, INT8 QDQ), and
  device latency replay (laptop M3, Raspberry Pi 5). Images are degraded at the raw pixels the
  policy sees. JPEG quality is a codec parameter, not a bitrate.
- **Baseline.** ONNX FP32, native image, 0 ms. It was run once in the latency plan and reused by
  explicit reference in the resolution, JPEG, and precision plans. The device replay axis has no
  baseline of its own; it is compared with the same 0 ms condition. The report checks that the
  shared baseline and the latency-plan 0 ms condition agree seed by seed.
- **Latency semantics.** One inference in flight. A fresh action chunk becomes available after
  the injected delay on simulated time (`discard_stale` arrival); queued actions are not delayed.
  Host inference wall time is never added.
- **Device replay.** Each device profile holds 1000 timed `ort_session_run` chunk latencies
  (50 warmups discarded). Each episode replays one contiguous 32-sample window starting at
  `floor(i * (N - 32) / (n_episodes - 1))`. This is not block resampling and not continuous
  full-trace replay. Only the samples an episode actually needs are consumed; the report shows
  full-profile, window, and consumed statistics separately.

## Statistics

- Success rate with 95% Wilson interval per condition.
- Threshold: 0.8 times the baseline success rate (project convention, not a safety criterion).
  A baseline below 0.5 would make breakpoint claims unsupported.
- Saved paired bootstrap (seed 20261012, 2000 resamples, pointwise coverage) of
  `p_condition - 0.8 * p_baseline`. This is a margin against the threshold, not a plain
  difference.
- The report also gives paired outcome counts per condition (both succeeded, lost, gained, neither)
  computed directly from the episode records.
- The phase is exploratory. Thirty episodes cannot locate an exact boundary, and an interval that
  spans the threshold is reported as unresolved.

## What the saved evidence shows and does not show

- No runtime errors or policy failures occurred. All non-success outcomes are 400-step timeouts.
- The INT8 graph ran in every episode. Its export validation does not require FP32 numerical
  parity; its recorded same-input chunk error is in `data/validation/export_validation_v2_int8.json`.
  It is a static QDQ model, not a full-model INT8 claim.
- The device replay axis has no saved experiment summary. Its counts, intervals, and paired tables
  are derived by `scripts/render_v1.py` from the episode records and labelled as derived.
- The saved resolution summary stored the same `ordered_value` (1.0) for every resolution, so the
  summary orders them by plan order (native, 240x320, 120x160, 60x80), which is mild to severe.
  The saved file is preserved unchanged.
- The Raspberry Pi 5 profile was captured while the firmware reported throttling for most
  telemetry samples (temperature up to 88.9 C). Cooling and power were not recorded. The replay
  therefore describes that throttled state.
- The run started from code revision `2754b60` with only untracked files present
  (`.cursor/`, `profiles/`). The Pi profile ran from a source archive rather than a git checkout.

## Not covered

H.264, other tasks or policies, confirmation with fresh seeds, physical hardware, and any claim
that these thresholds transfer to another setup.

See also `docs/v1_protocol.md` (fixed protocol) and `docs/methodology.md` (general rules).
