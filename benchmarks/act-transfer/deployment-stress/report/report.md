# ACT deployment stress benchmark: cube transfer under isolated stress

600 episode records, 30 paired episodes per condition, env seeds 5000–5029. Exploratory run.

## Summary

**What was evaluated.** One pretrained ACT policy (`lerobot/act_aloha_sim_transfer_cube_human`) on the ALOHA TransferCube task in simulation, run as ONNX Runtime on CPU. No physical robot is involved.

**What changed.** One stress at a time, everything else at baseline:

- Constant added delay: 0 ms (baseline), 40 ms, 80 ms, 160 ms, 320 ms, 640 ms
- Camera resolution: 480x640 native (baseline), 240x320, 120x160, 60x80
- JPEG quality: bypass (baseline), q95, q75, q50, q25, q10
- Numeric precision: FP32 (baseline), INT8 QDQ
- Device latency replay: 0 ms (baseline), laptop-m3 replay, rpi5 replay

**What the evidence shows.** Paired seeds: every condition faces the same episodes as the baseline.

- Baseline (ONNX FP32, native image, 0 ms delay): 25/30 (83.3%; Wilson 95% 66.4% to 92.7%).
- Constant added delay: first grid point below 0.8 x baseline is 320 ms at 18/30 (interval includes threshold).
- Camera resolution: first grid point below 0.8 x baseline is 60x80 at 10/30.
- JPEG quality: no condition fell below 0.8 x baseline.
- Numeric precision (categorical comparison, no ordering) against 0.8 x baseline: INT8 QDQ 24/30 not below.
- Device latency replay (categorical comparison, no ordering) against 0.8 x baseline: laptop-m3 replay 24/30 not below; rpi5 replay 9/30 below.

> Exploratory simulation run with 30 paired episodes per condition. Intervals are wide, the results have not been confirmed with fresh seeds, and a grid point below the threshold is not an exact boundary. The rpi5 replay reflects one throttled Pi state.

## Validation, errors and caveats

- Errors: 0 runtime errors or policy failures in 600 episode records. Outcomes: 445 success, 155 time_limit_truncation.
- Unsupported conditions: none skipped. INT8 export status is 'ok' and the INT8 graph ran in every planned episode. The INT8 graph has no FP32 numerical-parity requirement; see the provenance section for its recorded chunk error.
- Baseline strength: 25/30 (83.3%) against the project floor of 0.5. The baseline is not weak by that floor.
- Sample size: 30 paired episodes per condition. Intervals are wide; the run is exploratory and does not locate an exact boundary.
- Unresolved threshold comparisons (Wilson interval includes the threshold; the threshold is estimated from the same baseline, so this overlap is not a paired significance test): Constant added delay: 40 ms, 80 ms, 160 ms, 320 ms; Camera resolution: 120x160; JPEG quality: q95, q25, q10; Numeric precision: INT8 QDQ; Device latency replay: laptop-m3 replay.
- Resolution ordering: the saved summary stored the same ordered value for every resolution, so conditions are ordered by plan order (native, 240x320, 120x160, 60x80), which is mild to severe.
- Pi 5 timings were captured while throttling was reported in 304 of 393 telemetry samples; cooling and power were not recorded.

## Constant added delay

Stress variable: ms added to every fresh action chunk.

![Constant added delay](plots/success_latency_ms.png)

| Condition | Success | Rate | Wilson 95% | Status | Paired with baseline |
| --- | ---: | ---: | ---: | --- | --- |
| 0 ms (baseline) | 25/30 | 83.3% | 66.4% to 92.7% | ◆ baseline | reference |
| 40 ms | 25/30 | 83.3% | 66.4% to 92.7% | ○ not below threshold; interval spans threshold | lost 0; gained 0; both succeeded 25; neither 5 |
| 80 ms | 25/30 | 83.3% | 66.4% to 92.7% | ○ not below threshold; interval spans threshold | lost 0; gained 0; both succeeded 25; neither 5 |
| 160 ms | 23/30 | 76.7% | 59.1% to 88.2% | ○ not below threshold; interval spans threshold | lost 4; gained 2; both succeeded 21; neither 3 |
| 320 ms | 18/30 | 60.0% | 42.3% to 75.4% | ▼ below threshold; interval spans threshold | lost 9; gained 2; both succeeded 16; neither 3 |
| 640 ms | 14/30 | 46.7% | 30.2% to 63.9% | ▼ below threshold | lost 13; gained 2; both succeeded 12; neither 3 |

Saved summary: on this exploratory grid the first point below threshold is 320 ms; the observed crossing lies between 160 ms and 320 ms. Observed interval only; not an interpolated exact boundary.

Per-seed outcomes (● success, ○ none, ▼ lost, ▲ gained):

```
0 ms              ●●●●●●●○●○●●●●●●●●●○●●●○●●●○●●
40 ms             ●●●●●●●○●○●●●●●●●●●○●●●○●●●○●●
80 ms             ●●●●●●●○●○●●●●●●●●●○●●●○●●●○●●
160 ms            ●●●▼●●●○●▲●●●●●●●●▼○●▼●▲▼●●○●●
320 ms            ●●●▼●●▼▲●○●●●▼▼●●▼▼○●▼●○▼●●▲▼●
640 ms            ▼●▼▼●▼▼○●○▼●●▼▼●●▼●○●▼●▲▼●▼▲●▼
```

## Camera resolution

Stress variable: pixels, height x width (resampled back to model input).

![Camera resolution](plots/success_resolution.png)

| Condition | Success | Rate | Wilson 95% | Status | Paired with baseline |
| --- | ---: | ---: | ---: | --- | --- |
| 480x640 native (baseline) | 25/30 | 83.3% | 66.4% to 92.7% | ◆ baseline | reference |
| 240x320 | 26/30 | 86.7% | 70.3% to 94.7% | ○ not below threshold | lost 3; gained 4; both succeeded 22; neither 1 |
| 120x160 | 22/30 | 73.3% | 55.6% to 85.8% | ○ not below threshold; interval spans threshold | lost 6; gained 3; both succeeded 19; neither 2 |
| 60x80 | 10/30 | 33.3% | 19.2% to 51.2% | ▼ below threshold | lost 17; gained 2; both succeeded 8; neither 3 |

Saved summary: on this exploratory grid the first point below threshold is 60x80; the observed crossing lies between 120x160 and 60x80. Observed interval only; not an interpolated exact boundary.

Per-seed outcomes (● success, ○ none, ▼ lost, ▲ gained):

```
480x640 native    ●●●●●●●○●○●●●●●●●●●○●●●○●●●○●●
240x320           ●●●●▼●●○●▲●●●▼●●●●●▲▼●●▲●●●▲●●
120x160           ●●●▼▼●▼○●▲●●●▼●●●●●○●●▼▲●●●▲▼●
60x80             ●●▼▼▼▼●▲▼○▼●▼▼●▼▼▼▼○▼●▼○▼●●▲▼▼
```

## JPEG quality

Stress variable: OpenCV JPEG quality parameter (not a bitrate).

![JPEG quality](plots/success_jpeg.png)

| Condition | Success | Rate | Wilson 95% | Status | Paired with baseline |
| --- | ---: | ---: | ---: | --- | --- |
| bypass (baseline) | 25/30 | 83.3% | 66.4% to 92.7% | ◆ baseline | reference |
| q95 | 25/30 | 83.3% | 66.4% to 92.7% | ○ not below threshold; interval spans threshold | lost 1; gained 1; both succeeded 24; neither 4 |
| q75 | 26/30 | 86.7% | 70.3% to 94.7% | ○ not below threshold | lost 2; gained 3; both succeeded 23; neither 2 |
| q50 | 26/30 | 86.7% | 70.3% to 94.7% | ○ not below threshold | lost 3; gained 4; both succeeded 22; neither 1 |
| q25 | 25/30 | 83.3% | 66.4% to 92.7% | ○ not below threshold; interval spans threshold | lost 4; gained 4; both succeeded 21; neither 1 |
| q10 | 23/30 | 76.7% | 59.1% to 88.2% | ○ not below threshold; interval spans threshold | lost 7; gained 5; both succeeded 18; neither 0 |

Saved summary: no condition fell below the threshold within the tested range.

Per-seed outcomes (● success, ○ none, ▼ lost, ▲ gained):

```
bypass            ●●●●●●●○●○●●●●●●●●●○●●●○●●●○●●
q95               ●●●●●●●○●▲●●●●●●●▼●○●●●○●●●○●●
q75               ●●●●●●●▲●▲▼●●●●●●▼●○●●●○●●●▲●●
q50               ●●●●▼●●▲●▲●●●●●●●▼●○●●●▲▼●●▲●●
q25               ●●●●▼●●▲●▲●●●●●●●▼●○●●●▲●●▼▲▼●
q10               ●●●▼●●▼▲●▲▼▼●▼▼●▼●●▲●●●▲●●●▲●●
```

## Numeric precision

Stress variable: ONNX graph variant.

![Numeric precision](plots/success_precision.png)

| Condition | Success | Rate | Wilson 95% | Status | Paired with baseline |
| --- | ---: | ---: | ---: | --- | --- |
| FP32 (baseline) | 25/30 | 83.3% | 66.4% to 92.7% | ◆ baseline | reference |
| INT8 QDQ | 24/30 | 80.0% | 62.7% to 90.5% | ○ not below threshold; interval spans threshold | lost 4; gained 3; both succeeded 21; neither 2 |

Categorical comparison against the baseline: the conditions have no order, so no breakpoint is estimated. No condition's point estimate is below the threshold.

Per-seed outcomes (● success, ○ none, ▼ lost, ▲ gained):

```
FP32              ●●●●●●●○●○●●●●●●●●●○●●●○●●●○●●
INT8 QDQ          ●●▼●●●●○●▲●●▼●●●●▼▼▲●●●▲●●●○●●
```

## Device latency replay

Stress variable: measured ONNX Runtime chunk latency replayed from a device profile.

![Device latency replay](plots/success_device_profile.png)

| Condition | Success | Rate | Wilson 95% | Status | Paired with baseline |
| --- | ---: | ---: | ---: | --- | --- |
| 0 ms (baseline) | 25/30 | 83.3% | 66.4% to 92.7% | ◆ baseline | reference |
| laptop-m3 replay | 24/30 | 80.0% | 62.7% to 90.5% | ○ not below threshold; interval spans threshold | lost 2; gained 1; both succeeded 23; neither 4 |
| rpi5 replay | 9/30 | 30.0% | 16.7% to 47.9% | ▼ below threshold | lost 17; gained 1; both succeeded 8; neither 4 |

No saved experiment summary exists for this axis; counts, intervals, and paired tables are derived by this report from the episode records. Categorical comparison against the baseline: the conditions have no order, so no breakpoint is estimated. Point estimate below threshold: rpi5 replay.

Per-seed outcomes (● success, ○ none, ▼ lost, ▲ gained):

```
0 ms              ●●●●●●●○●○●●●●●●●●●○●●●○●●●○●●
laptop-m3 replay  ●●●▼●●●○●▲●●●●●●●●●○●▼●○●●●○●●
rpi5 replay       ▼▼▼▼●▼▼○●○▼●●▼▼●▼▼▼○●▼●○▼●▼▲▼▼
```

## Device timing

### laptop-m3

![laptop-m3 latency](plots/latency_trace_laptop-m3.png)

| Set | n | median ms | p95 ms | p99 ms |
| --- | ---: | ---: | ---: | ---: |
| full profile | 1000 | 108.2 | 114.9 | 135.0 |
| replay windows | 960 | 108.2 | 114.8 | 133.5 |
| consumed | 105 | 108.1 | 113.9 | 129.2 |

### rpi5

![rpi5 latency](plots/latency_trace_rpi5.png)

| Set | n | median ms | p95 ms | p99 ms |
| --- | ---: | ---: | ---: | ---: |
| full profile | 1000 | 706.2 | 917.1 | 972.6 |
| replay windows | 960 | 705.0 | 916.1 | 973.2 |
| consumed | 120 | 733.2 | 923.8 | 991.3 |

## Checks

- pass: packaged files match manifest sha256 (61 files hashed)
- pass: episode rows: planned seeds, policy RNG seeds, success flag vs outcome (600 rows)
- pass: no runtime errors or policy failures in any episode (outcomes: success, time_limit_truncation)
- pass: displayed counts: every axis lists each planned condition once (21 condition rows)
- pass: v0.1_full_latency: saved summary counts and Wilson intervals equal episode records (6 conditions)
- pass: v0.1_full_resolution: saved summary counts and Wilson intervals equal episode records (4 conditions)
- pass: v0.1_full_jpeg: saved summary counts and Wilson intervals equal episode records (6 conditions)
- pass: v0.1_full_precision: saved summary counts and Wilson intervals equal episode records (2 conditions)
- pass: shared baseline outcomes equal the latency_ms_0 condition seed by seed (90 referenced rows; differing: [])
- pass: latency axis: recorded constant delay equals the condition label (all episodes)
- pass: device replay: windows follow the locked design and consumed delays equal profile samples (2 profiles)
- pass: device profiles were measured on the locked FP32 graph (fp32 sha256 27bf9278030e)

## Condition IDs

| Axis | Condition | ID |
| --- | --- | --- |
| Constant added delay | 0 ms | `latency_ms_0` |
| Constant added delay | 40 ms | `latency_ms_40` |
| Constant added delay | 80 ms | `latency_ms_80` |
| Constant added delay | 160 ms | `latency_ms_160` |
| Constant added delay | 320 ms | `latency_ms_320` |
| Constant added delay | 640 ms | `latency_ms_640` |
| Camera resolution | 480x640 native | `resolution_native` |
| Camera resolution | 240x320 | `resolution_capture_240x320` |
| Camera resolution | 120x160 | `resolution_capture_120x160` |
| Camera resolution | 60x80 | `resolution_capture_60x80` |
| JPEG quality | bypass | `jpeg_bypass` |
| JPEG quality | q95 | `jpeg_q95` |
| JPEG quality | q75 | `jpeg_q75` |
| JPEG quality | q50 | `jpeg_q50` |
| JPEG quality | q25 | `jpeg_q25` |
| JPEG quality | q10 | `jpeg_q10` |
| Numeric precision | FP32 | `precision_onnx_fp32` |
| Numeric precision | INT8 QDQ | `precision_int8` |
| Device latency replay | 0 ms | `latency_ms_0` |
| Device latency replay | laptop-m3 replay | `device_profile_laptop-m3_fp32` |
| Device latency replay | rpi5 replay | `device_profile_rpi5_fp32` |
