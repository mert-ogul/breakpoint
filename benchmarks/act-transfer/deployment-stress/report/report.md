# ACT cube transfer under stress

810 episode records, 30 paired episodes per condition, env seeds 5000 to 5029.

## Summary

One pretrained ACT policy (`lerobot/act_aloha_sim_transfer_cube_human`) on the ALOHA TransferCube task, run as ONNX Runtime on CPU in simulation. No physical robot is involved.

Exploratory run with 30 paired episodes per condition; not confirmed with fresh seeds.

- Baseline (ONNX FP32, native image, 0 ms delay): 25/30 (83.3%; Wilson 95% 66.4% to 92.7%).
- Constant added delay: first tested point below threshold is 320 ms at 18/30 (interval includes threshold).
- Camera resolution: first tested point below threshold is 60x80 at 10/30.
- JPEG quality: no tested point is below the threshold.
- H.264 bitrate: first tested point below threshold is 250 kbps at 19/30 (interval includes threshold).
- Numeric precision (categorical comparison, no ordering) against 0.8 x baseline: INT8 QDQ 24/30 not below; INT4 weight-only 23/30 not below.
- Device latency replay (categorical comparison, no ordering) against 0.8 x baseline: laptop-m3 replay 24/30 not below; rpi5 replay 9/30 below.

## Validation, errors and caveats

- Errors: 0 runtime errors or policy failures in 810 episode records. Outcomes: 601 success, 209 time_limit_truncation.
- INT8 QDQ ran in all 30 planned episodes; its validation does not require FP32 parity.
- INT4 weight-only: 29 of 41 MatMul nodes as 4-bit (61.6% of weights, accuracy_level=0); max abs error 0.146, mean abs 0.00601. Not a full-model INT4 result; no INT4 device timing.
- H.264: same-step encode/decode, visual quality only, ABR (largest overshoot +118.2 kbps). PyAV 18.1.0, libavcodec 62.28.102, preset=veryfast, gop=30, tune=zerolatency.
- Baseline 25/30 (83.3%), above the 0.5 floor.
- Sample size: 30 paired episodes per condition. A grid point below the threshold is not an interpolated exact boundary.
- "Interval spans threshold" means the Wilson interval overlaps the threshold; that is an interval overlap, not a paired significance test. Conditions: Constant added delay: 40 ms, 80 ms, 160 ms, 320 ms; Camera resolution: 120x160; JPEG quality: q95, q25, q10; H.264 bitrate: 2000 kbps, 250 kbps; Numeric precision: INT8 QDQ, INT4 weight-only; Device latency replay: laptop-m3 replay.
- Resolution ordering: the saved summary stored the same ordered value for every resolution, so conditions are ordered by plan order (native, 240x320, 120x160, 60x80), which is mild to severe.
- Pi 5 timings were captured while throttling was reported in 304 of 393 telemetry samples; cooling and power were not recorded.

## Constant added delay

Stress variable: ms added to every fresh action chunk.

![Constant added delay](plots/success_latency_ms.png)

| Condition | Success | Rate | Wilson 95% | Status | Paired with baseline |
| --- | ---: | ---: | ---: | --- | --- |
| 0 ms (baseline) | 25/30 | 83.3% | 66.4% to 92.7% | baseline | reference |
| 40 ms | 25/30 | 83.3% | 66.4% to 92.7% | not below (interval spans threshold) | lost 0; gained 0; both succeeded 25; neither 5 |
| 80 ms | 25/30 | 83.3% | 66.4% to 92.7% | not below (interval spans threshold) | lost 0; gained 0; both succeeded 25; neither 5 |
| 160 ms | 23/30 | 76.7% | 59.1% to 88.2% | not below (interval spans threshold) | lost 4; gained 2; both succeeded 21; neither 3 |
| 320 ms | 18/30 | 60.0% | 42.3% to 75.4% | below (interval spans threshold) | lost 9; gained 2; both succeeded 16; neither 3 |
| 640 ms | 14/30 | 46.7% | 30.2% to 63.9% | below | lost 13; gained 2; both succeeded 12; neither 3 |

First tested point below threshold: 320 ms (between 160 ms and 320 ms).

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
| 480x640 native (baseline) | 25/30 | 83.3% | 66.4% to 92.7% | baseline | reference |
| 240x320 | 26/30 | 86.7% | 70.3% to 94.7% | not below | lost 3; gained 4; both succeeded 22; neither 1 |
| 120x160 | 22/30 | 73.3% | 55.6% to 85.8% | not below (interval spans threshold) | lost 6; gained 3; both succeeded 19; neither 2 |
| 60x80 | 10/30 | 33.3% | 19.2% to 51.2% | below | lost 17; gained 2; both succeeded 8; neither 3 |

First tested point below threshold: 60x80 (between 120x160 and 60x80).

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
| bypass (baseline) | 25/30 | 83.3% | 66.4% to 92.7% | baseline | reference |
| q95 | 25/30 | 83.3% | 66.4% to 92.7% | not below (interval spans threshold) | lost 1; gained 1; both succeeded 24; neither 4 |
| q75 | 26/30 | 86.7% | 70.3% to 94.7% | not below | lost 2; gained 3; both succeeded 23; neither 2 |
| q50 | 26/30 | 86.7% | 70.3% to 94.7% | not below | lost 3; gained 4; both succeeded 22; neither 1 |
| q25 | 25/30 | 83.3% | 66.4% to 92.7% | not below (interval spans threshold) | lost 4; gained 4; both succeeded 21; neither 1 |
| q10 | 23/30 | 76.7% | 59.1% to 88.2% | not below (interval spans threshold) | lost 7; gained 5; both succeeded 18; neither 0 |

No tested point is below the threshold.

Per-seed outcomes (● success, ○ none, ▼ lost, ▲ gained):

```
bypass            ●●●●●●●○●○●●●●●●●●●○●●●○●●●○●●
q95               ●●●●●●●○●▲●●●●●●●▼●○●●●○●●●○●●
q75               ●●●●●●●▲●▲▼●●●●●●▼●○●●●○●●●▲●●
q50               ●●●●▼●●▲●▲●●●●●●●▼●○●●●▲▼●●▲●●
q25               ●●●●▼●●▲●▲●●●●●●●▼●○●●●▲●●▼▲▼●
q10               ●●●▼●●▼▲●▲▼▼●▼▼●▼●●▲●●●▲●●●▲●●
```

## H.264 bitrate

Stress variable: requested libx264 bitrate in kbps; causal stream at 50 fps, visual quality only.

![H.264 bitrate](plots/success_h264.png)

| Condition | Success | Rate | Wilson 95% | Status | Paired with baseline |
| --- | ---: | ---: | ---: | --- | --- |
| bypass (baseline) | 25/30 | 83.3% | 66.4% to 92.7% | baseline | reference |
| 2000 kbps | 25/30 | 83.3% | 66.4% to 92.7% | not below (interval spans threshold) | lost 4; gained 4; both succeeded 21; neither 1 |
| 1000 kbps | 26/30 | 86.7% | 70.3% to 94.7% | not below | lost 2; gained 3; both succeeded 23; neither 2 |
| 500 kbps | 28/30 | 93.3% | 78.7% to 98.2% | not below | lost 1; gained 4; both succeeded 24; neither 1 |
| 250 kbps | 19/30 | 63.3% | 45.5% to 78.1% | below (interval spans threshold) | lost 9; gained 3; both succeeded 16; neither 2 |
| 100 kbps | 10/30 | 33.3% | 19.2% to 51.2% | below | lost 15; gained 0; both succeeded 10; neither 5 |

| Condition | Requested kbps | Measured kbps (median, min to max) | Mean encoded bytes / frame | Total encoded MB |
| --- | ---: | ---: | ---: | ---: |
| bypass (baseline) | n/a | n/a | n/a | n/a |
| 2000 kbps | 2000.0 | 2081.1 (2050.4 to 2118.2) | 5204.5 | 48.9 |
| 1000 kbps | 1000.0 | 1042.9 (998.2 to 1059.1) | 2600.4 | 24.0 |
| 500 kbps | 500.0 | 516.8 (503.7 to 528.0) | 1291.9 | 11.1 |
| 250 kbps | 250.0 | 253.9 (239.2 to 262.6) | 631.3 | 6.1 |
| 100 kbps | 100.0 | 99.7 (93.8 to 104.4) | 249.7 | 2.7 |

First tested point below threshold: 250 kbps (between 500 kbps and 250 kbps).

Per-seed outcomes (● success, ○ none, ▼ lost, ▲ gained):

```
bypass            ●●●●●●●○●○●●●●●●●●●○●●●○●●●○●●
2000 kbps         ●●▼●▼●●▲●○●●●●●●●▼●▲●●▼▲●●●▲●●
1000 kbps         ●●●●●●●▲●○●●●●●●●▼●○●●●▲▼●●▲●●
500 kbps          ●●●●▼●●▲●○●●●●●●●●●▲●●●▲●●●▲●●
250 kbps          ●●●●●▼●▲●▲▼●●▼●●▼▼▼○▼●▼○●●▼▲●●
100 kbps          ●●▼▼●●●○●○▼▼●▼●●▼▼▼○▼▼▼○●▼▼○▼▼
```

## Numeric precision

Stress variable: ONNX graph variant.

![Numeric precision](plots/success_precision.png)

| Condition | Success | Rate | Wilson 95% | Status | Paired with baseline |
| --- | ---: | ---: | ---: | --- | --- |
| FP32 (baseline) | 25/30 | 83.3% | 66.4% to 92.7% | baseline | reference |
| INT8 QDQ | 24/30 | 80.0% | 62.7% to 90.5% | not below (interval spans threshold) | lost 4; gained 3; both succeeded 21; neither 2 |
| INT4 weight-only | 23/30 | 76.7% | 59.1% to 88.2% | not below (interval spans threshold) | lost 3; gained 1; both succeeded 22; neither 4 |

Categorical comparison, no ordering: INT8 QDQ and INT4 weight-only; none is below the threshold.

Per-seed outcomes (● success, ○ none, ▼ lost, ▲ gained):

```
FP32              ●●●●●●●○●○●●●●●●●●●○●●●○●●●○●●
INT8 QDQ          ●●▼●●●●○●▲●●▼●●●●▼▼▲●●●▲●●●○●●
INT4 weight-only  ●●▼●●●●○●▲▼●●●●●●●▼○●●●○●●●○●●
```

## Device latency replay

Stress variable: measured ONNX Runtime chunk latency replayed from a device profile.

![Device latency replay](plots/success_device_profile.png)

| Condition | Success | Rate | Wilson 95% | Status | Paired with baseline |
| --- | ---: | ---: | ---: | --- | --- |
| 0 ms (baseline) | 25/30 | 83.3% | 66.4% to 92.7% | baseline | reference |
| laptop-m3 replay | 24/30 | 80.0% | 62.7% to 90.5% | not below (interval spans threshold) | lost 2; gained 1; both succeeded 23; neither 4 |
| rpi5 replay | 9/30 | 30.0% | 16.7% to 47.9% | below | lost 17; gained 1; both succeeded 8; neither 4 |

Categorical comparison, no ordering: laptop-m3 replay and rpi5 replay; rpi5 replay is below the threshold.

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

- pass: Packaged files match the manifest digests (77 files hashed)
- pass: Episode rows match planned seeds and success flags (810 rows)
- pass: No runtime errors or policy failures in any episode (outcomes: success, time_limit_truncation)
- pass: Every axis lists each planned condition once (28 condition rows)
- pass: v0.1_full_latency saved summary matches episode records (6 conditions)
- pass: v0.1_full_resolution saved summary matches episode records (4 conditions)
- pass: v0.1_full_jpeg saved summary matches episode records (6 conditions)
- pass: v0.1_full_h264 saved summary matches episode records (6 conditions)
- pass: v0.1_full_precision saved summary matches episode records (3 conditions)
- pass: Shared baseline outcomes match latency_ms_0 seed by seed (120 referenced rows; differing: [])
- pass: Latency axis recorded delay matches the condition label (all episodes)
- pass: Device replay windows and consumed delays match the profile samples (2 profiles)
- pass: Device profiles were measured on the locked FP32 graph (fp32 sha256 27bf9278030e)
- pass: INT4 episodes all completed (106 INT4 inference calls)
- pass: H.264 rows match frame counts, holds, and first keyframe (150 H.264 episodes)

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
| H.264 bitrate | bypass | `h264_bypass` |
| H.264 bitrate | 2000 kbps | `h264_2000kbps` |
| H.264 bitrate | 1000 kbps | `h264_1000kbps` |
| H.264 bitrate | 500 kbps | `h264_500kbps` |
| H.264 bitrate | 250 kbps | `h264_250kbps` |
| H.264 bitrate | 100 kbps | `h264_100kbps` |
| Numeric precision | FP32 | `precision_onnx_fp32` |
| Numeric precision | INT8 QDQ | `precision_int8` |
| Numeric precision | INT4 weight-only | `precision_int4_weight_only` |
| Device latency replay | 0 ms | `latency_ms_0` |
| Device latency replay | laptop-m3 replay | `device_profile_laptop-m3_fp32` |
| Device latency replay | rpi5 replay | `device_profile_rpi5_fp32` |
