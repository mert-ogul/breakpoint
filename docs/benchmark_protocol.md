# ACT deployment stress protocol

Thin scripts over the existing planner, runner, scheduler, and summarizer.
Public `breakpoint` CLI commands are unchanged.

Config source: `configs/act-transfer.json` (seeds, conditions, host ORT settings, trace-window
design, threshold ratio 0.8, baseline floor 0.5, scheduler semantics).

## Host settings (fixed across comparisons)

- Compute device: **CPU** (do not inherit CLI MPS default)
- ORT providers: `CPUExecutionProvider`
- Threads: intra-op **4**, inter-op **1**

## Trace-window design (smoke + full profiles)

For each device profile, one contiguous **32**-sample window per episode.
Start indices: `floor(i * (N - 32) / (n_episodes - 1))`, spread from `0` through
`N - 32`. Order inside each window is preserved; exhaustion fails. Window bounds
and consumed indices/delays are recorded. Used-sample stats are shown separately
from full-profile stats. **Not** block resampling; **not** continuous full-trace
replay.

## Profile on device (ONNX-only)

```bash
source .venv/bin/activate
# On Pi: pip install -r requirements-pi-profile.txt  (no torch/sim)

python scripts/profile_device.py \
  --device-label laptop-m3 \
  --export-dir artifacts/export/act-v1 \
  --output-dir profiles/laptop-m3/fp32 \
  --n-warmup 50 \
  --n-measured 1000 \
  --power-note "AC power" \
  --cooling-note "passive"

python scripts/profile_device.py \
  --device-label rpi5 \
  --export-dir artifacts/export/act-v1 \
  --output-dir profiles/rpi5/fp32 \
  --n-warmup 50 \
  --n-measured 1000 \
  --power-note "5V official supply" \
  --cooling-note "active heatsink"
```

Refuses overwrite of completed output dirs. Propagates real exit codes. When Git
is absent, reads `source_revision.txt` and marks archive context (no fabricated
clean status).

## Eval dry-run (validate only)

```bash
source .venv/bin/activate

python scripts/run_benchmark.py \
  --phase smoke \
  --out outputs/v1/smoke \
  --dry-run

python scripts/run_benchmark.py \
  --phase full \
  --out outputs/v1/full \
  --dry-run
```

Dry-run validates identity and prints plans/commands without creating run-result
directories or evaluating episodes.

## Eval (do not launch from this doc casually)

```bash
# Smoke: seeds 3000–3004, policy +10000; baseline, 640 ms, 60x80, JPEG 10,
# INT8 if runnable, both FP32 device traces.
python scripts/run_benchmark.py --phase smoke --out outputs/v1/smoke

# Full: seeds 5000–5029; isolated latency / resolution / JPEG / precision /
# device-profile axes. Resume matching lock only:
python scripts/run_benchmark.py --phase full --out outputs/v1/full
python scripts/run_benchmark.py --phase full --out outputs/v1/full --resume
```

Overrides (still identity-strict):

```bash
python scripts/run_benchmark.py --phase smoke --out outputs/v1/smoke \
  --export-dir PATH/TO/export \
  --profile-laptop PATH/TO/laptop.json \
  --profile-rpi5 PATH/TO/rpi5.json
```

Unsupported INT8 is recorded without fabricating measurements. Identical
baselines are reused by **explicit reference** only. Confirmation with fresh
seeds stays separate from exploration.

## Help

```bash
python scripts/run_benchmark.py --help
python scripts/profile_device.py --help
```
