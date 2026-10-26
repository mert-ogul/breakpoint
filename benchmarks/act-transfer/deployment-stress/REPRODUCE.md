# Reproduce

Run commands from the repository root. Each level needs more of the stack than the one before.

## 1. Render the saved example (report extra only)

No Torch, MuJoCo, network access, or model download.

```bash
python3.11 -m venv .venv && .venv/bin/python -m pip install -e '.[report]'
.venv/bin/python scripts/render_benchmark.py \
  --input benchmarks/act-transfer/deployment-stress/data \
  --out /tmp/breakpoint-report
open /tmp/breakpoint-report/index.html   # or xdg-open
```

The renderer checks the packaged files against `data/manifest.json`, recomputes every displayed
count and Wilson interval from `data/episodes/episodes.jsonl`, compares them with the saved
summaries, and refuses to write output if any check fails. The committed copy in
[`report/`](report/) was produced with `--out benchmarks/act-transfer/deployment-stress/report`.

## 2. Re-package from the raw run (local run directory required)

The raw control-step logs (about 390 MB with duplicates) are not in the repository. If you have
`outputs/v1/full`, `artifacts/export/act-v1`, and `profiles/*/fp32`:

```bash
.venv/bin/python scripts/package_benchmark.py \
  --input outputs/v1/full --out benchmarks/act-transfer/deployment-stress/data
```

It only reads those directories. [`data/large_evidence.json`](data/large_evidence.json) lists each
local-only file with its size and SHA-256.

## 3. Regenerate the comparison media (simulator stack)

Needs the stack from `docs/compatibility.md`, the exported FP32 graph in
`artifacts/export/act-v1`, Pillow, and `ffmpeg` with libx264 on `PATH`.

```bash
.venv/bin/python scripts/render_benchmark_media.py \
  --data benchmarks/act-transfer/deployment-stress/data --out benchmarks/act-transfer/deployment-stress/media
```

It re-runs only the selected baseline and 640 ms episodes (env seed 5000, policy RNG seed 15000)
through the same runner and settings, requires the outcome, step count, inference count, reward
and per-step timeline to equal the saved result (and compares actions when the raw log is
present), then writes `comparison.mp4`, `comparison.gif`, `social-preview.png`,
`selection.json`, and `verification.json`.

## 4. Re-run the evaluation (long)

Follow `docs/reproduce.md` (export and validation) and `docs/benchmark_protocol.md` (profiles and the
`scripts/run_benchmark.py --phase full` run).
