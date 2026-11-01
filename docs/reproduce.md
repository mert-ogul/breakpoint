# Reproduce the benchmark

This page runs the ACT transfer benchmark from a clean checkout: export, validation, device profiling, evaluation, packaging, report and optional media. Diffusion Policy on PushT has its own section below. To render the saved ACT results only, use the [README](../README.md#render-the-saved-report).

Run every command from the clone root. The commands call `.venv/bin/...` directly, so nothing needs to be activated.

## Prerequisites and cost

- Tested host: macOS 14.6 arm64, Python 3.11, CPU. Other hosts are untested ([compatibility](compatibility.md)).
- The first export needs network access: it fetches the pinned LeRobot checkout into `vendor/lerobot` (no git-lfs needed) and downloads the pinned Hub policy.
- The simulator and the export need the simulator install below. The Pi needs only the ONNX profile install.
- Disk: the full run writes about 580 MB under `outputs/` (364 MB of run reports plus 217 MB of duplicate `episodes.jsonl`, from [`large_evidence.json`](../benchmarks/act-transfer/deployment-stress/data/large_evidence.json)). The exported graphs and corpora add more under `artifacts/`.
- Run time was not recorded for the saved run. The full phase is 690 fresh episodes (810 episode rows including referenced baselines), so run `--dry-run` and the smoke phase first.
- The commands below write to `artifacts/export/act-transfer`, `profiles/` and `outputs/act-transfer/`. All are git-ignored. The saved evidence under `benchmarks/` is not touched.

## 1. Install the simulator environment

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-simulation.txt -r requirements-onnx.txt -e '.[dev,report,h264]'
.venv/bin/python -m pip check
export MUJOCO_GL=glfw
```

`MUJOCO_GL=glfw` is for macOS. Breakpoint also applies it automatically before it imports MuJoCo. The `h264` extra installs PyAV. H.264 also needs FFmpeg with libx264 on `PATH` (`brew install ffmpeg` or `sudo apt-get install -y ffmpeg`).

## 2. Export and validate the FP32 model

Seeds: calibration `9000` to `9007`, held-out evaluation `2000` to `2007`, closed-loop diagnosis `2100` and `2101`. The pools are disjoint. Later stress seeds must stay outside them.

```bash
.venv/bin/breakpoint export \
  --candidate act_aloha_transfer \
  --device cpu \
  --no-closed-loop \
  --output-dir artifacts/export/act-transfer

.venv/bin/breakpoint diagnose-closed-loop \
  --candidate act_aloha_transfer \
  --export-dir artifacts/export/act-transfer \
  --output-dir artifacts/diagnostics/act-transfer \
  --seeds 2100,2101 \
  --device cpu

.venv/bin/breakpoint write-export-validation \
  --export-dir artifacts/export/act-transfer \
  --diagnosis artifacts/diagnostics/act-transfer/closed_loop_diagnosis.json
```

`export` writes the FP32 and static QDQ INT8 graphs, the calibration and evaluation corpora, and `export_manifest.json`. INT4 and FP16 stay off unless `--int4` or `--fp16` is passed. `export` uses the default seed pools above. If INT8 cannot be built, the manifest records `errored` or `unsupported` and no other recipe is substituted.

`write-export-validation` writes `export_validation.json` (schema version 2). It refuses a diagnosis that does not match the FP32 graph hash, missing corpus or revision fields, non-finite or mis-shaped outputs, and incomplete episodes. Independent-trajectory equality is a diagnostic and is not a gate. INT8 has its own `export_validation_int8.json`, which does not authorize FP32.

An export made earlier at another path works if you pass that path explicitly (for example `--export-dir artifacts/export/act-v1`).

## 3. Add the INT4 weight-only variant

```bash
.venv/bin/breakpoint quantize-int4 \
  --export-dir artifacts/export/act-transfer
```

This writes `act_chunk_int4_matmul_nbits.onnx`, updates `export_manifest.json` with the `int4_weight_only` variant, and writes `export_validation_int4.json`. It refuses to overwrite an existing INT4 graph or manifest entry. A failed or unsupported recipe is recorded in the manifest and validation sidecar; no other recipe replaces it. The FP32 graph and its profiles are unchanged.

## 4. Profile the FP32 graph on each device

Profiling is ONNX Runtime only. It validates the export bundle first and refuses a graph without a passed `export_validation.json`. Defaults are 50 warm-up and 1000 measured calls. Add `--power-note` and `--cooling-note` to record the power source and cooling.

Laptop (the simulator environment from step 1):

```bash
.venv/bin/python scripts/profile_device.py \
  --device-label laptop-m3 \
  --export-dir artifacts/export/act-transfer \
  --output-dir profiles/laptop-m3/fp32
```

Raspberry Pi (Linux aarch64, Python 3.11). Do not install the simulator stack on the Pi. First copy these five files from the Mac export to a directory on the Pi, for example `~/act-export`:

```
act_chunk_fp32.onnx  export_manifest.json  export_validation.json
corpus_eval.npz      eval_chunks_onnx_fp32.npz
```

Then, in a clone of this repository on the Pi:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-pi-profile.txt -e '.[onnx]'
.venv/bin/python scripts/profile_device.py \
  --device-label rpi5 \
  --export-dir ~/act-export \
  --output-dir profiles/rpi5/fp32
```

If the Pi clone has no `.git` directory (a source archive), put the commit hash in `source_revision.txt` at the clone root. The profile then records an archive context and no clean-tree status.

Copy the Pi's `profiles/rpi5/fp32` directory to the same path on the Mac. The Pi records temperature and throttling telemetry through `vcgencmd` when it is available. A profile taken while the board throttles describes that state only.

## 5. Evaluate

The configuration is `configs/act-transfer.json`. Evaluation runs on CPU with ONNX Runtime (4 intra-op, 1 inter-op threads). Check the plan first. A dry run validates identities and creates no results:

```bash
.venv/bin/python scripts/run_benchmark.py --phase smoke --out outputs/act-transfer/smoke --dry-run
```

Smoke phase: env seeds `3000` to `3004`, the baseline, 640 ms, 60x80, JPEG 10, INT8 and INT4 if available, H.264 at 100 kbps if supported, and both device profiles.

```bash
.venv/bin/python scripts/run_benchmark.py --phase smoke --out outputs/act-transfer/smoke
```

Full phase: env seeds `5000` to `5029`, one isolated axis at a time, including the H.264 quality-only plan (`bypass`, then 2000, 1000, 500, 250 and 100 kbps).

```bash
.venv/bin/python scripts/run_benchmark.py --phase full --out outputs/act-transfer/full
```

H.264 needs FFmpeg with libx264 on `PATH` and the `h264` extra. Otherwise the axis is recorded as unsupported and left out of the plan. Do not upgrade FFmpeg or PyAV during a run: resume refuses a changed encoder lock.

Re-run the same command with `--resume` to continue an interrupted run. Resume refuses any change to the configuration, identities or seeds, and re-runs conditions that are not complete. The command exits nonzero if a condition is incomplete.

Use `--export-dir`, `--profile-laptop` and `--profile-rpi5` when the inputs are not at the configured paths. A run started with the old `artifacts/export/act-v1` export resumes only with `--export-dir artifacts/export/act-v1`.

## 6. Package and render the report

Packaging reads the run directory and writes a new package. It replaces an existing package only after the new one validates.

```bash
.venv/bin/python scripts/package_benchmark.py \
  --input outputs/act-transfer/full --out outputs/act-transfer/package
.venv/bin/python scripts/render_benchmark.py \
  --input outputs/act-transfer/package --out outputs/act-transfer/report
```

`package_benchmark.py` reads the export and profile locations from the run's lock. If they moved, pass `--export-dir` and `--profile DEVICE=PATH` (repeatable). Without `--out`, it writes the committed `benchmarks/act-transfer/deployment-stress/data`, so always pass `--out` here. Open `outputs/act-transfer/report/index.html`.

## 7. Comparison media (optional)

Needs the simulator environment, Pillow, matplotlib and `ffmpeg` with libx264 on `PATH` (`brew install ffmpeg` or `sudo apt-get install -y ffmpeg`).

```bash
.venv/bin/python scripts/render_benchmark_media.py \
  --data outputs/act-transfer/package --out outputs/act-transfer/comparison
```

It re-runs the baseline and 640 ms episodes of the selected seed with the saved settings and requires the outcome, step count, inference count, rewards and timeline to equal the saved result. It writes `comparison.mp4`, `comparison.gif`, `social-preview.png`, `selection.json` and `verification.json`. The export is read from `artifacts.export_dir` in the configuration unless `--export-dir` is given.

For the saved example, `--data` defaults to `benchmarks/act-transfer/deployment-stress/data` and the output to its `media/` directory.

## Diffusion Policy on PushT

Separate from ACT transfer: own config (`configs/diffusion-pusht.json`), seeds, package and report. Requires the simulator install from step 1 (includes `gym-pusht`). Commands write under `artifacts/export/diffusion_pusht`, `profiles/laptop-m3/diffusion_pusht` and `outputs/diffusion-pusht/`.

### 1. Export record

Records ONNX and device profiling as unsupported; does not build an ONNX graph.

```bash
.venv/bin/breakpoint export \
  --candidate diffusion_pusht \
  --device cpu \
  --output-dir artifacts/export/diffusion_pusht
```

### 2. Verify upstream agreement

Closed-loop L=0 scheduler path versus upstream `select_action` on seeds 2200 to 2204. Must pass before a run will resolve artifacts.

```bash
.venv/bin/breakpoint verify-upstream \
  --candidate diffusion_pusht \
  --device cpu \
  --torch-threads 4 \
  --output artifacts/export/diffusion_pusht/upstream_agreement.json
```

### 3. Profile action generation

Times one full PyTorch `generate_actions` call on real PushT observations (5 warm-ups, 50 measured by default). Use the same thread count as the config.

```bash
.venv/bin/python scripts/profile_diffusion.py \
  --device-label laptop-m3 \
  --device cpu \
  --torch-threads 4 \
  --output-dir profiles/laptop-m3/diffusion_pusht
```

### 4. Dry run, smoke and full run

```bash
.venv/bin/python scripts/run_benchmark.py \
  --config configs/diffusion-pusht.json \
  --phase smoke --out outputs/diffusion-pusht/smoke --dry-run

.venv/bin/python scripts/run_benchmark.py \
  --config configs/diffusion-pusht.json \
  --phase smoke --out outputs/diffusion-pusht/smoke

.venv/bin/python scripts/run_benchmark.py \
  --config configs/diffusion-pusht.json \
  --phase full --out outputs/diffusion-pusht/full
```

Smoke: env seeds 4000 to 4004, baseline, 800 ms, 24x24, JPEG 10. Full: env seeds 6000 to 6004, isolated delay, resolution and JPEG. Full-run wall time depends on the measured generation median from step 3. After baseline reuse the full phase has 70 fresh episodes; an order-of-magnitude estimate at the zero-latency cadence is:

`estimate_s ≈ 70 × (300 / 8) × median_s`

where `median_s` is the profile median for one full `generate_actions` call, 300 is the episode step limit and 8 is `n_action_steps`. Under long constant delay a fully stale chunk can trigger a new request every control step, so generations per episode can rise toward 300.

### 5. Package and render

```bash
.venv/bin/python scripts/package_benchmark.py \
  --input outputs/diffusion-pusht/full --out outputs/diffusion-pusht/package
.venv/bin/python scripts/render_benchmark.py \
  --input outputs/diffusion-pusht/package --out outputs/diffusion-pusht/report \
  --update-readme README.md --readme-block results-pusht
```

Pass `--out` on package; the script default is the ACT data directory. Open `outputs/diffusion-pusht/report/index.html`. Comparison media is not part of this candidate's flow.
