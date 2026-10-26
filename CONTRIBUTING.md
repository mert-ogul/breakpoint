# Contributing

Breakpoint is an experimental evaluation harness for one ACT cube-transfer task. Issues and small pull requests that fix bugs, tests or documentation are welcome. New policies, tasks or stress axes are out of scope for this release.

## Setup

Use Python 3.11 in a Git clone and run from the clone root. The commands call `.venv/bin/...` directly. Installation needs network access.

On Linux, install the CPU PyTorch wheels first. The `dev` extra pins `torch==2.6.0`, and a pin alone does not select CPU wheels, so the default index may install CUDA builds:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

On macOS, skip that step. Then install the development extras:

```bash
.venv/bin/python -m pip install -e '.[dev,onnx,report,h264]'
```

The H.264 tests need FFmpeg with libx264 on `PATH`. Install it before running them (`brew install ffmpeg` or `sudo apt-get install -y ffmpeg`). To run the simulator, see [docs/reproduce.md](docs/reproduce.md).

## Checks

These are the checks CI runs:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
BREAKPOINT_FAIL_ON_SKIP=1 .venv/bin/python -m pytest -m "not integration" -rs
```

- With `BREAKPOINT_FAIL_ON_SKIP=1`, a skipped test (a missing `onnx`, `matplotlib` or FFmpeg, for example) is a failure. Without it, `-rs` prints each skip with its reason.
- Tests marked `integration` need the simulator stack, the pinned LeRobot checkout and downloaded weights. CI deselects them. Run them with `-m integration`.
- CI also builds the wheel, installs it outside the checkout and renders the saved example, then compares `report.md` and the README results block with the committed copies.

## Saved evidence

Do not edit the files under `benchmarks/act-transfer/deployment-stress/data/` by hand. The `report/` directory and the README results block are generated from them:

```bash
.venv/bin/python scripts/render_benchmark.py \
  --input benchmarks/act-transfer/deployment-stress/data \
  --out benchmarks/act-transfer/deployment-stress/report --update-readme README.md
```

## Pull requests

Keep changes small. Describe what you ran and which checks you skipped. Contributions are accepted under the Apache License 2.0 (see [LICENSE](LICENSE)).
