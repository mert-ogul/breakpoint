# Contributing

Breakpoint is an experimental evaluation harness for one ACT cube-transfer task.
Issues and small pull requests that fix bugs, tests or documentation are welcome. New
policies, tasks or fault axes are out of scope for this release.

## Setup

Use Python 3.11 from a Git clone, run from the clone root. The commands call the virtual
environment's executables directly, so nothing needs to be activated. The
[README](README.md#supported-configurations) lists the other install routes.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,onnx,report,h264]'
```

`dev` pins CPU PyTorch 2.6.0. On Linux, install the CPU wheels first (as CI does) to avoid CUDA
wheels:

```bash
.venv/bin/python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

## Checks

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
BREAKPOINT_FAIL_ON_SKIP=1 .venv/bin/python -m pytest -m "not integration" -rs
```

- With `BREAKPOINT_FAIL_ON_SKIP=1` a test that skips (for example a missing `onnx` or
  `matplotlib`) is reported as a failure. CI sets it. Without it, skips are printed with
  their reason by `-rs`.
- Tests marked `integration` need the simulator stack, pinned LeRobot checkout and
  downloaded weights. CI deselects them. Run them locally with `-m integration`.
- Do not edit saved evidence under `benchmarks/act-transfer/deployment-stress/data/` by hand.
  `report/` and the README results block are generated from it:

  ```bash
  .venv/bin/python scripts/render_benchmark.py \
    --input benchmarks/act-transfer/deployment-stress/data \
    --out benchmarks/act-transfer/deployment-stress/report --update-readme README.md
  ```

## Pull requests

Keep changes small, describe what you ran, and say which checks you skipped and why.
Contributions are accepted under the Apache License 2.0 (see [LICENSE](LICENSE)).
