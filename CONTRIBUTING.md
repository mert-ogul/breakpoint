# Contributing

Breakpoint is an experimental v0.1 evaluation harness for one ACT cube-transfer task.
Issues and small pull requests that fix bugs, tests or documentation are welcome. New
policies, tasks or fault axes are out of scope for v0.1.

## Setup

Use Python 3.11 from a Git clone. See the install table in [README.md](README.md) for the
simulator and Raspberry Pi routes.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,onnx,report,h264]'
```

`dev` pins CPU PyTorch 2.6.0. On Linux, install `torch==2.6.0` and `torchvision==0.21.0`
from `https://download.pytorch.org/whl/cpu` first to avoid the CUDA wheels.

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
- Do not edit saved evidence under `benchmarks/act-transfer/v1/data/` by hand. The
  report is rebuilt from it with `scripts/render_benchmark.py`.

## Pull requests

Keep changes small, describe what you ran, and say which checks you skipped and why.
Contributions are accepted under the Apache License 2.0 (see [LICENSE](LICENSE)).
