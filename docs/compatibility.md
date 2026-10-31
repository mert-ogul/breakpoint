# Compatibility

This page lists the versions and platforms the benchmark was run on, the install routes, and fixes for the problems you are likely to hit. Commands run from the clone root.

## Tested platforms

| Role | Platform | Evidence |
| --- | --- | --- |
| Simulation, export, evaluation, laptop profile | macOS 14.6 arm64 (Apple M3), Python 3.11.17, CPU | The saved benchmark: 810 episode records, no runtime errors |
| Pi profile | Raspberry Pi on Linux aarch64 (kernel `6.18.50+rpt-rpi-2712`) | One saved profile, recorded while the firmware reported throttling. The board model string was not saved |
| Offline tests | Ubuntu with Python 3.11 | CI ([CONTRIBUTING.md](../CONTRIBUTING.md)) |

Other platforms, Python 3.12 and GPU hosts are untested. `requires-python` allows 3.11 and 3.12, but only 3.11 was exercised. Linux simulation was not tested.

## Tested versions

| Component | Version | Note |
| --- | --- | --- |
| LeRobot | commit `3c0a209f9fac4d2a57617e686a7f2a2309144ba2` | Vendor checkout in `vendor/lerobot`, fetched automatically. It is not a pip package |
| ACT policy | `lerobot/act_aloha_sim_transfer_cube_human` at `9e93545744656e57d20f8b2d631d5d72544002ce` | Downloaded from the Hub |
| torch / torchvision | 2.6.0 / 0.21.0 | |
| gymnasium / gym-aloha | 0.29.1 / 0.1.1 | |
| mujoco / dm-control | 2.3.7 / 1.0.14 | |
| onnx / onnxruntime | 1.18.0 / 1.22.1 | |
| numpy / opencv-python | 2.4.6 / 5.0.0.93 | |
| PyAV | 18.1.0 | From `av.__version__` on the tested host; installed with `.[h264]` |
| libavcodec | 62.28.102 | From `av.library_versions` on the tested host; encoding uses the libraries PyAV loads |
| FFmpeg with libx264 | host binary on `PATH` | Probed for H.264 support and used for comparison media |

The simulator pins are in [`requirements-simulation.txt`](../requirements-simulation.txt). [`constraints-macos-arm64.txt`](../constraints-macos-arm64.txt) is the full freeze of the macOS environment that passed `pip check`. Neither file is a lockfile for other platforms.

Scope of the evidence: only ACT transfer was exported, validated and evaluated. ACT insertion and Diffusion PushT load through the same code, but no episode, export or profile result exists for them. The ACT baseline of 25 of 30 successes (83.3%) is not a reproduction of the model card's 83% over 500 CUDA FP32 episodes. That reproduction was not run.

## Install routes

Use a Python 3.11 virtual environment (`python3.11 -m venv .venv`). Installation needs network access. Rendering a saved report does not.

| Goal | Command |
| --- | --- |
| Render a saved report | `.venv/bin/python -m pip install -e '.[report]'` |
| Simulation, export, evaluation (macOS) | `.venv/bin/python -m pip install -r requirements-simulation.txt -r requirements-onnx.txt -e '.[dev,report,h264]'` then `.venv/bin/python -m pip check` |
| Profile an exported graph on a Pi | `.venv/bin/python -m pip install -r requirements-pi-profile.txt -e '.[onnx]'` then `.venv/bin/python -m pip check` |
| Development and offline tests | See [CONTRIBUTING.md](../CONTRIBUTING.md) |

The Pi environment must not contain Torch, MuJoCo, gym or LeRobot. `requirements-pi-profile.txt` was checked on macOS arm64 only. The package list of the saved Pi run is in `data/profiles/rpi5/profile_run_meta.json` in the benchmark data.

An editable install writes `outputs/`, `artifacts/`, `vendor/` and `.cache/` inside the clone. A wheel install writes to `$BREAKPOINT_ROOT`, or `~/.local/share/breakpoint-eval` if it is unset, and never to `site-packages`.

## Troubleshooting

### MuJoCo fails on macOS with an OpenGL `dlopen` error

On macOS 14.6, `mujoco==2.3.7` cannot load its CGL backend. Breakpoint imports MuJoCo with GL disabled and then selects GLFW, which renders the task. Set `export MUJOCO_GL=glfw` when you run other MuJoCo code in the same environment. `osmesa` and `egl` are not valid on the macOS 2.3.7 wheels.

### The LeRobot checkout fails or is incomplete

Breakpoint fetches the pinned commit without git-lfs. A directory `vendor/lerobot` with `.git` but without `lerobot/__init__.py` is removed and fetched again. A checkout with local edits at another commit is not overwritten. Move it away yourself.

### DecodingError for device and use_amp on ACTConfig

The current Hub `main` configs need a newer LeRobot than the pinned commit. Use the pinned revision `9e935457…` above. Do not substitute `pip install lerobot` or a generic Transformers loader.

### Linux: the dev install pulls CUDA wheels

Install the CPU wheels first: see [CONTRIBUTING.md](../CONTRIBUTING.md).

### H.264 tests skip or fail

They need FFmpeg with libx264 on `PATH` and the `h264` extra. Install them with `brew install ffmpeg` or `sudo apt-get install -y ffmpeg`. Encoding uses the libraries PyAV loads; the `ffmpeg` CLI is only probed for availability and flag support.

### profile_device.py refuses the export

It needs `export_manifest.json`, `export_validation.json` with `passed: true` for the FP32 graph, `corpus_eval.npz`, `eval_chunks_onnx_fp32.npz` and the graph in one directory. Hashes must match. See [reproduce.md](reproduce.md#4-profile-the-fp32-graph-on-each-device).
