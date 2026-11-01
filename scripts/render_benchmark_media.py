#!/usr/bin/env python3
"""Re-record the selected baseline/stress pair and write the comparison media.

Needs the simulator stack (see docs/compatibility.md), ONNX Runtime or PyTorch per the
package lock, Pillow, matplotlib (for bundled fonts), and ffmpeg with libx264 on PATH.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from breakpoint_eval.benchmark_media import MediaVerificationError, make_media  # noqa: E402
from breakpoint_eval.benchmark_protocol import DEFAULT_CONFIG_REL  # noqa: E402

LOGGER = logging.getLogger("render_benchmark_media")
BENCHMARK_DIR = _ROOT / "benchmarks" / "act-transfer" / "deployment-stress"
CONFIG_PATH = _ROOT / DEFAULT_CONFIG_REL


def configured_export_dir() -> Path:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return _ROOT / config["artifacts"]["export_dir"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=BENCHMARK_DIR / "data", help="packaged data dir"
    )
    parser.add_argument(
        "--out", type=Path, default=BENCHMARK_DIR / "media", help="media output dir"
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="export directory with act_chunk_fp32.onnx for onnxruntime packages "
        "(default: artifacts.export_dir in configs/act-transfer.json)",
    )
    parser.add_argument(
        "--work",
        type=Path,
        default=_ROOT / "outputs" / "act-transfer" / "media",
        help="scratch dir for the re-recorded runs and PNG frames",
    )
    parser.add_argument("--width", type=int, default=1280, help="video width in pixels")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="output frame rate (default: real-time for the package control period)",
    )
    parser.add_argument("--hold-frames", type=int, default=15, help="repeat the last frames")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        verification = make_media(
            data_dir=args.data.resolve(),
            out_dir=args.out.resolve(),
            work_dir=args.work.resolve(),
            export_dir=(args.export_dir or configured_export_dir()).resolve(),
            width=args.width,
            fps=args.fps,
            hold_frames=args.hold_frames,
        )
    except MediaVerificationError as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info("media: %s", verification["media"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
