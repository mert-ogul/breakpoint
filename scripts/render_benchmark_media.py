#!/usr/bin/env python3
"""Re-record the selected v0.1 baseline/stress pair and write the comparison media.

Needs the simulator stack (see docs/compatibility.md), ONNX Runtime, Pillow, matplotlib
(for bundled fonts), and ffmpeg with libx264 on PATH.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from breakpoint_eval.benchmark_media import MediaVerificationError, make_media  # noqa: E402

LOGGER = logging.getLogger("render_benchmark_media")
V1_DIR = _ROOT / "benchmarks" / "act-transfer" / "v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=V1_DIR / "data", help="packaged data dir")
    parser.add_argument("--out", type=Path, default=V1_DIR / "media", help="media output dir")
    parser.add_argument(
        "--work",
        type=Path,
        default=_ROOT / "outputs" / "v1" / "media",
        help="scratch dir for the re-recorded runs and PNG frames",
    )
    parser.add_argument("--width", type=int, default=1280, help="video width in pixels")
    parser.add_argument("--fps", type=float, default=12.5, help="output frame rate")
    parser.add_argument("--hold-frames", type=int, default=15, help="repeat the last frames")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        verification = make_media(
            data_dir=args.data.resolve(),
            out_dir=args.out.resolve(),
            work_dir=args.work.resolve(),
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
