#!/usr/bin/env python3
"""Thin v0.1 device profiler (requires only breakpoint-eval[onnx])."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from breakpoint_eval.device_profile import DEFAULT_MEASURED, DEFAULT_WARMUPS  # noqa: E402
from breakpoint_eval.v1_profile import run_v1_device_profile  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="profile_v1",
        description=(
            "Profile validated onnx_fp32 with CPUExecutionProvider, "
            "4 intra-op / 1 inter-op threads. ONNX-only; no simulator/Torch."
        ),
    )
    p.add_argument("--device-label", required=True)
    p.add_argument("--export-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-warmup", type=int, default=DEFAULT_WARMUPS)
    p.add_argument("--n-measured", type=int, default=DEFAULT_MEASURED)
    p.add_argument("--power-note", default=None)
    p.add_argument("--cooling-note", default=None)
    p.add_argument("--repo-root", type=Path, default=_ROOT)
    p.add_argument("--candidate", default="act_aloha_transfer")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    try:
        path = run_v1_device_profile(
            device_label=args.device_label,
            export_dir=args.export_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            n_warmup=args.n_warmup,
            n_measured=args.n_measured,
            power_note=args.power_note,
            cooling_note=args.cooling_note,
            repo_root=args.repo_root.resolve(),
            candidate=args.candidate,
        )
    except FileExistsError as exc:
        logging.error("%s", exc)
        return 2
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
        logging.error("%s", exc)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
