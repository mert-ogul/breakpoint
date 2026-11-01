#!/usr/bin/env python3
"""Profile full Diffusion Policy action generation on real PushT observations."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from breakpoint_eval.diffusion_profile import (  # noqa: E402
    DEFAULT_MEASURED,
    DEFAULT_WARMUPS,
    DiffusionProfileRequest,
    run_diffusion_pytorch_profile,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="profile_diffusion",
        description=(
            "Time one full Diffusion Policy action-generation call "
            "(normalize, observation history, all denoising steps, unnormalize) "
            "on real PushT observations."
        ),
    )
    p.add_argument("--device-label", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cpu")
    p.add_argument("--torch-threads", type=int, default=None)
    p.add_argument("--n-warmup", type=int, default=DEFAULT_WARMUPS)
    p.add_argument("--n-measured", type=int, default=DEFAULT_MEASURED)
    p.add_argument("--candidate", default="diffusion_pusht")
    p.add_argument("--repo-root", type=Path, default=_ROOT)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    try:
        path = run_diffusion_pytorch_profile(
            DiffusionProfileRequest(
                candidate=args.candidate,
                output_dir=args.output_dir.resolve(),
                device=args.device,
                device_label=args.device_label,
                torch_threads=args.torch_threads,
                n_warmup=args.n_warmup,
                n_measured=args.n_measured,
                repo_root=args.repo_root.resolve(),
            )
        )
    except (FileNotFoundError, ValueError, RuntimeError, ImportError, OSError) as exc:
        logging.error("%s", exc)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
