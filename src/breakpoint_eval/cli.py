"""Thin CLI for Breakpoint. Subcommand: run."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from breakpoint_eval.camera_degrade import (
    ImageDegradeConfig,
    ImageMode,
    jpeg_smoke_configs,
    resolution_smoke_configs,
)
from breakpoint_eval.candidate_map import CANDIDATES
from breakpoint_eval.latency_scheduler import CHUNK_ARRIVAL_POLICY
from breakpoint_eval.run_baseline import RunRequest, build_seeds, parse_seed_list, run_baseline
from breakpoint_eval.stack import repo_root_from_package


def parse_capture_hw(text: str) -> tuple[int, int]:
    parts = [p.strip() for p in text.lower().replace("x", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("expected HxW or H,W")
    return int(parts[0]), int(parts[1])


def _add_run_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "run",
        help="Run a pinned candidate baseline through the Breakpoint episode runner",
    )
    parser.add_argument(
        "--candidate",
        default="act_aloha_transfer",
        choices=sorted(CANDIDATES),
        help="Direct model-to-environment mapping key",
    )
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated env seeds (paired across conditions later)",
    )
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for provenance + episode results",
    )
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="mps")
    parser.add_argument(
        "--video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write one mp4 per episode",
    )
    parser.add_argument(
        "--latency-ms",
        type=float,
        default=0.0,
        help="Constant simulated inference latency in milliseconds (not wall sleep)",
    )
    parser.add_argument(
        "--latency-sweep-ms",
        type=str,
        default=None,
        help="Comma-separated latency_ms values; writes one report subdir per value",
    )
    parser.add_argument(
        "--image-mode",
        choices=[m.value for m in ImageMode],
        default=ImageMode.NATIVE.value,
        help="Camera degradation at raw pixels (before policy normalization)",
    )
    parser.add_argument(
        "--capture-hw",
        type=str,
        default=None,
        help="Downsample capture size as HxW (e.g. 240x320); required for downsample mode",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
        help="JPEG quality 1-100 for jpeg mode (100 is still encoded, not identity)",
    )
    parser.add_argument(
        "--image-sweep",
        choices=["resolution", "jpeg"],
        default=None,
        help="Run an isolated visual-quality smoke sweep (separate from latency)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repo root containing vendor/ and .cache/",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="breakpoint", description="Breakpoint evaluation CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(sub)
    return parser


def _build_image_cfg(args: argparse.Namespace) -> ImageDegradeConfig:
    mode = ImageMode(args.image_mode)
    capture_h = capture_w = None
    if args.capture_hw is not None:
        capture_h, capture_w = parse_capture_hw(args.capture_hw)
    return ImageDegradeConfig(
        mode=mode,
        capture_height=capture_h,
        capture_width=capture_w,
        jpeg_quality=args.jpeg_quality,
    )


def _image_sweep_cfgs(kind: str) -> list[tuple[str, ImageDegradeConfig]]:
    if kind == "resolution":
        return resolution_smoke_configs()
    if kind == "jpeg":
        return jpeg_smoke_configs()
    raise ValueError(kind)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "run":
        parser.error(f"Unsupported command {args.command!r}")

    seeds_arg = parse_seed_list(args.seeds) if args.seeds is not None else None
    try:
        seeds = build_seeds(
            n_episodes=args.n_episodes,
            seeds=seeds_arg,
            start_seed=args.start_seed,
        )
    except ValueError as exc:
        parser.error(str(exc))

    repo_root = (args.repo_root or repo_root_from_package()).resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = repo_root / "artifacts" / "runs" / args.candidate
    output_dir = output_dir.resolve()

    if args.latency_ms < 0:
        parser.error("--latency-ms must be >= 0")

    try:
        base_image_cfg = _build_image_cfg(args)
    except ValueError as exc:
        parser.error(str(exc))

    latency_values_ms: list[float]
    if args.latency_sweep_ms:
        latency_values_ms = [
            float(x.strip()) for x in args.latency_sweep_ms.split(",") if x.strip()
        ]
        if not latency_values_ms:
            parser.error("--latency-sweep-ms is empty")
        if any(v < 0 for v in latency_values_ms):
            parser.error("--latency-sweep-ms values must be >= 0")
    else:
        latency_values_ms = [args.latency_ms]

    image_jobs: list[tuple[str | None, ImageDegradeConfig]]
    if args.image_sweep:
        if args.image_mode != ImageMode.NATIVE.value or args.capture_hw or args.jpeg_quality:
            parser.error("--image-sweep is mutually exclusive with explicit image mode flags")
        image_jobs = [(name, cfg) for name, cfg in _image_sweep_cfgs(args.image_sweep)]
    else:
        image_jobs = [(None, base_image_cfg)]

    if len(latency_values_ms) > 1 and len(image_jobs) > 1:
        parser.error("combine only one of --latency-sweep-ms or --image-sweep per invocation")

    exit_code = 0
    for latency_ms in latency_values_ms:
        for image_name, image_cfg in image_jobs:
            run_dir = output_dir
            if len(latency_values_ms) > 1:
                run_dir = output_dir / f"latency_ms_{latency_ms:g}"
            if image_name is not None:
                run_dir = output_dir / image_name
            report = run_baseline(
                RunRequest(
                    candidate=args.candidate,
                    seeds=seeds,
                    output_dir=run_dir,
                    device=args.device,
                    save_video=args.video,
                    latency_s=latency_ms / 1000.0,
                    image_cfg=image_cfg,
                    repo_root=repo_root,
                )
            )
            print(report)
            print(
                json.dumps(
                    {
                        "latency_ms": latency_ms,
                        "image_mode": image_cfg.mode.value,
                        "image_label": image_name,
                        "chunk_arrival_policy": CHUNK_ARRIVAL_POLICY,
                        "report": str(report),
                    }
                )
            )
            payload = json.loads(report.read_text(encoding="utf-8"))
            summary = payload.get("summary", {})
            provenance = payload.get("provenance", {})
            if provenance.get("env_created") is False:
                exit_code = 2
            if summary.get("n_runtime_error", 0) > 0 and summary.get("n_episodes_eligible", 0) == 0:
                exit_code = 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
