"""Thin CLI for Breakpoint. Subcommand: run."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from breakpoint_eval.candidate_map import CANDIDATES
from breakpoint_eval.run_baseline import RunRequest, build_seeds, parse_seed_list, run_baseline
from breakpoint_eval.stack import repo_root_from_package


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

    report = run_baseline(
        RunRequest(
            candidate=args.candidate,
            seeds=seeds,
            output_dir=output_dir,
            device=args.device,
            save_video=args.video,
            repo_root=repo_root,
        )
    )
    print(report)

    payload = json.loads(report.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    provenance = payload.get("provenance", {})
    if provenance.get("env_created") is False:
        return 2
    if summary.get("n_runtime_error", 0) > 0 and summary.get("n_episodes_eligible", 0) == 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
