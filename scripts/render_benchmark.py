#!/usr/bin/env python3
"""Render a packaged benchmark to HTML, Markdown, plots, and a data index."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

LOGGER = logging.getLogger("render_benchmark")
REPORT_EXTRA_MODULES = frozenset({"matplotlib", "numpy"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="packaged data directory")
    parser.add_argument("--out", type=Path, required=True, help="output directory for the report")
    parser.add_argument(
        "--update-readme",
        type=Path,
        help="rewrite the generated results block of this README from the packaged data",
    )
    parser.add_argument(
        "--readme-block",
        default="results",
        help='README marker name: "results" (ACT) or "results-pusht" (Diffusion PushT)',
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        from breakpoint_eval.benchmark_report import (
            ReportValidationError,
            render_benchmark_report,
            update_readme_results,
        )
        from breakpoint_eval.benchmark_report_data import ReportDataError
    except ModuleNotFoundError as exc:
        if exc.name not in REPORT_EXTRA_MODULES:
            raise
        LOGGER.error(
            "report dependency %r is missing; install with: pip install -e '.[report]'", exc.name
        )
        return 2
    try:
        written = render_benchmark_report(args.input, args.out)
        if args.update_readme is not None:
            changed = update_readme_results(
                args.input, args.update_readme, readme_block=args.readme_block
            )
            LOGGER.info("%s %s", args.update_readme, "updated" if changed else "already current")
    except (ReportValidationError, ReportDataError) as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info("open %s", (args.out / "index.html").resolve())
    LOGGER.info("%d files written", len(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
