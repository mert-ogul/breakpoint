#!/usr/bin/env python3
"""Render a packaged v0.1 evaluation to HTML, Markdown, plots, and a data index."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

LOGGER = logging.getLogger("render_v1")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="packaged data directory")
    parser.add_argument("--out", type=Path, required=True, help="output directory for the report")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        from breakpoint_eval.v1_report import ReportValidationError, render_report
        from breakpoint_eval.v1_report_data import ReportDataError
    except ImportError as exc:
        LOGGER.error("report dependencies are missing (%s); install with: pip install -e '.[report]'", exc)
        return 2
    try:
        written = render_report(args.input, args.out)
    except (ReportValidationError, ReportDataError) as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info("open %s", (args.out / "index.html").resolve())
    LOGGER.info("%d files written", len(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
