"""Source provenance helpers (standard library only).

Safe for Pi / ``[onnx]`` profile environments: must not import Torch, LeRobot,
MuJoCo, or any simulator stack.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("breakpoint_eval.source_revision")


def read_source_revision(repo_root: Path) -> dict[str, Any]:
    """Prefer Git; fall back to source_revision.txt for archive checkouts.

    When Git is absent, mark archive context explicitly and leave clean
    working-tree status unknown.
    """
    repo_root = Path(repo_root)
    git_dir = repo_root / ".git"
    if git_dir.exists():
        try:
            rev = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            dirty = bool(status.strip())
            return {
                "kind": "git",
                "revision": rev,
                "dirty": dirty,
                "status_porcelain": status,
                "archive_context": False,
            }
        except (OSError, subprocess.CalledProcessError) as exc:
            LOGGER.warning("git revision probe failed: %s", exc)
    archive = repo_root / "source_revision.txt"
    if archive.is_file():
        text = archive.read_text(encoding="utf-8").strip()
        return {
            "kind": "source_revision.txt",
            "revision": text or None,
            "dirty": None,
            "status_porcelain": None,
            "archive_context": True,
            "note": "Git absent; revision from source_revision.txt; clean status unknown",
        }
    return {
        "kind": "unavailable",
        "revision": None,
        "dirty": None,
        "status_porcelain": None,
        "archive_context": True,
        "note": "neither Git nor source_revision.txt available; revision unknown",
    }
