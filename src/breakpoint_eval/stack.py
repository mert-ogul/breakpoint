"""Pinned LeRobot / Hub stack helpers shared by feasibility and breakpoint run."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from breakpoint_eval.candidate_map import CANDIDATES, LEROBOT_CARD_COMMIT, CandidateSpec

LOGGER = logging.getLogger("breakpoint_eval.stack")

LEROBOT_GIT = "https://github.com/huggingface/lerobot.git"
LEROBOT_COMMIT = LEROBOT_CARD_COMMIT


def repo_root_from_package() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_lerobot_on_path(repo_root: Path) -> Path:
    """Ensure vendor/lerobot is at the card pin and importable."""
    dest = ensure_lerobot_checkout(repo_root)
    dest_str = str(dest)
    if dest_str not in sys.path:
        sys.path.insert(0, dest_str)
    return dest


def ensure_lerobot_checkout(repo_root: Path) -> Path:
    dest = repo_root / "vendor" / "lerobot"
    dest.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}

    if not (dest / ".git").exists():
        _run(["git", "clone", LEROBOT_GIT, str(dest)], env=env)

    _run(["git", "config", "--local", "filter.lfs.smudge", "cat"], cwd=dest, env=env)
    _run(["git", "config", "--local", "filter.lfs.clean", "cat"], cwd=dest, env=env)
    _run(["git", "config", "--local", "filter.lfs.required", "false"], cwd=dest, env=env)
    _run(["git", "fetch", "--depth", "1", "origin", LEROBOT_COMMIT], cwd=dest, env=env)
    _run(["git", "checkout", "-f", LEROBOT_COMMIT], cwd=dest, env=env)

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dest, text=True).strip()
    if head != LEROBOT_COMMIT:
        raise RuntimeError(f"Expected LeRobot HEAD {LEROBOT_COMMIT}, got {head}")
    return dest


def download_candidate_checkpoint(spec: CandidateSpec, cache_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download checkpoints "
            "(install via requirements-feasibility.txt or LeRobot)."
        ) from exc
    path = snapshot_download(
        repo_id=spec.hub_repo,
        revision=spec.hub_revision,
        cache_dir=str(cache_dir),
    )
    return Path(path)


def resolve_candidate(name: str) -> CandidateSpec:
    try:
        return CANDIDATES[name]
    except KeyError as exc:
        known = ", ".join(sorted(CANDIDATES))
        raise ValueError(f"Unknown candidate {name!r}. Known: {known}") from exc


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    LOGGER.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd, env=env)
