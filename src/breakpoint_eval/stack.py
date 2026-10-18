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

# Env override for writable workspace when the package is not a source checkout.
_BREAKPOINT_ROOT_ENV = "BREAKPOINT_ROOT"


def _is_under_site_packages(path: Path) -> bool:
    return any(part == "site-packages" for part in path.parts)


def _looks_like_checkout(root: Path) -> bool:
    """True when ``root`` is this project's source tree (writable checkout)."""
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return False
    return 'name = "breakpoint-eval"' in text or "name = 'breakpoint-eval'" in text


def default_user_workspace() -> Path:
    """User-writable workspace for caches/vendor/artifacts (never site-packages)."""
    override = os.environ.get(_BREAKPOINT_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg).expanduser().resolve() / "breakpoint-eval"
    return (Path.home() / ".local" / "share" / "breakpoint-eval").resolve()


def repo_root_from_package() -> Path:
    """Return a writable workspace root for vendor/, .cache/, and default artifacts.

    Editable / source checkouts resolve to the repository root. A non-editable
    install under site-packages must not receive user caches; use
    ``BREAKPOINT_ROOT`` or ``~/.local/share/breakpoint-eval`` instead.
    """
    pkg_file = Path(__file__).resolve()
    # src/breakpoint_eval/stack.py -> parents[2] == repo root
    src_layout_root = pkg_file.parents[2]
    if _looks_like_checkout(src_layout_root) and not _is_under_site_packages(src_layout_root):
        return src_layout_root
    # Flat install next to pyproject (unusual); still prefer a real checkout.
    flat_root = pkg_file.parents[1]
    if _looks_like_checkout(flat_root) and not _is_under_site_packages(flat_root):
        return flat_root
    if _is_under_site_packages(pkg_file):
        LOGGER.info(
            "Package lives under site-packages; using user workspace %s "
            "(set %s to override)",
            default_user_workspace(),
            _BREAKPOINT_ROOT_ENV,
        )
    return default_user_workspace()


def ensure_lerobot_on_path(repo_root: Path) -> Path:
    """Ensure vendor/lerobot is at the card pin and importable."""
    dest = ensure_lerobot_checkout(repo_root)
    dest_str = str(dest)
    if dest_str not in sys.path:
        sys.path.insert(0, dest_str)
    return dest


def _git_head(dest: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dest, text=True).strip()


def _git_is_dirty(dest: Path) -> bool:
    """True if the worktree or index has uncommitted changes (including untracked)."""
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=dest,
        text=True,
    )
    return bool(status.strip())


def _configure_lfs_filters(dest: Path, env: dict[str, str]) -> None:
    # Neutralize LFS filters when git-lfs is unavailable (common on minimal hosts).
    _run(["git", "config", "--local", "filter.lfs.smudge", "cat"], cwd=dest, env=env)
    _run(["git", "config", "--local", "filter.lfs.clean", "cat"], cwd=dest, env=env)
    _run(["git", "config", "--local", "filter.lfs.required", "false"], cwd=dest, env=env)


def ensure_lerobot_checkout(repo_root: Path) -> Path:
    """Clone or reuse vendor/lerobot at the card pin.

    Reuses an existing pin without re-fetching. Never force-checkouts over
    local modifications; refuses when the tree is dirty and at the wrong commit.
    """
    dest = repo_root / "vendor" / "lerobot"
    dest.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}

    if not (dest / ".git").exists():
        _run(["git", "clone", LEROBOT_GIT, str(dest)], env=env)
        _configure_lfs_filters(dest, env)
        _run(["git", "fetch", "--depth", "1", "origin", LEROBOT_COMMIT], cwd=dest, env=env)
        _run(["git", "checkout", LEROBOT_COMMIT], cwd=dest, env=env)
    else:
        _configure_lfs_filters(dest, env)
        head = _git_head(dest)
        dirty = _git_is_dirty(dest)
        if head == LEROBOT_COMMIT:
            if dirty:
                LOGGER.warning(
                    "vendor/lerobot is at pin %s with local modifications; "
                    "reusing without checkout",
                    LEROBOT_COMMIT,
                )
            else:
                LOGGER.info("Reusing existing vendor/lerobot@%s", LEROBOT_COMMIT)
            return dest
        if dirty:
            raise RuntimeError(
                f"vendor/lerobot is at {head} (want {LEROBOT_COMMIT}) with local "
                "modifications; refuse to overwrite. Commit/stash changes or remove "
                f"{dest}."
            )
        LOGGER.info(
            "vendor/lerobot at %s; checking out pin %s (clean tree)",
            head,
            LEROBOT_COMMIT,
        )
        _run(["git", "fetch", "--depth", "1", "origin", LEROBOT_COMMIT], cwd=dest, env=env)
        _run(["git", "checkout", LEROBOT_COMMIT], cwd=dest, env=env)

    head = _git_head(dest)
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
