"""Workspace path resolution and LeRobot vendor checkout safety."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from breakpoint_eval.stack import (
    LEROBOT_COMMIT,
    default_user_workspace,
    ensure_lerobot_checkout,
    repo_root_from_package,
)


def test_repo_root_from_editable_checkout() -> None:
    root = repo_root_from_package()
    assert (root / "pyproject.toml").is_file()
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "breakpoint-eval"' in text
    assert "site-packages" not in root.parts


def test_default_user_workspace_respects_breakpoint_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "ws"
    monkeypatch.setenv("BREAKPOINT_ROOT", str(target))
    assert default_user_workspace() == target.resolve()


def test_default_user_workspace_not_site_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BREAKPOINT_ROOT", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    ws = default_user_workspace()
    assert "site-packages" not in ws.parts
    assert ws == (Path.home() / ".local" / "share" / "breakpoint-eval").resolve()


def _init_git_repo(path: Path, *, commit_message: str = "init") -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, check=True)
    (path / "README").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "add", "README"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", commit_message], cwd=path, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def test_ensure_lerobot_reuses_pin_without_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "vendor" / "lerobot"
    _init_git_repo(dest)
    # Pretend HEAD is already the card pin.
    monkeypatch.setattr(
        "breakpoint_eval.stack._git_head",
        lambda _dest: LEROBOT_COMMIT,
    )
    calls: list[list[str]] = []

    def _record(cmd: list[str], *, cwd=None, env=None) -> None:  # noqa: ANN001
        calls.append(cmd)

    monkeypatch.setattr("breakpoint_eval.stack._run", _record)
    out = ensure_lerobot_checkout(tmp_path)
    assert out == dest
    assert not any(cmd[:2] == ["git", "fetch"] for cmd in calls)
    assert not any(cmd[:2] == ["git", "checkout"] for cmd in calls)
    # LFS filter config still runs (idempotent local config).
    assert any(cmd[:3] == ["git", "config", "--local"] for cmd in calls)


def test_ensure_lerobot_refuses_dirty_wrong_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "vendor" / "lerobot"
    _init_git_repo(dest)
    (dest / "dirty.txt").write_text("local edit\n", encoding="utf-8")
    monkeypatch.setattr(
        "breakpoint_eval.stack._git_head",
        lambda _dest: "0" * 40,
    )

    def _fail_if_checkout(cmd: list[str], *, cwd=None, env=None) -> None:  # noqa: ANN001
        if cmd[:2] == ["git", "checkout"]:
            raise AssertionError("must not checkout over dirty tree")

    monkeypatch.setattr("breakpoint_eval.stack._run", _fail_if_checkout)
    with pytest.raises(RuntimeError, match="refuse to overwrite"):
        ensure_lerobot_checkout(tmp_path)


def test_ensure_lerobot_no_force_flag_on_clean_retarget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "vendor" / "lerobot"
    _init_git_repo(dest)
    monkeypatch.setattr(
        "breakpoint_eval.stack._git_head",
        lambda _dest: "0" * 40,
    )
    checkouts: list[list[str]] = []

    def _record(cmd: list[str], *, cwd=None, env=None) -> None:  # noqa: ANN001
        if cmd[:2] == ["git", "checkout"]:
            checkouts.append(cmd)
        # After checkout, pretend we landed on the pin.
        if cmd[:2] == ["git", "checkout"]:
            monkeypatch.setattr(
                "breakpoint_eval.stack._git_head",
                lambda _dest: LEROBOT_COMMIT,
            )

    monkeypatch.setattr("breakpoint_eval.stack._run", _record)
    ensure_lerobot_checkout(tmp_path)
    assert checkouts
    assert "-f" not in checkouts[0]
    assert LEROBOT_COMMIT in checkouts[0]
