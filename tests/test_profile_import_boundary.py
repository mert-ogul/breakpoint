"""Regression: Pi / [onnx] profile path must not import Torch, LeRobot, or MuJoCo."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Roots that must never be pulled by the device-profile import graph.
_BANNED_ROOTS = frozenset(
    {
        "torch",
        "torchvision",
        "torchaudio",
        "mujoco",
        "dm_control",
        "lerobot",
        "gymnasium",
        "gym",
        "gym_aloha",
    }
)

# Modules the Pi profile path is allowed / expected to load.
_PROFILE_ENTRYPOINTS = (
    "breakpoint_eval.source_revision",
    "breakpoint_eval.latency_trace",
    "breakpoint_eval.device_profile",
    "breakpoint_eval.run_profile",
    "breakpoint_eval.v1_profile",
)


class _BanHeavyFinder:
    """Meta path finder that fails loudly if a banned package is imported."""

    def find_spec(self, fullname, path, target=None):  # noqa: ANN001
        root = fullname.split(".", 1)[0]
        if root in _BANNED_ROOTS:
            raise ImportError(
                f"profile import boundary violated: attempted to import {fullname!r}"
            )
        return None


def test_profile_modules_import_without_torch_sim() -> None:
    """Import the real profiling stack with banned packages blocked at the meta path."""
    finder = _BanHeavyFinder()
    sys.meta_path.insert(0, finder)
    # Drop already-loaded banned modules so a prior test cannot mask the boundary.
    for name in list(sys.modules):
        root = name.split(".", 1)[0]
        if root in _BANNED_ROOTS or name.startswith("breakpoint_eval."):
            # Keep candidate_map / seed_list free modules; purge profile-related and heavy.
            if root in _BANNED_ROOTS:
                del sys.modules[name]
            elif name in _PROFILE_ENTRYPOINTS or name.startswith(
                (
                    "breakpoint_eval.v1_profile",
                    "breakpoint_eval.device_profile",
                    "breakpoint_eval.latency_trace",
                    "breakpoint_eval.run_profile",
                    "breakpoint_eval.source_revision",
                    "breakpoint_eval.v1_protocol",  # must not be required by v1_profile
                )
            ):
                del sys.modules[name]
    try:
        for mod_name in _PROFILE_ENTRYPOINTS:
            importlib.import_module(mod_name)
        # Explicitly prove v1_profile does not need v1_protocol (torch chain).
        assert "breakpoint_eval.v1_protocol" not in sys.modules
        loaded_banned = sorted(
            name for name in sys.modules if name.split(".", 1)[0] in _BANNED_ROOTS
        )
        assert loaded_banned == []
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)


def test_profile_v1_script_help_subprocess_blocks_torch() -> None:
    """Subprocess imports profile_v1 with a torch ban (catches module-level leaks)."""
    ban = ",".join(sorted(_BANNED_ROOTS))
    code = f"""
import sys
from importlib.abc import MetaPathFinder

BANNED = set({ban!r}.split(","))

class Ban(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".", 1)[0] in BANNED:
            raise ImportError("banned:" + fullname)
        return None

sys.meta_path.insert(0, Ban())
sys.path.insert(0, {str(ROOT / "src")!r})
import runpy
sys.argv = ["profile_v1", "--help"]
try:
    runpy.run_path({str(ROOT / "scripts" / "profile_v1.py")!r}, run_name="__main__")
except SystemExit as exc:
    raise SystemExit(exc.code or 0)
"""
    py = ROOT / ".venv" / "bin" / "python"
    exe = str(py) if py.is_file() else sys.executable
    proc = subprocess.run(
        [exe, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--device-label" in proc.stdout
    assert "banned:" not in proc.stderr
    assert "No module named 'torch'" not in proc.stderr


def test_v1_profile_does_not_import_v1_protocol() -> None:
    import inspect

    import breakpoint_eval.v1_profile as mod

    src = inspect.getsource(mod)
    assert "v1_protocol" not in src
    assert "source_revision" in src
