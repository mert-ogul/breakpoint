"""The device-profile route must not import Torch, LeRobot, MuJoCo, or Gym.

Each check runs in a fresh interpreter with the banned packages blocked, so nothing here
depends on, or disturbs, modules loaded by other tests.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BANNED_ROOTS = (
    "torch",
    "torchvision",
    "torchaudio",
    "mujoco",
    "dm_control",
    "lerobot",
    "gymnasium",
    "gym",
    "gym_aloha",
)

PROFILE_MODULES = (
    "breakpoint_eval.source_revision",
    "breakpoint_eval.latency_trace",
    "breakpoint_eval.device_profile",
    "breakpoint_eval.run_profile",
    "breakpoint_eval.export_validation",
    "breakpoint_eval.benchmark_profile",
)

BAN_PRELUDE = f"""
import sys
from importlib.abc import MetaPathFinder

BANNED = set({BANNED_ROOTS!r})


class Ban(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".", 1)[0] in BANNED:
            raise ImportError("banned:" + fullname)
        return None


sys.meta_path.insert(0, Ban())
sys.path.insert(0, {str(ROOT / "src")!r})
"""


def run_banned(code: str) -> subprocess.CompletedProcess[str]:
    py = ROOT / ".venv" / "bin" / "python"
    exe = str(py) if py.is_file() else sys.executable
    return subprocess.run(
        [exe, "-c", BAN_PRELUDE + code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_profile_modules_import_without_heavy_packages() -> None:
    proc = run_banned(
        f"""
import importlib

for name in {PROFILE_MODULES!r}:
    importlib.import_module(name)
loaded = sorted(m for m in sys.modules if m.split(".", 1)[0] in BANNED)
assert loaded == [], loaded
assert "breakpoint_eval.benchmark_protocol" not in sys.modules
"""
    )
    assert proc.returncode == 0, proc.stderr


def test_profile_route_runs_without_heavy_packages(tmp_path: Path) -> None:
    """Calling the route, not only importing it, stays clear of the banned packages."""
    proc = run_banned(
        f"""
from pathlib import Path

from breakpoint_eval.benchmark_profile import resolve_validated_fp32_bundle
from breakpoint_eval.source_revision import read_source_revision

try:
    resolve_validated_fp32_bundle(Path({str(tmp_path)!r}))
except FileNotFoundError:
    pass
else:
    raise SystemExit("an empty export directory was accepted")
assert read_source_revision(Path({str(tmp_path)!r}))["kind"] == "unavailable"
loaded = sorted(m for m in sys.modules if m.split(".", 1)[0] in BANNED)
assert loaded == [], loaded
"""
    )
    assert proc.returncode == 0, proc.stderr


def test_profile_device_script_help_blocks_heavy_packages() -> None:
    proc = run_banned(
        f"""
import runpy

sys.argv = ["profile_device", "--help"]
try:
    runpy.run_path({str(ROOT / "scripts" / "profile_device.py")!r}, run_name="__main__")
except SystemExit as exc:
    raise SystemExit(exc.code or 0)
"""
    )
    assert proc.returncode == 0, proc.stderr
    assert "--device-label" in proc.stdout
    assert "banned:" not in proc.stderr
