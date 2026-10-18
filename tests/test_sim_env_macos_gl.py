"""macOS MuJoCo/GLFW prepare helper (no synthetic env)."""

from __future__ import annotations

import sys
import types

import pytest

from breakpoint_eval.sim_env import prepare_mujoco_macos_gl


def test_prepare_mujoco_macos_gl_noop_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("breakpoint_eval.sim_env.platform.system", lambda: "Linux")
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    prepare_mujoco_macos_gl()
    assert "MUJOCO_GL" not in __import__("os").environ


def test_prepare_mujoco_macos_gl_disable_then_glfw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("breakpoint_eval.sim_env.platform.system", lambda: "Darwin")
    monkeypatch.setenv("MUJOCO_GL", "glfw")
    # Drop any prior mujoco module so prepare runs.
    monkeypatch.delitem(sys.modules, "mujoco", raising=False)

    fake = types.ModuleType("mujoco")
    fake.__version__ = "2.3.7-test"

    real_import = __import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001
        if name == "mujoco":
            assert __import__("os").environ.get("MUJOCO_GL") == "disable"
            sys.modules["mujoco"] = fake
            return fake
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", _import)
    prepare_mujoco_macos_gl()
    assert __import__("os").environ.get("MUJOCO_GL") == "glfw"
    assert sys.modules["mujoco"] is fake


def test_prepare_mujoco_macos_gl_skips_when_already_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("breakpoint_eval.sim_env.platform.system", lambda: "Darwin")
    monkeypatch.setenv("MUJOCO_GL", "cgl")
    fake = types.ModuleType("mujoco")
    monkeypatch.setitem(sys.modules, "mujoco", fake)
    prepare_mujoco_macos_gl()
    assert __import__("os").environ.get("MUJOCO_GL") == "cgl"
