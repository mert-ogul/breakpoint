"""Dependency isolation: pure, profiling and report modules import without simulation stacks."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

BANNED_ROOTS = (
    "torch",
    "torchvision",
    "lerobot",
    "mujoco",
    "dm_control",
    "gymnasium",
    "gym_aloha",
)
# Internal modules that run policies or simulators; report code must not load them.
EXECUTION_MODULES = (
    "breakpoint_eval.act_export",
    "breakpoint_eval.policy_adapter",
    "breakpoint_eval.run_baseline",
    "breakpoint_eval.run_episode",
    "breakpoint_eval.sim_env",
    "breakpoint_eval.stack",
)


def _loaded_after_import(module: str) -> dict[str, list[str]]:
    """Import ``module`` in a clean interpreter with banned packages blocked."""
    script = textwrap.dedent(
        f"""
        import importlib, json, sys
        from importlib.abc import MetaPathFinder

        BANNED = {BANNED_ROOTS!r}

        class Ban(MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in BANNED:
                    raise ImportError("blocked " + name)

        sys.meta_path.insert(0, Ban())
        before = set(sys.modules)
        importlib.import_module({module!r})
        new = set(sys.modules) - before
        stdlib = sys.stdlib_module_names
        print(json.dumps({{
            "internal": sorted(m for m in new if m.startswith("breakpoint_eval.")),
            "third_party": sorted(
                {{m.split(".")[0] for m in new}}
                - set(stdlib) - {{"breakpoint_eval", "__main__"}}
                - {{m for m in new if m.startswith("_")}}
            ),
        }}))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


@pytest.mark.parametrize(
    "module",
    [
        "breakpoint_eval.source_revision",
        "breakpoint_eval.file_digest",
        "breakpoint_eval.artifact_names",
        "breakpoint_eval.operation_summary",
    ],
)
def test_pure_modules_use_only_the_standard_library(module: str) -> None:
    loaded = _loaded_after_import(module)
    assert loaded["third_party"] == []


def test_export_validation_does_not_load_torch() -> None:
    loaded = _loaded_after_import("breakpoint_eval.export_validation")
    assert "breakpoint_eval.act_export" not in loaded["internal"]


@pytest.mark.parametrize(
    "module", ["breakpoint_eval.benchmark_report_data", "breakpoint_eval.benchmark_report"]
)
def test_report_modules_do_not_load_execution_code(module: str) -> None:
    pytest.importorskip("matplotlib")
    loaded = _loaded_after_import(module)
    assert not set(loaded["internal"]) & set(EXECUTION_MODULES)


def test_single_run_report_does_not_load_execution_code() -> None:
    loaded = _loaded_after_import("breakpoint_eval.render_report")
    assert not set(loaded["internal"]) & set(EXECUTION_MODULES)
