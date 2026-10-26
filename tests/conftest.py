"""Turn skips into failures when ``BREAKPOINT_FAIL_ON_SKIP=1`` (set by CI)."""

from __future__ import annotations

import importlib.util
import os

import pytest

FAIL_ON_SKIP_ENV = "BREAKPOINT_FAIL_ON_SKIP"
ONNX_MODULES = ("onnx", "onnxruntime")


def _skip_reason(longrepr: object) -> str:
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return str(longrepr)


def _fail_if_skipped(report: pytest.CollectReport | pytest.TestReport) -> None:
    if os.environ.get(FAIL_ON_SKIP_ENV) == "1" and report.skipped:
        report.outcome = "failed"
        report.longrepr = f"skipped under {FAIL_ON_SKIP_ENV}=1: {_skip_reason(report.longrepr)}"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    report = yield
    _fail_if_skipped(report)
    return report


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector: pytest.Collector):
    report = yield
    _fail_if_skipped(report)
    return report


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    missing = [m for m in ONNX_MODULES if importlib.util.find_spec(m) is None]
    if not missing:
        return
    skip = pytest.mark.skip(reason=f"needs {', '.join(missing)} (pip install -e '.[onnx]')")
    for item in items:
        if item.get_closest_marker("requires_onnx"):
            item.add_marker(skip)
