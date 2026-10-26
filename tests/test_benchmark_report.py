"""Saved-example report: validation against episode records and the dependency boundary."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from breakpoint_eval.benchmark_report import ReportValidationError, render_report  # noqa: E402
from breakpoint_eval.benchmark_report_data import ReportDataError, load_report_data  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "act-transfer" / "v1" / "data"

pytestmark = pytest.mark.skipif(not DATA.is_dir(), reason="packaged v1 data not present")


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def refresh_manifest(data: Path) -> None:
    files = [
        {
            "path": p.relative_to(data).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in sorted(data.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    ]
    (data / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": files}))


def test_saved_example_passes_every_check() -> None:
    data = load_report_data(DATA)
    failed = [c.name for c in data.checks if not c.passed]
    assert failed == []
    baseline = data.axes[0].conditions[0]
    assert (baseline.successes, baseline.eligible) == (25, 30)
    assert len(data.rows) == 600


def test_render_does_not_edit_input_and_counts_match_records(tmp_path: Path) -> None:
    before = tree_hash(DATA)
    out = tmp_path / "report"
    render_report(DATA, out)
    assert tree_hash(DATA) == before
    index = json.loads((out / "data_index.json").read_text())
    rows = [
        json.loads(line) for line in (DATA / "episodes" / "episodes.jsonl").read_text().splitlines()
    ]
    for entry in index["displayed_counts"]:
        group = [
            r
            for r in rows
            if r["plan_id"] == entry["plan_id"] and r["condition_id"] == entry["condition_id"]
        ]
        if group:
            assert entry["successes"] == sum(r["success"] for r in group)
            assert entry["planned"] == len(group)
    html = (out / "index.html").read_text()
    assert "14/30" in html and "Read before the numbers" in html
    assert (out / "plots" / "success_latency_ms.svg").is_file()
    assert (out / "plots" / "success_latency_ms.png").is_file()


def test_changed_episode_record_is_refused(tmp_path: Path) -> None:
    copy = tmp_path / "data"
    shutil.copytree(DATA, copy)
    episodes = copy / "episodes" / "episodes.jsonl"
    lines = episodes.read_text().splitlines()
    row = json.loads(lines[0])
    row.update(
        success=not row["success"], outcome="time_limit_truncation" if row["success"] else "success"
    )
    lines[0] = json.dumps(row, sort_keys=True)
    episodes.write_text("\n".join(lines) + "\n")
    refresh_manifest(copy)
    with pytest.raises(ReportValidationError, match="saved summary counts"):
        render_report(copy, tmp_path / "report")
    assert not (tmp_path / "report" / "index.html").exists()


def test_tampered_file_fails_manifest_check(tmp_path: Path) -> None:
    copy = tmp_path / "data"
    shutil.copytree(DATA, copy)
    (copy / "episodes" / "episodes.jsonl").write_text("[]\n")
    with pytest.raises(ReportDataError, match="manifest"):
        render_report(copy, tmp_path / "report")


def test_render_script_runs_without_heavy_packages(tmp_path: Path) -> None:
    banned = [
        "torch",
        "torchvision",
        "cv2",
        "mujoco",
        "dm_control",
        "lerobot",
        "gymnasium",
        "gym_aloha",
        "onnxruntime",
        "onnx",
        "huggingface_hub",
        "requests",
    ]
    runner = f"""
import runpy, sys
from importlib.abc import MetaPathFinder
BANNED = {banned!r}
class Ban(MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in BANNED:
            raise ImportError("blocked " + name)
sys.meta_path.insert(0, Ban())
sys.argv = ["render_benchmark.py", "--input", {str(DATA)!r}, "--out", {str(tmp_path / "out")!r}]
try:
    runpy.run_path({str(ROOT / "scripts" / "render_benchmark.py")!r}, run_name="__main__")
except SystemExit as exc:
    sys.exit(exc.code)
"""
    proc = subprocess.run(
        [sys.executable, "-c", runner], capture_output=True, text=True, cwd=tmp_path, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "out" / "index.html").is_file()
