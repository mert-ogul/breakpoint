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

from breakpoint_eval.benchmark_report import (  # noqa: E402
    ReportValidationError,
    render_benchmark_report,
)
from breakpoint_eval.benchmark_report_data import ReportDataError, load_report_data  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "act-transfer" / "deployment-stress" / "data"

pytestmark = pytest.mark.skipif(not DATA.is_dir(), reason="packaged benchmark data not present")


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
    assert len(data.rows) == 810


def test_render_does_not_edit_input_and_counts_match_records(tmp_path: Path) -> None:
    before = tree_hash(DATA)
    out = tmp_path / "report"
    render_benchmark_report(DATA, out)
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
    assert "14/30" in html and "Validation, errors and caveats" in html
    assert "ACT cube transfer under stress" in html
    assert "eyebrow" not in html and "callout" not in html
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
    with pytest.raises(ReportValidationError, match="saved summary matches episode records"):
        render_benchmark_report(copy, tmp_path / "report")
    assert not (tmp_path / "report" / "index.html").exists()


def test_tampered_file_fails_manifest_check(tmp_path: Path) -> None:
    copy = tmp_path / "data"
    shutil.copytree(DATA, copy)
    (copy / "episodes" / "episodes.jsonl").write_text("[]\n")
    with pytest.raises(ReportDataError, match="manifest"):
        render_benchmark_report(copy, tmp_path / "report")


@pytest.mark.parametrize("manifest_text", ["", "{bad json", '{"files": 3}', '{"files": [{}]}'])
def test_malformed_manifest_is_a_report_error(tmp_path: Path, manifest_text: str) -> None:
    copy = tmp_path / "data"
    shutil.copytree(DATA, copy)
    (copy / "manifest.json").write_text(manifest_text)
    with pytest.raises(ReportDataError, match="not a valid manifest"):
        render_benchmark_report(copy, tmp_path / "report")


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


def test_readme_results_block_matches_saved_data(tmp_path: Path) -> None:
    from breakpoint_eval.benchmark_report import (
        README_RESULTS_END,
        README_RESULTS_START,
        load_validated_data,
        results_markdown,
        update_readme_results,
    )

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index(README_RESULTS_START) + len(README_RESULTS_START)
    committed = readme[start : readme.index(README_RESULTS_END)].strip("\n")
    assert committed == results_markdown(load_validated_data(DATA))

    stale = tmp_path / "README.md"
    stale.write_text(
        f"{README_RESULTS_START}\nSTALE-TABLE\n{README_RESULTS_END}\n", encoding="utf-8"
    )
    assert update_readme_results(DATA, stale) is True
    assert update_readme_results(DATA, stale) is False
    assert "STALE-TABLE" not in stale.read_text(encoding="utf-8")


def test_update_readme_requires_markers(tmp_path: Path) -> None:
    from breakpoint_eval.benchmark_report import update_readme_results

    readme = tmp_path / "README.md"
    readme.write_text("# no markers\n", encoding="utf-8")
    with pytest.raises(ReportValidationError, match="markers"):
        update_readme_results(DATA, readme)


# --- manifest, structure, wording, and table checks -------------------------------------


def copy_data(tmp_path: Path) -> Path:
    copy = tmp_path / "data"
    shutil.copytree(DATA, copy)
    return copy


def edit_json(path: Path, change) -> None:
    payload = json.loads(path.read_text())
    change(payload)
    path.write_text(json.dumps(payload))


def edit_manifest(copy: Path, change) -> None:
    edit_json(copy / "manifest.json", change)


def lock_path(copy: Path) -> Path:
    return next((copy / "protocol").glob("*protocol_lock.json"))


def edit_rows(copy: Path, change) -> None:
    path = copy / "episodes" / "episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    change(rows)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def refuses(copy: Path, tmp_path: Path, match: str) -> None:
    out = tmp_path / "report"
    with pytest.raises(ReportDataError, match=match):
        render_benchmark_report(copy, out)
    assert not out.exists()


def test_empty_manifest_is_refused(tmp_path: Path) -> None:
    copy = copy_data(tmp_path)
    (copy / "manifest.json").write_text('{"schema_version": 1, "files": []}')
    refuses(copy, tmp_path, "nonempty")


def test_unsupported_manifest_schema_is_refused(tmp_path: Path) -> None:
    copy = copy_data(tmp_path)
    edit_manifest(copy, lambda m: m.update(schema_version=2))
    refuses(copy, tmp_path, "schema_version")


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda m: m["files"].append(dict(m["files"][0])), "duplicate entry"),
        (lambda m: m["files"][0].update(path="/etc/hosts"), "plain relative path"),
        (lambda m: m["files"][0].update(path="../outside.json"), "plain relative path"),
        (lambda m: m["files"][0].update(path="episodes/../manifest.json"), "plain relative path"),
        (lambda m: m["files"][0].update(sha256="abc"), "64 lowercase hex"),
        (lambda m: m["files"][0].update(bytes=-1), "nonnegative integer"),
    ],
)
def test_unsafe_or_malformed_manifest_entries_are_refused(
    tmp_path: Path, change, match: str
) -> None:
    copy = copy_data(tmp_path)
    edit_manifest(copy, change)
    refuses(copy, tmp_path, match)


def test_declared_size_must_match(tmp_path: Path) -> None:
    copy = copy_data(tmp_path)
    edit_manifest(copy, lambda m: m["files"][0].update(bytes=m["files"][0]["bytes"] + 1))
    refuses(copy, tmp_path, "manifest")


def test_manifest_must_list_every_input_the_report_reads(tmp_path: Path) -> None:
    copy = copy_data(tmp_path)
    edit_manifest(
        copy,
        lambda m: m.update(
            files=[f for f in m["files"] if f["path"] != "profiles/rpi5/profile.json"]
        ),
    )
    refuses(copy, tmp_path, "does not list files the report reads")


def test_unlisted_plan_file_is_refused(tmp_path: Path) -> None:
    copy = copy_data(tmp_path)
    shutil.copy(
        next((copy / "protocol" / "plans").glob("*.json")),
        copy / "protocol" / "plans" / "extra.json",
    )
    refuses(copy, tmp_path, "does not list files the report reads")


def test_empty_episode_file_is_refused(tmp_path: Path) -> None:
    copy = copy_data(tmp_path)
    (copy / "episodes" / "episodes.jsonl").write_text("")
    refresh_manifest(copy)
    refuses(copy, tmp_path, "no episode rows")


def test_row_without_required_field_is_refused(tmp_path: Path) -> None:
    copy = copy_data(tmp_path)
    edit_rows(copy, lambda rows: rows[0].pop("seed"))
    refresh_manifest(copy)
    refuses(copy, tmp_path, "lacks required fields")


def test_missing_condition_rows_are_refused(tmp_path: Path) -> None:
    copy = copy_data(tmp_path)
    edit_rows(
        copy,
        lambda rows: rows.__setitem__(
            slice(None), [r for r in rows if r["condition_id"] != "jpeg_q75"]
        ),
    )
    refresh_manifest(copy)
    refuses(copy, tmp_path, "no episode rows for planned condition .*jpeg_q75")


def test_condition_without_eligible_episodes_is_refused(tmp_path: Path) -> None:
    copy = copy_data(tmp_path)

    def fail_all(rows: list[dict]) -> None:
        for r in rows:
            if r["condition_id"] == "jpeg_q75":
                r.update(outcome="runtime_error", success=False)

    edit_rows(copy, fail_all)
    refresh_manifest(copy)
    refuses(copy, tmp_path, "no eligible episodes")


def test_missing_plan_is_refused(tmp_path: Path) -> None:
    copy = copy_data(tmp_path)
    plan = next(p for p in (copy / "protocol" / "plans").glob("*.json") if "jpeg" in p.name)
    plan.unlink()
    refresh_manifest(copy)
    refuses(copy, tmp_path, "plans must cover")


def test_missing_profile_is_refused(tmp_path: Path) -> None:
    copy = copy_data(tmp_path)
    shutil.rmtree(copy / "profiles" / "rpi5")
    refresh_manifest(copy)
    refuses(copy, tmp_path, "does not list files the report reads")


def test_lock_without_recorded_statistics_is_refused(tmp_path: Path) -> None:
    copy = copy_data(tmp_path)
    edit_json(lock_path(copy), lambda lock: lock["statistics"].pop("breakpoint_ratio"))
    refresh_manifest(copy)
    refuses(copy, tmp_path, "missing statistics.breakpoint_ratio")


def test_threshold_ratio_comes_from_the_lock(tmp_path: Path) -> None:
    copy = copy_data(tmp_path)
    edit_json(lock_path(copy), lambda lock: lock["statistics"].update(breakpoint_ratio=0.7))
    refresh_manifest(copy)
    refuses(copy, tmp_path, "differ from the protocol lock")


def test_trace_window_must_match_the_plan(tmp_path: Path) -> None:
    copy = copy_data(tmp_path)
    edit_json(lock_path(copy), lambda lock: lock["trace_window"].update(length=16))
    refresh_manifest(copy)
    refuses(copy, tmp_path, "window_length")


def weak_baseline_data(tmp_path: Path):
    from breakpoint_eval.benchmark_report_data import load_report_data as load

    copy = copy_data(tmp_path)

    def weaken(rows: list[dict]) -> None:
        for r in rows:
            if r["condition_id"] == "latency_ms_0" or r["referenced_baseline"]:
                r.update(success=False, outcome="terminated_no_success")

    edit_rows(copy, weaken)
    refresh_manifest(copy)
    return load(copy)


def test_weak_baseline_suppresses_threshold_language(tmp_path: Path) -> None:
    from breakpoint_eval import benchmark_report as report

    data = weak_baseline_data(tmp_path)
    assert all(axis.suppressed_reason for axis in data.axes)
    text = "\n".join(report.observations(data))
    assert "first tested point" not in text and "below 0.8 x baseline" not in text
    assert text.count("threshold comparison not reported") == len(data.axes)
    for axis in data.axes:
        assert "not reported" in report.breakpoint_sentence(axis)
        assert report.unresolved_conditions(axis) == []
        assert all(
            report.status_text(axis, v) in {"baseline", "not assessed"} for v in axis.conditions
        )
    assert "below the 0.5 floor" in "\n".join(report.report_limitations(data))


def test_categorical_axes_are_not_described_as_crossings() -> None:
    from breakpoint_eval import benchmark_report as report

    data = report.load_validated_data(DATA)
    by_axis = {axis.axis: axis for axis in data.axes}
    assert not by_axis["precision"].ordered and not by_axis["device_profile"].ordered
    assert by_axis["latency_ms"].ordered and by_axis["h264"].ordered
    lines = report.observations(data)
    ordered = [line for line in lines if "first tested point" in line]
    assert len(ordered) == 3 and all("categorical" not in line for line in ordered)
    assert any("interval includes threshold" in line for line in ordered)
    for title in ("Numeric precision", "Device latency replay"):
        line = next(x for x in lines if x.startswith(title))
        assert "categorical comparison, no ordering" in line and "first tested" not in line
        assert "not below" in line or "below" in line
    for axis in (by_axis["precision"], by_axis["device_profile"]):
        sentence = report.breakpoint_sentence(axis)
        assert "Categorical comparison, no ordering" in sentence
    assert "First tested point below threshold" in report.breakpoint_sentence(by_axis["latency_ms"])
    assert "between" in report.breakpoint_sentence(by_axis["latency_ms"])


def table_widths(markdown: str) -> list[list[int]]:
    import re

    tables: list[list[int]] = []
    current: list[int] = []
    for line in markdown.splitlines():
        if line.startswith("|"):
            current.append(len(re.split(r"(?<!\\)\|", line.strip())[1:-1]))
        elif current:
            tables.append(current)
            current = []
    return tables + ([current] if current else [])


def test_markdown_tables_have_one_width_per_table() -> None:
    from breakpoint_eval import benchmark_report as report

    data = report.load_validated_data(DATA)
    tables = table_widths(report.render_markdown(data, Path("."), "plots"))
    assert len(tables) >= 7
    assert all(len(set(widths)) == 1 for widths in tables)
    assert tables[0][0] == 6


def test_markdown_cell_escapes_pipes() -> None:
    from breakpoint_eval import benchmark_report as report

    assert report.md_cell("a|b") == "a\\|b"
    assert table_widths("| x | " + report.md_cell("a|b") + " |\n")[0] == [2]
    data = report.load_validated_data(DATA)
    view = data.axes[0].conditions[1]
    assert "|" not in report.paired_text_plain(view)


def test_saved_example_counts_are_unchanged() -> None:
    from breakpoint_eval import benchmark_report as report

    data = report.load_validated_data(DATA)
    counts = {a.axis: [v.successes for v in a.conditions] for a in data.axes}
    assert counts == {
        "latency_ms": [25, 25, 25, 23, 18, 14],
        "resolution": [25, 26, 22, 10],
        "jpeg": [25, 25, 26, 26, 25, 23],
        "h264": [25, 25, 26, 28, 19, 10],
        "precision": [25, 24, 23],
        "device_profile": [25, 24, 9],
    }
    assert len(data.rows) == 810 and data.n_runs == 24
    assert data.trace_window_length == 32


def test_main_labels_are_readable_and_ids_stay_in_evidence() -> None:
    from breakpoint_eval import benchmark_report as report

    data = report.load_validated_data(DATA)
    for axis in data.axes:
        table = report.condition_table_html(axis, 30)
        assert not any(v.condition_id in table for v in axis.conditions)
        assert "(baseline)" in table
        assert not any(v.condition_id in report.breakpoint_sentence(axis) for v in axis.conditions)
    markdown = report.render_markdown(data, Path("."), "plots")
    assert "## Condition IDs" in markdown
    assert "`latency_ms_640`" in markdown.split("## Condition IDs")[1]
    assert "`latency_ms_640`" not in markdown.split("## Condition IDs")[0]


def test_short_label_int4_weight_only() -> None:
    from breakpoint_eval.benchmark_report_plots import short_label

    assert short_label("precision_int4_weight_only") == "INT4 weight-only"
    assert short_label("precision_int8") == "INT8 QDQ"


def test_short_label_resolution_native_uses_capture_hw() -> None:
    from breakpoint_eval.benchmark_report import plain_label, plain_label_view
    from breakpoint_eval.benchmark_report_data import ConditionView
    from breakpoint_eval.benchmark_report_plots import short_label

    assert short_label("resolution_native") == "480x640\nnative"
    assert short_label("resolution_native", capture_hw=[96, 96]) == "96x96\nnative"
    assert plain_label("resolution_native", capture_hw=[96, 96]) == "96x96 native"
    view = ConditionView(
        condition_id="resolution_native",
        label="native",
        referenced_baseline=False,
        is_baseline=True,
        rows=[{"capture_hw": [96, 96]}],
        successes=1,
        eligible=1,
        n_runtime_error=0,
        wilson={"low": 0.0, "high": 1.0},
    )
    assert plain_label_view(view) == "96x96 native"


def test_short_label_h264() -> None:
    from breakpoint_eval.benchmark_report_plots import short_label

    assert short_label("h264_bypass") == "bypass"
    assert short_label("h264_500kbps") == "500 kbps"
    assert short_label("h264_2000kbps") == "2000 kbps"


def test_h264_bytes_table_and_validate_rows() -> None:
    from breakpoint_eval.benchmark_report import h264_bytes_table_html, h264_bytes_table_markdown
    from breakpoint_eval.benchmark_report_data import (
        AxisView,
        ConditionView,
        measured_h264_kbps,
        validate_h264_rows,
    )

    rows = [
        {
            "axis": "h264",
            "condition_id": "h264_500kbps",
            "episode_index": 0,
            "n_control_steps": 10,
            "h264_requested_kbps": 500.0,
            "h264_encoded_bytes": 1250,
            "h264_frames": 10,
            "h264_actual_kbps": measured_h264_kbps(1250, 10, frame_rate_hz=50.0),
            "h264_holds": 0,
            "h264_first_frame_keyframe": True,
        },
        {
            "axis": "h264",
            "condition_id": "h264_500kbps",
            "episode_index": 1,
            "n_control_steps": 10,
            "h264_requested_kbps": 500.0,
            "h264_encoded_bytes": 1000,
            "h264_frames": 10,
            "h264_actual_kbps": measured_h264_kbps(1000, 10, frame_rate_hz=50.0),
            "h264_holds": 0,
            "h264_first_frame_keyframe": True,
        },
    ]
    view = ConditionView(
        condition_id="h264_500kbps",
        label="H.264 500 kbps",
        referenced_baseline=False,
        is_baseline=False,
        rows=rows,
        successes=2,
        eligible=2,
        n_runtime_error=0,
        wilson={"low": 0.3, "high": 0.9},
    )
    axis = AxisView(
        axis="h264",
        plan_id="v0.1_full_h264",
        title="H.264 bitrate",
        unit="kbps",
        phase="exploratory",
        baseline_condition_id="h264_bypass",
        ratio=0.8,
        baseline_floor=0.5,
        baseline_rate=1.0,
        threshold_rate=0.8,
        conditions=[view],
        derived=True,
        saved_breakpoint=None,
        saved_note=None,
        isolation_notes=[],
        ordered=True,
    )
    html = h264_bytes_table_html(axis)
    assert "500" in html
    assert "Mean encoded bytes" in html
    md = "\n".join(h264_bytes_table_markdown(axis))
    assert "Requested kbps" in md
    check = validate_h264_rows(rows)
    assert check is not None and check.passed
    assert "frame counts" in check.name
    assert "holds" in check.name
    assert "first keyframe" in check.name

    bad = dict(rows[0])
    bad["h264_holds"] = 1
    bad_check = validate_h264_rows([bad])
    assert bad_check is not None and not bad_check.passed


def test_int4_limitation_line_from_sidecar() -> None:
    from types import SimpleNamespace

    from breakpoint_eval.artifact_names import EXPORT_VALIDATION_INT4_NAME
    from breakpoint_eval.benchmark_report import int4_note

    sidecar = {
        "quantization_coverage": {
            "matmul_nbits_nodes": 29,
            "matmul_nodes_total": 41,
            "weight_params_int4_fraction": 0.6164908834826577,
            "recipe": {"accuracy_level": 0},
        },
        "runtime_checks": {
            "max_abs_error": 0.1459382176399231,
            "mean_abs_error": 0.006009796475399936,
        },
    }
    ran = SimpleNamespace(
        lock={"artifacts": {"int4_status": "ok"}},
        rows=[{"condition_id": "precision_int4_weight_only"}],
        validation_files={EXPORT_VALIDATION_INT4_NAME: sidecar},
    )
    line = int4_note(ran)
    assert line is not None
    assert "29 of 41 MatMul nodes" in line
    assert "61.6%" in line
    assert "accuracy_level=0" in line
    assert "0.146" in line
    assert "0.00601" in line
    assert "Not a full-model INT4 result" in line
    assert "no INT4 device timing" in line

    missing = SimpleNamespace(
        lock={
            "artifacts": {
                "int4_status": "missing",
                "int4_detail": "graph absent",
            }
        },
        rows=[],
        validation_files={},
    )
    skipped = int4_note(missing)
    assert skipped is not None
    assert "INT4 export status is 'missing'" in skipped
    assert "graph absent" in skipped
    assert "no INT4 measurement is reported" in skipped

    old = SimpleNamespace(
        lock={"artifacts": {"int8_status": "ok"}},
        rows=[],
        validation_files={},
    )
    assert int4_note(old) is None


def test_working_tree_porcelain_note() -> None:
    from breakpoint_eval.benchmark_report import working_tree_porcelain_note

    assert working_tree_porcelain_note("") == "clean"
    assert working_tree_porcelain_note(None) == "clean"
    assert working_tree_porcelain_note("\n") == "clean"
    note = working_tree_porcelain_note(" M src/breakpoint_eval/experiment_plan.py\n")
    assert note == "<code> M src/breakpoint_eval/experiment_plan.py</code>"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _episode_row(
    *,
    axis: str,
    plan_id: str,
    condition_id: str,
    episode_index: int,
    seed: int,
    success: bool,
    latency_s: float = 0.0,
    referenced: bool = False,
) -> dict:
    return {
        "axis": axis,
        "plan_id": plan_id,
        "condition_id": condition_id,
        "candidate": "diffusion_pusht",
        "referenced_baseline": referenced,
        "episode_index": episode_index,
        "seed": seed,
        "policy_rng_seed": seed + 10000,
        "outcome": "success" if success else "time_limit_truncation",
        "success": success,
        "runtime_error": None,
        "n_control_steps": 10,
        "sim_time_s": 1.0,
        "success_step_index": 5 if success else None,
        "final_step_sim_time_s": 1.0,
        "n_inference_calls": 2,
        "n_fresh_chunks": 2,
        "n_underflows": 0,
        "n_missed_deadlines": 0,
        "sum_reward": 1.0,
        "max_reward": 1.0,
        "latency_mode": "constant",
        "constant_latency_s": latency_s,
        "image_mode": "native",
        "capture_hw": [96, 96],
        "jpeg_quality": None,
        "trace_window_start": None,
        "trace_window_length": None,
        "consumed_trace_indices": None,
        "consumed_delays_s": None,
        "source_report": f"outputs/{plan_id}/{condition_id}/run_report.json",
        "h264_requested_kbps": None,
        "h264_encoded_bytes": None,
        "h264_frames": None,
        "h264_actual_kbps": None,
        "h264_holds": None,
        "h264_keyframes": None,
        "h264_first_frame_keyframe": None,
    }


def build_synthetic_pytorch_package(root: Path) -> Path:
    """Minimal Diffusion PushT package: three axes, timing, agreement, manifest."""
    from breakpoint_eval.experiment_summary import wilson_interval

    data = root / "data"
    seeds = [6000, 6001, 6002]
    checkpoint_files = {
        "model.safetensors": "11" * 32,
        "config.json": "22" * 32,
    }
    phase_config = {
        "phase": "exploratory",
        "env_seed_start": 6000,
        "n_episodes": 3,
        "policy_seed_offset": 10000,
        "latency_ms": [0, 100, 800],
        "resolution_hw": [[96, 96], [24, 24]],
        "jpeg_qualities": ["bypass", 10],
        "isolate_axes": True,
    }
    lock = {
        "schema_version": 1,
        "backend": "pytorch",
        "candidate": "diffusion_pusht",
        "protocol_name": "diffusion_pusht_stress",
        "protocol_version": "0.1",
        "phase": "full",
        "phase_config": phase_config,
        "artifacts": {
            "export_dir": str(root / "export"),
            "checkpoint_files": checkpoint_files,
            "upstream_agreement_path": str(root / "upstream_agreement.json"),
            "upstream_agreement_sha256": "aa" * 32,
            "timing_profile_path": str(root / "timing.json"),
            "timing_profile_sha256": "bb" * 32,
        },
        "host": {"compute_device": "cpu", "torch_num_threads": 4},
        "source_revision": {"revision": "deadbeef", "dirty": False, "status_porcelain": ""},
        "scheduler": {"chunk_arrival_policy": "discard_stale"},
        "statistics": {
            "breakpoint_ratio": 0.8,
            "baseline_success_floor": 0.5,
            "bootstrap_seed": 1,
            "n_bootstrap": 100,
        },
        "trace_window": {"length": 32},
        "baseline_reuse": {"fingerprint": {"latency_s": 0.0}},
    }
    plans = [
        {
            "plan_id": "v0.1_full_latency",
            "axis": "latency_ms",
            "phase": "exploratory",
            "candidate": "diffusion_pusht",
            "baseline_condition_id": "latency_ms_0",
            "env_seeds": seeds,
            "policy_rng_seeds": [s + 10000 for s in seeds],
            "conditions": [
                {
                    "condition_id": f"latency_ms_{ms}",
                    "label": f"{ms} ms",
                    "params": {"latency_s": ms / 1000.0, "latency_ms": ms},
                }
                for ms in (0, 100, 800)
            ],
        },
        {
            "plan_id": "v0.1_full_resolution",
            "axis": "resolution",
            "phase": "exploratory",
            "candidate": "diffusion_pusht",
            "baseline_condition_id": "resolution_native",
            "env_seeds": seeds,
            "policy_rng_seeds": [s + 10000 for s in seeds],
            "conditions": [
                {
                    "condition_id": "resolution_native",
                    "label": "native 96x96",
                    "params": {"capture_hw": [96, 96]},
                },
                {
                    "condition_id": "resolution_capture_24x24",
                    "label": "24x24",
                    "params": {"capture_hw": [24, 24]},
                },
            ],
        },
        {
            "plan_id": "v0.1_full_jpeg",
            "axis": "jpeg",
            "phase": "exploratory",
            "candidate": "diffusion_pusht",
            "baseline_condition_id": "jpeg_bypass",
            "env_seeds": seeds,
            "policy_rng_seeds": [s + 10000 for s in seeds],
            "conditions": [
                {"condition_id": "jpeg_bypass", "label": "bypass", "params": {"bypass": True}},
                {
                    "condition_id": "jpeg_q10",
                    "label": "q10",
                    "params": {"jpeg_quality": 10, "bypass": False},
                },
            ],
        },
    ]
    rows: list[dict] = []
    for plan in plans:
        for condition in plan["conditions"]:
            latency_s = float(condition.get("params", {}).get("latency_s") or 0.0)
            referenced = (
                plan["axis"] != "latency_ms"
                and condition["condition_id"] == plan["baseline_condition_id"]
            )
            # Shared baseline is latency_ms_0; other axes reference it for their baseline row set.
            if referenced:
                # resolution/jpeg baselines are real local baselines in this synthetic package.
                referenced = False
            for i, seed in enumerate(seeds):
                success = not (condition["condition_id"] == "latency_ms_800" and i == 2)
                rows.append(
                    _episode_row(
                        axis=plan["axis"],
                        plan_id=plan["plan_id"],
                        condition_id=condition["condition_id"],
                        episode_index=i,
                        seed=seed,
                        success=success,
                        latency_s=latency_s,
                        referenced=referenced,
                    )
                )

    def summary_for(plan: dict) -> dict:
        conditions = []
        for condition in plan["conditions"]:
            group = [
                r
                for r in rows
                if r["plan_id"] == plan["plan_id"]
                and r["condition_id"] == condition["condition_id"]
            ]
            successes = sum(1 for r in group if r["success"])
            eligible = len(group)
            wilson = wilson_interval(successes, eligible)
            conditions.append(
                {
                    "condition_id": condition["condition_id"],
                    "successes": successes,
                    "eligible": eligible,
                    "n_runtime_error": 0,
                    "planned": eligible,
                    "success_rate": successes / eligible,
                    "wilson_95": {"low": wilson["low"], "high": wilson["high"]},
                }
            )
        baseline_id = plan["baseline_condition_id"]
        return {
            "plan_id": plan["plan_id"],
            "ratio": 0.8,
            "baseline_success_floor": 0.5,
            "baseline_condition_id": baseline_id,
            "conditions": conditions,
            "paired_contrasts_vs_baseline": [
                {
                    "condition_id": c["condition_id"],
                    "n_pairs": 3,
                    "point": 0.0,
                    "low": -0.1,
                    "high": 0.1,
                }
                for c in plan["conditions"]
                if c["condition_id"] != baseline_id
            ],
            "axis_summary": {"breakpoint": None, "note": None, "points": []},
            "isolation_notes": [],
        }

    _write_json(data / "protocol" / "protocol_lock.json", lock)
    for plan in plans:
        _write_json(data / "protocol" / "plans" / f"{plan['plan_id']}.json", plan)
        _write_json(
            data / "summaries" / f"{plan['plan_id']}.experiment_summary.json", summary_for(plan)
        )
    (data / "episodes").mkdir(parents=True)
    (data / "episodes" / "episodes.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
    )
    agreement = {
        "passed": True,
        "candidate": "diffusion_pusht",
        "checkpoint_files": checkpoint_files,
        "determinism": {
            "passed": True,
            "max_abs_action_difference": 0.0,
            "seed": 2200,
        },
        "diffusion": {"num_inference_steps": 100},
        "per_seed": [
            {
                "seed": 2200 + i,
                "passed": True,
                "max_abs_action_difference": 0.0,
                "success": True,
            }
            for i in range(5)
        ],
    }
    _write_json(data / "validation" / "upstream_agreement.json", agreement)
    _write_json(
        data / "validation" / "export_manifest.json",
        {
            "candidate": "diffusion_pusht",
            "checkpoint_files": checkpoint_files,
            "control_period_s": 0.1,
            "hub_repo": "lerobot/diffusion_pusht",
            "hub_revision": "202c8695947a458d8ce08ce1c0219f2588d60b33",
            "inference": {"num_inference_steps": 100},
        },
    )
    samples = [0.4, 0.5, 0.55, 0.6, 0.8]
    _write_json(
        data / "timing" / "laptop-m3.json",
        {
            "device_label": "laptop-m3",
            "model": {"checkpoint_files": checkpoint_files},
            "measurements": {
                "inference_latency_s": samples,
                "n_measured": len(samples),
                "summary": {
                    "count": len(samples),
                    "median_s": 0.55,
                    "p95_s": 0.8,
                    "p99_s": 0.8,
                },
            },
            "runtime": {
                "device": "cpu",
                "torch_threads": 4,
                "torch_version": "2.6.0",
                "timed_boundary": "pytorch_generate_actions_full_denoise",
            },
        },
    )
    _write_json(
        data / "identities" / "identities.json",
        {"checkpoint_files": checkpoint_files, "lock_keys": sorted(lock)},
    )
    _write_json(data / "identities" / "condition_resolution.json", [])
    _write_json(
        data / "large_evidence.json",
        {"schema_version": 1, "note": "synthetic", "files": []},
    )
    refresh_manifest(data)
    return data


def test_synthetic_pytorch_package_report(tmp_path: Path) -> None:
    from breakpoint_eval.benchmark_report import (
        action_generation_timing_sentences,
        evaluated_paragraph,
        render_benchmark_report,
        report_title,
        update_readme_results,
    )
    from breakpoint_eval.benchmark_report_data import (
        load_report_data,
        required_axes_from_lock,
        required_profiles_from_lock,
    )

    data_dir = build_synthetic_pytorch_package(tmp_path)
    data = load_report_data(data_dir)
    assert required_axes_from_lock(data.lock) == ("latency_ms", "resolution", "jpeg")
    assert required_profiles_from_lock(data.lock) == ()
    assert [a.axis for a in data.axes] == ["latency_ms", "resolution", "jpeg"]
    assert data.profiles == []
    assert len(data.timings) == 1
    assert data.timings[0].device_label == "laptop-m3"
    check_names = [c.name for c in data.checks]
    assert any(name.startswith("Upstream agreement passed on 5 seeds") for name in check_names)
    assert any("lock candidate" in name for name in check_names)
    assert any("Timing profile checkpoint hashes" in name for name in check_names)
    assert all(c.passed for c in data.checks)

    out = tmp_path / "report"
    render_benchmark_report(data_dir, out)
    md = (out / "report.md").read_text(encoding="utf-8")
    html = (out / "index.html").read_text(encoding="utf-8")
    assert report_title(data) == "Diffusion Policy on PushT under stress"
    assert "Diffusion Policy on PushT under stress" in md
    assert "lerobot/diffusion_pusht" in evaluated_paragraph(data)
    assert "Action-generation timing" in md
    assert "Action-generation timing" in html
    sentences = action_generation_timing_sentences(data, data.timings[0])
    assert "100 denoising steps" in sentences[0]
    assert "800" in sentences[1] and "100" in sentences[1]
    assert "not replayed into episodes" in sentences[2]
    assert "No ONNX" in md
    for banned in ("ONNX Runtime", "ONNX FP32", "rpi5", "INT8"):
        assert banned not in md
        assert banned not in html

    readme = tmp_path / "README.md"
    readme.write_text(
        "<!-- results-pusht:start (generated by scripts/render_benchmark.py --update-readme) -->\n"
        "OLD\n"
        "<!-- results-pusht:end -->\n",
        encoding="utf-8",
    )
    assert update_readme_results(data_dir, readme, readme_block="results-pusht") is True
    text = readme.read_text(encoding="utf-8")
    assert "OLD" not in text
    assert "Constant added delay" in text
    assert "ONNX Runtime" not in text
