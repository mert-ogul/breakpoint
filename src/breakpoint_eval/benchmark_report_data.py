"""Load a packaged benchmark run and validate every displayed count against episode records.

Reads only the packaged data directory (see scripts/package_benchmark.py). No Torch, MuJoCo,
network, or model access.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from breakpoint_eval.artifact_names import (
    EXPORT_VALIDATION_INT8_NAME,
    EXPORT_VALIDATION_NAME,
    PROTOCOL_LOCK_NAME,
    require_artifact,
    resolve_artifact,
)
from breakpoint_eval.episode_outcome import EpisodeOutcome
from breakpoint_eval.experiment_summary import wilson_interval
from breakpoint_eval.latency_trace import contiguous_window_starts

LOGGER = logging.getLogger("breakpoint_eval.benchmark_report_data")

AXIS_TITLES = {
    "latency_ms": "Constant added delay",
    "resolution": "Camera resolution",
    "jpeg": "JPEG quality",
    "precision": "Numeric precision",
    "device_profile": "Device latency replay",
}
AXIS_UNITS = {
    "latency_ms": "ms added to every fresh action chunk",
    "resolution": "pixels, height x width (resampled back to model input)",
    "jpeg": "OpenCV JPEG quality parameter (not a bitrate)",
    "precision": "ONNX graph variant",
    "device_profile": "measured ONNX Runtime chunk latency replayed from a device profile",
}
AXIS_ORDER = ("latency_ms", "resolution", "jpeg", "precision", "device_profile")
CONTROL_PERIOD_S = 0.02
PROFILE_WINDOW_LENGTH = 32
RATE_TOLERANCE = 1e-9


class ReportDataError(RuntimeError):
    """Raised when packaged data cannot be loaded."""


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ConditionView:
    condition_id: str
    label: str
    referenced_baseline: bool
    is_baseline: bool
    rows: list[dict[str, Any]]
    successes: int
    eligible: int
    n_runtime_error: int
    wilson: dict[str, Any]
    below_threshold: bool | None = None
    threshold_inside_interval: bool | None = None
    paired: dict[str, int] | None = None
    saved_contrast: dict[str, Any] | None = None
    ordered_value: float | None = None


@dataclass
class AxisView:
    axis: str
    plan_id: str
    title: str
    unit: str
    phase: str
    baseline_condition_id: str
    ratio: float
    baseline_floor: float
    baseline_rate: float
    threshold_rate: float
    conditions: list[ConditionView]
    derived: bool
    saved_breakpoint: dict[str, Any] | None
    saved_note: str | None
    isolation_notes: list[str]
    saved_summary: dict[str, Any] | None = None


@dataclass
class ProfileView:
    device: str
    samples_s: np.ndarray
    full_stats: dict[str, float]
    window_stats: dict[str, float]
    used_stats: dict[str, float]
    window_starts: list[int]
    consumed_indices: list[int]
    telemetry: dict[str, Any]
    meta: dict[str, Any]
    profile_path: str
    model_sha256: str
    environment_notes: dict[str, Any]


@dataclass
class ReportData:
    data_dir: Path
    axes: list[AxisView]
    profiles: list[ProfileView]
    identities: dict[str, Any]
    lock: dict[str, Any]
    large_evidence: list[dict[str, Any]]
    validation_files: dict[str, dict[str, Any]]
    manifest_sha256: str
    rows: list[dict[str, Any]]
    checks: list[Check] = field(default_factory=list)

    @property
    def n_runs(self) -> int:
        return len({row["source_report"] for row in self.rows})


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min_s": float(arr.min()),
        "median_s": float(np.median(arr)),
        "p95_s": float(np.percentile(arr, 95)),
        "p99_s": float(np.percentile(arr, 99)),
        "max_s": float(arr.max()),
    }


def verify_manifest(data_dir: Path) -> tuple[Check, str]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise ReportDataError(
            f"{manifest_path} is missing; is this a packaged benchmark data directory?"
        )
    try:
        manifest = load_json(manifest_path)
        entries = [(entry["path"], entry["sha256"]) for entry in manifest["files"]]
    except (ValueError, KeyError, TypeError) as exc:
        raise ReportDataError(f"{manifest_path} is not a valid manifest: {exc!r}") from exc
    bad: list[str] = []
    for rel_path, expected_sha256 in entries:
        path = data_dir / rel_path
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            bad.append(rel_path)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    detail = f"{len(entries)} files hashed" if not bad else f"mismatch: {bad[:5]}"
    return Check("packaged files match manifest sha256", not bad, detail), manifest_sha


def load_rows(data_dir: Path) -> list[dict[str, Any]]:
    path = data_dir / "episodes" / "episodes.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def rows_by_condition(rows: list[dict[str, Any]], plan_id: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["plan_id"] == plan_id:
            grouped.setdefault(row["condition_id"], []).append(row)
    for group in grouped.values():
        group.sort(key=lambda r: r["episode_index"])
    return grouped


def count_successes(rows: list[dict[str, Any]]) -> tuple[int, int, int]:
    errors = sum(1 for r in rows if r["outcome"] == EpisodeOutcome.RUNTIME_ERROR.value)
    successes = sum(
        1 for r in rows if r["success"] and r["outcome"] == EpisodeOutcome.SUCCESS.value
    )
    return successes, len(rows) - errors, errors


def paired_counts(baseline: list[dict[str, Any]], stress: list[dict[str, Any]]) -> dict[str, int]:
    base = {r["seed"]: r["success"] for r in baseline}
    counts = {"both_success": 0, "baseline_only": 0, "stress_only": 0, "both_fail": 0}
    for row in stress:
        b, s = base[row["seed"]], row["success"]
        key = (
            "both_success"
            if b and s
            else "baseline_only"
            if b
            else "stress_only"
            if s
            else "both_fail"
        )
        counts[key] += 1
    return counts


def condition_view(
    *,
    entry: dict[str, Any],
    rows: list[dict[str, Any]],
    baseline_id: str,
    saved: dict[str, Any] | None,
    contrasts: dict[str, dict[str, Any]],
) -> ConditionView:
    successes, eligible, errors = count_successes(rows)
    return ConditionView(
        condition_id=entry["condition_id"],
        label=entry["label"],
        referenced_baseline=bool(rows[0]["referenced_baseline"]),
        is_baseline=entry["condition_id"] == baseline_id,
        rows=rows,
        successes=successes,
        eligible=eligible,
        n_runtime_error=errors,
        wilson=wilson_interval(successes, eligible),
        saved_contrast=contrasts.get(entry["condition_id"]),
        ordered_value=(saved or {}).get("ordered_value"),
    )


def build_axis(
    data_dir: Path, rows: list[dict[str, Any]], plan: dict[str, Any], baseline_override: str | None
) -> AxisView:
    plan_id = plan["plan_id"]
    summary_path = data_dir / "summaries" / f"{plan_id}.experiment_summary.json"
    summary = load_json(summary_path) if summary_path.exists() else None
    grouped = rows_by_condition(rows, plan_id)
    baseline_id = summary["baseline_condition_id"] if summary else baseline_override
    ratio = summary["ratio"] if summary else 0.8
    floor = summary["baseline_success_floor"] if summary else 0.5
    contrasts = {
        c["condition_id"]: c for c in (summary or {}).get("paired_contrasts_vs_baseline", [])
    }
    axis_summary = (summary or {}).get("axis_summary", {})
    saved_points = {
        p["condition_id"]: p for p in axis_summary.get("points", axis_summary.get("recipes", []))
    }
    views = [
        condition_view(
            entry=entry,
            rows=grouped[entry["condition_id"]],
            baseline_id=baseline_id,
            saved=saved_points.get(entry["condition_id"]),
            contrasts=contrasts,
        )
        for entry in plan["conditions"]
    ]
    baseline_rows = next(v.rows for v in views if v.is_baseline) if baseline_id in grouped else None
    if baseline_rows is None:
        baseline_rows = rows_by_condition(rows, "v0.1_full_latency")[baseline_id]
        baseline_view = condition_view(
            entry={"condition_id": baseline_id, "label": "shared baseline: constant delay 0 ms"},
            rows=baseline_rows,
            baseline_id=baseline_id,
            saved=None,
            contrasts={},
        )
        views.insert(0, baseline_view)
    baseline_rate = (
        next(v for v in views if v.is_baseline).successes
        / next(v for v in views if v.is_baseline).eligible
    )
    threshold = ratio * baseline_rate
    for view in views:
        rate = view.successes / view.eligible
        view.below_threshold = rate < threshold and not view.is_baseline
        low, high = view.wilson["low"], view.wilson["high"]
        view.threshold_inside_interval = (not view.is_baseline) and low <= threshold <= high
        view.paired = None if view.is_baseline else paired_counts(baseline_rows, view.rows)
    return AxisView(
        axis=plan["axis"],
        plan_id=plan_id,
        title=AXIS_TITLES[plan["axis"]],
        unit=AXIS_UNITS[plan["axis"]],
        phase=plan["phase"],
        baseline_condition_id=baseline_id,
        ratio=ratio,
        baseline_floor=floor,
        baseline_rate=baseline_rate,
        threshold_rate=threshold,
        conditions=views,
        derived=summary is None,
        saved_breakpoint=axis_summary.get("breakpoint") if summary else None,
        saved_note=axis_summary.get("note") if summary else None,
        isolation_notes=(summary or {}).get("isolation_notes", plan.get("notes", [])),
        saved_summary=summary,
    )


def parse_throttle(stdout: str | None) -> int | None:
    match = re.search(r"0x([0-9a-fA-F]+)", stdout or "")
    return int(match.group(1), 16) if match else None


def parse_temperature_c(stdout: str | None) -> float | None:
    match = re.search(r"temp=([0-9.]+)", stdout or "")
    return float(match.group(1)) if match else None


def summarize_telemetry(telemetry: dict[str, Any]) -> dict[str, Any]:
    samples = telemetry["during"]["samples"]
    flags = [parse_throttle(s["throttled"].get("stdout")) for s in samples]
    temps = [parse_temperature_c(s["temperature"].get("stdout")) for s in samples]
    flags_ok = [f for f in flags if f is not None]
    temps_ok = [t for t in temps if t is not None]
    return {
        "n_samples": len(samples),
        "n_with_throttle_reading": len(flags_ok),
        "under_voltage_now": sum(1 for f in flags_ok if f & 0x1),
        "arm_frequency_capped_now": sum(1 for f in flags_ok if f & 0x2),
        "throttled_now": sum(1 for f in flags_ok if f & 0x4),
        "soft_temperature_limit_now": sum(1 for f in flags_ok if f & 0x8),
        "start_flags_hex": _hex(parse_throttle(telemetry["before"]["throttled"].get("stdout"))),
        "end_flags_hex": _hex(parse_throttle(telemetry["after"]["throttled"].get("stdout"))),
        "temperature_min_c": min(temps_ok) if temps_ok else None,
        "temperature_max_c": max(temps_ok) if temps_ok else None,
    }


def _hex(value: int | None) -> str | None:
    return None if value is None else f"0x{value:x}"


def build_profile(data_dir: Path, device: str, rows: list[dict[str, Any]]) -> ProfileView:
    folder = data_dir / "profiles" / device
    profile = load_json(folder / "profile.json")
    meta = load_json(folder / "profile_run_meta.json")
    samples = np.asarray(profile["measurements"]["inference_latency_s"], dtype=np.float64)
    device_rows = [r for r in rows if r["condition_id"] == f"device_profile_{device}_fp32"]
    starts = sorted({r["trace_window_start"] for r in device_rows})
    window_values = np.concatenate([samples[s : s + PROFILE_WINDOW_LENGTH] for s in starts])
    consumed = [i for r in device_rows for i in r["consumed_trace_indices"]]
    return ProfileView(
        device=device,
        samples_s=samples,
        full_stats=sample_stats(samples),
        window_stats=sample_stats(window_values),
        used_stats=sample_stats(samples[consumed]),
        window_starts=starts,
        consumed_indices=sorted(set(consumed)),
        telemetry=summarize_telemetry(meta["telemetry"]) if device == "rpi5" else {},
        meta=meta,
        profile_path=f"profiles/{device}/profile.json",
        model_sha256=profile["model"]["sha256"],
        environment_notes=profile["environment_notes"],
    )


def validate_summaries(axes: list[AxisView]) -> list[Check]:
    checks: list[Check] = []
    for axis in axes:
        if axis.derived:
            continue
        saved = {c["condition_id"]: c for c in axis.saved_summary["conditions"]}
        mismatches: list[str] = []
        for view in axis.conditions:
            record = saved[view.condition_id]
            expected = (
                record["successes"],
                record["eligible"],
                record["n_runtime_error"],
                record["planned"],
            )
            actual = (view.successes, view.eligible, view.n_runtime_error, len(view.rows))
            if expected != actual:
                mismatches.append(f"{view.condition_id}: saved {expected} vs records {actual}")
            if abs(record["success_rate"] - view.successes / view.eligible) > RATE_TOLERANCE:
                mismatches.append(f"{view.condition_id}: success_rate")
            for key in ("low", "high"):
                if abs(record["wilson_95"][key] - view.wilson[key]) > RATE_TOLERANCE:
                    mismatches.append(f"{view.condition_id}: wilson {key}")
            if view.saved_contrast and view.saved_contrast["n_pairs"] != view.eligible:
                mismatches.append(f"{view.condition_id}: n_pairs")
        checks.append(
            Check(
                f"{axis.plan_id}: saved summary counts and Wilson intervals equal episode records",
                not mismatches,
                "; ".join(mismatches) or f"{len(axis.conditions)} conditions",
            )
        )
    return checks


def validate_rows(
    axes: list[AxisView], rows: list[dict[str, Any]], plans: list[dict[str, Any]]
) -> list[Check]:
    checks: list[Check] = []
    problems: list[str] = []
    for plan in plans:
        grouped = rows_by_condition(rows, plan["plan_id"])
        expected_seeds = plan["env_seeds"]
        expected_policy = plan["policy_rng_seeds"]
        for condition_id, group in grouped.items():
            if [r["seed"] for r in group] != expected_seeds:
                problems.append(f"{plan['plan_id']}/{condition_id}: env seeds differ from plan")
            if [r["policy_rng_seed"] for r in group] != expected_policy:
                problems.append(f"{plan['plan_id']}/{condition_id}: policy RNG seeds differ")
    for row in rows:
        if row["success"] != (row["outcome"] == EpisodeOutcome.SUCCESS.value):
            problems.append(
                f"{row['condition_id']}#{row['episode_index']}: success/outcome mismatch"
            )
    checks.append(
        Check(
            "episode rows: planned seeds, policy RNG seeds, success flag vs outcome",
            not problems,
            "; ".join(problems[:5]) or f"{len(rows)} rows",
        )
    )
    checks.append(
        Check(
            "no runtime errors or policy failures in any episode",
            all(
                r["outcome"] in {"success", "terminated_no_success", "time_limit_truncation"}
                for r in rows
            ),
            "outcomes: " + ", ".join(sorted({r["outcome"] for r in rows})),
        )
    )
    checks.append(
        Check(
            "displayed counts: every axis lists each planned condition once",
            all(len({v.condition_id for v in a.conditions}) == len(a.conditions) for a in axes),
            f"{sum(len(a.conditions) for a in axes)} condition rows",
        )
    )
    return checks


def validate_baseline_reuse(rows: list[dict[str, Any]]) -> Check:
    own = {
        r["seed"]: r["outcome"]
        for r in rows
        if r["condition_id"] == "latency_ms_0" and not r["referenced_baseline"]
    }
    shared = [r for r in rows if r["referenced_baseline"]]
    differing = sorted(
        {(r["condition_id"], r["seed"]) for r in shared if own[r["seed"]] != r["outcome"]}
    )
    return Check(
        "shared baseline outcomes equal the latency axis 0 ms condition seed by seed",
        not differing,
        f"{len(shared)} referenced rows; differing: {differing[:5]}",
    )


def validate_replay(
    axes: list[AxisView], profiles: list[ProfileView], rows: list[dict[str, Any]]
) -> list[Check]:
    checks: list[Check] = []
    problems: list[str] = []
    for profile in profiles:
        cond = f"device_profile_{profile.device}_fp32"
        group = sorted(
            (r for r in rows if r["condition_id"] == cond), key=lambda r: r["episode_index"]
        )
        starts = contiguous_window_starts(
            profile.samples_s.size, window_length=PROFILE_WINDOW_LENGTH, n_episodes=len(group)
        )
        for row, start in zip(group, starts, strict=True):
            idx = row["consumed_trace_indices"]
            delays = row["consumed_delays_s"]
            if row["trace_window_start"] != start:
                problems.append(f"{cond}#{row['episode_index']}: window start")
            if any(not start <= i < start + PROFILE_WINDOW_LENGTH for i in idx):
                problems.append(f"{cond}#{row['episode_index']}: index outside window")
            if not np.array_equal(profile.samples_s[idx], np.asarray(delays)):
                problems.append(
                    f"{cond}#{row['episode_index']}: delays differ from profile samples"
                )
            if len(idx) != row["n_inference_calls"]:
                problems.append(f"{cond}#{row['episode_index']}: consumed count != inference calls")
    checks.append(
        Check(
            "device replay: windows follow the locked design and consumed delays equal profile samples",
            not problems,
            "; ".join(problems[:5]) or f"{len(profiles)} profiles",
        )
    )
    return checks


def validate_constant_latency(rows: list[dict[str, Any]]) -> Check:
    bad = [
        f"{r['condition_id']}#{r['episode_index']}"
        for r in rows
        if r["axis"] == "latency_ms"
        and abs(r["constant_latency_s"] * 1000 - float(r["condition_id"].rsplit("_", 1)[1])) > 1e-6
    ]
    return Check(
        "latency axis: recorded constant delay equals the condition label",
        not bad,
        "; ".join(bad[:5]) or "all episodes",
    )


def validate_identities(lock: dict[str, Any], profiles: list[ProfileView]) -> Check:
    fp32 = lock["artifacts"]["fp32_sha256"]
    bad = [p.device for p in profiles if p.model_sha256 != fp32]
    return Check(
        "device profiles were measured on the locked FP32 graph",
        not bad,
        f"fp32 sha256 {fp32[:12]}",
    )


def load_report_data(data_dir: Path) -> ReportData:
    data_dir = data_dir.resolve()
    manifest_check, manifest_sha = verify_manifest(data_dir)
    if not manifest_check.passed:
        raise ReportDataError(f"{manifest_check.name}: {manifest_check.detail}")
    rows = load_rows(data_dir)
    plans = sorted(
        (load_json(p) for p in (data_dir / "protocol" / "plans").glob("*.json")),
        key=lambda p: AXIS_ORDER.index(p["axis"]),
    )
    axes = [build_axis(data_dir, rows, plan, "latency_ms_0") for plan in plans]
    lock = load_json(require_artifact(data_dir / "protocol", PROTOCOL_LOCK_NAME))
    profiles = [build_profile(data_dir, d, rows) for d in ("laptop-m3", "rpi5")]
    checks = [
        manifest_check,
        *validate_rows(axes, rows, plans),
        *validate_summaries(axes),
        validate_baseline_reuse(rows),
        validate_constant_latency(rows),
        *validate_replay(axes, profiles, rows),
        validate_identities(lock, profiles),
    ]
    validation_files = {}
    for name in (EXPORT_VALIDATION_NAME, EXPORT_VALIDATION_INT8_NAME):
        path = resolve_artifact(data_dir / "validation", name)
        if path is not None:
            validation_files[name] = load_json(path)
    return ReportData(
        data_dir=data_dir,
        axes=axes,
        profiles=profiles,
        identities=load_json(data_dir / "identities" / "identities.json"),
        lock=lock,
        large_evidence=load_json(data_dir / "large_evidence.json")["files"],
        validation_files=validation_files,
        manifest_sha256=manifest_sha,
        rows=rows,
        checks=checks,
    )
