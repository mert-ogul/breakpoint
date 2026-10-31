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
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from breakpoint_eval.artifact_names import (
    EXPORT_VALIDATION_INT4_NAME,
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
    "h264": "H.264 bitrate",
    "precision": "Numeric precision",
    "device_profile": "Device latency replay",
}
AXIS_UNITS = {
    "latency_ms": "ms added to every fresh action chunk",
    "resolution": "pixels, height x width (resampled back to model input)",
    "jpeg": "OpenCV JPEG quality parameter (not a bitrate)",
    "h264": "requested libx264 bitrate in kbps; causal stream at 50 fps, visual quality only",
    "precision": "ONNX graph variant",
    "device_profile": "measured ONNX Runtime chunk latency replayed from a device profile",
}
# Required axes always present; h264 is optional and inserts after jpeg when present.
REQUIRED_AXES = ("latency_ms", "resolution", "jpeg", "precision", "device_profile")
AXIS_ORDER = ("latency_ms", "resolution", "jpeg", "h264", "precision", "device_profile")
RATE_TOLERANCE = 1e-9
# Axes whose conditions are ordered from mild to severe; the others compare categories.
ORDERED_AXES = frozenset({"latency_ms", "resolution", "jpeg", "h264"})
REQUIRED_PROFILES = ("laptop-m3", "rpi5")
MANIFEST_SCHEMA_VERSION = 1
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
ROW_KEYS = (
    "plan_id",
    "condition_id",
    "axis",
    "episode_index",
    "seed",
    "policy_rng_seed",
    "outcome",
    "success",
    "referenced_baseline",
    "source_report",
    "constant_latency_s",
    "trace_window_start",
    "consumed_trace_indices",
    "consumed_delays_s",
)
PLAN_KEYS = ("plan_id", "axis", "phase", "conditions", "env_seeds", "policy_rng_seeds")
# Lock fields the renderer reads, as dotted paths.
LOCK_PATHS = (
    "candidate",
    "artifacts.fp32_sha256",
    "artifacts.int8_sha256",
    "artifacts.int8_status",
    "host.ort_providers",
    "host.intra_op_num_threads",
    "host.inter_op_num_threads",
    "host.compute_device",
    "source_revision.revision",
    "source_revision.dirty",
    "scheduler.chunk_arrival_policy",
    "phase_config.policy_seed_offset",
    "statistics.bootstrap_seed",
    "statistics.n_bootstrap",
    "statistics.breakpoint_ratio",
    "statistics.baseline_success_floor",
    "trace_window.length",
    "baseline_reuse.fingerprint.latency_s",
)


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
    ordered: bool = False
    suppressed_reason: str | None = None


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


@dataclass(frozen=True)
class SharedBaseline:
    """The condition that other axes reuse by reference, found from the lock fingerprint."""

    plan_id: str
    condition_id: str
    label: str


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
    trace_window_length: int = 0
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


def _manifest_entries(manifest_path: Path, data_dir: Path) -> list[dict[str, Any]]:
    """Parse and structurally validate the manifest; raise ReportDataError if unusable."""

    def invalid(reason: str) -> ReportDataError:
        return ReportDataError(f"{manifest_path} is not a valid manifest: {reason}")

    try:
        manifest = load_json(manifest_path)
    except ValueError as exc:
        raise invalid(repr(exc)) from exc
    if not isinstance(manifest, dict):
        raise invalid("not a JSON object")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise invalid(
            f"schema_version {manifest.get('schema_version')!r}, supported is "
            f"{MANIFEST_SCHEMA_VERSION}"
        )
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise invalid("files must be a nonempty list")
    seen: set[str] = set()
    root = data_dir.resolve()
    for entry in entries:
        if not isinstance(entry, dict):
            raise invalid(f"entry {entry!r} is not an object")
        rel = entry.get("path")
        if not isinstance(rel, str) or not rel or "\\" in rel:
            raise invalid(f"unusable path {rel!r}")
        pure = PurePosixPath(rel)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or pure.as_posix() != rel
            or rel == "manifest.json"
        ):
            raise invalid(f"path {rel!r} is not a plain relative path inside the package")
        if not (root / rel).resolve().is_relative_to(root):
            raise invalid(f"path {rel!r} escapes the package")
        if rel in seen:
            raise invalid(f"duplicate entry for {rel!r}")
        seen.add(rel)
        digest = entry.get("sha256")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise invalid(f"{rel}: sha256 is not 64 lowercase hex digits")
        size = entry.get("bytes")
        if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 0):
            raise invalid(f"{rel}: bytes is not a nonnegative integer")
    return entries


def required_inputs(data_dir: Path) -> list[str]:
    """Package files the renderer reads; each must be listed in the manifest."""
    required = ["episodes/episodes.jsonl", "identities/identities.json", "large_evidence.json"]
    try:
        lock = resolve_artifact(data_dir / "protocol", PROTOCOL_LOCK_NAME)
        validation = [
            resolve_artifact(data_dir / "validation", name)
            for name in (
                EXPORT_VALIDATION_NAME,
                EXPORT_VALIDATION_INT8_NAME,
                EXPORT_VALIDATION_INT4_NAME,
            )
        ]
    except ValueError as exc:
        raise ReportDataError(str(exc)) from exc
    if lock is None:
        raise ReportDataError(f"{data_dir / 'protocol'} has no {PROTOCOL_LOCK_NAME}")
    required.append(lock.relative_to(data_dir).as_posix())
    required += [p.relative_to(data_dir).as_posix() for p in validation if p is not None]
    plans = sorted((data_dir / "protocol" / "plans").glob("*.json"))
    if not plans:
        raise ReportDataError(f"{data_dir / 'protocol' / 'plans'} has no plan files")
    required += [p.relative_to(data_dir).as_posix() for p in plans]
    required += [
        p.relative_to(data_dir).as_posix()
        for p in sorted((data_dir / "summaries").glob("*.experiment_summary.json"))
    ]
    for device in REQUIRED_PROFILES:
        required += [f"profiles/{device}/profile.json", f"profiles/{device}/profile_run_meta.json"]
    required += [
        p.relative_to(data_dir).as_posix()
        for p in sorted((data_dir / "protocol").glob("*_unsupported.json"))
    ]
    return required


def verify_manifest(data_dir: Path) -> tuple[Check, str]:
    """Check the manifest and every listed file; package integrity, not proof of the results."""
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise ReportDataError(
            f"{manifest_path} is missing; is this a packaged benchmark data directory?"
        )
    entries = _manifest_entries(manifest_path, data_dir)
    listed = {entry["path"] for entry in entries}
    unlisted = [rel for rel in required_inputs(data_dir) if rel not in listed]
    if unlisted:
        raise ReportDataError(f"manifest does not list files the report reads: {unlisted[:5]}")
    bad: list[str] = []
    for entry in entries:
        path = data_dir / entry["path"]
        if not path.is_file():
            bad.append(entry["path"])
            continue
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"] or (
            entry.get("bytes") is not None and len(content) != entry["bytes"]
        ):
            bad.append(entry["path"])
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    detail = f"{len(entries)} files hashed" if not bad else f"mismatch: {bad[:5]}"
    return Check("Packaged files match the manifest digests", not bad, detail), manifest_sha


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


def _require_path(payload: dict[str, Any], dotted: str, where: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ReportDataError(f"{where} is missing {dotted}")
        value = value[part]
    return value


def find_shared_baseline(plans: list[dict[str, Any]], lock: dict[str, Any]) -> SharedBaseline:
    """The latency-axis condition whose delay equals the locked baseline fingerprint."""
    latency_s = lock["baseline_reuse"]["fingerprint"]["latency_s"]
    found = [
        SharedBaseline(plan["plan_id"], c["condition_id"], c["label"])
        for plan in plans
        if plan["axis"] == "latency_ms"
        for c in plan["conditions"]
        if c.get("params", {}).get("latency_s") == latency_s
    ]
    if len(found) != 1:
        raise ReportDataError(
            f"expected exactly one latency condition with the locked baseline delay "
            f"{latency_s}; found {[f.condition_id for f in found]}"
        )
    return found[0]


def validate_structure(
    plans: list[dict[str, Any]], rows: list[dict[str, Any]], lock: dict[str, Any]
) -> None:
    """Reject packages that lack the plans, conditions, seeds, rows, or profiles the report uses."""
    for dotted in LOCK_PATHS:
        _require_path(lock, dotted, "protocol lock")
    window_length = lock["trace_window"]["length"]
    if not isinstance(window_length, int) or window_length < 1:
        raise ReportDataError(f"protocol lock trace_window.length {window_length!r} is not usable")
    if not rows:
        raise ReportDataError("episodes.jsonl has no episode rows")
    for row in rows:
        if not isinstance(row, dict) or any(key not in row for key in ROW_KEYS):
            raise ReportDataError(f"episode row lacks required fields: {str(row)[:120]}")
    axes: list[str] = []
    for plan in plans:
        missing = [key for key in PLAN_KEYS if key not in plan]
        if missing:
            raise ReportDataError(f"plan {plan.get('plan_id')!r} lacks {missing}")
        if plan["axis"] not in AXIS_TITLES:
            raise ReportDataError(f"plan {plan['plan_id']} has unsupported axis {plan['axis']!r}")
        axes.append(plan["axis"])
        if not plan["conditions"] or not plan["env_seeds"]:
            raise ReportDataError(f"plan {plan['plan_id']} has no conditions or no seeds")
        if len(plan["env_seeds"]) != len(plan["policy_rng_seeds"]):
            raise ReportDataError(f"plan {plan['plan_id']} env and policy seed counts differ")
        for condition in plan["conditions"]:
            if "condition_id" not in condition or "label" not in condition:
                raise ReportDataError(f"plan {plan['plan_id']} has a condition without id/label")
            recorded = condition.get("params", {}).get("window_length")
            if plan["axis"] == "device_profile" and recorded != window_length:
                raise ReportDataError(
                    f"{condition['condition_id']} window_length {recorded!r} differs from "
                    f"the locked trace window {window_length}"
                )
    if len(set(axes)) != len(axes):
        raise ReportDataError(f"duplicate axis in plans: {axes}")
    missing_required = [a for a in REQUIRED_AXES if a not in axes]
    if missing_required:
        raise ReportDataError(
            f"plans must cover each of {list(REQUIRED_AXES)} once; missing {missing_required}"
        )
    if len({plan["plan_id"] for plan in plans}) != len(plans):
        raise ReportDataError("duplicate plan_id in protocol/plans")
    planned = {plan["plan_id"]: {c["condition_id"] for c in plan["conditions"]} for plan in plans}
    for row in rows:
        if row["condition_id"] not in planned.get(row["plan_id"], ()):
            raise ReportDataError(
                f"episode row for unplanned condition {row['plan_id']}/{row['condition_id']}"
            )
    for plan in plans:
        grouped = rows_by_condition(rows, plan["plan_id"])
        for condition in plan["conditions"]:
            group = grouped.get(condition["condition_id"])
            if not group:
                raise ReportDataError(
                    f"no episode rows for planned condition "
                    f"{plan['plan_id']}/{condition['condition_id']}"
                )
            if count_successes(group)[1] == 0:
                raise ReportDataError(
                    f"{plan['plan_id']}/{condition['condition_id']} has no eligible episodes"
                )
    find_shared_baseline(plans, lock)
    for device in REQUIRED_PROFILES:
        if not any(r["condition_id"] == f"device_profile_{device}_fp32" for r in rows):
            raise ReportDataError(f"no device replay rows for required profile {device}")


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


def suppression_reason(
    views: list[ConditionView], baseline_rate: float, floor: float, summary: dict[str, Any] | None
) -> str | None:
    """Why threshold comparisons must not be reported for this axis, or None."""
    if baseline_rate < floor:
        return f"baseline success rate {baseline_rate:.1%} is below the {floor:g} floor"
    if any(v.n_runtime_error or v.eligible != len(v.rows) for v in views):
        return "runtime errors leave some conditions with incomplete evidence"
    saved = (summary or {}).get("axis_summary", {})
    if saved.get("headline_suppressed"):
        return str(saved.get("reason") or "saved summary suppressed the headline")
    return None


def build_axis(
    data_dir: Path,
    rows: list[dict[str, Any]],
    plan: dict[str, Any],
    lock: dict[str, Any],
    shared: SharedBaseline,
) -> AxisView:
    plan_id = plan["plan_id"]
    summary_path = data_dir / "summaries" / f"{plan_id}.experiment_summary.json"
    summary = load_json(summary_path) if summary_path.exists() else None
    grouped = rows_by_condition(rows, plan_id)
    baseline_id = plan.get("baseline_condition_id") or shared.condition_id
    ratio = float(lock["statistics"]["breakpoint_ratio"])
    floor = float(lock["statistics"]["baseline_success_floor"])
    if summary and (
        summary["ratio"] != ratio
        or summary["baseline_success_floor"] != floor
        or summary["baseline_condition_id"] != baseline_id
    ):
        raise ReportDataError(f"{plan_id}: saved summary settings differ from the protocol lock")
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
    if baseline_id in grouped:
        baseline_rows = grouped[baseline_id]
    else:
        baseline_rows = rows_by_condition(rows, shared.plan_id)[baseline_id]
        views.insert(
            0,
            condition_view(
                entry={
                    "condition_id": baseline_id,
                    "label": f"shared baseline: {shared.label}",
                },
                rows=baseline_rows,
                baseline_id=baseline_id,
                saved=None,
                contrasts={},
            ),
        )
    baseline = next(v for v in views if v.is_baseline)
    baseline_rate = baseline.successes / baseline.eligible
    threshold = ratio * baseline_rate
    reason = suppression_reason(views, baseline_rate, floor, summary)
    for view in views:
        rate = view.successes / view.eligible
        low, high = view.wilson["low"], view.wilson["high"]
        view.paired = None if view.is_baseline else paired_counts(baseline_rows, view.rows)
        if reason is not None:
            view.below_threshold = False
            view.threshold_inside_interval = False
            continue
        view.below_threshold = rate < threshold and not view.is_baseline
        view.threshold_inside_interval = (not view.is_baseline) and low <= threshold <= high
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
        ordered=plan["axis"] in ORDERED_AXES,
        suppressed_reason=reason,
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


def build_profile(
    data_dir: Path, device: str, rows: list[dict[str, Any]], window_length: int
) -> ProfileView:
    folder = data_dir / "profiles" / device
    profile = load_json(folder / "profile.json")
    meta = load_json(folder / "profile_run_meta.json")
    samples = np.asarray(profile["measurements"]["inference_latency_s"], dtype=np.float64)
    device_rows = [r for r in rows if r["condition_id"] == f"device_profile_{device}_fp32"]
    if any(
        r["trace_window_start"] is None or r["consumed_trace_indices"] is None for r in device_rows
    ):
        raise ReportDataError(f"{device}: replay rows lack trace window or consumed indices")
    starts = sorted({r["trace_window_start"] for r in device_rows})
    window_values = np.concatenate([samples[s : s + window_length] for s in starts])
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
            record = saved.get(view.condition_id)
            if record is None:
                mismatches.append(f"{view.condition_id}: missing from saved summary")
                continue
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
                f"{axis.plan_id} saved summary matches episode records",
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
            "Episode rows match planned seeds and success flags",
            not problems,
            "; ".join(problems[:5]) or f"{len(rows)} rows",
        )
    )
    checks.append(
        Check(
            "No runtime errors or policy failures in any episode",
            all(
                r["outcome"] in {"success", "terminated_no_success", "time_limit_truncation"}
                for r in rows
            ),
            "outcomes: " + ", ".join(sorted({r["outcome"] for r in rows})),
        )
    )
    checks.append(
        Check(
            "Every axis lists each planned condition once",
            all(len({v.condition_id for v in a.conditions}) == len(a.conditions) for a in axes),
            f"{sum(len(a.conditions) for a in axes)} condition rows",
        )
    )
    return checks


def validate_baseline_reuse(rows: list[dict[str, Any]], shared: SharedBaseline) -> Check:
    own = {
        r["seed"]: r["outcome"]
        for r in rows
        if r["condition_id"] == shared.condition_id and not r["referenced_baseline"]
    }
    referenced = [r for r in rows if r["referenced_baseline"]]
    differing = sorted(
        {(r["condition_id"], r["seed"]) for r in referenced if own.get(r["seed"]) != r["outcome"]}
    )
    return Check(
        f"Shared baseline outcomes match {shared.condition_id} seed by seed",
        not differing,
        f"{len(referenced)} referenced rows; differing: {differing[:5]}",
    )


def validate_replay(
    profiles: list[ProfileView], rows: list[dict[str, Any]], window_length: int
) -> list[Check]:
    checks: list[Check] = []
    problems: list[str] = []
    for profile in profiles:
        cond = f"device_profile_{profile.device}_fp32"
        group = sorted(
            (r for r in rows if r["condition_id"] == cond), key=lambda r: r["episode_index"]
        )
        starts = contiguous_window_starts(
            profile.samples_s.size, window_length=window_length, n_episodes=len(group)
        )
        for row, start in zip(group, starts, strict=True):
            idx = row["consumed_trace_indices"]
            delays = row["consumed_delays_s"]
            if row["trace_window_start"] != start:
                problems.append(f"{cond}#{row['episode_index']}: window start")
            if any(not start <= i < start + window_length for i in idx):
                problems.append(f"{cond}#{row['episode_index']}: index outside window")
            if not np.array_equal(profile.samples_s[idx], np.asarray(delays)):
                problems.append(
                    f"{cond}#{row['episode_index']}: delays differ from profile samples"
                )
            if len(idx) != row["n_inference_calls"]:
                problems.append(f"{cond}#{row['episode_index']}: consumed count != inference calls")
    checks.append(
        Check(
            "Device replay windows and consumed delays match the profile samples",
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
        "Latency axis recorded delay matches the condition label",
        not bad,
        "; ".join(bad[:5]) or "all episodes",
    )


def validate_identities(lock: dict[str, Any], profiles: list[ProfileView]) -> Check:
    fp32 = lock["artifacts"]["fp32_sha256"]
    bad = [p.device for p in profiles if p.model_sha256 != fp32]
    return Check(
        "Device profiles were measured on the locked FP32 graph",
        not bad,
        f"fp32 sha256 {fp32[:12]}",
    )


def validate_int4_episodes(rows: list[dict[str, Any]]) -> Check | None:
    """Every INT4 episode completed with no policy_failed or runtime_error.

    OrtActChunkSession raises on a non-finite chunk, which the episode records as
    policy_failed, so this covers finite closed-loop outputs.
    """
    int4_rows = [r for r in rows if r["condition_id"] == "precision_int4_weight_only"]
    if not int4_rows:
        return None
    complete = frozenset({"success", "terminated_no_success", "time_limit_truncation"})
    bad = [
        f"#{r['episode_index']} outcome={r.get('outcome')!r}"
        for r in int4_rows
        if r.get("outcome") not in complete or r.get("runtime_error")
    ]
    n_calls = sum(int(r.get("n_inference_calls") or 0) for r in int4_rows)
    return Check(
        "INT4 episodes all completed",
        not bad,
        f"{n_calls} INT4 inference calls" if not bad else f"bad: {bad[:5]}; {n_calls} calls",
    )


def measured_h264_kbps(encoded_bytes: int, frames: int, *, frame_rate_hz: float) -> float:
    """Recompute measured ABR kbps from total encoded bytes and frame count."""
    if frames < 1 or frame_rate_hz <= 0:
        raise ValueError("h264 measured kbps requires frames >= 1 and frame_rate_hz > 0")
    return float(encoded_bytes) * 8.0 * float(frame_rate_hz) / float(frames) / 1000.0


def validate_h264_rows(rows: list[dict[str, Any]]) -> Check | None:
    """Check packaged H.264 rows: frames match control steps, zero holds, first keyframe."""
    h264_rows = [
        r for r in rows if r.get("axis") == "h264" and r.get("h264_encoded_bytes") is not None
    ]
    if not h264_rows:
        return None
    problems: list[str] = []
    for row in h264_rows:
        label = f"{row['condition_id']}#{row['episode_index']}"
        frames = row.get("h264_frames")
        if frames != row.get("n_control_steps"):
            problems.append(f"{label}: h264_frames != n_control_steps")
            continue
        if int(row.get("h264_holds") or 0) != 0:
            problems.append(f"{label}: h264_holds is not 0")
        if row.get("h264_first_frame_keyframe") is not True:
            problems.append(f"{label}: first_frame_keyframe is not true")
    return Check(
        "H.264 rows match frame counts, holds, and first keyframe",
        not problems,
        "; ".join(problems[:5]) or f"{len(h264_rows)} H.264 episodes",
    )


def load_report_data(data_dir: Path) -> ReportData:
    data_dir = data_dir.resolve()
    manifest_check, manifest_sha = verify_manifest(data_dir)
    if not manifest_check.passed:
        raise ReportDataError(f"{manifest_check.name}: {manifest_check.detail}")
    try:
        return _load_verified(data_dir, manifest_check, manifest_sha)
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError, StopIteration) as exc:
        raise ReportDataError(f"malformed packaged data: {type(exc).__name__}: {exc}") from exc


def _load_verified(data_dir: Path, manifest_check: Check, manifest_sha: str) -> ReportData:
    rows = load_rows(data_dir)
    plans = sorted(
        (load_json(p) for p in (data_dir / "protocol" / "plans").glob("*.json")),
        key=lambda p: (
            AXIS_ORDER.index(p["axis"]) if p.get("axis") in AXIS_ORDER else len(AXIS_ORDER)
        ),
    )
    lock = load_json(require_artifact(data_dir / "protocol", PROTOCOL_LOCK_NAME))
    validate_structure(plans, rows, lock)
    window_length = lock["trace_window"]["length"]
    shared = find_shared_baseline(plans, lock)
    axes = [build_axis(data_dir, rows, plan, lock, shared) for plan in plans]
    profiles = [build_profile(data_dir, d, rows, window_length) for d in REQUIRED_PROFILES]
    checks = [
        manifest_check,
        *validate_rows(axes, rows, plans),
        *validate_summaries(axes),
        validate_baseline_reuse(rows, shared),
        validate_constant_latency(rows),
        *validate_replay(profiles, rows, window_length),
        validate_identities(lock, profiles),
    ]
    int4_check = validate_int4_episodes(rows)
    if int4_check is not None:
        checks.append(int4_check)
    h264_check = validate_h264_rows(rows)
    if h264_check is not None:
        checks.append(h264_check)
    validation_files = {}
    for name in (
        EXPORT_VALIDATION_NAME,
        EXPORT_VALIDATION_INT8_NAME,
        EXPORT_VALIDATION_INT4_NAME,
    ):
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
        trace_window_length=window_length,
        checks=checks,
    )
