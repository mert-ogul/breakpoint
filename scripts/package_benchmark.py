#!/usr/bin/env python3
"""Package a completed benchmark run into benchmarks/act-transfer/deployment-stress/data (read-only on inputs)."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from breakpoint_eval.artifact_names import (  # noqa: E402
    BENCHMARK_CONFIG_NAME,
    EXPORT_VALIDATION_INT4_NAME,
    EXPORT_VALIDATION_INT8_NAME,
    EXPORT_VALIDATION_NAME,
    PROTOCOL_LOCK_NAME,
    require_artifact,
    resolve_artifact,
)
from breakpoint_eval.benchmark_protocol import (  # noqa: E402
    condition_incomplete_reason,
    resolve_profile_by_identity,
)
from breakpoint_eval.benchmark_report_data import measured_h264_kbps  # noqa: E402
from breakpoint_eval.experiment_plan import ExperimentPlan, load_experiment_plan  # noqa: E402
from breakpoint_eval.file_digest import sha256_file  # noqa: E402
from breakpoint_eval.latency_trace import load_latency_profile  # noqa: E402

LOGGER = logging.getLogger("package_benchmark")

DEFAULT_FULL = REPO_ROOT / "outputs" / "act-transfer" / "full"
DEFAULT_OUT = REPO_ROOT / "benchmarks" / "act-transfer" / "deployment-stress" / "data"
REFERENCE_NAME = "baseline_reference.json"
# "resumed_complete" was written by runs resumed before records kept their original status.
COMPLETED_STATUSES = frozenset({"complete", "resumed_complete"})
HISTORICAL_NOTE = (
    "Files under historical/ and the verbatim summaries/protocol files keep the absolute "
    "paths of the machine that produced them. Portable links use relative paths only."
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_verbatim(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return resolved.relative_to(REPO_ROOT).as_posix()
    return resolved.as_posix()


def _reference_dir_name(condition_dir: Path) -> str:
    reference = load_json(condition_dir / REFERENCE_NAME)
    return Path(reference["reference_dir"]).name


def condition_report_path(full_root: Path, plan_id: str, record: dict[str, Any]) -> Path:
    """Report file a record stands for: its own report, or the shared baseline it references."""
    condition_id = record["condition_id"]
    status = record["status"]
    condition_dir = full_root / "axes" / plan_id / "conditions" / condition_id
    if status not in COMPLETED_STATUSES | {"referenced_baseline"}:
        raise ValueError(
            f"{plan_id}/{condition_id} has status {status!r}"
            + (f": {record['error']}" if record.get("error") else "")
        )
    if (condition_dir / REFERENCE_NAME).is_file():
        shared = full_root / "shared_baselines" / _reference_dir_name(condition_dir)
        return shared / "run_report.json"
    if status == "referenced_baseline":
        raise ValueError(
            f"{plan_id}/{condition_id} is a baseline reference without {REFERENCE_NAME}"
        )
    return condition_dir / "run_report.json"


def resolve_condition(
    full_root: Path, plan: ExperimentPlan, record: dict[str, Any]
) -> tuple[Path, bool]:
    """Validate a finished record against its plan; return (report path, referenced)."""
    condition = next(c for c in plan.conditions if c.condition_id == record["condition_id"])
    report_path = condition_report_path(full_root, plan.plan_id, record)
    condition_dir = full_root / "axes" / plan.plan_id / "conditions" / condition.condition_id
    referenced = (condition_dir / REFERENCE_NAME).is_file()
    shared = report_path.parent if referenced else full_root / "shared_baselines"
    reason = condition_incomplete_reason(
        condition_dir, plan=plan, condition=condition, shared_baseline_dir=shared
    )
    if reason is not None:
        raise ValueError(f"{plan.plan_id}/{condition.condition_id} is incomplete: {reason}")
    return report_path, referenced


def first_success_step(steps: list[dict[str, Any]]) -> int | None:
    for step in steps:
        if step.get("is_success"):
            return int(step["control_step_index"])
    return None


def _h264_compact_fields(episode: dict[str, Any]) -> dict[str, Any]:
    """Sum H.264 camera stats into compact row fields; None when the episode is not H.264."""
    empty = {
        "h264_requested_kbps": None,
        "h264_encoded_bytes": None,
        "h264_frames": None,
        "h264_actual_kbps": None,
        "h264_holds": None,
        "h264_keyframes": None,
        "h264_first_frame_keyframe": None,
    }
    summary = episode.get("h264_summary")
    if not isinstance(summary, dict):
        return empty
    cameras = summary.get("cameras")
    if not isinstance(cameras, dict) or not cameras:
        return empty
    cams = [c for c in cameras.values() if isinstance(c, dict)]
    if not cams:
        return empty
    encoded = sum(int(c.get("encoded_bytes_total") or 0) for c in cams)
    frames = sum(int(c.get("n_frames_in") or 0) for c in cams)
    holds = sum(
        int(c.get("n_startup_holds") or 0) + int(c.get("n_last_frame_holds") or 0) for c in cams
    )
    keyframes = sum(int(c.get("n_keyframes") or 0) for c in cams)
    requested_block = summary.get("requested") or {}
    requested = requested_block.get("bitrate_kbps")
    if requested is None:
        requested = cams[0].get("requested_bitrate_kbps")
    frame_rate_hz = requested_block.get("frame_rate_hz")
    if frames > 0 and frame_rate_hz is None:
        raise ValueError("h264_summary.requested.frame_rate_hz is required to measure ABR kbps")
    actual = (
        measured_h264_kbps(encoded, frames, frame_rate_hz=float(frame_rate_hz))
        if frames > 0
        else None
    )
    return {
        "h264_requested_kbps": float(requested) if requested is not None else None,
        "h264_encoded_bytes": encoded,
        "h264_frames": frames,
        "h264_actual_kbps": actual,
        "h264_holds": holds,
        "h264_keyframes": keyframes,
        "h264_first_frame_keyframe": all(c.get("first_frame_keyframe") is True for c in cams),
    }


def compact_episode_row(
    *,
    axis: str,
    plan_id: str,
    condition_id: str,
    referenced: bool,
    source: str,
    episode: dict[str, Any],
) -> dict[str, Any]:
    steps = episode["steps"]
    trace = episode.get("latency_trace_provenance") or {}
    image = episode.get("image_degrade_summary") or {}
    control_period_s = float(episode["control_period_s"])
    return {
        "axis": axis,
        "plan_id": plan_id,
        "condition_id": condition_id,
        "referenced_baseline": referenced,
        "episode_index": episode["episode_index"],
        "seed": episode["seed"],
        "policy_rng_seed": episode["policy_rng_seed"],
        "outcome": episode["outcome"],
        "success": episode["success"],
        "runtime_error": episode["runtime_error"],
        "n_control_steps": episode["n_control_steps"],
        "sim_time_s": round(episode["n_control_steps"] * control_period_s, 6),
        "success_step_index": first_success_step(steps),
        "final_step_sim_time_s": steps[-1]["sim_time_s"] if steps else None,
        "n_inference_calls": episode["n_inference_calls"],
        "n_fresh_chunks": episode["n_fresh_chunks"],
        "n_underflows": episode["n_underflows"],
        "n_missed_deadlines": episode["n_missed_deadlines"],
        "sum_reward": episode["sum_reward"],
        "max_reward": episode["max_reward"],
        "latency_mode": episode["latency_mode"],
        "constant_latency_s": episode["latency_s"],
        "image_mode": episode["image_mode"],
        "capture_hw": image.get("capture_hw"),
        "jpeg_quality": image.get("jpeg_quality"),
        "trace_window_start": trace.get("window_start"),
        "trace_window_length": trace.get("window_length"),
        "consumed_trace_indices": trace.get("consumed_trace_indices"),
        "consumed_delays_s": trace.get("consumed_delays_s"),
        "source_report": source,
        **_h264_compact_fields(episode),
    }


def check_episodes_jsonl_duplicates_report(report: dict[str, Any], jsonl_path: Path) -> None:
    lines = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    if lines != report["episodes"]:
        raise ValueError(f"{jsonl_path} does not equal run_report episodes")


def _unsupported_int8_problem(plan_id: str, record: dict[str, Any], lock: dict[str, Any]) -> None:
    """Accept an unsupported record only for INT8 when the lock recorded INT8 as not runnable."""
    condition_id = record["condition_id"]
    if condition_id != "precision_int8":
        raise ValueError(f"{plan_id}/{condition_id} is unsupported but is not the INT8 condition")
    if lock["artifacts"].get("int8_status") == "ok":
        raise ValueError(
            f"{plan_id}/{condition_id} is unsupported but the lock records INT8 as runnable"
        )
    if record.get("measurements_fabricated") is not False:
        raise ValueError(f"{plan_id}/{condition_id} does not state measurements_fabricated=false")


def collect_episode_rows(
    full_root: Path,
    large: list[dict[str, Any]],
    lock: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (episode rows, per-condition resolution records) for every planned condition."""
    rows: list[dict[str, Any]] = []
    resolutions: list[dict[str, Any]] = []
    hashed: set[Path] = set()
    for plan_entry in load_json(full_root / "run_records.json"):
        plan_id = plan_entry["plan_id"]
        plan = load_experiment_plan(full_root / "plans" / f"{plan_id}.json")
        records = {r["condition_id"]: r for r in plan_entry["records"]}
        planned = [c.condition_id for c in plan.conditions]
        if sorted(records) != sorted(planned):
            raise ValueError(
                f"{plan_id}: run records {sorted(records)} do not match planned conditions "
                f"{sorted(planned)}"
            )
        for condition_id in planned:
            record = records[condition_id]
            if record["status"] == "unsupported":
                _unsupported_int8_problem(plan_id, record, lock)
                resolutions.append(
                    {
                        "plan_id": plan_id,
                        "condition_id": condition_id,
                        "record_status": "unsupported",
                        "resolution": "unsupported",
                        "int8_status": lock["artifacts"].get("int8_status"),
                        "measurements_fabricated": False,
                    }
                )
                continue
            report_path, referenced = resolve_condition(full_root, plan, record)
            report = load_json(report_path)
            source = repo_relative(report_path)
            for episode in report["episodes"]:
                rows.append(
                    compact_episode_row(
                        axis=plan.axis,
                        plan_id=plan_id,
                        condition_id=condition_id,
                        referenced=referenced,
                        source=source,
                        episode=episode,
                    )
                )
            resolutions.append(
                {
                    "plan_id": plan_id,
                    "condition_id": condition_id,
                    "record_status": record["status"],
                    "resolution": "referenced_baseline" if referenced else "complete",
                    "source_report": source,
                }
            )
            if report_path in hashed:
                continue
            hashed.add(report_path)
            large.append(large_entry(report_path, "canonical_control_step_log"))
            jsonl = report_path.parent / "episodes.jsonl"
            if jsonl.exists():
                check_episodes_jsonl_duplicates_report(report, jsonl)
                large.append(large_entry(jsonl, "duplicate_of_run_report_episodes"))
            LOGGER.info("episodes %s/%s: %d", plan_id, condition_id, len(report["episodes"]))
    return rows, resolutions


def large_entry(path: Path, role: str) -> dict[str, Any]:
    return {
        "path": repo_relative(path),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "in_repository": False,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def package_protocol(full_root: Path, out: Path) -> None:
    for name in (BENCHMARK_CONFIG_NAME, PROTOCOL_LOCK_NAME):
        source = require_artifact(full_root, name)
        copy_verbatim(source, out / "protocol" / source.name)
    copy_verbatim(full_root / "run_records.json", out / "protocol" / "run_records.json")
    for unsupported in sorted(full_root.glob("*_unsupported.json")):
        copy_verbatim(unsupported, out / "protocol" / unsupported.name)
    for plan in sorted((full_root / "plans").glob("*.json")):
        copy_verbatim(plan, out / "protocol" / "plans" / plan.name)


def package_summaries(full_root: Path, out: Path) -> None:
    for summary in sorted(full_root.glob("axes/*/experiment_summary.json")):
        copy_verbatim(summary, out / "summaries" / f"{summary.parent.name}.experiment_summary.json")


def package_historical(full_root: Path, out: Path) -> None:
    for cond in sorted(full_root.glob("axes/*/conditions/*")):
        for name in ("provenance.json", "planned_settings.json"):
            src = cond / name
            if src.exists():
                copy_verbatim(src, out / "historical" / cond.parts[-3] / cond.name / name)
    for fingerprint in sorted(full_root.glob("shared_baselines/*/baseline_fingerprint.json")):
        copy_verbatim(fingerprint, out / "historical" / "shared_baselines" / fingerprint.name)


def resolve_export_dir(lock: dict[str, Any], override: Path | None) -> Path:
    export_dir = Path(override) if override else Path(lock["artifacts"]["export_dir"])
    if not export_dir.is_dir():
        raise FileNotFoundError(
            f"export directory not found: {export_dir}; pass --export-dir to relocate it"
        )
    return export_dir.resolve()


def locked_model(export_dir: Path, lock: dict[str, Any], label: str) -> Path:
    """Locate a locked model by file name in the export directory and check its hash."""
    artifacts = lock["artifacts"]
    path = export_dir / Path(artifacts[f"{label}_path"]).name
    if not path.is_file():
        raise FileNotFoundError(f"locked {label} model missing: {path}")
    digest = sha256_file(path)
    if digest != artifacts[f"{label}_sha256"]:
        raise ValueError(f"{path} sha256 {digest} does not match the locked sha256")
    return path


def package_validation(
    export_dir: Path, lock: dict[str, Any], out: Path, large: list[dict[str, Any]]
) -> None:
    int8_ok = lock["artifacts"].get("int8_status") == "ok"
    int4_ok = lock["artifacts"].get("int4_status") == "ok"
    sources = [
        require_artifact(export_dir, EXPORT_VALIDATION_NAME),
        export_dir / "export_manifest.json",
        export_dir / "corpus_eval.json",
    ]
    if int8_ok:
        sources.append(require_artifact(export_dir, EXPORT_VALIDATION_INT8_NAME))
    else:
        found = resolve_artifact(export_dir, EXPORT_VALIDATION_INT8_NAME)
        if found is not None:
            sources.append(found)
    if int4_ok:
        sources.append(require_artifact(export_dir, EXPORT_VALIDATION_INT4_NAME))
    calibration = export_dir / "corpus_calibration.json"
    if int8_ok or calibration.is_file():
        sources.append(calibration)
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"missing export file: {source}")
        copy_verbatim(source, out / "validation" / source.name)
    models = [("fp32", locked_model(export_dir, lock, "fp32"))]
    if int8_ok:
        models.append(("int8", locked_model(export_dir, lock, "int8")))
    if int4_ok:
        models.append(("int4", locked_model(export_dir, lock, "int4")))
    for _, model in models:
        large.append(large_entry(model, "model_or_corpus_artifact"))
    large.append(large_entry(export_dir / "corpus_eval.npz", "model_or_corpus_artifact"))
    for name in ("corpus_calibration.npz", "eval_chunks_onnx_fp32.npz"):
        if (export_dir / name).is_file():
            large.append(large_entry(export_dir / name, "model_or_corpus_artifact"))


def resolve_profile(key: str, lock: dict[str, Any], override: Path | None) -> Path:
    """Locate a locked profile and require the same identity as in the run lock."""
    artifacts = lock["artifacts"]
    identity = artifacts["profile_identities"][key]
    if override is None:
        path = Path(artifacts["profiles"][key])
        if not path.is_file():
            raise FileNotFoundError(
                f"locked profile for {key} not found at {path}; pass --profile {key}=PATH"
            )
    elif Path(override).is_dir():
        path = resolve_profile_by_identity(
            Path(override),
            device_label=identity["device_label"],
            candidate=identity["candidate"],
            variant_label=identity["variant_label"],
            expected_model_sha256=identity["model_sha256"],
        )
    else:
        path = Path(override)
    if load_latency_profile(path).identity() != identity:
        raise ValueError(f"profile {path} does not match the locked identity for {key}")
    return path.resolve()


def package_profiles(
    lock: dict[str, Any],
    overrides: dict[str, Path],
    out: Path,
    large: list[dict[str, Any]],
) -> None:
    unknown = set(overrides) - set(lock["artifacts"]["profile_identities"])
    if unknown:
        raise ValueError(f"--profile names devices not in the run lock: {sorted(unknown)}")
    for device in lock["artifacts"]["profile_identities"]:
        profile = resolve_profile(device, lock, overrides.get(device))
        src = profile.parent
        meta = load_json(src / "profile_run_meta.json")
        raw_telemetry = load_json(src / "telemetry_raw.json")
        if raw_telemetry != meta["telemetry"]:
            raise ValueError(
                f"{device}: telemetry_raw.json differs from profile_run_meta telemetry"
            )
        target = out / "profiles" / device
        copy_verbatim(profile, target / "profile.json")
        copy_verbatim(src / "profile_run_meta.json", target / "profile_run_meta.json")
        copy_verbatim(src / "profile_run_complete.json", target / "profile_run_complete.json")
        for extra, role in (
            ("telemetry_raw.json", "duplicate_of_profile_run_meta_telemetry"),
            (profile.with_suffix(".npz").name, "duplicate_of_profile_samples"),
        ):
            large.append(large_entry(src / extra, role))


def package_identities(lock: dict[str, Any], resolutions: list[dict[str, Any]], out: Path) -> None:
    artifacts = lock["artifacts"]
    identities = {
        "model_fp32": {
            "sha256": artifacts["fp32_sha256"],
            "recipe_id": artifacts["fp32_recipe_id"],
        },
        "model_int8_qdq": {
            "sha256": artifacts["int8_sha256"],
            "status": artifacts["int8_status"],
            "detail": artifacts.get("int8_detail"),
        },
        "profile_identities": artifacts["profile_identities"],
        "lock_keys": sorted(lock),
    }
    if "int4_status" in artifacts:
        identities["model_int4_weight_only"] = {
            "sha256": artifacts.get("int4_sha256"),
            "status": artifacts["int4_status"],
            "detail": artifacts.get("int4_detail"),
        }
    write_json(out / "identities" / "identities.json", identities)
    write_json(out / "identities" / "condition_resolution.json", resolutions)


def build_manifest(out: Path) -> dict[str, Any]:
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    return {
        "schema_version": 1,
        "note": HISTORICAL_NOTE,
        "files": [
            {
                "path": p.relative_to(out).as_posix(),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
            for p in files
        ],
    }


def verify_package(staged: Path) -> None:
    """Re-read the staged package: every listed file matches, and episodes are present."""
    manifest = load_json(staged / "manifest.json")
    listed = {entry["path"]: entry for entry in manifest["files"]}
    on_disk = {
        p.relative_to(staged).as_posix()
        for p in staged.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    if set(listed) != on_disk:
        raise ValueError("staged package files differ from its manifest")
    for path, entry in listed.items():
        if sha256_file(staged / path) != entry["sha256"]:
            raise ValueError(f"staged file {path} does not match its manifest sha256")
    episodes = staged / "episodes" / "episodes.jsonl"
    if not episodes.is_file() or not episodes.read_text(encoding="utf-8").strip():
        raise ValueError("staged package has no episode rows")


def refuse_overlap(out: Path, inputs: list[Path]) -> None:
    for path in inputs:
        if out == path or out.is_relative_to(path) or path.is_relative_to(out):
            raise ValueError(f"output {out} overlaps input {path}; choose a separate --out")


def replace_directory(staged: Path, out: Path) -> None:
    """Swap staged into place; restore the previous directory if the swap fails."""
    if not out.exists():
        staged.rename(out)
        return
    backup = Path(tempfile.mkdtemp(prefix=f".{out.name}.", suffix=".previous", dir=out.parent))
    backup.rmdir()
    out.rename(backup)
    try:
        staged.rename(out)
    except OSError:
        backup.rename(out)
        raise
    shutil.rmtree(backup)


def build_package(
    full_root: Path,
    out: Path,
    lock: dict[str, Any],
    export_dir: Path,
    profile_overrides: dict[str, Path],
) -> int:
    large: list[dict[str, Any]] = []
    rows, resolutions = collect_episode_rows(full_root, large, lock)
    write_jsonl(out / "episodes" / "episodes.jsonl", rows)
    package_protocol(full_root, out)
    package_summaries(full_root, out)
    package_historical(full_root, out)
    package_validation(export_dir, lock, out, large)
    package_profiles(lock, profile_overrides, out, large)
    package_identities(lock, resolutions, out)
    write_json(
        out / "large_evidence.json",
        {
            "schema_version": 1,
            "note": "Not copied into the repository. Present locally under the listed paths.",
            "files": large,
        },
    )
    write_json(out / "manifest.json", build_manifest(out))
    return len(rows)


def package(
    full_root: Path,
    out: Path,
    *,
    export_dir: Path | None = None,
    profile_overrides: dict[str, Path] | None = None,
) -> int:
    """Build the package in a sibling directory and replace `out` only after it verifies."""
    lock = load_json(require_artifact(full_root, PROTOCOL_LOCK_NAME))
    export = resolve_export_dir(lock, export_dir)
    overrides = {k: Path(v).resolve() for k, v in (profile_overrides or {}).items()}
    refuse_overlap(out, [full_root, export, *overrides.values()])
    if out.exists() and not (out / "manifest.json").exists():
        raise ValueError(f"refusing to replace {out}: no manifest.json from a previous package")
    out.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{out.name}.", suffix=".staging", dir=out.parent))
    try:
        n_rows = build_package(full_root, staged, lock, export, overrides)
        verify_package(staged)
        replace_directory(staged, out)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    LOGGER.info("packaged %d episode rows -> %s", n_rows, out)
    return 0


def _parse_profile(value: str) -> tuple[str, Path]:
    device, separator, path = value.partition("=")
    if not separator or not device or not path:
        raise argparse.ArgumentTypeError("expected DEVICE=PATH")
    return device, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_FULL, help="completed full-run root")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="packaged data directory")
    parser.add_argument(
        "--export-dir", type=Path, default=None, help="export directory if not at its locked path"
    )
    parser.add_argument(
        "--profile",
        type=_parse_profile,
        action="append",
        default=[],
        metavar="DEVICE=PATH",
        help="profile JSON or directory if not at its locked path (repeatable)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        return package(
            args.input.resolve(),
            args.out.resolve(),
            export_dir=args.export_dir,
            profile_overrides=dict(args.profile),
        )
    except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
