#!/usr/bin/env python3
"""Package a completed benchmark run into benchmarks/act-transfer/deployment-stress/data (read-only on inputs)."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from breakpoint_eval.artifact_names import (  # noqa: E402
    BENCHMARK_CONFIG_NAME,
    EXPORT_VALIDATION_INT8_NAME,
    EXPORT_VALIDATION_NAME,
    PROTOCOL_LOCK_NAME,
    require_artifact,
)
from breakpoint_eval.file_digest import sha256_file  # noqa: E402

LOGGER = logging.getLogger("package_benchmark")

DEFAULT_FULL = REPO_ROOT / "outputs" / "v1" / "full"
DEFAULT_OUT = REPO_ROOT / "benchmarks" / "act-transfer" / "deployment-stress" / "data"
EXPORT_DIR = REPO_ROOT / "artifacts" / "export" / "act-v1"
PROFILE_DEVICES = ("laptop-m3", "rpi5")
EXPORT_VALIDATION_FILES = (
    EXPORT_VALIDATION_NAME,
    EXPORT_VALIDATION_INT8_NAME,
    "export_manifest.json",
    "corpus_eval.json",
    "corpus_calibration.json",
)
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
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def condition_report_path(full_root: Path, plan_id: str, record: dict[str, Any]) -> Path:
    if record["status"] == "referenced_baseline":
        reference_dir = Path(record["reference_dir"]).name
        return full_root / "shared_baselines" / reference_dir / "run_report.json"
    if record["status"] != "complete":
        raise ValueError(f"{plan_id}/{record['condition_id']} has status {record['status']!r}")
    return full_root / "axes" / plan_id / "conditions" / record["condition_id"] / "run_report.json"


def first_success_step(steps: list[dict[str, Any]]) -> int | None:
    for step in steps:
        if step.get("is_success"):
            return int(step["control_step_index"])
    return None


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
    }


def check_episodes_jsonl_duplicates_report(report: dict[str, Any], jsonl_path: Path) -> None:
    lines = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    if lines != report["episodes"]:
        raise ValueError(f"{jsonl_path} does not equal run_report episodes")


def collect_episode_rows(full_root: Path, large: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hashed: set[Path] = set()
    for plan in load_json(full_root / "run_records.json"):
        plan_id = plan["plan_id"]
        axis = load_json(full_root / "plans" / f"{plan_id}.json")["axis"]
        for record in plan["records"]:
            report_path = condition_report_path(full_root, plan_id, record)
            report = load_json(report_path)
            referenced = record["status"] == "referenced_baseline"
            source = repo_relative(report_path)
            for episode in report["episodes"]:
                rows.append(
                    compact_episode_row(
                        axis=axis,
                        plan_id=plan_id,
                        condition_id=record["condition_id"],
                        referenced=referenced,
                        source=source,
                        episode=episode,
                    )
                )
            if report_path in hashed:
                continue
            hashed.add(report_path)
            large.append(large_entry(report_path, "canonical_control_step_log"))
            jsonl = report_path.parent / "episodes.jsonl"
            if jsonl.exists():
                check_episodes_jsonl_duplicates_report(report, jsonl)
                large.append(large_entry(jsonl, "duplicate_of_run_report_episodes"))
            LOGGER.info(
                "episodes %s/%s: %d", plan_id, record["condition_id"], len(report["episodes"])
            )
    return rows


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
    fingerprint = (
        full_root / "shared_baselines" / "default_fp32_native_L0" / "baseline_fingerprint.json"
    )
    copy_verbatim(fingerprint, out / "historical" / "shared_baselines" / fingerprint.name)


def package_validation(out: Path, large: list[dict[str, Any]]) -> None:
    for name in EXPORT_VALIDATION_FILES:
        source = (
            require_artifact(EXPORT_DIR, name)
            if name in (EXPORT_VALIDATION_NAME, EXPORT_VALIDATION_INT8_NAME)
            else EXPORT_DIR / name
        )
        copy_verbatim(source, out / "validation" / source.name)
    for name in (
        "act_chunk_fp32.onnx",
        "act_chunk_int8_qdq.onnx",
        "corpus_eval.npz",
        "corpus_calibration.npz",
        "eval_chunks_onnx_fp32.npz",
    ):
        large.append(large_entry(EXPORT_DIR / name, "model_or_corpus_artifact"))


def package_profiles(out: Path, large: list[dict[str, Any]]) -> None:
    for device in PROFILE_DEVICES:
        src = REPO_ROOT / "profiles" / device / "fp32"
        profile = next(src.glob("act_aloha_transfer__*.json"))
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


def package_identities(full_root: Path, out: Path) -> None:
    lock = load_json(require_artifact(full_root, PROTOCOL_LOCK_NAME))
    artifacts = lock["artifacts"]
    for label, key in (("fp32", "fp32_sha256"), ("int8", "int8_sha256")):
        path = Path(artifacts[f"{label}_path"])
        local = EXPORT_DIR / path.name
        if sha256_file(local) != artifacts[key]:
            raise ValueError(f"{local} does not match locked sha256")
    identities = {
        "model_fp32": {
            "sha256": artifacts["fp32_sha256"],
            "recipe_id": artifacts["fp32_recipe_id"],
        },
        "model_int8_qdq": {"sha256": artifacts["int8_sha256"], "status": artifacts["int8_status"]},
        "profile_identities": artifacts["profile_identities"],
        "lock_keys": sorted(lock),
    }
    write_json(out / "identities" / "identities.json", identities)


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


def package(full_root: Path, out: Path) -> int:
    if out.exists():
        if not (out / "manifest.json").exists():
            raise ValueError(f"refusing to replace {out}: no manifest.json from a previous package")
        shutil.rmtree(out)
    large: list[dict[str, Any]] = []
    rows = collect_episode_rows(full_root, large)
    write_jsonl(out / "episodes" / "episodes.jsonl", rows)
    package_protocol(full_root, out)
    package_summaries(full_root, out)
    package_historical(full_root, out)
    package_validation(out, large)
    package_profiles(out, large)
    package_identities(full_root, out)
    write_json(
        out / "large_evidence.json",
        {
            "schema_version": 1,
            "note": "Not copied into the repository. Present locally under the listed paths.",
            "files": large,
        },
    )
    write_json(out / "manifest.json", build_manifest(out))
    LOGGER.info("packaged %d episode rows -> %s", len(rows), out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_FULL, help="completed full-run root")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="packaged data directory")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return package(args.input.resolve(), args.out.resolve())


if __name__ == "__main__":
    sys.exit(main())
