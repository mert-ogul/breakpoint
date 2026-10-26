#!/usr/bin/env python3
"""Run the ACT deployment stress protocol over the existing planner, runner and summarizer."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import shutil
import sys
from pathlib import Path

# Prefer installed package; fall back to src/ for editable-free checkouts.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from breakpoint_eval.artifact_names import (  # noqa: E402
    BENCHMARK_CONFIG_NAME,
    PROTOCOL_LOCK_NAME,
    resolve_artifact,
)
from breakpoint_eval.benchmark_protocol import (  # noqa: E402
    DEFAULT_CONFIG_REL,
    assert_lock_matches,
    build_full_axis_plans,
    build_smoke_plan,
    condition_completed,
    condition_is_shared_baseline,
    condition_report_problem,
    describe_plan_commands,
    freeze_protocol_lock,
    host_settings_from_config,
    load_benchmark_config,
    resolve_benchmark_artifacts,
    shared_baseline_problem,
    write_plan_bundle,
    write_unsupported_int8_record,
)
from breakpoint_eval.experiment_summary import (  # noqa: E402
    summarize_from_plan_file,
)
from breakpoint_eval.render_report import render_report  # noqa: E402
from breakpoint_eval.run_baseline import run_baseline  # noqa: E402
from breakpoint_eval.run_plan_execute import condition_run_request  # noqa: E402
from breakpoint_eval.source_revision import read_source_revision  # noqa: E402

LOGGER = logging.getLogger("run_benchmark")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_benchmark",
        description="Run the fixed ACT deployment stress protocol (smoke or full phase).",
    )
    p.add_argument("--phase", choices=("smoke", "full"), required=True)
    p.add_argument("--out", type=Path, required=True, help="Run output root")
    p.add_argument(
        "--config",
        type=Path,
        default=_ROOT / DEFAULT_CONFIG_REL,
        help="Protocol config (default: configs/act-transfer.json)",
    )
    p.add_argument("--export-dir", type=Path, default=None, help="Override export dir")
    p.add_argument(
        "--profile-laptop",
        type=Path,
        default=None,
        help="Override laptop profile JSON or directory",
    )
    p.add_argument(
        "--profile-rpi5",
        type=Path,
        default=None,
        help="Override Pi profile JSON or directory",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print plans/commands; do not create run-result dirs or evaluate",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume only completed matching conditions; refuse identity/config changes",
    )
    p.add_argument("--repo-root", type=Path, default=_ROOT)
    return p.parse_args(argv)


def _shared_baseline_dir(out: Path) -> Path:
    return out / "shared_baselines" / "default_fp32_native_L0"


def _execute_condition(
    *,
    plan,
    condition,
    condition_dir: Path,
    repo_root: Path,
    host,
    artifacts,
    shared_baseline: Path | None,
    baseline_report: Path | None,
) -> dict:
    condition_dir.mkdir(parents=True, exist_ok=True)
    if (
        shared_baseline is not None
        and baseline_report is not None
        and condition_is_shared_baseline(condition)
        and condition.condition_id != "latency_ms_0"
        and baseline_report.is_file()
    ):
        problem = shared_baseline_problem(shared_baseline, plan=plan)
        if problem is not None:
            LOGGER.error("condition %s: referenced baseline: %s", condition.condition_id, problem)
            return {
                "condition_id": condition.condition_id,
                "status": "incomplete",
                "error": f"referenced baseline: {problem}",
                "reference_report": str(baseline_report),
            }
        ref = {
            "status": "referenced_baseline",
            "reference_report": str(baseline_report),
            "reference_dir": str(shared_baseline),
            "condition_id": condition.condition_id,
            "by_explicit_reference_only": True,
            "measurements_fabricated": False,
        }
        (condition_dir / "baseline_reference.json").write_text(
            json.dumps(ref, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        dest = condition_dir / "run_report.json"
        shutil.copy2(baseline_report, dest)
        return {**ref, "report": str(dest)}

    onnx_model = artifacts.fp32_path
    if condition.axis == "precision" and condition.params.get("variant_label") == "int8":
        if artifacts.int8_status != "ok" or artifacts.int8_path is None:
            rec = {
                "condition_id": condition.condition_id,
                "status": "unsupported",
                "detail": artifacts.int8_detail,
                "measurements_fabricated": False,
            }
            (condition_dir / "condition_unsupported.json").write_text(
                json.dumps(rec, indent=2) + "\n", encoding="utf-8"
            )
            return rec
        onnx_model = artifacts.int8_path

    req = condition_run_request(
        plan=plan,
        condition=condition,
        condition_dir=condition_dir,
        repo_root=repo_root,
        device=host.compute_device,
        onnx_model=onnx_model,
        latency_profile=None,
        save_video=False,
        ort_providers=list(host.ort_providers),
        ort_intra_op_num_threads=host.intra_op_num_threads,
        ort_inter_op_num_threads=host.inter_op_num_threads,
    )
    planned = {
        "condition_id": condition.condition_id,
        "axis": condition.axis,
        "params": condition.params,
        "env_seeds": req.seeds,
        "policy_rng_seeds": req.policy_rng_seeds,
        "onnx_model": str(req.onnx_model) if req.onnx_model else None,
        "device": host.compute_device,
        "ort_providers": host.ort_providers,
        "ort_intra_op_num_threads": host.intra_op_num_threads,
        "ort_inter_op_num_threads": host.inter_op_num_threads,
        "latency_window_starts": req.latency_window_starts,
        "latency_window_length": req.latency_window_length,
    }
    (condition_dir / "planned_settings.json").write_text(
        json.dumps(planned, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        if req.latency_profile is not None:
            req = dataclasses.replace(
                req, save_video=False, latency_s=0.0, latency_end_of_trace="fail"
            )
        report = run_baseline(req)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        err = f"{type(exc).__name__}: {exc}"
        LOGGER.error("condition %s failed: %s", condition.condition_id, err)
        (condition_dir / "condition_error.json").write_text(
            json.dumps({"error": err, "planned": planned}, indent=2) + "\n",
            encoding="utf-8",
        )
        return {"condition_id": condition.condition_id, "status": "errored", "error": err}

    problem = condition_report_problem(report, plan, condition)
    if problem is not None:
        LOGGER.error("condition %s incomplete: %s", condition.condition_id, problem)
        (condition_dir / "condition_error.json").write_text(
            json.dumps({"error": problem, "report": str(report), "planned": planned}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return {
            "condition_id": condition.condition_id,
            "status": "incomplete",
            "error": problem,
            "report": str(report),
            "planned": planned,
        }

    if condition.condition_id == "latency_ms_0" and shared_baseline is not None:
        shared_baseline.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report, shared_baseline / "run_report.json")
        (shared_baseline / "baseline_fingerprint.json").write_text(
            json.dumps(
                {
                    "condition_id": "latency_ms_0",
                    "report": str(shared_baseline / "run_report.json"),
                    "by_explicit_reference_only": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return {
        "condition_id": condition.condition_id,
        "status": "complete",
        "report": str(report),
        "planned": planned,
    }


def _resumed_record(condition_dir: Path, condition, report: Path) -> dict:
    """Rebuild the record of a finished condition, keeping whether it reused the baseline."""
    reference_path = condition_dir / "baseline_reference.json"
    if reference_path.is_file():
        record = json.loads(reference_path.read_text(encoding="utf-8"))
    else:
        record = {"condition_id": condition.condition_id, "status": "complete"}
        planned_path = condition_dir / "planned_settings.json"
        if planned_path.is_file():
            record["planned"] = json.loads(planned_path.read_text(encoding="utf-8"))
    return {**record, "report": str(report), "resumed": True}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        config = load_benchmark_config(args.config.resolve())
        host = host_settings_from_config(config)
        if host.compute_device != "cpu":
            raise ValueError("the protocol requires host.compute_device=cpu (no MPS inheritance)")

        overrides: dict[str, Path] = {}
        if args.profile_laptop is not None:
            overrides["laptop-m3"] = args.profile_laptop.resolve()
        if args.profile_rpi5 is not None:
            overrides["rpi5"] = args.profile_rpi5.resolve()

        artifacts = resolve_benchmark_artifacts(
            config,
            repo_root=repo_root,
            export_dir=args.export_dir.resolve() if args.export_dir else None,
            profile_overrides=overrides or None,
        )
        source_revision = read_source_revision(repo_root)
        lock = freeze_protocol_lock(
            config=config,
            artifacts=artifacts,
            source_revision=source_revision,
            host=host,
            phase=args.phase,
        )

        if args.phase == "smoke":
            plans = [build_smoke_plan(config, artifacts)]
        else:
            plans = build_full_axis_plans(config, artifacts)
    except (ValueError, FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        LOGGER.error("%s", exc)
        return 2

    if args.dry_run:
        print(json.dumps({"dry_run": True, "lock": lock, "n_plans": len(plans)}, indent=2))
        for plan in plans:
            axis_out = args.out / "axes" / plan.plan_id
            for line in describe_plan_commands(plan, axis_out=axis_out):
                print(line)
            print(
                json.dumps(
                    {
                        "plan_id": plan.plan_id,
                        "axis": plan.axis,
                        "n_episodes": plan.n_episodes,
                        "env_seeds": [s.env_seed for s in plan.seed_pairs],
                        "policy_rng_seeds": [s.policy_rng_seed for s in plan.seed_pairs],
                        "conditions": [c.to_dict() for c in plan.conditions],
                        "baseline_condition_id": plan.baseline_condition_id,
                    },
                    indent=2,
                )
            )
        if artifacts.int8_status != "ok":
            print(
                json.dumps(
                    {
                        "int8": {
                            "status": artifacts.int8_status,
                            "detail": artifacts.int8_detail,
                            "measurements_fabricated": False,
                        }
                    },
                    indent=2,
                )
            )
        print("# dry-run complete: no run-result directories created, no episodes evaluated")
        return 0

    out = args.out.resolve()
    lock_path = out / PROTOCOL_LOCK_NAME
    if args.resume:
        lock_path = resolve_artifact(out, PROTOCOL_LOCK_NAME)
        if lock_path is None:
            raise SystemExit(f"--resume requires existing {out / PROTOCOL_LOCK_NAME}")
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        assert_lock_matches(existing, lock)
    else:
        if out.exists() and any(out.iterdir()) and not args.resume:
            raise SystemExit(
                f"--out is not empty: {out}. Pass --resume for matching locks, "
                "or choose a fresh directory."
            )
        out.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shutil.copy2(args.config.resolve(), out / BENCHMARK_CONFIG_NAME)

    if artifacts.int8_status != "ok":
        write_unsupported_int8_record(out, artifacts=artifacts)

    plan_paths = write_plan_bundle(plans, out / "plans")
    shared = _shared_baseline_dir(out)
    baseline_report = shared / "run_report.json"
    incomplete = False
    all_records: list[dict] = []

    for plan, plan_path in zip(plans, plan_paths, strict=True):
        axis_out = out / "axes" / plan.plan_id
        axis_out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plan_path, axis_out / "experiment_plan.json")
        records: list[dict] = []
        condition_reports: dict[str, Path] = {}

        for condition in plan.conditions:
            condition_dir = axis_out / "conditions" / condition.condition_id
            if args.resume and condition_completed(
                condition_dir, plan=plan, condition=condition, shared_baseline_dir=shared
            ):
                LOGGER.info("resume skip completed %s", condition.condition_id)
                report = condition_dir / "run_report.json"
                records.append(_resumed_record(condition_dir, condition, report))
                condition_reports[condition.condition_id] = report
                continue

            rec = _execute_condition(
                plan=plan,
                condition=condition,
                condition_dir=condition_dir,
                repo_root=repo_root,
                host=host,
                artifacts=artifacts,
                shared_baseline=shared,
                baseline_report=baseline_report if baseline_report.is_file() else None,
            )
            records.append(rec)
            incomplete_condition = rec.get("status") in {"errored", "incomplete"}
            incomplete = incomplete or incomplete_condition
            report_path = condition_dir / "run_report.json"
            if report_path.is_file() and not incomplete_condition:
                condition_reports[condition.condition_id] = report_path
            if (shared / "run_report.json").is_file():
                baseline_report = shared / "run_report.json"

        (axis_out / "condition_records.json").write_text(
            json.dumps(records, indent=2) + "\n", encoding="utf-8"
        )
        all_records.append({"plan_id": plan.plan_id, "records": records})

        if plan.baseline_condition_id and plan.baseline_condition_id in condition_reports:
            mapping = dict(condition_reports)
            try:
                summarize_from_plan_file(
                    axis_out / "experiment_plan.json",
                    mapping,
                    output_path=axis_out / "experiment_summary.json",
                    ratio=float(lock["statistics"]["breakpoint_ratio"]),
                    baseline_success_floor=float(lock["statistics"]["baseline_success_floor"]),
                )
            except (ValueError, FileNotFoundError, KeyError) as exc:
                LOGGER.error("summary failed for %s: %s", plan.plan_id, exc)
                incomplete = True
            try:
                render_report(axis_out)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("render_report failed: %s", exc)

    (out / "run_records.json").write_text(
        json.dumps(all_records, indent=2) + "\n", encoding="utf-8"
    )
    return 2 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
