"""Orchestrate ``breakpoint run``: resolve model, plan, execute, save diagnostics."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from breakpoint_eval.camera_degrade import ImageDegradeConfig, ImageMode
from breakpoint_eval.candidate_map import CandidateSpec
from breakpoint_eval.latency_trace import (
    LatencyProfile,
    load_latency_profile,
    profiling_command_hint,
    require_unique_device_profile,
)
from breakpoint_eval.model_resolve import resolve_model_to_candidate
from breakpoint_eval.run_baseline import RunRequest, build_seeds, run_baseline
from breakpoint_eval.stack import repo_root_from_package

LOGGER = logging.getLogger("breakpoint_eval.run_orchestrate")


@dataclass
class ResolvedRunPlan:
    model_input: str
    candidate: str
    spec: CandidateSpec
    n_episodes: int
    seeds: list[int]
    output_dir: Path
    repo_root: Path
    cache_dir: Path
    compute_device: str
    latency_mode: str  # constant | trace_replay
    latency_s: float
    latency_profile_path: str | None
    latency_profile_identity: dict[str, Any] | None
    image_mode: str
    dry_run: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolved_model": self.model_input,
            "candidate": self.candidate,
            "hub_repo": self.spec.hub_repo,
            "hub_revision": self.spec.hub_revision,
            "plan_id": f"{self.candidate}__run__n{self.n_episodes}",
            "n_episodes": self.n_episodes,
            "env_seeds": list(self.seeds),
            "output_dir": str(self.output_dir),
            "repo_root": str(self.repo_root),
            "cache_dir": str(self.cache_dir),
            "compute_device": self.compute_device,
            "latency_mode": self.latency_mode,
            "latency_s": self.latency_s if self.latency_mode == "constant" else None,
            "latency_profile_path": self.latency_profile_path,
            "latency_profile_identity": self.latency_profile_identity,
            "image_mode": self.image_mode,
            "dry_run": self.dry_run,
            "notes": list(self.notes),
            "semantics": {
                "simulated_task_outcomes": True,
                "physical_robot_execution": False,
                "hardware_timing_from_profile_only": self.latency_mode == "trace_replay",
            },
        }


class MissingPrerequisiteError(FileNotFoundError):
    """Actionable missing artifact / profile / cache prerequisite."""


def resolve_run_plan(
    *,
    model: str,
    n_episodes: int,
    output_dir: Path,
    device: str,
    compute_device: str | None = None,
    latency_ms: float = 0.0,
    latency_profile: Path | None = None,
    profile_device: str | None = None,
    profile_variant: str | None = None,
    profile_recipe_id: str | None = None,
    seeds: list[int] | None = None,
    start_seed: int = 1000,
    image_mode: str = ImageMode.NATIVE.value,
    repo_root: Path | None = None,
    dry_run: bool = False,
) -> tuple[ResolvedRunPlan, LatencyProfile | None]:
    candidate, spec = resolve_model_to_candidate(model)
    root = (repo_root or repo_root_from_package()).resolve()
    out = Path(output_dir).resolve()
    cache_dir = root / ".cache" / "huggingface"
    seed_list = seeds or build_seeds(n_episodes=n_episodes, seeds=None, start_seed=start_seed)
    if len(seed_list) != n_episodes:
        raise ValueError("seeds length must match n_episodes")

    # Device / profile resolution
    profile_label = profile_device
    if device == "rpi5":
        profile_label = "rpi5"
    host_compute = compute_device or ("cpu" if device == "rpi5" else device)
    if host_compute not in {"cuda", "cpu", "mps"}:
        raise ValueError(f"invalid compute device {host_compute!r}")

    profile_obj: LatencyProfile | None = None
    profile_path: str | None = None
    identity: dict[str, Any] | None = None
    latency_mode = "constant"
    notes: list[str] = []

    if latency_profile is not None and profile_label is not None:
        raise ValueError("use only one of --latency-profile or --device/--profile-device replay")
    if latency_profile is not None:
        if latency_ms != 0.0:
            raise ValueError("constant --latency-ms is distinct from --latency-profile replay")
        path = Path(latency_profile).resolve()
        if not path.is_file():
            raise MissingPrerequisiteError(
                f"latency profile not found: {path}. "
                "Pass an existing profile JSON or generate one with: "
                + profiling_command_hint(
                    device_label="DEVICE",
                    candidate=candidate,
                    recipe_id=profile_recipe_id or "RECIPE_ID",
                )
            )
        profile_obj = load_latency_profile(path)
        profile_path = str(path)
        identity = profile_obj.identity()
        latency_mode = "trace_replay"
        notes.append(f"Explicit latency profile: {path}")
    elif profile_label is not None:
        if latency_ms != 0.0:
            raise ValueError("constant --latency-ms is distinct from device-profile replay")
        profiles_dir = root / "profiles" / profile_label
        try:
            path = require_unique_device_profile(
                profiles_dir,
                device_label=profile_label,
                candidate=candidate,
                variant_label=profile_variant,
                precision_recipe=profile_recipe_id,
            )
        except FileNotFoundError as exc:
            raise MissingPrerequisiteError(str(exc)) from exc
        profile_obj = load_latency_profile(path)
        profile_path = str(path)
        identity = profile_obj.identity()
        latency_mode = "trace_replay"
        notes.append(f"Resolved device profile from {profiles_dir}: {path.name}")
    else:
        notes.append(f"Constant simulated latency_ms={latency_ms}")

    plan = ResolvedRunPlan(
        model_input=model,
        candidate=candidate,
        spec=spec,
        n_episodes=n_episodes,
        seeds=seed_list,
        output_dir=out,
        repo_root=root,
        cache_dir=cache_dir,
        compute_device=host_compute,
        latency_mode=latency_mode,
        latency_s=0.0 if latency_mode == "trace_replay" else float(latency_ms) / 1000.0,
        latency_profile_path=profile_path,
        latency_profile_identity=identity,
        image_mode=image_mode,
        dry_run=dry_run,
        notes=notes,
    )
    return plan, profile_obj


def write_run_manifest(plan: ResolvedRunPlan, *, extra: dict[str, Any] | None = None) -> Path:
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    payload = plan.to_dict()
    if extra:
        payload.update(extra)
    path = plan.output_dir / "run_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_partial_diagnostics(
    output_dir: Path,
    *,
    plan: dict[str, Any],
    completed: list[dict[str, Any]],
    error: str | None,
    interrupted: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "partial_diagnostics.json"
    payload = {
        "interrupted": interrupted,
        "error": error,
        "completed_steps": completed,
        "plan": plan,
        "saved_at_unix_s": time.time(),
        "note": "Partial diagnostics after interruption or execution error; not a complete report.",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def execute_resolved_run(
    plan: ResolvedRunPlan,
    profile: LatencyProfile | None,
    *,
    save_video: bool = False,
    image_cfg: ImageDegradeConfig | None = None,
    latency_end_of_trace: str = "fail",
    latency_block_length: int | None = None,
    latency_block_seed: int | None = None,
) -> tuple[Path, int]:
    """Run the resolved plan. Returns (run_report_path, exit_code).

    Exit code is non-zero for execution errors, not for low policy success.
    """
    write_run_manifest(plan)
    if plan.dry_run:
        LOGGER.info("dry-run: skipping episode execution")
        return plan.output_dir / "run_manifest.json", 0

    cfg = image_cfg or ImageDegradeConfig(mode=ImageMode(plan.image_mode))
    completed: list[dict[str, Any]] = []
    try:
        run_dir = plan.output_dir
        if plan.latency_mode == "trace_replay":
            run_dir = plan.output_dir / "latency_profile_replay"
        report = run_baseline(
            RunRequest(
                candidate=plan.candidate,
                seeds=plan.seeds,
                output_dir=run_dir,
                device=plan.compute_device,
                save_video=save_video,
                latency_s=plan.latency_s,
                latency_profile=profile,
                latency_end_of_trace=latency_end_of_trace,
                latency_block_length=latency_block_length,
                latency_block_seed=latency_block_seed,
                image_cfg=cfg,
                repo_root=plan.repo_root,
            )
        )
        payload = json.loads(report.read_text(encoding="utf-8"))
        summary = payload.get("summary") or {}
        provenance = payload.get("provenance") or {}
        completed.append(
            {
                "report": str(report),
                "success_rate": summary.get("success_rate"),
                "n_runtime_error": summary.get("n_runtime_error"),
                "env_created": provenance.get("env_created"),
            }
        )
        write_run_manifest(
            plan,
            extra={"completed_steps": completed, "primary_report": str(report)},
        )
        # Execution errors (not low success): env failed or all episodes runtime-errored.
        exit_code = 0
        if provenance.get("env_created") is False:
            exit_code = 2
        if summary.get("n_runtime_error", 0) > 0 and summary.get("n_episodes_eligible", 0) == 0:
            exit_code = 2
        return report, exit_code
    except KeyboardInterrupt:
        path = write_partial_diagnostics(
            plan.output_dir,
            plan=plan.to_dict(),
            completed=completed,
            error="KeyboardInterrupt",
            interrupted=True,
        )
        LOGGER.error("Interrupted; wrote %s", path)
        raise
    except Exception as exc:  # noqa: BLE001 — boundary: persist diagnostics
        path = write_partial_diagnostics(
            plan.output_dir,
            plan=plan.to_dict(),
            completed=completed,
            error=f"{type(exc).__name__}: {exc}",
            interrupted=False,
        )
        LOGGER.exception("Execution error; wrote %s", path)
        return path, 2
