"""Execute ``breakpoint run --plan PATH --out DIR`` condition-by-condition."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from breakpoint_eval.camera_degrade import ImageDegradeConfig, ImageMode
from breakpoint_eval.experiment_plan import ExperimentPlan, load_experiment_plan
from breakpoint_eval.experiment_summary import summarize_from_plan_file
from breakpoint_eval.export_runtime import load_export_manifest, resolve_onnx_runtime
from breakpoint_eval.latency_trace import load_latency_profile
from breakpoint_eval.render_report import render_report
from breakpoint_eval.run_baseline import RunRequest, run_baseline

LOGGER = logging.getLogger("breakpoint_eval.run_plan_execute")


def _refuse_or_prepare_out(output_dir: Path, *, allow_existing: bool) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists():
        nonempty = any(output_dir.iterdir())
        if nonempty and not allow_existing:
            raise FileExistsError(
                f"output directory is not empty: {output_dir}. "
                "Pass --allow-existing-out to reuse it, or choose a fresh --out path."
            )
    output_dir.mkdir(parents=True, exist_ok=True)


def _variant_onnx_path(export_dir: Path, variant_label: str) -> Path:
    manifest = load_export_manifest(export_dir / "export_manifest.json")
    for variant in manifest.get("variants") or []:
        if variant.get("label") == variant_label and variant.get("status") == "ok":
            path = variant.get("path")
            if path is None:
                break
            candidate = Path(path)
            if not candidate.is_file():
                candidate = export_dir / Path(path).name
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(
        f"no ok export variant {variant_label!r} under {export_dir}; "
        "run breakpoint export and pass --onnx-model from that directory"
    )


def _image_cfg_from_params(params: dict[str, Any]) -> ImageDegradeConfig:
    mode = str(params.get("mode") or ImageMode.NATIVE.value)
    if mode == ImageMode.NATIVE.value or params.get("bypass"):
        return ImageDegradeConfig(mode=ImageMode.NATIVE)
    if mode == ImageMode.DOWNSAMPLE.value:
        hw = params.get("capture_hw")
        if not hw or len(hw) != 2:
            raise ValueError(f"downsample condition missing capture_hw: {params}")
        return ImageDegradeConfig(
            mode=ImageMode.DOWNSAMPLE,
            capture_height=int(hw[0]),
            capture_width=int(hw[1]),
        )
    if mode == ImageMode.JPEG.value:
        return ImageDegradeConfig(
            mode=ImageMode.JPEG,
            jpeg_quality=int(params["jpeg_quality"]),
        )
    if mode == ImageMode.H264.value:
        return ImageDegradeConfig(
            mode=ImageMode.H264,
            h264_bitrate_kbps=float(params["h264_bitrate_kbps"]),
            frame_rate_hz=float(params.get("frame_rate_hz") or 50.0),
            h264_gop=int(params.get("h264_gop") or 30),
            h264_preset=str(params.get("h264_preset") or "veryfast"),
            h264_latency_mode=str(params.get("h264_latency_mode") or "quality_only"),
            h264_deployment_buffer_frames=int(params.get("h264_deployment_buffer_frames") or 0),
        )
    raise ValueError(f"unsupported image condition mode {mode!r}")


def _condition_run_request(
    *,
    plan: ExperimentPlan,
    condition: Any,
    condition_dir: Path,
    repo_root: Path,
    device: str,
    onnx_model: Path | None,
    latency_profile: Path | None,
    save_video: bool,
) -> RunRequest:
    params = dict(condition.params)
    env_seeds = [s.env_seed for s in plan.seed_pairs]
    policy_seeds = [s.policy_rng_seed for s in plan.seed_pairs]
    axis = condition.axis

    latency_s = 0.0
    profile = None
    image_cfg = ImageDegradeConfig(mode=ImageMode.NATIVE)
    model_path = onnx_model

    if axis == "latency_ms":
        latency_s = float(params["latency_s"])
        if latency_profile is not None:
            raise ValueError(
                "latency_ms plan axis uses constant delay; do not combine with --latency-profile"
            )
    elif axis == "precision":
        if onnx_model is None:
            raise ValueError("precision plan axis requires --onnx-model from an export directory")
        export_dir = Path(onnx_model).resolve().parent
        model_path = _variant_onnx_path(export_dir, str(params["variant_label"]))
        # Validate identity for the selected variant.
        resolve_onnx_runtime(
            model_path,
            expected_candidate=plan.candidate,
            require_export_validation=True,
        )
        if latency_profile is not None:
            profile = load_latency_profile(Path(latency_profile).resolve())
    elif axis in {"resolution", "jpeg", "h264"}:
        image_cfg = _image_cfg_from_params(params)
    elif axis == "combined":
        # Combined plans encode nested params; require explicit supported keys.
        if "latency_s" in params:
            latency_s = float(params["latency_s"])
        if "variant_label" in params:
            if onnx_model is None:
                raise ValueError("combined precision requires --onnx-model")
            model_path = _variant_onnx_path(
                Path(onnx_model).resolve().parent, str(params["variant_label"])
            )
        if params.get("mode") and params.get("mode") != ImageMode.NATIVE.value:
            image_cfg = _image_cfg_from_params(params)
    else:
        raise ValueError(f"unsupported plan axis {axis!r}")

    return RunRequest(
        candidate=plan.candidate,
        seeds=env_seeds,
        output_dir=condition_dir,
        device=device,
        save_video=save_video,
        latency_s=latency_s,
        latency_profile=profile,
        image_cfg=image_cfg,
        repo_root=repo_root,
        onnx_model=model_path,
        policy_rng_seeds=policy_seeds,
    )


def execute_experiment_plan(
    *,
    plan_path: Path,
    output_dir: Path,
    repo_root: Path,
    device: str,
    compute_device: str | None = None,
    onnx_model: Path | None = None,
    latency_profile: Path | None = None,
    profile_device: str | None = None,
    profile_variant: str | None = None,
    profile_recipe_id: str | None = None,
    save_video: bool = False,
    dry_run: bool = False,
    allow_existing_out: bool = False,
    latency_end_of_trace: str = "fail",
    latency_block_length: int | None = None,
    latency_block_seed: int | None = None,
) -> int:
    """Run each plan condition into ``output_dir/conditions/<id>/``.

    Returns nonzero if any condition is incomplete (runtime errors / missing report)
    or if planned vs effective settings fail validation.
    """
    if profile_device is not None or profile_recipe_id is not None or profile_variant is not None:
        # Keep plan execution deterministic: explicit --latency-profile only for now.
        if latency_profile is None:
            raise ValueError(
                "plan execution with device replay requires explicit --latency-profile "
                "(profile-device resolution is not used in --plan mode)"
            )
    _refuse_or_prepare_out(output_dir, allow_existing=allow_existing_out)
    plan = load_experiment_plan(plan_path)
    plan_copy = output_dir / "experiment_plan.json"
    shutil.copy2(plan_path, plan_copy)

    host = compute_device or ("cpu" if device == "rpi5" else device)
    if host not in {"cuda", "cpu", "mps"}:
        raise ValueError(f"invalid compute device {host!r}")

    condition_reports: dict[str, Path] = {}
    records: list[dict[str, Any]] = []
    incomplete = False

    for condition in plan.conditions:
        condition_dir = output_dir / "conditions" / condition.condition_id
        condition_dir.mkdir(parents=True, exist_ok=True)
        req = _condition_run_request(
            plan=plan,
            condition=condition,
            condition_dir=condition_dir,
            repo_root=repo_root,
            device=host,
            onnx_model=onnx_model,
            latency_profile=latency_profile,
            save_video=save_video,
        )
        planned = {
            "condition_id": condition.condition_id,
            "axis": condition.axis,
            "params": condition.params,
            "env_seeds": req.seeds,
            "policy_rng_seeds": req.policy_rng_seeds,
            "onnx_model": str(req.onnx_model) if req.onnx_model else None,
            "latency_s": req.latency_s,
            "latency_profile": str(latency_profile) if latency_profile else None,
            "image_mode": req.image_cfg.mode.value if req.image_cfg else "native",
            "device": host,
        }
        (condition_dir / "planned_settings.json").write_text(
            json.dumps(planned, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if dry_run:
            records.append({"condition_id": condition.condition_id, "dry_run": True, **planned})
            continue
        try:
            if req.latency_profile is not None:
                req = RunRequest(
                    candidate=req.candidate,
                    seeds=req.seeds,
                    output_dir=req.output_dir,
                    device=req.device,
                    save_video=req.save_video,
                    latency_s=0.0,
                    latency_profile=req.latency_profile,
                    latency_end_of_trace=latency_end_of_trace,
                    latency_block_length=latency_block_length,
                    latency_block_seed=latency_block_seed,
                    image_cfg=req.image_cfg,
                    repo_root=req.repo_root,
                    onnx_model=req.onnx_model,
                    policy_rng_seeds=req.policy_rng_seeds,
                )
            report = run_baseline(req)
        except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
            incomplete = True
            err = f"{type(exc).__name__}: {exc}"
            LOGGER.error("condition %s failed: %s", condition.condition_id, err)
            (condition_dir / "condition_error.json").write_text(
                json.dumps({"error": err, "planned": planned}, indent=2) + "\n",
                encoding="utf-8",
            )
            records.append(
                {
                    "condition_id": condition.condition_id,
                    "status": "errored",
                    "error": err,
                    "planned": planned,
                }
            )
            continue

        payload = json.loads(report.read_text(encoding="utf-8"))
        provenance = payload.get("provenance") or {}
        summary = payload.get("summary") or {}
        effective = {
            "policy_backend": provenance.get("policy_backend"),
            "onnx_runtime": provenance.get("onnx_runtime"),
            "latency_mode": provenance.get("latency_mode"),
            "latency_s": provenance.get("latency_s"),
            "latency_profile_identity": provenance.get("latency_profile_identity"),
            "image_degrade": provenance.get("image_degrade"),
            "env_seeds": provenance.get("env_seeds") or provenance.get("seeds"),
            "policy_rng_seeds": provenance.get("policy_rng_seeds"),
            "device": provenance.get("device"),
        }
        mismatches: list[str] = []
        if planned["env_seeds"] != effective.get("env_seeds"):
            mismatches.append("env_seeds")
        if planned["policy_rng_seeds"] != effective.get("policy_rng_seeds"):
            mismatches.append("policy_rng_seeds")
        if planned["onnx_model"] is not None:
            runtime = effective.get("onnx_runtime") or {}
            if str(runtime.get("model_path")) != planned["onnx_model"]:
                mismatches.append("onnx_model")
            if effective.get("policy_backend") != "onnxruntime":
                mismatches.append("policy_backend")
        if mismatches:
            incomplete = True
            LOGGER.error(
                "condition %s planned vs effective mismatch: %s",
                condition.condition_id,
                mismatches,
            )
        if summary.get("n_runtime_error", 0) > 0:
            incomplete = True
        if provenance.get("env_created") is False:
            incomplete = True
        condition_reports[condition.condition_id] = report
        records.append(
            {
                "condition_id": condition.condition_id,
                "status": "complete" if not mismatches else "mismatched",
                "report": str(report),
                "planned": planned,
                "effective": effective,
                "mismatches": mismatches,
                "summary": summary,
            }
        )

    records_path = output_dir / "condition_records.json"
    records_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    if dry_run:
        LOGGER.info("dry-run plan execution wrote planned settings only")
        return 0

    # Summarize from on-disk reports keyed by condition id.
    mapping = {cid: path for cid, path in condition_reports.items() if path.is_file()}
    try:
        summarize_from_plan_file(
            plan_copy,
            mapping,
            output_path=output_dir / "experiment_summary.json",
        )
    except (ValueError, FileNotFoundError, KeyError) as exc:
        incomplete = True
        LOGGER.error("experiment summary failed: %s", exc)

    try:
        render_report(output_dir)
    except Exception as exc:  # noqa: BLE001 — report is best-effort after records exist
        LOGGER.warning("render_report failed: %s", exc)

    return 2 if incomplete else 0
