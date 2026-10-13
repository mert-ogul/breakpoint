"""Thin CLI for Breakpoint. Subcommands: run, report, export, profile, plan, summarize."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from breakpoint_eval.camera_degrade import (
    ImageDegradeConfig,
    ImageMode,
    h264_smoke_configs,
    jpeg_smoke_configs,
    resolution_smoke_configs,
)
from breakpoint_eval.candidate_map import CANDIDATES
from breakpoint_eval.device_profile import DEFAULT_MEASURED, DEFAULT_WARMUPS
from breakpoint_eval.experiment_plan import (
    N_EPISODES_FULL,
    N_EPISODES_SMOKE,
    ExperimentAxis,
    PlanPhase,
    make_isolated_plan,
    write_experiment_plan,
)
from breakpoint_eval.experiment_summary import (
    DEFAULT_BASELINE_SUCCESS_FLOOR,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_BREAKPOINT_RATIO,
    summarize_from_plan_file,
)
from breakpoint_eval.latency_scheduler import CHUNK_ARRIVAL_POLICY
from breakpoint_eval.latency_trace import (
    load_latency_profile,
    require_unique_device_profile,
)
from breakpoint_eval.render_report import render_report
from breakpoint_eval.run_baseline import RunRequest, build_seeds, parse_seed_list, run_baseline
from breakpoint_eval.run_export import ExportRequest, run_export
from breakpoint_eval.run_orchestrate import (
    MissingPrerequisiteError,
    execute_resolved_run,
    resolve_run_plan,
    write_partial_diagnostics,
    write_run_manifest,
)
from breakpoint_eval.run_profile import ProfileCliRequest, run_profile
from breakpoint_eval.stack import repo_root_from_package


def parse_capture_hw(text: str) -> tuple[int, int]:
    parts = [p.strip() for p in text.lower().replace("x", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("expected HxW or H,W")
    return int(parts[0]), int(parts[1])


def _add_run_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "run",
        help="Run a pinned candidate / Hub model through the Breakpoint episode runner",
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Candidate key or Hub repo (e.g. lerobot/act_aloha_sim_transfer_cube_human)",
    )
    parser.add_argument(
        "--candidate",
        default=None,
        choices=sorted(CANDIDATES),
        help="Candidate key (alternative to positional model)",
    )
    parser.add_argument(
        "--episodes",
        "--n-episodes",
        dest="n_episodes",
        type=int,
        default=None,
        help="Number of paired episodes",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated env seeds (paired across conditions later)",
    )
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument(
        "--out",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=None,
        help="Directory for provenance + episode results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve plan / profile / cache paths and exit without running episodes",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "mps", "rpi5"],
        default="mps",
        help=(
            "Policy compute device, or rpi5 to resolve measured latency from "
            "profiles/rpi5/ (policy still runs on host; default host compute=cpu)"
        ),
    )
    parser.add_argument(
        "--compute-device",
        choices=["cuda", "cpu", "mps"],
        default=None,
        help="Host policy compute device when --device rpi5 (default: cpu)",
    )
    parser.add_argument(
        "--video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write one mp4 per episode",
    )
    parser.add_argument(
        "--latency-ms",
        type=float,
        default=0.0,
        help="Constant simulated inference latency in milliseconds (not wall sleep)",
    )
    parser.add_argument(
        "--latency-sweep-ms",
        type=str,
        default=None,
        help="Comma-separated constant latency_ms values (distinct from device-profile replay)",
    )
    parser.add_argument(
        "--latency-profile",
        type=Path,
        default=None,
        help="Explicit path to a measured device latency profile JSON",
    )
    parser.add_argument(
        "--profile-device",
        type=str,
        default=None,
        help="Resolve an unambiguous profile from profiles/<label>/ (alternative to --device rpi5)",
    )
    parser.add_argument(
        "--profile-variant",
        type=str,
        default=None,
        help="Variant label filter when resolving --profile-device (e.g. onnx_fp32)",
    )
    parser.add_argument(
        "--profile-recipe-id",
        type=str,
        default=None,
        help="Precision recipe filter when resolving --profile-device",
    )
    parser.add_argument(
        "--latency-end-of-trace",
        choices=["fail", "cycle"],
        default="fail",
        help="Behavior when a measured latency trace is exhausted (default: fail)",
    )
    parser.add_argument(
        "--latency-block-length",
        type=int,
        default=None,
        help="Optional block-resampling length (requires --latency-block-seed)",
    )
    parser.add_argument(
        "--latency-block-seed",
        type=int,
        default=None,
        help="Seed for optional block resampling of a measured trace",
    )
    parser.add_argument(
        "--image-mode",
        choices=[m.value for m in ImageMode],
        default=ImageMode.NATIVE.value,
        help="Camera degradation at raw pixels (before policy normalization)",
    )
    parser.add_argument(
        "--capture-hw",
        type=str,
        default=None,
        help="Downsample capture size as HxW (e.g. 240x320); required for downsample mode",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
        help="JPEG quality 1-100 for jpeg mode (100 is still encoded, not identity)",
    )
    parser.add_argument(
        "--h264-bitrate-kbps",
        type=float,
        default=None,
        help="Target H.264 bitrate (kbps) for h264 mode (requires FFmpeg+libx264+PyAV)",
    )
    parser.add_argument(
        "--h264-gop",
        type=int,
        default=30,
        help="H.264 GOP / keyframe interval (frames)",
    )
    parser.add_argument(
        "--h264-preset",
        type=str,
        default="veryfast",
        help="libx264 preset (validated against installed FFmpeg)",
    )
    parser.add_argument(
        "--h264-latency-mode",
        choices=["quality_only", "deployment"],
        default="quality_only",
        help=(
            "quality_only: visual quality, codec/transport delay excluded; "
            "deployment: model codec buffering delay separately from inference latency"
        ),
    )
    parser.add_argument(
        "--h264-deployment-buffer-frames",
        type=int,
        default=0,
        help="Deployment-mode codec buffer delay in control frames (not added to latency_s)",
    )
    parser.add_argument(
        "--image-sweep",
        choices=["resolution", "jpeg", "h264"],
        default=None,
        help="Run an isolated visual-quality smoke sweep (separate from latency)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repo root containing vendor/ and .cache/",
    )


def _add_export_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "export",
        help="Export verified ACT tensor path to ONNX; FP32 parity then optional quant variants",
    )
    parser.add_argument(
        "--candidate",
        default="act_aloha_transfer",
        choices=sorted(CANDIDATES),
        help="Pinned candidate (ACT exports ONNX; Diffusion records unsupported ONNX)",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cpu")
    parser.add_argument(
        "--calibration-seeds",
        type=str,
        default=None,
        help="Comma-separated calibration seeds (disjoint from eval)",
    )
    parser.add_argument(
        "--eval-seeds",
        type=str,
        default=None,
        help="Comma-separated held-out evaluation seeds",
    )
    parser.add_argument(
        "--closed-loop-seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for paired closed-loop compare",
    )
    parser.add_argument(
        "--synthetic-corpus",
        action="store_true",
        help="Skip simulator capture; use shape-matched synthetic observations",
    )
    parser.add_argument("--no-int8", action="store_true")
    parser.add_argument("--no-int4", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--no-closed-loop", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=None)


def _add_profile_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "profile",
        help=(
            "Measure ONNX Runtime policy inference latency on this device "
            "(no simulator install required)"
        ),
    )
    parser.add_argument(
        "--candidate",
        default="act_aloha_transfer",
        choices=sorted(CANDIDATES),
        help="Pinned candidate (Diffusion ORT profiling is unsupported)",
    )
    parser.add_argument("--model", type=Path, required=True, help="Exported ONNX model path")
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Representative recorded inputs (.npz from breakpoint export)",
    )
    parser.add_argument(
        "--reference-chunks",
        type=Path,
        default=None,
        help="Optional reference action chunks for pre-timing output validation",
    )
    parser.add_argument("--device-label", type=str, required=True, help="e.g. rpi5, mac_m3")
    parser.add_argument(
        "--recipe-id",
        type=str,
        required=True,
        help="Pinned precision recipe id (e.g. torch_onnx_export_fp32_opset17)",
    )
    parser.add_argument("--variant-label", type=str, default="onnx_fp32")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-warmup", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--n-measured", type=int, default=DEFAULT_MEASURED)
    parser.add_argument(
        "--providers",
        type=str,
        default="CPUExecutionProvider",
        help="Comma-separated ORT execution providers",
    )
    parser.add_argument("--intra-op-threads", type=int, default=None)
    parser.add_argument("--inter-op-threads", type=int, default=None)
    parser.add_argument("--chunk-horizon", type=int, default=None)
    parser.add_argument("--hub-revision", type=str, default=None)
    parser.add_argument("--cooling-note", type=str, default=None)
    parser.add_argument("--power-note", type=str, default=None)
    parser.add_argument("--load-note", type=str, default=None)
    parser.add_argument("--repo-root", type=Path, default=None)


def _add_plan_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "plan",
        help="Write a fixed isolated-axis experiment plan (paired seeds + condition grid)",
    )
    parser.add_argument(
        "--candidate",
        default="act_aloha_transfer",
        choices=sorted(CANDIDATES),
    )
    parser.add_argument(
        "--axis",
        required=True,
        choices=[a.value for a in ExperimentAxis if a is not ExperimentAxis.COMBINED],
    )
    parser.add_argument(
        "--phase",
        choices=[p.value for p in PlanPhase],
        default=PlanPhase.EXPLORATORY.value,
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=None,
        help=f"Default full={N_EPISODES_FULL}, smoke={N_EPISODES_SMOKE}",
    )
    parser.add_argument("--env-seed-start", type=int, default=1000)
    parser.add_argument("--policy-seed-offset", type=int, default=1_000_000)
    parser.add_argument("--confirms-plan-id", type=str, default=None)
    parser.add_argument("--plan-id", type=str, default=None)
    parser.add_argument("--output", type=Path, required=True, help="Path for experiment_plan.json")


def _add_report_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "report",
        help="Render summary.md / report.html / plots from a saved RUN_DIR (no sim/model)",
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory containing run_manifest.json and/or run_report.json records",
    )


def _add_summarize_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "summarize",
        help="Summarize existing run_report.json records against an experiment plan",
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        metavar="ID=PATH",
        help="Map condition_id to run_report.json (repeatable)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=DEFAULT_BREAKPOINT_RATIO,
        help="Below-threshold when p_c < ratio * p_baseline (default 0.8)",
    )
    parser.add_argument(
        "--baseline-success-floor",
        type=float,
        default=DEFAULT_BASELINE_SUCCESS_FLOOR,
        help="Weak-baseline floor for headline suppression (default 0.5)",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--output", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="breakpoint", description="Breakpoint evaluation CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(sub)
    _add_report_parser(sub)
    _add_export_parser(sub)
    _add_profile_parser(sub)
    _add_plan_parser(sub)
    _add_summarize_parser(sub)
    return parser


def _build_image_cfg(args: argparse.Namespace) -> ImageDegradeConfig:
    mode = ImageMode(args.image_mode)
    capture_h = capture_w = None
    if args.capture_hw is not None:
        capture_h, capture_w = parse_capture_hw(args.capture_hw)
    frame_rate_hz = None
    if mode is ImageMode.H264:
        # Default control rate matches ALOHA; overridden when env fps is known at run time.
        frame_rate_hz = 50.0
    return ImageDegradeConfig(
        mode=mode,
        capture_height=capture_h,
        capture_width=capture_w,
        jpeg_quality=args.jpeg_quality,
        frame_rate_hz=frame_rate_hz,
        h264_bitrate_kbps=args.h264_bitrate_kbps,
        h264_gop=args.h264_gop,
        h264_preset=args.h264_preset,
        h264_latency_mode=args.h264_latency_mode,
        h264_deployment_buffer_frames=args.h264_deployment_buffer_frames,
    )


def _image_sweep_cfgs(kind: str) -> list[tuple[str, ImageDegradeConfig]]:
    if kind == "resolution":
        return resolution_smoke_configs()
    if kind == "jpeg":
        return jpeg_smoke_configs()
    if kind == "h264":
        from breakpoint_eval.h264_ffmpeg import require_ffmpeg_h264

        require_ffmpeg_h264()
        return h264_smoke_configs()
    raise ValueError(kind)


def _cmd_export(args: argparse.Namespace) -> int:
    repo_root = (args.repo_root or repo_root_from_package()).resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = repo_root / "artifacts" / "export" / args.candidate
    try:
        calib = parse_seed_list(args.calibration_seeds) if args.calibration_seeds else None
        eval_seeds = parse_seed_list(args.eval_seeds) if args.eval_seeds else None
        closed = parse_seed_list(args.closed_loop_seeds) if args.closed_loop_seeds else None
        manifest = run_export(
            ExportRequest(
                candidate=args.candidate,
                output_dir=output_dir.resolve(),
                device=args.device,
                calibration_seeds=calib,
                eval_seeds=eval_seeds,
                closed_loop_seeds=closed,
                prefer_simulator_corpus=not args.synthetic_corpus,
                attempt_int8=not args.no_int8,
                attempt_int4=not args.no_int4,
                attempt_fp16=not args.no_fp16,
                run_closed_loop=not args.no_closed_loop,
                repo_root=repo_root,
            )
        )
    except (ImportError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 2
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    print(manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "onnx_fp32_parity_passed": payload.get("comparisons", {})
                .get("onnx_fp32_vs_pytorch_fp32", {})
                .get("passed"),
                "variants": [
                    {"label": v["label"], "status": v["status"], "recipe_id": v["recipe_id"]}
                    for v in payload.get("variants", [])
                ],
                "closed_loop_status": payload.get("closed_loop_onnx_fp32_vs_pytorch", {}).get(
                    "status"
                ),
            }
        )
    )
    if not payload.get("comparisons", {}).get("onnx_fp32_vs_pytorch_fp32", {}).get("passed"):
        return 1
    return 0


def _resolve_latency_profile(
    args: argparse.Namespace,
    repo_root: Path,
    *,
    candidate: str | None = None,
):
    profile_device = args.profile_device
    if args.device == "rpi5":
        if profile_device is not None and profile_device != "rpi5":
            raise ValueError("--device rpi5 conflicts with --profile-device")
        profile_device = "rpi5"
    if args.latency_profile is not None and profile_device is not None:
        raise ValueError("use only one of --latency-profile or --device/--profile-device replay")
    if args.latency_profile is not None:
        return load_latency_profile(Path(args.latency_profile).resolve())
    if profile_device is not None:
        cand = candidate or args.candidate
        if cand is None:
            raise ValueError("candidate required to resolve device latency profile")
        profiles_dir = repo_root / "profiles" / profile_device
        path = require_unique_device_profile(
            profiles_dir,
            device_label=profile_device,
            candidate=cand,
            variant_label=args.profile_variant,
            precision_recipe=args.profile_recipe_id,
        )
        return load_latency_profile(path)
    return None


def _host_compute_device(args: argparse.Namespace) -> str:
    if args.device == "rpi5":
        return args.compute_device or "cpu"
    if args.compute_device is not None and args.compute_device != args.device:
        raise ValueError("--compute-device only applies with --device rpi5")
    return args.device


def _cmd_profile(args: argparse.Namespace) -> int:
    repo_root = (args.repo_root or repo_root_from_package()).resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = repo_root / "profiles" / args.device_label
    providers = tuple(p.strip() for p in args.providers.split(",") if p.strip())
    if not providers:
        logging.error("--providers is empty")
        return 2
    try:
        path = run_profile(
            ProfileCliRequest(
                model_path=args.model.resolve(),
                corpus_path=args.corpus.resolve(),
                output_dir=output_dir.resolve(),
                device_label=args.device_label,
                candidate=args.candidate,
                recipe_id=args.recipe_id,
                variant_label=args.variant_label,
                reference_chunks_path=(
                    args.reference_chunks.resolve() if args.reference_chunks else None
                ),
                n_warmup=args.n_warmup,
                n_measured=args.n_measured,
                providers=providers,
                intra_op_num_threads=args.intra_op_threads,
                inter_op_num_threads=args.inter_op_threads,
                cooling_note=args.cooling_note,
                power_note=args.power_note,
                load_note=args.load_note,
                hub_revision=args.hub_revision,
                chunk_horizon=args.chunk_horizon,
            )
        )
    except (ImportError, FileNotFoundError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 2
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("measurements", {}).get("summary", {})
    print(path)
    print(
        json.dumps(
            {
                "profile": str(path),
                "device_label": payload.get("device_label"),
                "n_warmup": payload.get("measurements", {}).get("n_warmup"),
                "n_measured": summary.get("count"),
                "median_s": summary.get("median_s"),
                "p95_s": summary.get("p95_s"),
                "p99_s": summary.get("p99_s"),
                "validation_passed": payload.get("output_validation", {}).get("passed"),
                "timed_boundary": payload.get("timed_boundary"),
            }
        )
    )
    return 0


def _cmd_run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    model = args.model or args.candidate
    if model is None:
        parser.error("provide a positional MODEL or --candidate")
    if args.model and args.candidate and args.model != args.candidate:
        # Allow hub repo positional with --candidate omitted; conflict if both differ.
        from breakpoint_eval.model_resolve import resolve_model_to_candidate

        try:
            resolved_key, _ = resolve_model_to_candidate(args.model)
        except ValueError as exc:
            parser.error(str(exc))
        if resolved_key != args.candidate:
            parser.error("positional model and --candidate disagree")

    if args.latency_block_length is not None and args.latency_block_seed is None:
        parser.error("--latency-block-length requires --latency-block-seed")
    if args.latency_block_seed is not None and args.latency_block_length is None:
        parser.error("--latency-block-seed requires --latency-block-length")
    if args.latency_ms < 0:
        parser.error("--latency-ms must be >= 0")

    repo_root = (args.repo_root or repo_root_from_package()).resolve()
    seeds_arg = parse_seed_list(args.seeds) if args.seeds is not None else None
    n_episodes = args.n_episodes
    if n_episodes is None and seeds_arg is not None:
        n_episodes = len(seeds_arg)
    if n_episodes is None:
        n_episodes = 1

    try:
        seeds = build_seeds(
            n_episodes=n_episodes,
            seeds=seeds_arg,
            start_seed=args.start_seed,
        )
    except ValueError as exc:
        parser.error(str(exc))

    # Resolve candidate early for default output path.
    from breakpoint_eval.model_resolve import resolve_model_to_candidate

    try:
        candidate_key, _spec = resolve_model_to_candidate(model)
    except ValueError as exc:
        logging.error("%s", exc)
        return 2

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = repo_root / "artifacts" / "runs" / candidate_key
    output_dir = output_dir.resolve()

    using_sweep = bool(args.latency_sweep_ms or args.image_sweep)
    using_profile = (
        args.latency_profile is not None or args.profile_device is not None or args.device == "rpi5"
    )
    if using_profile and (args.latency_ms != 0.0 or args.latency_sweep_ms):
        parser.error(
            "constant --latency-ms/--latency-sweep-ms is distinct from measured "
            "device replay; do not combine with --latency-profile/--device rpi5"
        )

    # Preferred orchestration path (single condition / dry-run / rpi5 / explicit profile).
    if not using_sweep:
        try:
            plan, profile = resolve_run_plan(
                model=model,
                n_episodes=n_episodes,
                output_dir=output_dir,
                device=args.device,
                compute_device=args.compute_device,
                latency_ms=args.latency_ms,
                latency_profile=args.latency_profile,
                profile_device=args.profile_device,
                profile_variant=args.profile_variant,
                profile_recipe_id=args.profile_recipe_id,
                seeds=seeds,
                start_seed=args.start_seed,
                image_mode=args.image_mode,
                repo_root=repo_root,
                dry_run=args.dry_run,
            )
        except MissingPrerequisiteError as exc:
            logging.error("%s", exc)
            return 2
        except ValueError as exc:
            logging.error("%s", exc)
            return 2

        try:
            base_image_cfg = _build_image_cfg(args)
        except ValueError as exc:
            parser.error(str(exc))
        if base_image_cfg.mode is ImageMode.H264:
            try:
                from breakpoint_eval.h264_ffmpeg import require_ffmpeg_h264

                require_ffmpeg_h264()
            except Exception as exc:  # noqa: BLE001
                logging.error("%s", exc)
                return 2

        manifest = write_run_manifest(plan)
        print(manifest)
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        if args.dry_run:
            return 0

        try:
            report, exit_code = execute_resolved_run(
                plan,
                profile,
                save_video=args.video,
                image_cfg=base_image_cfg,
                latency_end_of_trace=args.latency_end_of_trace,
                latency_block_length=args.latency_block_length,
                latency_block_seed=args.latency_block_seed,
            )
        except KeyboardInterrupt:
            return 130
        print(report)
        print(
            json.dumps(
                {
                    "latency_mode": plan.latency_mode,
                    "latency_profile": plan.latency_profile_path,
                    "chunk_arrival_policy": CHUNK_ARRIVAL_POLICY,
                    "report": str(report),
                    "exit_code": exit_code,
                }
            )
        )
        return exit_code

    # Sweep path (constant-delay or image sweeps) — still writes a run_manifest.
    try:
        compute_device = _host_compute_device(args)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        latency_profile = _resolve_latency_profile(args, repo_root, candidate=candidate_key)
    except (ValueError, FileNotFoundError, OSError) as exc:
        logging.error("%s", exc)
        return 2

    try:
        base_image_cfg = _build_image_cfg(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.latency_sweep_ms:
        latency_values_ms = [
            float(x.strip()) for x in args.latency_sweep_ms.split(",") if x.strip()
        ]
        if not latency_values_ms:
            parser.error("--latency-sweep-ms is empty")
        if any(v < 0 for v in latency_values_ms):
            parser.error("--latency-sweep-ms values must be >= 0")
    else:
        latency_values_ms = [args.latency_ms]

    if args.image_sweep:
        if (
            args.image_mode != ImageMode.NATIVE.value
            or args.capture_hw
            or args.jpeg_quality
            or args.h264_bitrate_kbps
        ):
            parser.error("--image-sweep is mutually exclusive with explicit image mode flags")
        try:
            image_jobs = [(name, cfg) for name, cfg in _image_sweep_cfgs(args.image_sweep)]
        except Exception as exc:  # noqa: BLE001 — surface missing FFmpeg clearly
            from breakpoint_eval.h264_ffmpeg import H264UnsupportedError

            if isinstance(exc, H264UnsupportedError) or args.image_sweep == "h264":
                logging.error("%s", exc)
                return 2
            raise
    else:
        image_jobs = [(None, base_image_cfg)]
        if base_image_cfg.mode is ImageMode.H264:
            try:
                from breakpoint_eval.h264_ffmpeg import require_ffmpeg_h264

                require_ffmpeg_h264()
            except Exception as exc:  # noqa: BLE001
                logging.error("%s", exc)
                return 2

    if len(latency_values_ms) > 1 and len(image_jobs) > 1:
        parser.error("combine only one of --latency-sweep-ms or --image-sweep per invocation")

    try:
        plan, _profile = resolve_run_plan(
            model=model,
            n_episodes=n_episodes,
            output_dir=output_dir,
            device=compute_device,
            compute_device=compute_device,
            latency_ms=0.0,
            seeds=seeds,
            repo_root=repo_root,
            dry_run=args.dry_run,
        )
    except (ValueError, MissingPrerequisiteError) as exc:
        logging.error("%s", exc)
        return 2
    plan.notes.append("sweep orchestration: constant-delay and/or image-mode grid")
    write_run_manifest(
        plan,
        extra={
            "latency_sweep_ms": latency_values_ms,
            "image_sweep": args.image_sweep,
            "n_jobs": len(latency_values_ms) * len(image_jobs),
        },
    )
    print(output_dir / "run_manifest.json")
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    exit_code = 0
    completed: list[dict[str, Any]] = []
    try:
        for latency_ms in latency_values_ms:
            for image_name, image_cfg in image_jobs:
                run_dir = output_dir
                if len(latency_values_ms) > 1:
                    run_dir = output_dir / f"latency_ms_{latency_ms:g}"
                if image_name is not None:
                    run_dir = output_dir / image_name
                report = run_baseline(
                    RunRequest(
                        candidate=candidate_key,
                        seeds=seeds,
                        output_dir=run_dir,
                        device=compute_device,
                        save_video=args.video,
                        latency_s=latency_ms / 1000.0,
                        latency_profile=latency_profile,
                        latency_end_of_trace=args.latency_end_of_trace,
                        latency_block_length=args.latency_block_length,
                        latency_block_seed=args.latency_block_seed,
                        image_cfg=image_cfg,
                        repo_root=repo_root,
                    )
                )
                print(report)
                print(
                    json.dumps(
                        {
                            "latency_ms": latency_ms,
                            "latency_mode": "constant",
                            "image_mode": image_cfg.mode.value,
                            "image_label": image_name,
                            "chunk_arrival_policy": CHUNK_ARRIVAL_POLICY,
                            "report": str(report),
                        }
                    )
                )
                payload = json.loads(report.read_text(encoding="utf-8"))
                summary = payload.get("summary", {})
                provenance = payload.get("provenance", {})
                completed.append({"report": str(report), "summary": summary})
                if provenance.get("env_created") is False:
                    exit_code = 2
                if (
                    summary.get("n_runtime_error", 0) > 0
                    and summary.get("n_episodes_eligible", 0) == 0
                ):
                    exit_code = 2
    except KeyboardInterrupt:
        write_partial_diagnostics(
            output_dir,
            plan=plan.to_dict(),
            completed=completed,
            error="KeyboardInterrupt",
            interrupted=True,
        )
        return 130
    return exit_code


def _cmd_report(args: argparse.Namespace) -> int:
    try:
        bundle = render_report(args.run_dir.resolve())
    except (FileNotFoundError, ImportError, ValueError, OSError) as exc:
        logging.error("%s", exc)
        return 2
    print(bundle.report_html)
    print(
        json.dumps(
            {
                "summary_md": str(bundle.summary_md),
                "report_html": str(bundle.report_html),
                "plots_dir": str(bundle.plots_dir),
                "index": bundle.index,
            }
        )
    )
    return 0


def _parse_condition_maps(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected ID=PATH, got {item!r}")
        cid, path_s = item.split("=", 1)
        cid = cid.strip()
        if not cid:
            raise ValueError(f"empty condition id in {item!r}")
        out[cid] = Path(path_s).expanduser().resolve()
    return out


def _cmd_plan(args: argparse.Namespace) -> int:
    try:
        plan = make_isolated_plan(
            axis=ExperimentAxis(args.axis),
            candidate=args.candidate,
            phase=PlanPhase(args.phase),
            n_episodes=args.n_episodes,
            env_seed_start=args.env_seed_start,
            policy_seed_offset=args.policy_seed_offset,
            plan_id=args.plan_id,
            confirms_plan_id=args.confirms_plan_id,
        )
        path = write_experiment_plan(plan, args.output.resolve())
    except ValueError as exc:
        logging.error("%s", exc)
        return 2
    print(path)
    print(
        json.dumps(
            {
                "plan_id": plan.plan_id,
                "axis": plan.axis,
                "phase": plan.phase,
                "n_episodes": plan.n_episodes,
                "n_conditions": len(plan.conditions),
                "baseline_condition_id": plan.baseline_condition_id,
            }
        )
    )
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    try:
        mapping = _parse_condition_maps(args.condition)
        summary = summarize_from_plan_file(
            args.plan.resolve(),
            mapping,
            output_path=args.output.resolve(),
            ratio=args.ratio,
            baseline_success_floor=args.baseline_success_floor,
            bootstrap_seed=args.bootstrap_seed,
        )
    except (ValueError, OSError, KeyError) as exc:
        logging.error("%s", exc)
        return 2
    axis = summary.get("axis_summary") or {}
    print(args.output.resolve())
    print(
        json.dumps(
            {
                "plan_id": summary.get("plan_id"),
                "phase": summary.get("phase"),
                "headline_suppressed": axis.get("headline_suppressed"),
                "breakpoint": axis.get("breakpoint"),
                "n_conditions": len(summary.get("conditions") or []),
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _cmd_run(parser, args)
    if args.command == "report":
        return _cmd_report(args)
    if args.command == "export":
        return _cmd_export(args)
    if args.command == "profile":
        return _cmd_profile(args)
    if args.command == "plan":
        return _cmd_plan(args)
    if args.command == "summarize":
        return _cmd_summarize(args)
    parser.error(f"Unsupported command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
