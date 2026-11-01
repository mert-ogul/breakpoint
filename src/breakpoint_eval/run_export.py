"""Orchestrate ACT ONNX export, corpus, FP32 parity, and quantized variants."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from breakpoint_eval.act_export import (
    DEFAULT_OPSET,
    RECIPE_ONNX_FP32,
    OrtActChunkSession,
    OrtPolicyAdapter,
    VariantArtifact,
    build_act_chunk_module,
    chunk_info_from_policy,
    export_act_onnx_fp32,
    inspect_onnx_graph,
    require_onnx,
    write_json,
)
from breakpoint_eval.export_corpus import (
    PLANNED_CORPUS_CONTROL_STEPS,
    SYNTHETIC_SOURCE,
    SYNTHETIC_UNSUITABLE_NOTE,
    build_synthetic_corpus,
    capture_simulator_rollout_corpus,
    chunk_dict_by_sample_id,
    save_corpus,
    write_corpus_sidecar,
)
from breakpoint_eval.export_parity import (
    ParityTolerances,
    evaluate_corpus_parity,
    paired_closed_loop_rollouts,
    run_ort_chunks,
    run_pytorch_chunks,
)
from breakpoint_eval.export_quantize import (
    quantize_int4_weight_only,
    quantize_int8_static_qdq,
    try_fp16_convert,
)
from breakpoint_eval.export_tolerances import (
    ONNX_FP32_CLOSED_LOOP_ATOL,
    ONNX_FP32_CLOSED_LOOP_RTOL,
    ONNX_FP32_PARITY_ACTION_UNIT_MEAN_ABS,
    ONNX_FP32_PARITY_ATOL,
    ONNX_FP32_PARITY_MAX_ABS,
    ONNX_FP32_PARITY_MEAN_ABS,
    ONNX_FP32_PARITY_RTOL,
)
from breakpoint_eval.policy_adapter import PolicyAdapter
from breakpoint_eval.sim_env import control_period_s, load_policy, make_vector_env
from breakpoint_eval.stack import (
    LEROBOT_COMMIT,
    download_candidate_checkpoint,
    ensure_lerobot_on_path,
    repo_root_from_package,
    resolve_candidate,
)

LOGGER = logging.getLogger("breakpoint_eval.run_export")

# Disjoint seed pools (calibration never overlaps evaluation / diagnosis).
# Closed-loop diagnosis uses 2100–2101 via diagnose-closed-loop (not export eval).
DEFAULT_CALIBRATION_SEEDS = list(range(9000, 9008))
DEFAULT_EVAL_SEEDS = list(range(2000, 2008))
DEFAULT_CLOSED_LOOP_SEEDS = [2100, 2101]


@dataclass(frozen=True)
class ExportRequest:
    candidate: str = "act_aloha_transfer"
    output_dir: Path | None = None
    device: str = "cpu"
    calibration_seeds: list[int] | None = None
    eval_seeds: list[int] | None = None
    closed_loop_seeds: list[int] | None = None
    prefer_simulator_corpus: bool = True
    samples_per_episode: int = 2
    attempt_int8: bool = True
    # Release path disables INT4/FP16 by default; opt in explicitly if needed.
    attempt_int4: bool = False
    attempt_fp16: bool = False
    run_closed_loop: bool = True
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    repo_root: Path | None = None
    opset: int = DEFAULT_OPSET


def _assert_disjoint(calib: list[int], eval_seeds: list[int]) -> None:
    overlap = set(calib) & set(eval_seeds)
    if overlap:
        raise ValueError(f"calibration and evaluation seeds overlap: {sorted(overlap)}")


def build_export_validation(
    *,
    open_loop_passed: bool,
    closed_loop: dict[str, Any],
    closed_loop_requested: bool,
    corpus_suitable_as_deployment_evidence: bool,
) -> dict[str, Any]:
    """Assemble export_validation. ``passed`` means deployment evidence, not diagnostics."""
    closed_loop_completed = bool(closed_loop.get("completed"))
    closed_loop_passed = closed_loop.get("passed")
    diagnostic_parity_ok = bool(open_loop_passed)
    deployment_passed = (
        open_loop_passed
        and corpus_suitable_as_deployment_evidence
        and closed_loop_completed
        and closed_loop_passed is True
    )
    reasons: list[str] = []
    if not open_loop_passed:
        reasons.append("open_loop_fp32_parity_failed")
    if not corpus_suitable_as_deployment_evidence:
        reasons.append("corpus_not_suitable_as_deployment_evidence")
    if not closed_loop_requested:
        reasons.append("closed_loop_not_requested")
    elif not closed_loop_completed:
        reasons.append("closed_loop_not_completed")
    elif closed_loop_passed is not True:
        reasons.append("closed_loop_not_passed")
        kind = closed_loop.get("failure_kind")
        if kind:
            reasons.append(f"closed_loop_failure_kind={kind}")
    return {
        "open_loop_fp32_parity_passed": open_loop_passed,
        "closed_loop_requested": closed_loop_requested,
        "closed_loop_completed": closed_loop_completed,
        "closed_loop_passed": closed_loop_passed,
        "closed_loop_failure_kind": closed_loop.get("failure_kind"),
        "closed_loop_comparison_kind": closed_loop.get("comparison_kind"),
        "corpus_suitable_as_deployment_evidence": corpus_suitable_as_deployment_evidence,
        "tensor_only_is_not_closed_loop_validation": True,
        "diagnostic_parity_ok": diagnostic_parity_ok,
        "passed": deployment_passed,
        "not_passed_reasons": [] if deployment_passed else reasons,
        "note": (
            "Export validation.passed is true only for deployment evidence: simulator "
            "corpus + FP32 open-loop parity + completed closed-loop pass. "
            "diagnostic_parity_ok may be true for synthetic or skipped-closed-loop "
            "diagnostic exports without counting as validated deployment evidence. "
            "Closed-loop compares independently controlled episodes (same seeds, "
            "separate resets); it is not same-observation numerical export parity."
        ),
    }


def run_export(req: ExportRequest) -> Path:
    spec = resolve_candidate(req.candidate)
    repo_root = (req.repo_root or repo_root_from_package()).resolve()
    ensure_lerobot_on_path(repo_root)
    output_dir = (req.output_dir or (repo_root / "artifacts" / "export" / req.candidate)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if spec.policy_type == "diffusion":
        from breakpoint_eval.diffusion_export import run_diffusion_export_record

        # ONNX/device/precision unsupported; PyTorch axes remain. No ACT timing reuse.
        return run_diffusion_export_record(
            candidate=req.candidate,
            output_dir=output_dir,
            device=req.device,
            repo_root=repo_root,
        )
    if spec.policy_type != "act":
        raise ValueError(
            f"breakpoint export supports act and diffusion candidates; "
            f"got {req.candidate!r} policy_type={spec.policy_type}"
        )

    require_onnx()

    calib_seeds = list(req.calibration_seeds or DEFAULT_CALIBRATION_SEEDS)
    eval_seeds = list(req.eval_seeds or DEFAULT_EVAL_SEEDS)
    closed_seeds = list(req.closed_loop_seeds or DEFAULT_CLOSED_LOOP_SEEDS)
    _assert_disjoint(calib_seeds, eval_seeds)
    _assert_disjoint(calib_seeds, closed_seeds)

    # Tolerances recorded into the manifest before any numeric comparison below.
    declared_tolerances = {
        "onnx_fp32_parity_atol": ONNX_FP32_PARITY_ATOL,
        "onnx_fp32_parity_rtol": ONNX_FP32_PARITY_RTOL,
        "onnx_fp32_parity_max_abs": ONNX_FP32_PARITY_MAX_ABS,
        "onnx_fp32_parity_mean_abs": ONNX_FP32_PARITY_MEAN_ABS,
        "onnx_fp32_parity_action_unit_mean_abs": ONNX_FP32_PARITY_ACTION_UNIT_MEAN_ABS,
        "onnx_fp32_closed_loop_atol": ONNX_FP32_CLOSED_LOOP_ATOL,
        "onnx_fp32_closed_loop_rtol": ONNX_FP32_CLOSED_LOOP_RTOL,
        "source_module": "breakpoint_eval.export_tolerances",
    }

    checkpoint_dir = download_candidate_checkpoint(spec, repo_root / ".cache" / "huggingface")
    period_s = control_period_s(spec)
    policy = load_policy(spec, str(checkpoint_dir), req.device)
    module = build_act_chunk_module(policy)
    chunk_info = chunk_info_from_policy(policy, period_s)

    corpus_meta: dict[str, Any] = {}
    eval_samples = []
    calib_samples = []
    if req.prefer_simulator_corpus:
        capture_horizon = max(PLANNED_CORPUS_CONTROL_STEPS) + 1
        eval_samples, eval_meta = capture_simulator_rollout_corpus(
            seeds=eval_seeds,
            candidate=req.candidate,
            repo_root=repo_root,
            policy=policy,
            control_period_s=period_s,
            device=req.device,
            samples_per_episode=req.samples_per_episode,
            capture_horizon_steps=capture_horizon,
            planned_steps=PLANNED_CORPUS_CONTROL_STEPS,
        )
        calib_samples, calib_meta = capture_simulator_rollout_corpus(
            seeds=calib_seeds,
            candidate=req.candidate,
            repo_root=repo_root,
            policy=policy,
            control_period_s=period_s,
            device=req.device,
            samples_per_episode=req.samples_per_episode,
            capture_horizon_steps=capture_horizon,
            planned_steps=PLANNED_CORPUS_CONTROL_STEPS,
        )
        corpus_meta["evaluation_capture"] = eval_meta
        corpus_meta["calibration_capture"] = calib_meta
        if not eval_samples:
            raise RuntimeError(
                "simulator evaluation corpus capture failed; refusing synthetic fallback. "
                f"detail={eval_meta}. Pass --synthetic-corpus only for non-deployment tests."
            )
        if not calib_samples:
            raise RuntimeError(
                "simulator calibration corpus capture failed; refusing synthetic fallback. "
                f"detail={calib_meta}. Pass --synthetic-corpus only for non-deployment tests."
            )
        corpus_meta["corpus_mode"] = "simulator_rollout"
        corpus_meta["suitable_as_deployment_evidence"] = True
    else:
        eval_samples = build_synthetic_corpus(eval_seeds)
        calib_samples = build_synthetic_corpus(calib_seeds)
        corpus_meta["corpus_mode"] = SYNTHETIC_SOURCE
        corpus_meta["suitable_as_deployment_evidence"] = False
        corpus_meta["synthetic_note"] = SYNTHETIC_UNSUITABLE_NOTE

    eval_manifest = save_corpus(output_dir / "corpus_eval.npz", eval_samples, role="evaluation")
    calib_manifest = save_corpus(
        output_dir / "corpus_calibration.npz", calib_samples, role="calibration"
    )
    write_corpus_sidecar(eval_manifest, output_dir / "corpus_eval.json")
    write_corpus_sidecar(calib_manifest, output_dir / "corpus_calibration.json")

    example = eval_samples[0]
    example_images = torch.from_numpy(example.images_top)
    example_state = torch.from_numpy(example.state)
    if req.device != "cpu":
        example_images = example_images.to(req.device)
        example_state = example_state.to(req.device)
        module = module.to(req.device)

    fp32_path = output_dir / "act_chunk_fp32.onnx"
    io_spec = export_act_onnx_fp32(
        module,
        output_path=fp32_path,
        example_images=example_images.cpu(),
        example_state=example_state.cpu(),
        opset=req.opset,
    )
    # Keep module on CPU for parity vs ORT CPU.
    module = module.to("cpu").eval()
    fp32_graph = inspect_onnx_graph(fp32_path)
    providers = list(req.providers)
    fp32_session = OrtActChunkSession(fp32_path, providers=providers)

    pytorch_chunks = run_pytorch_chunks(module, eval_samples)
    ort_fp32_chunks = run_ort_chunks(fp32_session, eval_samples)
    fp32_vs_pytorch = evaluate_corpus_parity(
        reference_chunks=pytorch_chunks,
        candidate_chunks=ort_fp32_chunks,
        seeds=[s.seed for s in eval_samples],
        sample_ids=[s.sample_id for s in eval_samples],
        reference_label="pytorch_fp32",
        candidate_label="onnx_fp32",
        tolerances=ParityTolerances(),
        enforce_fp32_gates=True,
    )
    if not fp32_vs_pytorch.passed:
        LOGGER.error(
            "ONNX FP32 parity failed (investigate export first): max_abs=%s mean_abs=%s",
            fp32_vs_pytorch.max_abs_error,
            fp32_vs_pytorch.mean_abs_error,
        )

    variants: list[VariantArtifact] = [
        VariantArtifact(
            label="onnx_fp32",
            recipe_id=RECIPE_ONNX_FP32,
            path=str(fp32_path),
            status="ok" if fp32_vs_pytorch.passed else "errored",
            detail=None if fp32_vs_pytorch.passed else "failed declared FP32 parity gates",
            graph=fp32_graph,
            effective_weight_precision="float32",
            effective_activation_precision="float32",
            quantization_coverage={"full_model_int8_claim": False},
            runtime={
                "providers": providers,
                "graph_sha256": fp32_graph["sha256"],
            },
        )
    ]

    comparisons: dict[str, Any] = {
        "onnx_fp32_vs_pytorch_fp32": fp32_vs_pytorch.to_dict(),
    }

    # Quantized variants: compare to ONNX FP32 baseline; show PyTorch separately.
    if req.attempt_int8:
        int8 = quantize_int8_static_qdq(
            fp32_path,
            output_dir / "act_chunk_int8_qdq.onnx",
            calibration_samples=calib_samples,
            calibration_sha256=calib_manifest.sha256,
            providers=providers,
        )
        variants.append(int8)
        if int8.status == "ok" and int8.path:
            int8_sess = OrtActChunkSession(Path(int8.path), providers=providers)
            int8_chunks = run_ort_chunks(int8_sess, eval_samples)
            comparisons["int8_vs_onnx_fp32"] = evaluate_corpus_parity(
                reference_chunks=ort_fp32_chunks,
                candidate_chunks=int8_chunks,
                seeds=[s.seed for s in eval_samples],
                sample_ids=[s.sample_id for s in eval_samples],
                reference_label="onnx_fp32",
                candidate_label="int8",
                enforce_fp32_gates=False,
            ).to_dict()
            comparisons["int8_vs_pytorch_fp32"] = evaluate_corpus_parity(
                reference_chunks=pytorch_chunks,
                candidate_chunks=int8_chunks,
                seeds=[s.seed for s in eval_samples],
                sample_ids=[s.sample_id for s in eval_samples],
                reference_label="pytorch_fp32",
                candidate_label="int8",
                enforce_fp32_gates=False,
            ).to_dict()
        elif int8.status != "ok":
            LOGGER.warning(
                "INT8 variant status=%s detail=%s",
                int8.status,
                int8.detail,
            )

    if req.attempt_int4:
        int4 = quantize_int4_weight_only(
            fp32_path,
            output_dir / "act_chunk_int4_matmul_nbits.onnx",
            providers=providers,
            probe_sample=eval_samples[0],
        )
        variants.append(int4)
        if int4.status == "ok" and int4.path:
            int4_sess = OrtActChunkSession(Path(int4.path), providers=providers)
            int4_chunks = run_ort_chunks(int4_sess, eval_samples)
            comparisons["int4_weight_only_vs_onnx_fp32"] = evaluate_corpus_parity(
                reference_chunks=ort_fp32_chunks,
                candidate_chunks=int4_chunks,
                seeds=[s.seed for s in eval_samples],
                sample_ids=[s.sample_id for s in eval_samples],
                reference_label="onnx_fp32",
                candidate_label="int4_weight_only",
                enforce_fp32_gates=False,
            ).to_dict()
            comparisons["int4_weight_only_vs_pytorch_fp32"] = evaluate_corpus_parity(
                reference_chunks=pytorch_chunks,
                candidate_chunks=int4_chunks,
                seeds=[s.seed for s in eval_samples],
                sample_ids=[s.sample_id for s in eval_samples],
                reference_label="pytorch_fp32",
                candidate_label="int4_weight_only",
                enforce_fp32_gates=False,
            ).to_dict()

    if req.attempt_fp16:
        fp16 = try_fp16_convert(
            fp32_path,
            output_dir / "act_chunk_fp16.onnx",
            providers=providers,
            probe_sample=eval_samples[0],
        )
        variants.append(fp16)
        if fp16.status == "ok" and fp16.path:
            fp16_sess = OrtActChunkSession(Path(fp16.path), providers=providers)
            fp16_chunks = run_ort_chunks(fp16_sess, eval_samples)
            comparisons["fp16_vs_onnx_fp32"] = evaluate_corpus_parity(
                reference_chunks=ort_fp32_chunks,
                candidate_chunks=fp16_chunks,
                seeds=[s.seed for s in eval_samples],
                reference_label="onnx_fp32",
                candidate_label="fp16",
                enforce_fp32_gates=False,
            ).to_dict()
            comparisons["fp16_vs_pytorch_fp32"] = evaluate_corpus_parity(
                reference_chunks=pytorch_chunks,
                candidate_chunks=fp16_chunks,
                seeds=[s.seed for s in eval_samples],
                reference_label="pytorch_fp32",
                candidate_label="fp16",
                enforce_fp32_gates=False,
            ).to_dict()

    closed_loop: dict[str, Any]
    if req.run_closed_loop:
        try:
            env = make_vector_env(spec, n_envs=1)
        except (OSError, ImportError, ModuleNotFoundError, RuntimeError) as exc:
            closed_loop = {
                "status": "errored",
                "detail": f"{type(exc).__name__}: {exc}",
                "note": "closed-loop requires simulator; unknown",
                "passed": False,
                "completed": False,
            }
        else:
            try:
                policy.cpu()
                pt_adapter = PolicyAdapter(policy, control_period_s=period_s, device="cpu")
                ort_adapter = OrtPolicyAdapter(
                    fp32_session,
                    chunk_info=chunk_info,
                    image_key="observation.images.top",
                )
                report = paired_closed_loop_rollouts(
                    pytorch_adapter=pt_adapter,
                    onnx_adapter=ort_adapter,
                    env=env,
                    seeds=closed_seeds,
                )
                closed_loop = report.to_dict()
                closed_loop["completed"] = closed_loop.get("status") == "ok"
            finally:
                env.close()
    else:
        closed_loop = {
            "status": "skipped",
            "detail": "run_closed_loop=False",
            "passed": None,
            "completed": False,
            "note": ("Open-loop tensor parity alone is not completed closed-loop validation."),
        }

    export_validation = build_export_validation(
        open_loop_passed=bool(fp32_vs_pytorch.passed),
        closed_loop=closed_loop,
        closed_loop_requested=req.run_closed_loop,
        corpus_suitable_as_deployment_evidence=bool(
            corpus_meta.get("suitable_as_deployment_evidence")
        ),
    )

    import onnx
    import onnxruntime as ort

    manifest: dict[str, Any] = {
        "candidate": req.candidate,
        "hub_repo": spec.hub_repo,
        "hub_revision": spec.hub_revision,
        "lerobot_commit": LEROBOT_COMMIT,
        "python": sys.version,
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "device_policy_load": req.device,
        "opset": req.opset,
        "io_spec": {
            "image_key": io_spec.image_key,
            "image_shape": list(io_spec.image_shape),
            "state_shape": list(io_spec.state_shape),
            "action_chunk_shape": list(io_spec.action_chunk_shape),
        },
        "export_boundary": {
            "inside_graph": [
                "normalize_inputs",
                "stack_cameras",
                "ACT.model_eval_zero_latent",
                "slice_n_action_steps",
                "unnormalize_outputs",
            ],
            "outside_graph": [
                "preprocess_observation",
                "action_queue",
                "temporal_ensembler",
                "camera_degrade",
                "latency_scheduler",
            ],
        },
        "declared_tolerances_before_measurement": declared_tolerances,
        "corpus": {
            "evaluation": eval_manifest.to_dict(),
            "calibration": calib_manifest.to_dict(),
            "meta": corpus_meta,
            "seeds_disjoint": True,
        },
        "chunk_info": {
            "chunk_size": chunk_info.chunk_size,
            "n_action_steps": chunk_info.n_action_steps,
            "n_obs_steps": chunk_info.n_obs_steps,
            "temporal_ensemble_coeff": chunk_info.temporal_ensemble_coeff,
            "control_period_s": chunk_info.control_period_s,
        },
        "variants": [v.to_dict() for v in variants],
        "comparisons": comparisons,
        "closed_loop_onnx_fp32_vs_pytorch": closed_loop,
        "export_validation": export_validation,
        "notes": [
            "Precision labels alone never drive results; each variant uses its weights/runtime.",
            "INT8 recipe is static QDQ.",
            "INT4 is weight-only MatMulNBits when supported; not native INT4 activation timing.",
            "Quantized success-rate preservation is not required; valid labeled compute is.",
            "Reference chunk NPZ keys are sample_id values (seed{seed}_t{step}).",
            SYNTHETIC_UNSUITABLE_NOTE,
        ],
    }
    manifest_path = output_dir / "export_manifest.json"
    write_json(manifest_path, manifest)
    # Reference chunks keyed by sample_id for profile validation.
    np.savez_compressed(
        output_dir / "eval_chunks_pytorch_fp32.npz",
        **chunk_dict_by_sample_id(eval_samples, pytorch_chunks),
    )
    np.savez_compressed(
        output_dir / "eval_chunks_onnx_fp32.npz",
        **chunk_dict_by_sample_id(eval_samples, ort_fp32_chunks),
    )
    if req.attempt_int8:
        from breakpoint_eval.export_validation import write_int8_validation_from_export

        int8_val = write_int8_validation_from_export(output_dir)
        LOGGER.info("Wrote INT8 validation sidecar %s", int8_val)
    if req.attempt_int4:
        from breakpoint_eval.export_validation import write_int4_validation_from_export

        int4_val = write_int4_validation_from_export(output_dir)
        LOGGER.info("Wrote INT4 validation sidecar %s", int4_val)
    LOGGER.info("Wrote export manifest %s", manifest_path)
    return manifest_path
