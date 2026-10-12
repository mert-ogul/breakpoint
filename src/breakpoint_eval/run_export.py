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
    build_synthetic_corpus,
    save_corpus,
    try_build_simulator_corpus,
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

# Disjoint seed pools (calibration never overlaps evaluation).
DEFAULT_CALIBRATION_SEEDS = list(range(9000, 9008))
DEFAULT_EVAL_SEEDS = list(range(1000, 1008))
DEFAULT_CLOSED_LOOP_SEEDS = [1000]


@dataclass(frozen=True)
class ExportRequest:
    candidate: str = "act_aloha_transfer"
    output_dir: Path | None = None
    device: str = "cpu"
    calibration_seeds: list[int] | None = None
    eval_seeds: list[int] | None = None
    closed_loop_seeds: list[int] | None = None
    prefer_simulator_corpus: bool = True
    attempt_int8: bool = True
    attempt_int4: bool = True
    attempt_fp16: bool = True
    run_closed_loop: bool = True
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    repo_root: Path | None = None
    opset: int = DEFAULT_OPSET


def _assert_disjoint(calib: list[int], eval_seeds: list[int]) -> None:
    overlap = set(calib) & set(eval_seeds)
    if overlap:
        raise ValueError(f"calibration and evaluation seeds overlap: {sorted(overlap)}")


def run_export(req: ExportRequest) -> Path:
    require_onnx()
    if req.candidate != "act_aloha_transfer" and not req.candidate.startswith("act_"):
        raise ValueError(
            f"breakpoint export currently supports verified ACT adapters; got {req.candidate!r}"
        )
    spec = resolve_candidate(req.candidate)
    if spec.policy_type != "act":
        raise ValueError(f"candidate {req.candidate!r} is policy_type={spec.policy_type}, not act")

    repo_root = (req.repo_root or repo_root_from_package()).resolve()
    ensure_lerobot_on_path(repo_root)
    output_dir = (req.output_dir or (repo_root / "artifacts" / "export" / req.candidate)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
        eval_samples, eval_meta = try_build_simulator_corpus(
            seeds=eval_seeds,
            candidate=req.candidate,
            repo_root=repo_root,
            device=req.device,
        )
        calib_samples, calib_meta = try_build_simulator_corpus(
            seeds=calib_seeds,
            candidate=req.candidate,
            repo_root=repo_root,
            device=req.device,
        )
        corpus_meta["evaluation_capture"] = eval_meta
        corpus_meta["calibration_capture"] = calib_meta

    if not eval_samples:
        eval_samples = build_synthetic_corpus(eval_seeds)
        corpus_meta["evaluation_fallback"] = "synthetic_shape_matched"
    if not calib_samples:
        calib_samples = build_synthetic_corpus(calib_seeds)
        corpus_meta["calibration_fallback"] = "synthetic_shape_matched"

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
                reference_label="onnx_fp32",
                candidate_label="int8",
                enforce_fp32_gates=False,
            ).to_dict()
            comparisons["int8_vs_pytorch_fp32"] = evaluate_corpus_parity(
                reference_chunks=pytorch_chunks,
                candidate_chunks=int8_chunks,
                seeds=[s.seed for s in eval_samples],
                reference_label="pytorch_fp32",
                candidate_label="int8",
                enforce_fp32_gates=False,
            ).to_dict()

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
                reference_label="onnx_fp32",
                candidate_label="int4_weight_only",
                enforce_fp32_gates=False,
            ).to_dict()
            comparisons["int4_weight_only_vs_pytorch_fp32"] = evaluate_corpus_parity(
                reference_chunks=pytorch_chunks,
                candidate_chunks=int4_chunks,
                seeds=[s.seed for s in eval_samples],
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
                "note": "closed-loop requires simulator; not fabricated",
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
            finally:
                env.close()
    else:
        closed_loop = {"status": "skipped", "detail": "run_closed_loop=False"}

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
        "notes": [
            "Precision labels alone never drive results; each variant uses its weights/runtime.",
            "INT8 recipe is static QDQ; not silently replaced after failure.",
            "INT4 is weight-only MatMulNBits when supported; not native INT4 activation timing.",
            "Quantized success-rate preservation is not required; valid labeled compute is.",
        ],
    }
    manifest_path = output_dir / "export_manifest.json"
    write_json(manifest_path, manifest)
    # Also dump numpy reference chunks for inspection.
    np.savez_compressed(
        output_dir / "eval_chunks_pytorch_fp32.npz",
        **{f"seed_{s.seed}": c for s, c in zip(eval_samples, pytorch_chunks, strict=True)},
    )
    np.savez_compressed(
        output_dir / "eval_chunks_onnx_fp32.npz",
        **{f"seed_{s.seed}": c for s, c in zip(eval_samples, ort_fp32_chunks, strict=True)},
    )
    LOGGER.info("Wrote export manifest %s", manifest_path)
    return manifest_path
