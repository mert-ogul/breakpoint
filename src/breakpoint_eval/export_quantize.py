"""Quantization recipes for exported ACT ONNX."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from breakpoint_eval.act_export import (
    RECIPE_ORT_FP16,
    RECIPE_ORT_INT4_MATMUL_NBITS,
    RECIPE_ORT_INT8_STATIC_QDQ,
    OrtActChunkSession,
    VariantArtifact,
    create_ort_session,
    inspect_onnx_graph,
    require_onnx,
    sha256_file,
    write_json,
)
from breakpoint_eval.export_corpus import CorpusSample, load_corpus

LOGGER = logging.getLogger("breakpoint_eval.export_quantize")

_INT4_GRAPH_NAME = "act_chunk_int4_matmul_nbits.onnx"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON via a temp file in the same directory, then os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{path.stem}_",
        suffix=".json",
        dir=str(path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        write_json(tmp_path, payload)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.is_file():
            tmp_path.unlink()
        raise


def _corpus_parity_vs_reference(
    ref_path: Path,
    *,
    candidate_chunks: list[np.ndarray],
    eval_samples: list[CorpusSample],
    sample_ids: list[str],
    reference_label: str,
    candidate_label: str,
    missing_keys_message: str | None = None,
) -> dict[str, Any]:
    """Load a reference .npz, require every sample_id, return parity as a dict."""
    from breakpoint_eval.export_parity import evaluate_corpus_parity

    ref = np.load(ref_path)
    missing = [sid for sid in sample_ids if sid not in ref.files]
    if missing:
        if missing_keys_message is None:
            raise ValueError(
                f"{ref_path.name} missing sample_id keys required by evaluation corpus: "
                f"{missing[:5]}"
            )
        raise ValueError(f"{missing_keys_message}{missing[:5]}")
    reference_chunks = [np.asarray(ref[sid], dtype=np.float32) for sid in sample_ids]
    return evaluate_corpus_parity(
        reference_chunks=reference_chunks,
        candidate_chunks=candidate_chunks,
        seeds=[s.seed for s in eval_samples],
        sample_ids=sample_ids,
        reference_label=reference_label,
        candidate_label=candidate_label,
        enforce_fp32_gates=False,
    ).to_dict()


class _CorpusCalibrationReader:
    """onnxruntime CalibrationDataReader over a fixed corpus."""

    def __init__(self, samples: list[CorpusSample]) -> None:
        from onnxruntime.quantization import CalibrationDataReader

        # Subclass dynamically so import stays optional at module import time.
        self._samples = samples
        self._index = 0
        self._base = CalibrationDataReader

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._index >= len(self._samples):
            return None
        sample = self._samples[self._index]
        self._index += 1
        return {
            "observation_images_top": np.asarray(sample.images_top, dtype=np.float32),
            "observation_state": np.asarray(sample.state, dtype=np.float32),
        }


def _make_calibration_reader(samples: list[CorpusSample]) -> Any:
    from onnxruntime.quantization import CalibrationDataReader

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._inner = _CorpusCalibrationReader(samples)

        def get_next(self) -> dict[str, np.ndarray] | None:
            return self._inner.get_next()

    return Reader()


def _runtime_meta(providers: list[str]) -> dict[str, Any]:
    import onnxruntime as ort

    return {
        "onnxruntime_version": ort.__version__,
        "requested_providers": providers,
        "available_providers": ort.get_available_providers(),
    }


def _coverage_from_graph(graph: dict[str, Any]) -> dict[str, Any]:
    ops = graph.get("op_counts", {})
    quant_ops = {
        k: v
        for k, v in ops.items()
        if k
        in {
            "QuantizeLinear",
            "DequantizeLinear",
            "QLinearConv",
            "QLinearMatMul",
            "MatMulNBits",
            "ConvInteger",
            "MatMulInteger",
            "DynamicQuantizeLinear",
        }
    }
    return {
        "op_counts": ops,
        "quant_related_ops": quant_ops,
        "has_matmul_nbits": bool(ops.get("MatMulNBits")),
        "has_qdq": bool(ops.get("QuantizeLinear") or ops.get("DequantizeLinear")),
        "remaining_float_matmul": int(ops.get("MatMul", 0)),
        "remaining_float_conv": int(ops.get("Conv", 0)),
        "remaining_float_gemm": int(ops.get("Gemm", 0)),
    }


def _node_attr_int(node: Any, name: str, default: int | None = None) -> int | None:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def _float_initializer_element_count(model: Any) -> int:
    from onnx import numpy_helper

    total = 0
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype.kind == "f":
            total += int(arr.size)
    return total


def _matmul_nbits_original_name(node_name: str) -> str:
    suffix = "_Q4"
    if node_name.endswith(suffix):
        return node_name[: -len(suffix)]
    return node_name


def _build_int4_quantization_coverage(
    fp32_path: Path,
    quant_path: Path,
    *,
    block_size: int,
    bits: int,
    symmetric: bool,
    op_types_to_quantize: tuple[str, ...],
    quant_format_name: str,
) -> dict[str, Any]:
    import onnx

    fp32_model = onnx.load(str(fp32_path))
    quant_model = onnx.load(str(quant_path))
    graph = inspect_onnx_graph(quant_path)
    coverage = _coverage_from_graph(graph)

    init_shapes = {init.name: tuple(init.dims) for init in fp32_model.graph.initializer}
    const_weight_names: list[str] = []
    float_matmul_nodes: list[dict[str, str]] = []
    matmul_nodes_total = 0
    for node in fp32_model.graph.node:
        if node.op_type != "MatMul":
            continue
        matmul_nodes_total += 1
        weight_name = node.input[1] if len(node.input) > 1 else ""
        if weight_name not in init_shapes:
            float_matmul_nodes.append(
                {"name": node.name, "reason": "weight is not a constant initializer"}
            )
        elif len(init_shapes[weight_name]) != 2:
            float_matmul_nodes.append({"name": node.name, "reason": "weight is not 2D"})
        else:
            const_weight_names.append(node.name)

    nbits_nodes = [node for node in quant_model.graph.node if node.op_type == "MatMulNBits"]
    quantized_from_nbits = {_matmul_nbits_original_name(node.name) for node in nbits_nodes}
    quantized_nodes = [name for name in const_weight_names if name in quantized_from_nbits]
    for name in const_weight_names:
        if name not in quantized_from_nbits:
            float_matmul_nodes.append({"name": name, "reason": "not converted by quantizer"})

    accuracy_levels: list[int] = []
    weight_params_int4 = 0
    for node in nbits_nodes:
        k = int(_node_attr_int(node, "K", 0) or 0)
        n = int(_node_attr_int(node, "N", 0) or 0)
        weight_params_int4 += k * n
        level = _node_attr_int(node, "accuracy_level", None)
        accuracy_levels.append(0 if level is None else int(level))
    if accuracy_levels and len(set(accuracy_levels)) == 1:
        accuracy_level: int | list[int] = accuracy_levels[0]
    elif accuracy_levels:
        accuracy_level = accuracy_levels
    else:
        accuracy_level = 0

    weight_params_total = _float_initializer_element_count(fp32_model)
    fraction = (
        float(weight_params_int4) / float(weight_params_total) if weight_params_total > 0 else 0.0
    )
    float_gemm_nodes = [node.name for node in fp32_model.graph.node if node.op_type == "Gemm"]
    float_conv_nodes = [node.name for node in fp32_model.graph.node if node.op_type == "Conv"]

    return {
        **coverage,
        "recipe": {
            "algorithm": "RTN",
            "bits": bits,
            "block_size": block_size,
            "symmetric": symmetric,
            "op_types_to_quantize": list(op_types_to_quantize),
            "quant_format": quant_format_name,
            "accuracy_level": accuracy_level,
        },
        "matmul_nodes_total": matmul_nodes_total,
        "matmul_nodes_const_weight": len(const_weight_names),
        "matmul_nbits_nodes": len(nbits_nodes),
        "quantized_nodes": quantized_nodes,
        "float_matmul_nodes": float_matmul_nodes,
        "float_gemm_nodes": float_gemm_nodes,
        "float_conv_nodes": float_conv_nodes,
        "weight_params_total": weight_params_total,
        "weight_params_int4": weight_params_int4,
        "weight_params_int4_fraction": fraction,
        "full_model_int4_claim": False,
    }


def quantize_int8_static_qdq(
    fp32_path: Path,
    output_path: Path,
    *,
    calibration_samples: list[CorpusSample],
    calibration_sha256: str,
    providers: list[str] | None = None,
) -> VariantArtifact:
    """Static QDQ INT8 with MinMax calibration."""
    require_onnx()
    providers = providers or ["CPUExecutionProvider"]
    recipe = RECIPE_ORT_INT8_STATIC_QDQ
    if not calibration_samples:
        return VariantArtifact(
            label="int8",
            recipe_id=recipe,
            path=None,
            status="errored",
            detail="calibration corpus empty",
            runtime=_runtime_meta(providers),
        )
    try:
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        quantize_static(
            model_input=str(fp32_path),
            model_output=str(output_path),
            calibration_data_reader=_make_calibration_reader(calibration_samples),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )
        graph = inspect_onnx_graph(output_path)
        coverage = _coverage_from_graph(graph)
        # Probe actual EP on the produced graph.
        session = create_ort_session(output_path, providers=providers)
        _ = session.run(
            None,
            {
                "observation_images_top": calibration_samples[0].images_top,
                "observation_state": calibration_samples[0].state,
            },
        )
        return VariantArtifact(
            label="int8",
            recipe_id=recipe,
            path=str(output_path),
            status="ok",
            detail=(
                f"calibration_sha256={calibration_sha256}; "
                "QDQ static quantization — not a claim that every op is integer-only"
            ),
            graph=graph,
            effective_weight_precision="QInt8 (QDQ; Conv/MatMul may execute via ORT fused kernels)",
            effective_activation_precision="QUInt8 (QDQ)",
            quantization_coverage={
                **coverage,
                "calibration_sha256": calibration_sha256,
                "calibration_n": len(calibration_samples),
                "partial_ok_label": True,
                "full_model_int8_claim": False,
            },
            runtime={
                **_runtime_meta(providers),
                "session_providers": session.get_providers(),
                "graph_sha256": graph["sha256"],
            },
        )
    except Exception as exc:  # noqa: BLE001  capability boundary
        LOGGER.exception("INT8 static quantization failed")
        return VariantArtifact(
            label="int8",
            recipe_id=recipe,
            path=None,
            status="errored",
            detail=f"{type(exc).__name__}: {exc}",
            runtime=_runtime_meta(providers),
        )


def quantize_int4_weight_only(
    fp32_path: Path,
    output_path: Path,
    *,
    providers: list[str] | None = None,
    probe_sample: CorpusSample | None = None,
) -> VariantArtifact:
    """ORT MatMulNBits weight-only INT4 (RTN)."""
    providers = providers or ["CPUExecutionProvider"]
    recipe = RECIPE_ORT_INT4_MATMUL_NBITS
    block_size = 128
    bits = 4
    symmetric = True
    op_types_to_quantize = ("MatMul",)

    def _failed_artifact(status: str, detail: str) -> VariantArtifact:
        return VariantArtifact(
            label="int4_weight_only",
            recipe_id=recipe,
            path=None,
            status=status,
            detail=detail,
            runtime={
                **_runtime_meta(providers),
                "native_int4_execution_claim": False,
            },
        )

    try:
        require_onnx()
        import onnx
        from onnxruntime.quantization import QuantFormat
        from onnxruntime.quantization.matmul_nbits_quantizer import (
            DefaultWeightOnlyQuantConfig,
            MatMulNBitsQuantizer,
        )

        cfg = DefaultWeightOnlyQuantConfig(
            block_size=block_size,
            is_symmetric=symmetric,
            accuracy_level=None,
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=op_types_to_quantize,
            bits=bits,
        )
        quantizer = MatMulNBitsQuantizer(str(fp32_path), algo_config=cfg)
        quantizer.process()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save_model(quantizer.model.model, str(output_path))
        graph = inspect_onnx_graph(output_path)
        coverage = _build_int4_quantization_coverage(
            Path(fp32_path),
            output_path,
            block_size=block_size,
            bits=bits,
            symmetric=symmetric,
            op_types_to_quantize=op_types_to_quantize,
            quant_format_name=QuantFormat.QOperator.name,
        )
        runtime_base = {
            **_runtime_meta(providers),
            "native_int4_execution_claim": False,
        }
        if not coverage["has_matmul_nbits"]:
            return VariantArtifact(
                label="int4_weight_only",
                recipe_id=recipe,
                path=str(output_path),
                status="unsupported",
                detail="graph contains no MatMulNBits nodes after weight-only pass",
                graph=graph,
                effective_weight_precision=None,
                effective_activation_precision="float32",
                quantization_coverage=coverage,
                runtime=runtime_base,
            )
        try:
            session = create_ort_session(output_path, providers=providers)
        except Exception as exc:  # noqa: BLE001
            return VariantArtifact(
                label="int4_weight_only",
                recipe_id=recipe,
                path=str(output_path),
                status="unsupported",
                detail=(
                    f"MatMulNBits graph produced but EP load failed "
                    f"({type(exc).__name__}: {exc}); not labeled native INT4 execution"
                ),
                graph=graph,
                effective_weight_precision="int4_weights_stored",
                effective_activation_precision="float32",
                quantization_coverage=coverage,
                runtime=runtime_base,
            )
        if probe_sample is not None:
            _ = session.run(
                None,
                {
                    "observation_images_top": probe_sample.images_top,
                    "observation_state": probe_sample.state,
                },
            )
        return VariantArtifact(
            label="int4_weight_only",
            recipe_id=recipe,
            path=str(output_path),
            status="ok",
            detail=(
                "Weight-only MatMulNBits INT4; activations remain float. "
                "Not native INT4 activation execution; do not invent INT4 hardware timings."
            ),
            graph=graph,
            effective_weight_precision="int4 (MatMulNBits block_size=128, symmetric)",
            effective_activation_precision="float32",
            quantization_coverage=coverage,
            runtime={
                **runtime_base,
                "session_providers": session.get_providers(),
                "graph_sha256": graph["sha256"],
            },
        )
    except ImportError as exc:
        LOGGER.exception("INT4 weight-only quantization unavailable")
        return _failed_artifact("unsupported", f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001  capability boundary
        LOGGER.exception("INT4 weight-only quantization failed")
        return _failed_artifact("errored", f"{type(exc).__name__}: {exc}")


def try_fp16_convert(
    fp32_path: Path,
    output_path: Path,
    *,
    providers: list[str] | None = None,
    probe_sample: CorpusSample | None = None,
) -> VariantArtifact:
    """Optional FP16 cast. Capability-checked; not a mandatory CPU speedup."""
    require_onnx()
    providers = providers or ["CPUExecutionProvider"]
    recipe = RECIPE_ORT_FP16
    try:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16

        model = onnx.load(str(fp32_path))
        model_fp16 = convert_float_to_float16(model, keep_io_types=True)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model_fp16, str(output_path))
        graph = inspect_onnx_graph(output_path)
        try:
            session = create_ort_session(output_path, providers=providers)
        except Exception as exc:  # noqa: BLE001
            return VariantArtifact(
                label="fp16",
                recipe_id=recipe,
                path=str(output_path),
                status="unsupported",
                detail=f"FP16 graph saved but EP load failed: {type(exc).__name__}: {exc}",
                graph=graph,
                effective_weight_precision="float16",
                effective_activation_precision="float16 (I/O kept float32 if keep_io_types)",
                quantization_coverage=_coverage_from_graph(graph),
                runtime=_runtime_meta(providers),
            )
        if probe_sample is not None:
            _ = session.run(
                None,
                {
                    "observation_images_top": probe_sample.images_top,
                    "observation_state": probe_sample.state,
                },
            )
        return VariantArtifact(
            label="fp16",
            recipe_id=recipe,
            path=str(output_path),
            status="ok",
            detail="FP16 weight/activation cast via ORT transformers helper; optional path",
            graph=graph,
            effective_weight_precision="float16",
            effective_activation_precision="float16 (I/O float32)",
            quantization_coverage=_coverage_from_graph(graph),
            runtime={
                **_runtime_meta(providers),
                "session_providers": session.get_providers(),
                "graph_sha256": sha256_file(output_path),
            },
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("FP16 conversion failed")
        return VariantArtifact(
            label="fp16",
            recipe_id=recipe,
            path=None,
            status="unsupported",
            detail=f"{type(exc).__name__}: {exc}",
            runtime=_runtime_meta(providers),
        )


def add_int4_variant(
    export_dir: Path,
    *,
    providers: tuple[str, ...] | list[str] = ("CPUExecutionProvider",),
) -> Path:
    """Add MatMulNBits INT4 beside an existing validated FP32 export."""
    from breakpoint_eval.export_parity import run_ort_chunks
    from breakpoint_eval.export_runtime import resolve_onnx_runtime
    from breakpoint_eval.export_validation import write_int4_validation_from_export

    export_dir = Path(export_dir).resolve()
    manifest_path = export_dir / "export_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing export_manifest.json in {export_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"export manifest must be a JSON object: {manifest_path}")

    variants = [v for v in (manifest.get("variants") or []) if isinstance(v, dict)]
    fp32_ok = [v for v in variants if v.get("label") == "onnx_fp32" and v.get("status") == "ok"]
    if len(fp32_ok) != 1:
        raise ValueError(
            f"export manifest must list exactly one ok onnx_fp32 variant; found {len(fp32_ok)}"
        )
    fp32 = fp32_ok[0]
    fp32_name = Path(str(fp32.get("path") or "act_chunk_fp32.onnx")).name
    fp32_path = export_dir / fp32_name
    if not fp32_path.is_file():
        raise FileNotFoundError(f"onnx_fp32 graph not found in export dir: {fp32_path}")
    file_sha = sha256_file(fp32_path)
    manifest_sha = str((fp32.get("graph") or {}).get("sha256") or "")
    if not manifest_sha or manifest_sha != file_sha:
        raise ValueError(
            "onnx_fp32 manifest sha256 does not match file in export dir: "
            f"manifest={manifest_sha!r} file={file_sha!r}"
        )
    resolve_onnx_runtime(fp32_path, require_export_validation=True)

    if any(v.get("label") == "int4_weight_only" for v in variants):
        raise ValueError(
            f"export manifest already lists int4_weight_only; refuse overwrite in {export_dir}"
        )
    int4_path = export_dir / _INT4_GRAPH_NAME
    if int4_path.is_file():
        raise ValueError(f"{_INT4_GRAPH_NAME} already exists in {export_dir}; refuse overwrite")

    eval_meta = (manifest.get("corpus") or {}).get("evaluation") or {}
    corpus_path = export_dir / "corpus_eval.npz"
    if not corpus_path.is_file():
        raise FileNotFoundError(f"missing corpus_eval.npz in {export_dir}")
    corpus_sha = sha256_file(corpus_path)
    expected_sha = str(eval_meta.get("sha256") or "")
    if not expected_sha or corpus_sha != expected_sha:
        raise ValueError(
            "corpus_eval.npz sha256 does not match manifest corpus.evaluation.sha256: "
            f"file={corpus_sha!r} manifest={expected_sha!r}"
        )
    eval_samples = load_corpus(corpus_path)
    expected_ids = [str(x) for x in (eval_meta.get("sample_ids") or [])]
    actual_ids = [s.sample_id for s in eval_samples]
    if not expected_ids or actual_ids != expected_ids:
        raise ValueError(
            "corpus_eval.npz sample_ids do not match manifest corpus.evaluation.sample_ids"
        )
    if not eval_samples:
        raise ValueError("evaluation corpus is empty")

    provider_list = list(providers) or ["CPUExecutionProvider"]
    int4 = quantize_int4_weight_only(
        fp32_path,
        int4_path,
        providers=provider_list,
        probe_sample=eval_samples[0],
    )

    comparisons = dict(manifest.get("comparisons") or {})
    if int4.status == "ok" and int4.path:
        onnx_ref_path = export_dir / "eval_chunks_onnx_fp32.npz"
        if not onnx_ref_path.is_file():
            raise FileNotFoundError(
                f"missing eval_chunks_onnx_fp32.npz in {export_dir}; "
                "required for INT4 vs ONNX FP32 corpus comparison"
            )
        int4_sess = OrtActChunkSession(Path(int4.path), providers=provider_list)
        int4_chunks = run_ort_chunks(int4_sess, eval_samples)
        comparisons["int4_weight_only_vs_onnx_fp32"] = _corpus_parity_vs_reference(
            onnx_ref_path,
            candidate_chunks=int4_chunks,
            eval_samples=eval_samples,
            sample_ids=actual_ids,
            reference_label="onnx_fp32",
            candidate_label="int4_weight_only",
        )
        pytorch_ref_path = export_dir / "eval_chunks_pytorch_fp32.npz"
        if pytorch_ref_path.is_file():
            comparisons["int4_weight_only_vs_pytorch_fp32"] = _corpus_parity_vs_reference(
                pytorch_ref_path,
                candidate_chunks=int4_chunks,
                eval_samples=eval_samples,
                sample_ids=actual_ids,
                reference_label="pytorch_fp32",
                candidate_label="int4_weight_only",
                missing_keys_message="eval_chunks_pytorch_fp32.npz missing sample_id keys: ",
            )
    elif int4.status != "ok":
        LOGGER.warning(
            "INT4 variant status=%s detail=%s",
            int4.status,
            int4.detail,
        )

    updated = dict(manifest)
    updated["variants"] = list(variants) + [int4.to_dict()]
    updated["comparisons"] = comparisons
    _atomic_write_json(manifest_path, updated)

    return write_int4_validation_from_export(export_dir)
