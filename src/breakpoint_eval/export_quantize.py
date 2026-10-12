"""Quantization recipes for exported ACT ONNX. No silent recipe switching."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from breakpoint_eval.act_export import (
    RECIPE_ORT_FP16,
    RECIPE_ORT_INT4_MATMUL_NBITS,
    RECIPE_ORT_INT8_STATIC_QDQ,
    VariantArtifact,
    create_ort_session,
    inspect_onnx_graph,
    require_onnx,
    sha256_file,
)
from breakpoint_eval.export_corpus import CorpusSample

LOGGER = logging.getLogger("breakpoint_eval.export_quantize")


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


def quantize_int8_static_qdq(
    fp32_path: Path,
    output_path: Path,
    *,
    calibration_samples: list[CorpusSample],
    calibration_sha256: str,
    providers: list[str] | None = None,
) -> VariantArtifact:
    """Static QDQ INT8 with MinMax calibration. Does not fall back to another recipe."""
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
    except Exception as exc:  # noqa: BLE001 — capability boundary
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
    """ORT MatMulNBits weight-only INT4. Unsupported/run failure is explicit."""
    require_onnx()
    providers = providers or ["CPUExecutionProvider"]
    recipe = RECIPE_ORT_INT4_MATMUL_NBITS
    try:
        import onnx
        from onnxruntime.quantization.matmul_nbits_quantizer import (
            DefaultWeightOnlyQuantConfig,
            MatMulNBitsQuantizer,
        )

        cfg = DefaultWeightOnlyQuantConfig(
            block_size=128,
            is_symmetric=True,
            bits=4,
        )
        quantizer = MatMulNBitsQuantizer(str(fp32_path), algo_config=cfg)
        quantizer.process()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save_model(quantizer.model.model, str(output_path))
        graph = inspect_onnx_graph(output_path)
        coverage = _coverage_from_graph(graph)
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
                runtime=_runtime_meta(providers),
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
            quantization_coverage={
                **coverage,
                "matmul_nbits_nodes": int(graph["op_counts"].get("MatMulNBits", 0)),
                "excluded_const_matmul_note": (
                    "MatMuls without const weights and Gemm/Conv remain float"
                ),
                "full_model_int4_claim": False,
            },
            runtime={
                **_runtime_meta(providers),
                "session_providers": session.get_providers(),
                "graph_sha256": graph["sha256"],
                "native_int4_execution_claim": False,
            },
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("INT4 weight-only quantization failed")
        return VariantArtifact(
            label="int4_weight_only",
            recipe_id=recipe,
            path=None,
            status="unsupported",
            detail=f"{type(exc).__name__}: {exc}",
            runtime=_runtime_meta(providers),
        )


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
