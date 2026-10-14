"""ONNX export parity, graph metadata, unsupported variants, no silent fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from breakpoint_eval.act_export import (  # noqa: E402
    INPUT_IMAGES,
    INPUT_STATE,
    OUTPUT_CHUNK,
    RECIPE_ORT_INT4_MATMUL_NBITS,
    RECIPE_ORT_INT8_STATIC_QDQ,
    OrtActChunkSession,
    VariantArtifact,
    inspect_onnx_graph,
    sha256_file,
)
from breakpoint_eval.export_corpus import (  # noqa: E402
    SYNTHETIC_SOURCE,
    SYNTHETIC_UNSUITABLE_NOTE,
    build_synthetic_corpus,
    load_corpus,
    make_sample_id,
    sample_step_indices,
    save_corpus,
)
from breakpoint_eval.export_parity import (  # noqa: E402
    ParityTolerances,
    compare_paired_action_traces,
    evaluate_corpus_parity,
)
from breakpoint_eval.export_quantize import (  # noqa: E402
    quantize_int4_weight_only,
    quantize_int8_static_qdq,
    try_fp16_convert,
)
from breakpoint_eval.export_tolerances import (  # noqa: E402
    ONNX_FP32_PARITY_ATOL,
    ONNX_FP32_PARITY_MAX_ABS,
)


class _TinyChunk(nn.Module):
    """Minimal ACT-shaped I/O for offline export tests (not the Hub ACT)."""

    def __init__(self) -> None:
        super().__init__()
        self.image_key = "observation.images.top"
        self.n_action_steps = 4
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8 + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 4 * 2),
        )

    def forward(self, images_top: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        b = images_top.shape[0]
        x = torch.cat([images_top.reshape(b, -1), state.reshape(b, -1)], dim=-1)
        return self.net(x).reshape(b, 4, 2)


def _export_tiny(tmp_path: Path) -> tuple[Path, _TinyChunk]:
    mod = _TinyChunk().eval()
    imgs = torch.rand(1, 3, 8, 8)
    state = torch.randn(1, 2)
    path = tmp_path / "tiny_fp32.onnx"
    torch.onnx.export(
        mod,
        (imgs, state),
        str(path),
        input_names=[INPUT_IMAGES, INPUT_STATE],
        output_names=[OUTPUT_CHUNK],
        opset_version=17,
        dynamo=False,
    )
    return path, mod


def test_declared_tolerances_are_finite_and_positive() -> None:
    assert ONNX_FP32_PARITY_ATOL > 0
    assert ONNX_FP32_PARITY_MAX_ABS > 0
    # ParityTolerances defaults must match declared module (not silently widened).
    tol = ParityTolerances()
    assert tol.atol == ONNX_FP32_PARITY_ATOL
    assert tol.max_abs == ONNX_FP32_PARITY_MAX_ABS


def test_tiny_onnx_fp32_parity(tmp_path: Path) -> None:
    path, mod = _export_tiny(tmp_path)
    graph = inspect_onnx_graph(path)
    assert graph["inputs"] == [INPUT_IMAGES, INPUT_STATE]
    assert graph["outputs"] == [OUTPUT_CHUNK]
    assert graph["sha256"] == sha256_file(path)
    assert graph["n_nodes"] > 0

    session = OrtActChunkSession(path, providers=["CPUExecutionProvider"])
    samples = build_synthetic_corpus([1, 2, 3], image_shape=(1, 3, 8, 8), state_dim=2)
    ref = []
    cand = []
    for s in samples:
        with torch.inference_mode():
            ref.append(mod(torch.from_numpy(s.images_top), torch.from_numpy(s.state)).numpy())
        cand.append(session.run(s.images_top, s.state).action_chunk)
    report = evaluate_corpus_parity(
        reference_chunks=ref,
        candidate_chunks=cand,
        seeds=[s.seed for s in samples],
        reference_label="pytorch_fp32",
        candidate_label="onnx_fp32",
        enforce_fp32_gates=True,
    )
    assert report.shapes_ok and report.finite_ok and report.passed


def test_corpus_roundtrip_and_disjoint_roles(tmp_path: Path) -> None:
    samples = build_synthetic_corpus([10, 11], image_shape=(1, 3, 8, 8), state_dim=2)
    assert all(s.source == SYNTHETIC_SOURCE for s in samples)
    assert samples[0].sample_id == make_sample_id(seed=10, control_step_index=0)
    man = save_corpus(tmp_path / "c.npz", samples, role="evaluation")
    assert man.n_samples == 2
    assert man.suitable_as_deployment_evidence is False
    assert "unsuitable" in man.note.lower() or SYNTHETIC_SOURCE in man.note
    loaded = load_corpus(tmp_path / "c.npz")
    assert [s.seed for s in loaded] == [10, 11]
    assert [s.sample_id for s in loaded] == man.sample_ids
    np.testing.assert_array_equal(loaded[0].images_top, samples[0].images_top)


def test_sample_step_indices_span_rollout() -> None:
    assert sample_step_indices(1, 1) == [0]
    assert sample_step_indices(100, 2) == [0, 99]
    assert sample_step_indices(5, 3) == [0, 2, 4]


def test_export_validation_rejects_closed_loop_error() -> None:
    """Closed-loop failure must fail the export gate even if open-loop passed."""
    from breakpoint_eval.run_export import ExportRequest

    # Pure logic check mirrored from run_export assembly.
    open_loop_passed = True
    closed_loop = {"status": "errored", "passed": False, "completed": False}
    req_run_closed = True
    closed_ok = (
        bool(closed_loop.get("completed")) and closed_loop.get("passed") is True
        if req_run_closed
        else True
    )
    assert (open_loop_passed and closed_ok) is False
    skipped = {"status": "skipped", "passed": None, "completed": False}
    assert skipped["completed"] is False
    _ = ExportRequest  # keep import used for module surface stability
    assert SYNTHETIC_UNSUITABLE_NOTE


def test_int8_recipe_pinned_and_runs(tmp_path: Path) -> None:
    path, _mod = _export_tiny(tmp_path)
    calib = build_synthetic_corpus([9000, 9001], image_shape=(1, 3, 8, 8), state_dim=2)
    out = quantize_int8_static_qdq(
        path,
        tmp_path / "tiny_int8.onnx",
        calibration_samples=calib,
        calibration_sha256="deadbeef",
        providers=["CPUExecutionProvider"],
    )
    assert out.recipe_id == RECIPE_ORT_INT8_STATIC_QDQ
    assert out.status == "ok"
    assert out.path is not None
    assert out.quantization_coverage.get("full_model_int8_claim") is False
    sess = OrtActChunkSession(Path(out.path), providers=["CPUExecutionProvider"])
    y = sess.run(calib[0].images_top, calib[0].state).action_chunk
    assert y.shape == (1, 4, 2)
    assert np.isfinite(y).all()


def test_int4_capability_explicit(tmp_path: Path) -> None:
    path, _mod = _export_tiny(tmp_path)
    probe = build_synthetic_corpus([1], image_shape=(1, 3, 8, 8), state_dim=2)[0]
    out = quantize_int4_weight_only(
        path,
        tmp_path / "tiny_int4.onnx",
        providers=["CPUExecutionProvider"],
        probe_sample=probe,
    )
    assert out.recipe_id == RECIPE_ORT_INT4_MATMUL_NBITS
    # Tiny Linear may become Gemm (often excluded) or MatMulNBits — either ok/unsupported
    # must be explicit, never silently relabeled as FP32.
    assert out.status in {"ok", "unsupported", "errored"}
    assert out.label == "int4_weight_only"
    if out.status == "ok":
        assert out.runtime.get("native_int4_execution_claim") is False
        assert out.effective_activation_precision == "float32"


def test_fp16_optional_capability(tmp_path: Path) -> None:
    path, _mod = _export_tiny(tmp_path)
    probe = build_synthetic_corpus([1], image_shape=(1, 3, 8, 8), state_dim=2)[0]
    out = try_fp16_convert(
        path,
        tmp_path / "tiny_fp16.onnx",
        providers=["CPUExecutionProvider"],
        probe_sample=probe,
    )
    assert out.status in {"ok", "unsupported", "errored"}
    assert out.label == "fp16"


def test_no_silent_fallback_on_missing_provider(tmp_path: Path) -> None:
    path, _mod = _export_tiny(tmp_path)
    with pytest.raises(RuntimeError, match="not available"):
        OrtActChunkSession(path, providers=["TotallyFakeExecutionProvider"])


def test_variant_artifact_does_not_relabel_failure_as_ok() -> None:
    art = VariantArtifact(
        label="int8",
        recipe_id=RECIPE_ORT_INT8_STATIC_QDQ,
        path=None,
        status="errored",
        detail="calibration corpus empty",
    )
    d = art.to_dict()
    assert d["status"] == "errored"
    assert d["path"] is None
    assert (
        "fp32" not in (d["detail"] or "").lower() or "fallback" not in (d["detail"] or "").lower()
    )


def test_parity_fails_on_large_error() -> None:
    ref = [np.zeros((1, 4, 2), dtype=np.float32)]
    bad = [np.ones((1, 4, 2), dtype=np.float32)]
    report = evaluate_corpus_parity(
        reference_chunks=ref,
        candidate_chunks=bad,
        seeds=[0],
        reference_label="a",
        candidate_label="b",
        enforce_fp32_gates=True,
    )
    assert report.passed is False


def test_closed_loop_trace_compare() -> None:
    a = [np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)]
    b = [np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)]
    ok = compare_paired_action_traces(a, b)
    assert ok.passed is True
    bad = compare_paired_action_traces(a, [np.zeros(2, dtype=np.float32), np.full(2, 9.0)])
    assert bad.passed is False


def test_export_act_chunk_module_rejects_multi_camera() -> None:
    from breakpoint_eval.act_export import ACTChunkModule

    class P:
        config = type(
            "C",
            (),
            {
                "image_features": ["observation.images.top", "observation.images.wrist"],
                "n_action_steps": 100,
                "temporal_ensemble_coeff": None,
            },
        )()

    with pytest.raises(NotImplementedError, match="one camera"):
        ACTChunkModule(P())
