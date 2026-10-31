"""ONNX export parity, graph metadata, unsupported variants, no silent fallback."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from breakpoint_eval.act_export import (
    INPUT_IMAGES,
    INPUT_STATE,
    OUTPUT_CHUNK,
    RECIPE_ONNX_FP32,
    RECIPE_ORT_INT4_MATMUL_NBITS,
    RECIPE_ORT_INT8_STATIC_QDQ,
    OrtActChunkSession,
    VariantArtifact,
    create_ort_session,
    inspect_onnx_graph,
    sha256_file,
)
from breakpoint_eval.export_corpus import (
    SYNTHETIC_SOURCE,
    SYNTHETIC_UNSUITABLE_NOTE,
    build_synthetic_corpus,
    load_corpus,
    make_sample_id,
    sample_step_indices,
    save_corpus,
)
from breakpoint_eval.export_parity import (
    ParityTolerances,
    compare_paired_action_traces,
    evaluate_corpus_parity,
)
from breakpoint_eval.export_quantize import (
    add_int4_variant,
    quantize_int4_weight_only,
    quantize_int8_static_qdq,
    try_fp16_convert,
)
from breakpoint_eval.export_tolerances import (
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


@pytest.mark.requires_onnx
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


def test_planned_reachable_steps_no_padding() -> None:
    from breakpoint_eval.export_corpus import (
        PLANNED_CORPUS_CONTROL_STEPS,
        planned_reachable_steps,
    )

    assert planned_reachable_steps(0) == []
    assert planned_reachable_steps(1) == [0]
    assert planned_reachable_steps(50) == [0, 49]
    assert planned_reachable_steps(100) == [0, 49, 99]
    assert planned_reachable_steps(400) == list(PLANNED_CORPUS_CONTROL_STEPS)
    # Early terminate before later planned times: do not invent missing steps.
    assert 349 not in planned_reachable_steps(200)


def test_export_validation_rejects_closed_loop_error() -> None:
    """Closed-loop failure must fail the export gate even if open-loop passed."""
    from breakpoint_eval.run_export import build_export_validation

    failed = build_export_validation(
        open_loop_passed=True,
        closed_loop={
            "status": "errored",
            "passed": False,
            "completed": False,
            "failure_kind": "runtime_error",
        },
        closed_loop_requested=True,
        corpus_suitable_as_deployment_evidence=True,
    )
    assert failed["passed"] is False
    assert failed["diagnostic_parity_ok"] is True
    assert "closed_loop_not_completed" in failed["not_passed_reasons"]


def test_export_validation_rejects_skipped_closed_loop() -> None:
    from breakpoint_eval.run_export import build_export_validation

    skipped = build_export_validation(
        open_loop_passed=True,
        closed_loop={"status": "skipped", "passed": None, "completed": False},
        closed_loop_requested=False,
        corpus_suitable_as_deployment_evidence=True,
    )
    assert skipped["passed"] is False
    assert skipped["diagnostic_parity_ok"] is True
    assert "closed_loop_not_requested" in skipped["not_passed_reasons"]


def test_export_validation_rejects_synthetic_corpus() -> None:
    from breakpoint_eval.run_export import build_export_validation

    synthetic = build_export_validation(
        open_loop_passed=True,
        closed_loop={
            "status": "ok",
            "passed": True,
            "completed": True,
            "comparison_kind": "independent_rollout_action_trace",
        },
        closed_loop_requested=True,
        corpus_suitable_as_deployment_evidence=False,
    )
    assert synthetic["passed"] is False
    assert synthetic["diagnostic_parity_ok"] is True
    assert "corpus_not_suitable_as_deployment_evidence" in synthetic["not_passed_reasons"]
    assert SYNTHETIC_UNSUITABLE_NOTE


def test_export_validation_passes_only_full_deployment_gate() -> None:
    from breakpoint_eval.run_export import build_export_validation

    ok = build_export_validation(
        open_loop_passed=True,
        closed_loop={
            "status": "ok",
            "passed": True,
            "completed": True,
            "comparison_kind": "independent_rollout_action_trace",
        },
        closed_loop_requested=True,
        corpus_suitable_as_deployment_evidence=True,
    )
    assert ok["passed"] is True
    assert ok["not_passed_reasons"] == []


def test_closed_loop_reports_per_seed_and_unmatched_tails() -> None:
    from breakpoint_eval.export_parity import (
        FAILURE_KIND_LENGTH_MISMATCH,
        FAILURE_KIND_TRAJECTORY_DIVERGENCE,
        compare_action_traces_with_tails,
    )

    a = [np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)]
    b_short = [np.zeros(2, dtype=np.float32)]
    mismatch = compare_action_traces_with_tails(a, b_short)
    assert mismatch["length_mismatch"] is True
    assert mismatch["unmatched_tail_steps"] == 1
    assert mismatch["passed"] is False
    assert mismatch["failure_kind"] == FAILURE_KIND_LENGTH_MISMATCH

    b_div = [np.zeros(2, dtype=np.float32), np.full(2, 9.0, dtype=np.float32)]
    diverged = compare_action_traces_with_tails(a, b_div)
    assert diverged["passed"] is False
    assert diverged["first_divergence_step"] == 1
    assert diverged["failure_kind"] == FAILURE_KIND_TRAJECTORY_DIVERGENCE
    assert diverged["length_mismatch"] is False


@pytest.mark.requires_onnx
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


def _write_int4_coverage_graph(path: Path) -> tuple[int, int]:
    """FP32 graph: const MatMul (K>=128), activation MatMul, and Gemm."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    k, n = 128, 8
    weight = np.random.default_rng(0).standard_normal((k, n), dtype=np.float32)
    gemm_w = np.random.default_rng(1).standard_normal((n, n), dtype=np.float32)
    gemm_b = np.zeros(n, dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["X", "W"], ["Y1"], name="mm_const"),
            helper.make_node("MatMul", ["Y1", "Act"], ["Y2"], name="mm_act"),
            helper.make_node("Gemm", ["Y2", "Wg", "Bg"], ["Y"], name="gemm1"),
        ],
        "int4_coverage",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, k]),
            helper.make_tensor_value_info("Act", TensorProto.FLOAT, [n, n]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, n])],
        [
            numpy_helper.from_array(weight, "W"),
            numpy_helper.from_array(gemm_w, "Wg"),
            numpy_helper.from_array(gemm_b, "Bg"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return k, n


def _write_act_matmul_export(tmp_path: Path) -> tuple[Path, list, list]:
    """ACT-shaped FP32 export with a const MatMul (K>=128) for add_int4_variant."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    k, hidden, out_dim = 192, 128, 8
    w1 = np.random.default_rng(2).standard_normal((k, hidden), dtype=np.float32)
    w2 = np.random.default_rng(3).standard_normal((hidden, out_dim), dtype=np.float32)
    b2 = np.zeros(out_dim, dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Flatten", [INPUT_IMAGES], ["flat"], name="flatten", axis=1),
            helper.make_node("MatMul", ["flat", "W1"], ["h"], name="mm_const"),
            helper.make_node("Gemm", ["h", "W2", "B2"], ["flat_out"], name="gemm_out"),
            helper.make_node(
                "Reshape",
                ["flat_out", "shape"],
                [OUTPUT_CHUNK],
                name="reshape_out",
            ),
        ],
        "act_int4",
        [
            helper.make_tensor_value_info(INPUT_IMAGES, TensorProto.FLOAT, [1, 3, 8, 8]),
            helper.make_tensor_value_info(INPUT_STATE, TensorProto.FLOAT, [1, 2]),
        ],
        [helper.make_tensor_value_info(OUTPUT_CHUNK, TensorProto.FLOAT, [1, 4, 2])],
        [
            numpy_helper.from_array(w1, "W1"),
            numpy_helper.from_array(w2, "W2"),
            numpy_helper.from_array(b2, "B2"),
            numpy_helper.from_array(np.asarray([1, 4, 2], dtype=np.int64), "shape"),
        ],
    )
    # Keep state as a graph input even if unused so OrtActChunkSession binds both feeds.
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    onnx.checker.check_model(model)
    model_path = tmp_path / "act_chunk_fp32.onnx"
    onnx.save(model, str(model_path))
    samples = build_synthetic_corpus([2000, 2001], image_shape=(1, 3, 8, 8), state_dim=2)
    sess = OrtActChunkSession(model_path, providers=["CPUExecutionProvider"])
    chunks = [sess.run(s.images_top, s.state).action_chunk for s in samples]
    return model_path, samples, chunks


@pytest.mark.requires_onnx
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


@pytest.mark.requires_onnx
def test_int4_weight_only_quantizes_const_matmul_only(tmp_path: Path) -> None:
    fp32 = tmp_path / "coverage_fp32.onnx"
    k, n = _write_int4_coverage_graph(fp32)
    out = quantize_int4_weight_only(
        fp32,
        tmp_path / "coverage_int4.onnx",
        providers=["CPUExecutionProvider"],
        probe_sample=None,
    )
    assert out.status == "ok"
    assert out.path is not None
    cov = out.quantization_coverage
    assert cov["recipe"]["algorithm"] == "RTN"
    assert cov["recipe"]["bits"] == 4
    assert cov["recipe"]["block_size"] == 128
    assert cov["recipe"]["symmetric"] is True
    assert cov["recipe"]["op_types_to_quantize"] == ["MatMul"]
    assert cov["recipe"]["quant_format"] == "QOperator"
    assert cov["recipe"]["accuracy_level"] == 0
    assert cov["matmul_nodes_total"] == 2
    assert cov["matmul_nodes_const_weight"] == 1
    assert cov["matmul_nbits_nodes"] == 1
    assert cov["quantized_nodes"] == ["mm_const"]
    assert cov["float_gemm_nodes"] == ["gemm1"]
    assert cov["full_model_int4_claim"] is False
    reasons = {entry["name"]: entry["reason"] for entry in cov["float_matmul_nodes"]}
    assert reasons["mm_act"] == "weight is not a constant initializer"
    assert out.runtime.get("native_int4_execution_claim") is False
    assert out.runtime.get("session_providers")
    session = create_ort_session(Path(out.path), providers=["CPUExecutionProvider"])
    y = session.run(
        None,
        {
            "X": np.zeros((1, k), dtype=np.float32),
            "Act": np.eye(n, dtype=np.float32),
        },
    )[0]
    assert np.isfinite(y).all()


@pytest.mark.requires_onnx
def test_int4_import_error_unsupported_other_errored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fp32 = tmp_path / "coverage_fp32.onnx"
    _write_int4_coverage_graph(fp32)

    monkeypatch.setattr(
        "breakpoint_eval.export_quantize.require_onnx",
        lambda: (_ for _ in ()).throw(ImportError("onnxruntime missing")),
    )
    missing = quantize_int4_weight_only(fp32, tmp_path / "a.onnx")
    assert missing.status == "unsupported"
    assert "ImportError" in (missing.detail or "")

    monkeypatch.setattr("breakpoint_eval.export_quantize.require_onnx", lambda: None)

    import onnxruntime.quantization.matmul_nbits_quantizer as nbits_mod

    class _Boom:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("quantizer exploded")

    monkeypatch.setattr(nbits_mod, "MatMulNBitsQuantizer", _Boom)
    errored = quantize_int4_weight_only(fp32, tmp_path / "b.onnx")
    assert errored.status == "errored"
    assert "RuntimeError" in (errored.detail or "")


@pytest.mark.requires_onnx
def test_add_int4_variant_appends_and_refuses_second(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path, samples, chunks = _write_act_matmul_export(tmp_path)
    digest = sha256_file(model_path)
    corpus_path = tmp_path / "corpus_eval.npz"
    corpus_manifest = save_corpus(corpus_path, samples, role="evaluation")
    np.savez_compressed(
        tmp_path / "eval_chunks_onnx_fp32.npz",
        **{s.sample_id: c for s, c in zip(samples, chunks, strict=True)},
    )
    manifest = {
        "candidate": "act_aloha_transfer",
        "hub_revision": "deadbeef",
        "io_spec": {
            "image_shape": [1, 3, 8, 8],
            "state_shape": [1, 2],
            "action_chunk_shape": [1, 4, 2],
        },
        "chunk_info": {"n_action_steps": 4},
        "corpus": {
            "evaluation": corpus_manifest.to_dict(),
        },
        "export_validation": {
            "passed": True,
            "corpus_suitable_as_deployment_evidence": True,
        },
        "comparisons": {},
        "variants": [
            {
                "label": "onnx_fp32",
                "recipe_id": RECIPE_ONNX_FP32,
                "path": str(model_path),
                "status": "ok",
                "graph": {"sha256": digest},
            }
        ],
    }
    (tmp_path / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        "breakpoint_eval.export_runtime.resolve_onnx_runtime",
        lambda *args, **kwargs: None,
    )
    sidecar = add_int4_variant(tmp_path, providers=("CPUExecutionProvider",))
    assert sidecar.name == "export_validation_int4.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["variant_label"] == "int4_weight_only"
    updated = json.loads((tmp_path / "export_manifest.json").read_text(encoding="utf-8"))
    assert any(v.get("label") == "int4_weight_only" for v in updated["variants"])
    assert "int4_weight_only_vs_onnx_fp32" in updated["comparisons"]
    with pytest.raises(ValueError, match="refuse overwrite"):
        add_int4_variant(tmp_path, providers=("CPUExecutionProvider",))


@pytest.mark.requires_onnx
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


@pytest.mark.requires_onnx
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
