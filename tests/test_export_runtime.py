"""ONNX runtime identity resolution and profile mismatch rejection."""

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
    sha256_file,
)
from breakpoint_eval.export_runtime import (
    apply_policy_rng_seed,
    resolve_onnx_runtime,
)
from breakpoint_eval.latency_trace import (
    TIMED_BOUNDARY_ORT_RUN,
    assert_profile_compatible,
    load_latency_profile,
)


class _Tiny(nn.Module):
    def forward(self, images_top: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        b = images_top.shape[0]
        return torch.zeros(b, 4, 2)


def _write_export_dir(tmp_path: Path) -> tuple[Path, Path, str]:
    mod = _Tiny().eval()
    imgs = torch.rand(1, 3, 8, 8)
    state = torch.randn(1, 2)
    model_path = tmp_path / "act_chunk_fp32.onnx"
    torch.onnx.export(
        mod,
        (imgs, state),
        str(model_path),
        input_names=[INPUT_IMAGES, INPUT_STATE],
        output_names=[OUTPUT_CHUNK],
        opset_version=17,
        dynamo=False,
    )
    digest = sha256_file(model_path)
    manifest = {
        "candidate": "act_aloha_transfer",
        "hub_revision": "deadbeef",
        "io_spec": {
            "image_shape": [1, 3, 8, 8],
            "state_shape": [1, 2],
            "action_chunk_shape": [1, 4, 2],
        },
        "chunk_info": {"n_action_steps": 4},
        "corpus": {"evaluation": {"sha256": "corpushash"}},
        "export_validation": {"passed": True},
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
    man_path = tmp_path / "export_manifest.json"
    man_path.write_text(json.dumps(manifest), encoding="utf-8")
    return model_path, man_path, digest


@pytest.mark.requires_onnx
def test_resolve_onnx_runtime_from_manifest(tmp_path: Path) -> None:
    model_path, _, digest = _write_export_dir(tmp_path)
    resolved = resolve_onnx_runtime(model_path, expected_candidate="act_aloha_transfer")
    assert resolved.model_sha256 == digest
    assert resolved.recipe_id == RECIPE_ONNX_FP32
    assert resolved.variant_label == "onnx_fp32"
    assert resolved.chunk_horizon == 4


@pytest.mark.requires_onnx
def test_resolve_rejects_unvalidated_export(tmp_path: Path) -> None:
    model_path, man_path, _ = _write_export_dir(tmp_path)
    payload = json.loads(man_path.read_text())
    payload["export_validation"] = {"passed": False}
    man_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="export_validation"):
        resolve_onnx_runtime(model_path, expected_candidate="act_aloha_transfer")


@pytest.mark.requires_onnx
def test_resolve_rejects_synthetic_passed_bypass(tmp_path: Path) -> None:
    """diagnostic_parity_ok / forged passed must not unlock deployment runtime."""
    model_path, man_path, _ = _write_export_dir(tmp_path)
    payload = json.loads(man_path.read_text())
    payload["export_validation"] = {
        "passed": True,
        "diagnostic_parity_ok": True,
        "corpus_suitable_as_deployment_evidence": False,
        "closed_loop_requested": False,
        "closed_loop_completed": False,
        "closed_loop_passed": None,
    }
    man_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not suitable as deployment evidence"):
        resolve_onnx_runtime(model_path, expected_candidate="act_aloha_transfer")


@pytest.mark.requires_onnx
def test_resolve_rejects_skipped_closed_loop_bypass(tmp_path: Path) -> None:
    model_path, man_path, _ = _write_export_dir(tmp_path)
    payload = json.loads(man_path.read_text())
    payload["export_validation"] = {
        "passed": True,
        "diagnostic_parity_ok": True,
        "corpus_suitable_as_deployment_evidence": True,
        "closed_loop_requested": False,
        "closed_loop_completed": False,
        "closed_loop_passed": None,
    }
    man_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="closed-loop was not requested"):
        resolve_onnx_runtime(model_path, expected_candidate="act_aloha_transfer")


@pytest.mark.requires_onnx
def test_resolve_rejects_failed_closed_loop_bypass(tmp_path: Path) -> None:
    model_path, man_path, _ = _write_export_dir(tmp_path)
    payload = json.loads(man_path.read_text())
    payload["export_validation"] = {
        "passed": True,
        "diagnostic_parity_ok": True,
        "corpus_suitable_as_deployment_evidence": True,
        "closed_loop_requested": True,
        "closed_loop_completed": True,
        "closed_loop_passed": False,
        "closed_loop_failure_kind": "independent_trajectory_divergence",
    }
    man_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="closed-loop did not pass"):
        resolve_onnx_runtime(model_path, expected_candidate="act_aloha_transfer")


@pytest.mark.requires_onnx
def test_resolve_rejects_path_match_when_manifest_hash_disagrees(tmp_path: Path) -> None:
    model_path, man_path, digest = _write_export_dir(tmp_path)
    payload = json.loads(man_path.read_text())
    payload["variants"][0]["graph"]["sha256"] = "0" * 64
    assert payload["variants"][0]["graph"]["sha256"] != digest
    man_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="model-hash mismatch"):
        resolve_onnx_runtime(model_path, expected_candidate="act_aloha_transfer")


@pytest.mark.requires_onnx
def test_resolve_int8_refuses_fp32_validation_sidecar(tmp_path: Path) -> None:
    model_path, man_path, digest = _write_export_dir(tmp_path)
    # Relabel the only graph as int8; FP32 v2 sidecar must not authorize it.
    payload = json.loads(man_path.read_text())
    payload["variants"][0]["label"] = "int8"
    payload["variants"][0]["recipe_id"] = (
        "onnxruntime_static_qdq_uint8_act_qint8_weight_minmax_calibrate"
    )
    man_path.write_text(json.dumps(payload), encoding="utf-8")
    from breakpoint_eval.act_export import RECIPE_ONNX_FP32
    from breakpoint_eval.export_validation import (
        VALIDATION_FILENAME,
        ValidationBinding,
        build_export_validation_v2,
    )

    v2 = build_export_validation_v2(
        binding=ValidationBinding(
            model_sha256=digest,
            corpus_eval_sha256="corpushash",
            hub_revision="deadbeef",
            candidate="act_aloha_transfer",
            recipe_id=RECIPE_ONNX_FP32,
            variant_label="onnx_fp32",
            code_revision="x",
            code_dirty=False,
            uncommitted_paths=(),
        ),
        open_loop_report={"passed": True, "finite_ok": True, "shapes_ok": True},
        shadow_reports=[
            {
                "seed": 2100,
                "passed_atol": True,
                "finite_ok": True,
                "shapes_ok": True,
                "compares_every_control_step": True,
                "covers_every_fresh_chunk_refresh": True,
                "covers_every_inference_call": True,
                "shadow_controls_simulation": False,
                "n_fresh_chunks": 1,
                "n_inference_calls": 1,
            }
        ],
        independent_seed_reports=[
            {
                "seed": 2100,
                "pytorch_outcome": "success",
                "onnx_outcome": "success",
                "pytorch_success": True,
                "onnx_success": True,
                "pytorch_terminated": True,
                "onnx_terminated": True,
                "pytorch_truncated": False,
                "onnx_truncated": False,
                "pytorch_n_control_steps": 10,
                "onnx_n_control_steps": 10,
                "passed": True,
            }
        ],
        corpus_suitable_as_deployment_evidence=True,
    )
    (tmp_path / VALIDATION_FILENAME).write_text(json.dumps(v2), encoding="utf-8")
    with pytest.raises(ValueError, match="export_validation_int8"):
        resolve_onnx_runtime(model_path, expected_candidate="act_aloha_transfer")


@pytest.mark.requires_onnx
def test_profile_rejects_pytorch_relabel_via_hash_mismatch(tmp_path: Path) -> None:
    model_path, _, digest = _write_export_dir(tmp_path)
    profile = {
        "schema_version": 1,
        "device_label": "local",
        "candidate": "act_aloha_transfer",
        "hub_revision": "deadbeef",
        "timed_boundary": TIMED_BOUNDARY_ORT_RUN,
        "chunk_horizon": 4,
        "model": {
            "sha256": digest,
            "precision_recipe": RECIPE_ONNX_FP32,
            "variant_label": "onnx_fp32",
        },
        "input_contract": {
            "shapes": {INPUT_IMAGES: [1, 3, 8, 8], INPUT_STATE: [1, 2]},
            "corpus_sha256": "corpushash",
        },
        "output_validation": {"passed": True},
        "measurements": {
            "n_measured": 2,
            "inference_latency_s": [0.01, 0.02],
            "summary": {
                "count": 2,
                "median_s": 0.015,
                "p95_s": float(np.percentile([0.01, 0.02], 95)),
                "p99_s": float(np.percentile([0.01, 0.02], 99)),
            },
        },
        "runtime": {},
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    prof = load_latency_profile(path)
    assert_profile_compatible(
        prof,
        candidate="act_aloha_transfer",
        model_sha256=digest,
        precision_recipe=RECIPE_ONNX_FP32,
        variant_label="onnx_fp32",
        chunk_horizon=4,
        hub_revision="deadbeef",
        input_shapes={INPUT_IMAGES: [1, 3, 8, 8], INPUT_STATE: [1, 2]},
    )
    with pytest.raises(ValueError, match="precision_recipe"):
        assert_profile_compatible(
            prof,
            candidate="act_aloha_transfer",
            model_sha256=digest,
            precision_recipe="onnxruntime_static_qdq_uint8_act_qint8_weight_minmax_calibrate",
            variant_label="int8",
            chunk_horizon=4,
        )


def test_apply_policy_rng_seed_sets_torch() -> None:
    apply_policy_rng_seed(123)
    a = torch.randn(2)
    apply_policy_rng_seed(123)
    b = torch.randn(2)
    assert torch.allclose(a, b)


def test_plan_refuses_existing_out(tmp_path: Path) -> None:
    from breakpoint_eval.run_plan_execute import _refuse_or_prepare_out

    out = tmp_path / "out"
    out.mkdir()
    (out / "marker").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        _refuse_or_prepare_out(out, allow_existing=False)
    _refuse_or_prepare_out(out, allow_existing=True)
