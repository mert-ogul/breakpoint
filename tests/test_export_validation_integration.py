"""export_validation schema v2 accepted through resolve_onnx_runtime (needs Torch and ORT)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from test_export_validation import RECIPE_ONNX_FP32, _binding, _seed_outcome, _shadow
from torch import nn

from breakpoint_eval.act_export import INPUT_IMAGES, INPUT_STATE, OUTPUT_CHUNK, sha256_file
from breakpoint_eval.export_runtime import resolve_onnx_runtime
from breakpoint_eval.export_validation import build_export_validation_v2


class _Tiny(nn.Module):
    def forward(self, images_top: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        b = images_top.shape[0]
        return torch.zeros(b, 4, 2)


def _write_export_with_v2(tmp_path: Path, *, v2_passed: bool) -> Path:
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
    # Legacy v1 still fails independent closed-loop; v2 is the deployment gate.
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
        "export_validation": {
            "passed": False,
            "closed_loop_passed": False,
            "closed_loop_completed": True,
            "closed_loop_requested": True,
            "corpus_suitable_as_deployment_evidence": True,
            "open_loop_fp32_parity_passed": True,
        },
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
    (tmp_path / "export_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    v2 = build_export_validation_v2(
        binding=_binding(
            model_sha256=digest, corpus_eval_sha256="corpushash", hub_revision="deadbeef"
        ),
        open_loop_report={"passed": True, "finite_ok": True, "shapes_ok": True},
        shadow_reports=[_shadow()] if v2_passed else [_shadow(passed_atol=False)],
        independent_seed_reports=[_seed_outcome()],
        corpus_suitable_as_deployment_evidence=True,
    )
    assert v2["passed"] is v2_passed
    (tmp_path / "export_validation.json").write_text(json.dumps(v2), encoding="utf-8")
    return model_path


@pytest.mark.requires_onnx
def test_resolve_accepts_v2_when_v1_closed_loop_failed(tmp_path: Path) -> None:
    model_path = _write_export_with_v2(tmp_path, v2_passed=True)
    resolved = resolve_onnx_runtime(model_path, expected_candidate="act_aloha_transfer")
    assert resolved.export_validation.get("validation_schema") == "v2"
    assert resolved.export_validation.get("passed") is True


@pytest.mark.requires_onnx
def test_resolve_rejects_failed_v2(tmp_path: Path) -> None:
    model_path = _write_export_with_v2(tmp_path, v2_passed=False)
    with pytest.raises(ValueError, match="export_validation.passed"):
        resolve_onnx_runtime(model_path, expected_candidate="act_aloha_transfer")
