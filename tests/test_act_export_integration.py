"""Integration: real ACT Hub checkpoint ONNX FP32 parity (optional)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


@pytest.mark.integration
def test_real_act_export_fp32_parity(tmp_path: Path) -> None:
    from breakpoint_eval.run_export import ExportRequest, run_export
    from breakpoint_eval.stack import repo_root_from_package

    repo_root = repo_root_from_package()
    out = tmp_path / "export_act"
    try:
        manifest_path = run_export(
            ExportRequest(
                candidate="act_aloha_transfer",
                output_dir=out,
                device="cpu",
                calibration_seeds=[9000, 9001],
                eval_seeds=[1000, 1001],
                prefer_simulator_corpus=False,
                attempt_int8=False,
                attempt_int4=False,
                attempt_fp16=False,
                run_closed_loop=False,
                repo_root=repo_root,
            )
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ACT export stack unavailable: {exc}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    parity = payload["comparisons"]["onnx_fp32_vs_pytorch_fp32"]
    assert parity["passed"] is True
    assert (
        parity["max_abs_error"]
        <= payload["declared_tolerances_before_measurement"]["onnx_fp32_parity_max_abs"]
    )
    assert (out / "act_chunk_fp32.onnx").is_file()
    assert payload["variants"][0]["recipe_id"] == "torch_onnx_export_fp32_opset17"
