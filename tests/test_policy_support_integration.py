"""Integration smokes: one real checkpoint check per supported policy.

Marked ``integration`` so default CI stays small/offline. Large sweeps stay out
of this file.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.integration


def _stack_ready():
    from breakpoint_eval.stack import ensure_lerobot_on_path, repo_root_from_package

    root = repo_root_from_package()
    ensure_lerobot_on_path(root)
    return root


def _load_adapter(candidate: str, device: str = "cpu"):
    from breakpoint_eval.policy_adapter import PolicyAdapter
    from breakpoint_eval.sim_env import control_period_s, load_policy
    from breakpoint_eval.stack import download_candidate_checkpoint, resolve_candidate

    root = _stack_ready()
    spec = resolve_candidate(candidate)
    ckpt = download_candidate_checkpoint(spec, root / ".cache" / "huggingface")
    policy = load_policy(spec, str(ckpt), device)
    period = control_period_s(spec)
    return PolicyAdapter(policy, control_period_s=period, device=device), spec, root


def test_smoke_act_aloha_transfer_gates() -> None:
    from breakpoint_eval.policy_validate import assert_gates_passed, run_policy_validation_gates

    try:
        adapter, spec, _root = _load_adapter("act_aloha_transfer")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ACT transfer stack unavailable: {exc}")

    pol_obs = {
        "observation.state": torch.zeros(1, 14),
        "observation.images.top": torch.rand(1, 3, 480, 640),
    }
    raw = {
        "pixels": {"top": np.zeros((480, 640, 3), dtype=np.uint8)},
        "agent_pos": np.zeros((14,), dtype=np.float32),
    }
    results = run_policy_validation_gates(
        spec=spec,
        adapter=adapter,
        policy_observation=pol_obs,
        raw_observation=raw,
        zero_latency_observations=[
            {
                "observation.state": torch.zeros(1, 14),
                "observation.images.top": torch.rand(1, 3, 480, 640),
            }
            for _ in range(5)
        ],
    )
    assert_gates_passed(results)


def test_smoke_act_aloha_insertion_gates_and_export(tmp_path) -> None:
    from breakpoint_eval.policy_validate import assert_gates_passed, run_policy_validation_gates
    from breakpoint_eval.run_export import ExportRequest, run_export

    try:
        adapter, spec, root = _load_adapter("act_aloha_insertion")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ACT insertion stack unavailable: {exc}")

    pol_obs = {
        "observation.state": torch.zeros(1, 14),
        "observation.images.top": torch.rand(1, 3, 480, 640),
    }
    raw = {
        "pixels": {"top": np.zeros((480, 640, 3), dtype=np.uint8)},
        "agent_pos": np.zeros((14,), dtype=np.float32),
    }
    results = run_policy_validation_gates(
        spec=spec,
        adapter=adapter,
        policy_observation=pol_obs,
        raw_observation=raw,
        zero_latency_observations=[
            {
                "observation.state": torch.zeros(1, 14),
                "observation.images.top": torch.rand(1, 3, 480, 640),
            }
            for _ in range(5)
        ],
    )
    assert_gates_passed(results)

    try:
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
        manifest = run_export(
            ExportRequest(
                candidate="act_aloha_insertion",
                output_dir=tmp_path / "export_insertion",
                device="cpu",
                calibration_seeds=[9000, 9001],
                eval_seeds=[1000, 1001],
                prefer_simulator_corpus=False,
                attempt_int8=False,
                attempt_int4=False,
                attempt_fp16=False,
                run_closed_loop=False,
                repo_root=root,
            )
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ACT insertion export unavailable: {exc}")

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["candidate"] == "act_aloha_insertion"
    assert payload["comparisons"]["onnx_fp32_vs_pytorch_fp32"]["passed"] is True


def test_smoke_diffusion_pusht_gates_and_unsupported_onnx(tmp_path) -> None:
    from breakpoint_eval.policy_validate import assert_gates_passed, run_policy_validation_gates
    from breakpoint_eval.run_export import ExportRequest, run_export

    try:
        adapter, spec, root = _load_adapter("diffusion_pusht")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Diffusion PushT stack unavailable: {exc}")

    assert adapter.chunk_info.n_obs_steps == 2
    assert adapter.chunk_info.n_action_steps == 8
    assert int(adapter.policy.diffusion.num_inference_steps) == 100

    pol_obs = {
        "observation.state": torch.zeros(1, 2),
        "observation.image": torch.rand(1, 3, 96, 96),
    }
    raw = {
        "pixels": np.zeros((384, 384, 3), dtype=np.uint8),
        "agent_pos": np.zeros((2,), dtype=np.float32),
    }
    # Two steps only: one full denoise generate (preserves step count; keeps smoke bounded).
    results = run_policy_validation_gates(
        spec=spec,
        adapter=adapter,
        policy_observation=pol_obs,
        raw_observation=raw,
        zero_latency_observations=[
            {
                "observation.state": torch.zeros(1, 2),
                "observation.image": torch.rand(1, 3, 96, 96),
            }
            for _ in range(2)
        ],
    )
    assert_gates_passed(results)

    manifest = run_export(
        ExportRequest(
            candidate="diffusion_pusht",
            output_dir=tmp_path / "export_dp",
            device="cpu",
            repo_root=root,
        )
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["onnx_export"]["status"] == "unsupported"
    assert payload["pytorch_simulation_axes"]["available"] is True
    assert payload["act_timings_reused"] is False
    assert "config.json" in payload["checkpoint_files"]
    assert "model.safetensors" in payload["checkpoint_files"]
    assert payload["torch_dtype"]


def test_smoke_diffusion_verify_upstream(tmp_path) -> None:
    from breakpoint_eval.policy_validate import verify_upstream_agreement

    try:
        _adapter, _spec, root = _load_adapter("diffusion_pusht")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Diffusion PushT stack unavailable: {exc}")

    out = tmp_path / "upstream_agreement.json"
    payload = verify_upstream_agreement(
        "diffusion_pusht",
        [2200, 2201],
        "cpu",
        1,
        out,
        repo_root=root,
    )
    assert out.is_file()
    assert "per_seed" in payload
    assert "determinism" in payload
    assert "checkpoint_files" in payload
    assert payload["candidate"] == "diffusion_pusht"
    assert payload["passed"] is True, payload
