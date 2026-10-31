"""Diffusion Policy export boundary.

Full action generation includes the Python noise-scheduler loop. Exporting only
the denoising network is not a complete policy-inference graph; that case must be
labeled as mixed runtime. When ONNX export is unsupported, PyTorch simulation
axes remain available and ONNX/device/precision combinations are marked unsupported.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from breakpoint_eval.act_export import write_json
from breakpoint_eval.diffusion_adapter import (
    TIMED_BOUNDARY_DIFFUSION_GENERATE,
    diffusion_inference_provenance,
    pytorch_full_generation_provenance,
)
from breakpoint_eval.sim_env import control_period_s, load_policy
from breakpoint_eval.stack import (
    LEROBOT_COMMIT,
    download_candidate_checkpoint,
    ensure_lerobot_on_path,
    repo_root_from_package,
    resolve_candidate,
)

LOGGER = logging.getLogger("breakpoint_eval.diffusion_export")

UNSUPPORTED_REASON = (
    "Diffusion Policy fresh inference is a multi-step denoising loop "
    "(observation encode + noise scheduler + UNet × num_inference_steps + "
    "unnormalize). A single-shot ONNX action-chunk graph matching that boundary "
    "is not supported. PyTorch simulation axes remain available. If only the "
    "denoising UNet is exported later, profile the Python scheduler loop with it "
    "and label the runtime as mixed — never treat one UNet pass as policy latency."
)


def run_diffusion_export_record(
    *,
    candidate: str = "diffusion_pusht",
    output_dir: Path | None = None,
    device: str = "cpu",
    repo_root: Path | None = None,
) -> Path:
    """Write an export manifest that records ONNX as unsupported for Diffusion."""
    spec = resolve_candidate(candidate)
    if spec.policy_type != "diffusion":
        raise ValueError(f"expected diffusion candidate, got {candidate!r} ({spec.policy_type})")

    root = (repo_root or repo_root_from_package()).resolve()
    ensure_lerobot_on_path(root)
    out = (output_dir or (root / "artifacts" / "export" / candidate)).resolve()
    out.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = download_candidate_checkpoint(spec, root / ".cache" / "huggingface")
    period_s = control_period_s(spec)
    policy = load_policy(spec, str(checkpoint_dir), device)
    inference = diffusion_inference_provenance(policy)
    runtime = pytorch_full_generation_provenance()

    payload: dict[str, Any] = {
        "candidate": candidate,
        "hub_repo": spec.hub_repo,
        "hub_revision": spec.hub_revision,
        "lerobot_commit": LEROBOT_COMMIT,
        "policy_type": "diffusion",
        "env_type": spec.env_type,
        "env_task": spec.env_task,
        "control_period_s": period_s,
        "checkpoint_dir": str(checkpoint_dir),
        "device": device,
        "inference": inference,
        "onnx_export": {
            "status": "unsupported",
            "reason": UNSUPPORTED_REASON,
            "variants": [],
        },
        "precision_recipes": {
            "onnx_fp32": "unsupported",
            "int8": "unsupported",
            "int4_weight_only": "unsupported",
            "fp16": "unsupported",
        },
        "device_profile": {
            "status": "unsupported",
            "reason": (
                "No ORT action-chunk model to profile. Use PyTorch full "
                f"action-generation timing ({TIMED_BOUNDARY_DIFFUSION_GENERATE}) "
                "or a explicitly labeled mixed UNet+scheduler path. "
                "Do not copy ACT profile latencies."
            ),
            "pytorch_timing_boundary": runtime.to_dict(),
        },
        "pytorch_simulation_axes": {
            "available": True,
            "latency": True,
            "camera_degrade": True,
            "combined_stress": True,
            "note": "Stress sweeps may use the PyTorch DiffusionPolicy adapter.",
        },
        "act_timings_reused": False,
    }
    path = out / "export_manifest.json"
    write_json(path, payload)
    LOGGER.info("Wrote diffusion export record (ONNX unsupported): %s", path)
    return path
