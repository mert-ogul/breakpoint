"""Tested support matrix for pinned Hub candidates (not a plugin registry)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

GateValue = Literal["yes", "no", "not_attempted", "unsupported"]


@dataclass(frozen=True)
class CandidateSupport:
    candidate: str
    policy_type: str
    env_task: str
    loaded: GateValue
    ran: GateValue
    reproduced_baseline: GateValue
    camera_degrade: GateValue
    zero_latency_equivalence: GateValue
    onnx_fp32: GateValue
    onnx_int8: GateValue
    onnx_int4: GateValue
    device_profile: GateValue
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Evidence-backed status only. Never invent success rates or copy timings across rows.
SUPPORT_MATRIX: dict[str, CandidateSupport] = {
    "act_aloha_transfer": CandidateSupport(
        candidate="act_aloha_transfer",
        policy_type="act",
        env_task="AlohaTransferCube-v0",
        loaded="yes",
        ran="no",
        reproduced_baseline="no",
        camera_degrade="yes",
        zero_latency_equivalence="yes",
        onnx_fp32="yes",
        onnx_int8="yes",
        onnx_int4="yes",
        device_profile="yes",
        notes=(
            "ran blocked by MuJoCo GL on some hosts. ONNX/device axes verified on "
            "ACT transfer only; do not reuse those timings for other candidates."
        ),
    ),
    "act_aloha_insertion": CandidateSupport(
        candidate="act_aloha_insertion",
        policy_type="act",
        env_task="AlohaInsertion-v0",
        loaded="yes",
        ran="no",
        reproduced_baseline="no",
        camera_degrade="yes",
        zero_latency_equivalence="yes",
        onnx_fp32="yes",
        onnx_int8="not_attempted",
        onnx_int4="not_attempted",
        device_profile="not_attempted",
        notes=(
            "Shares ACT adapter/export path with transfer; FP32 parity smoked on "
            "this checkpoint. Separate env success semantics (peg/pin). "
            "Do not pool success with transfer."
        ),
    ),
    "diffusion_pusht": CandidateSupport(
        candidate="diffusion_pusht",
        policy_type="diffusion",
        env_task="PushT-v0",
        loaded="yes",
        ran="not_attempted",
        reproduced_baseline="no",
        camera_degrade="yes",
        zero_latency_equivalence="yes",
        onnx_fp32="unsupported",
        onnx_int8="unsupported",
        onnx_int4="unsupported",
        device_profile="unsupported",
        notes=(
            "PyTorch simulation axes retained. Full action-generation latency "
            "(all denoising steps) is the timing boundary; ORT action-chunk export "
            "is unsupported. Do not copy ACT device timings."
        ),
    ),
}


def support_row(candidate: str) -> CandidateSupport:
    try:
        return SUPPORT_MATRIX[candidate]
    except KeyError as exc:
        known = ", ".join(sorted(SUPPORT_MATRIX))
        raise ValueError(f"Unknown candidate {candidate!r}. Known: {known}") from exc


def support_matrix_table() -> list[dict[str, Any]]:
    return [SUPPORT_MATRIX[k].to_dict() for k in sorted(SUPPORT_MATRIX)]
