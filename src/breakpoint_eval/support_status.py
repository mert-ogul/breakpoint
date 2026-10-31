"""Support matrix for pinned Hub candidates (not a plugin registry).

Status vocabulary:
- yes: retained evidence exists for that gate
- no: attempted and failed / blocked, with recorded reason
- not_attempted: no retained attempt
- unsupported: explicitly out of scope for this candidate
- unverified: implementation or synthetic tests exist, but no retained
  real-simulation / hardware evidence for the claim
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

GateValue = Literal["yes", "no", "not_attempted", "unsupported", "unverified"]


@dataclass(frozen=True)
class CandidateSupport:
    candidate: str
    policy_type: str
    env_task: str
    loaded: GateValue
    ran: GateValue
    reproduced_baseline: GateValue
    # Implementation available vs synthetic unit checks vs real sim validation.
    camera_impl: GateValue
    camera_synthetic: GateValue
    camera_sim: GateValue
    zero_latency_impl: GateValue
    zero_latency_synthetic: GateValue
    zero_latency_sim: GateValue
    onnx_fp32: GateValue
    onnx_int8: GateValue
    onnx_int4: GateValue
    device_profile: GateValue
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Evidence-backed status only. Success rates and hardware runs come from loaded
# evidence; timings are not copied across rows. No physical-robot experiments are claimed.
SUPPORT_MATRIX: dict[str, CandidateSupport] = {
    "act_aloha_transfer": CandidateSupport(
        candidate="act_aloha_transfer",
        policy_type="act",
        env_task="AlohaTransferCube-v0",
        loaded="yes",
        ran="yes",
        reproduced_baseline="no",
        camera_impl="yes",
        camera_synthetic="yes",
        camera_sim="yes",
        zero_latency_impl="yes",
        zero_latency_synthetic="yes",
        zero_latency_sim="yes",
        onnx_fp32="yes",
        onnx_int8="yes",
        onnx_int4="yes",
        device_profile="yes",
        notes=(
            "deployment stress benchmark packaged in benchmarks/act-transfer/deployment-stress: 810 CPU episodes "
            "(seeds 5000-5029), ONNX FP32 baseline 25/30, static QDQ INT8 24/30, "
            "INT4 weight-only 23/30, constant delay, resolution, JPEG, H.264 bitrate, "
            "and laptop-m3 / rpi5 latency replay. "
            "Card-scale reproduction not claimed. The rpi5 profile was captured under "
            "reported throttling. Do not reuse timings for other candidates. "
            "No hardware experiments claimed."
        ),
    ),
    "act_aloha_insertion": CandidateSupport(
        candidate="act_aloha_insertion",
        policy_type="act",
        env_task="AlohaInsertion-v0",
        loaded="unverified",
        ran="not_attempted",
        reproduced_baseline="no",
        camera_impl="yes",
        camera_synthetic="yes",
        camera_sim="unverified",
        zero_latency_impl="yes",
        zero_latency_synthetic="yes",
        zero_latency_sim="unverified",
        onnx_fp32="unverified",
        onnx_int8="not_attempted",
        onnx_int4="not_attempted",
        device_profile="not_attempted",
        notes=(
            "Shares ACT adapter/export implementation with transfer; integration "
            "smoke exercised load/export but no retained load_evidence artifact. "
            "Do not pool success with transfer. No hardware experiments claimed."
        ),
    ),
    "diffusion_pusht": CandidateSupport(
        candidate="diffusion_pusht",
        policy_type="diffusion",
        env_task="PushT-v0",
        loaded="unverified",
        ran="not_attempted",
        reproduced_baseline="no",
        camera_impl="yes",
        camera_synthetic="yes",
        camera_sim="unverified",
        zero_latency_impl="yes",
        zero_latency_synthetic="yes",
        zero_latency_sim="unverified",
        onnx_fp32="unsupported",
        onnx_int8="unsupported",
        onnx_int4="unsupported",
        device_profile="unsupported",
        notes=(
            "Implementation and synthetic/stub gates exist; real PushT episode "
            "ran/reproduced not claimed. ORT action-chunk export unsupported. "
            "Do not copy ACT device timings. No hardware experiments claimed."
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
