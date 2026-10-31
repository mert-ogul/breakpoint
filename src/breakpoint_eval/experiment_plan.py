"""Fixed multi-condition experiment plans (seeds, grids, isolation rules).

Defaults match docs/methodology.md. Grids are configurable experimental choices,
not known failure thresholds.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

# Sample counts (always recorded explicitly in the plan artifact).
N_EPISODES_SMOKE = 5
N_EPISODES_FULL = 100

# Suggested starting grids (methodology.md).
DEFAULT_LATENCY_MS = (0, 20, 40, 60, 80, 120, 200)
# Categories rather than a continuous bit-depth axis.
DEFAULT_PRECISION_VARIANTS = ("onnx_fp32", "int8", "int4_weight_only")
# Linear size fractions of native ALOHA 480x640; actual HxW recorded per level.
NATIVE_HW = (480, 640)
DEFAULT_RESOLUTION_LINEAR = (1.0, 0.5, 0.25)  # native, 1/2, 1/4
# bypass = native uncompressed; qualities are experimental choices.
DEFAULT_JPEG_QUALITIES: tuple[str | int, ...] = ("bypass", 95, 75, 50, 30, 10)
# Quality-only H.264 bitrate ladder (kbps); bypass = native uncompressed.
DEFAULT_H264_BITRATES_KBPS: tuple[str | float, ...] = (
    "bypass",
    2000.0,
    1000.0,
    500.0,
    250.0,
    100.0,
)

DEFAULT_ENV_SEED_START = 1000
DEFAULT_POLICY_SEED_OFFSET = 1_000_000


class ExperimentAxis(StrEnum):
    LATENCY_MS = "latency_ms"
    PRECISION = "precision"
    RESOLUTION = "resolution"
    JPEG = "jpeg"
    H264 = "h264"
    DEVICE_PROFILE = "device_profile"
    COMBINED = "combined"


class PlanPhase(StrEnum):
    EXPLORATORY = "exploratory"
    CONFIRMATION = "confirmation"
    SMOKE = "smoke"


@dataclass(frozen=True)
class ConditionSpec:
    condition_id: str
    axis: str
    params: dict[str, Any]
    label: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeSeedPair:
    episode_index: int
    env_seed: int
    policy_rng_seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentPlan:
    """One fixed plan revision: paired seeds + condition list + isolation notes."""

    plan_id: str
    candidate: str
    phase: str
    n_episodes: int
    axis: str
    seed_pairs: list[EpisodeSeedPair]
    conditions: list[ConditionSpec]
    isolation_notes: list[str] = field(default_factory=list)
    baseline_condition_id: str | None = None
    confirms_plan_id: str | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.n_episodes < 1:
            raise ValueError("n_episodes must be >= 1")
        if len(self.seed_pairs) != self.n_episodes:
            raise ValueError("seed_pairs length must equal n_episodes")
        if not self.conditions:
            raise ValueError("conditions must be non-empty")
        ids = [c.condition_id for c in self.conditions]
        if len(ids) != len(set(ids)):
            raise ValueError("condition_id values must be unique")
        if self.baseline_condition_id is not None:
            if self.baseline_condition_id not in ids:
                raise ValueError("baseline_condition_id missing from conditions")

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "candidate": self.candidate,
            "phase": self.phase,
            "n_episodes": self.n_episodes,
            "axis": self.axis,
            "seed_pairs": [s.to_dict() for s in self.seed_pairs],
            "env_seeds": [s.env_seed for s in self.seed_pairs],
            "policy_rng_seeds": [s.policy_rng_seed for s in self.seed_pairs],
            "conditions": [c.to_dict() for c in self.conditions],
            "baseline_condition_id": self.baseline_condition_id,
            "confirms_plan_id": self.confirms_plan_id,
            "isolation_notes": list(self.isolation_notes),
            "notes": list(self.notes),
            "conventions": {
                "n_episodes_smoke": N_EPISODES_SMOKE,
                "n_episodes_full_default": N_EPISODES_FULL,
                "hundred_episodes_not_precise_boundary": True,
                "grids_are_experimental_choices_not_known_thresholds": True,
                "no_optional_stopping_for_headline": True,
                "combined_only_after_isolated": True,
                "effects_not_assumed_additive": True,
            },
        }


def build_seed_pairs(
    n_episodes: int,
    *,
    env_seed_start: int = DEFAULT_ENV_SEED_START,
    policy_seed_offset: int = DEFAULT_POLICY_SEED_OFFSET,
    env_seeds: list[int] | None = None,
    policy_rng_seeds: list[int] | None = None,
) -> list[EpisodeSeedPair]:
    if n_episodes < 1:
        raise ValueError("n_episodes must be >= 1")
    if env_seeds is not None and len(env_seeds) != n_episodes:
        raise ValueError("env_seeds length must match n_episodes")
    if policy_rng_seeds is not None and len(policy_rng_seeds) != n_episodes:
        raise ValueError("policy_rng_seeds length must match n_episodes")
    env = env_seeds or [env_seed_start + i for i in range(n_episodes)]
    policy = policy_rng_seeds or [e + policy_seed_offset for e in env]
    return [
        EpisodeSeedPair(episode_index=i, env_seed=env[i], policy_rng_seed=policy[i])
        for i in range(n_episodes)
    ]


def resolution_hw(
    linear_scale: float, *, native_hw: tuple[int, int] = NATIVE_HW
) -> tuple[int, int]:
    if linear_scale <= 0 or linear_scale > 1:
        raise ValueError("linear_scale must be in (0, 1]")
    h = max(1, int(round(native_hw[0] * linear_scale)))
    w = max(1, int(round(native_hw[1] * linear_scale)))
    return h, w


def latency_conditions(
    latency_ms: tuple[int, ...] = DEFAULT_LATENCY_MS,
) -> list[ConditionSpec]:
    out: list[ConditionSpec] = []
    for ms in latency_ms:
        if ms < 0:
            raise ValueError("latency_ms must be >= 0")
        out.append(
            ConditionSpec(
                condition_id=f"latency_ms_{ms}",
                axis=ExperimentAxis.LATENCY_MS.value,
                params={
                    "latency_ms": ms,
                    "latency_s": ms / 1000.0,
                    "precision_held": "onnx_fp32",
                    "timing_schedule": "constant_added_inference_delay",
                },
                label=f"constant delay {ms} ms (FP32 held)",
            )
        )
    return out


def precision_conditions(
    variants: tuple[str, ...] = DEFAULT_PRECISION_VARIANTS,
    *,
    timing_schedule: str = "constant_latency_ms_0",
) -> list[ConditionSpec]:
    out: list[ConditionSpec] = []
    for v in variants:
        out.append(
            ConditionSpec(
                condition_id=f"precision_{v}",
                axis=ExperimentAxis.PRECISION.value,
                params={
                    "variant_label": v,
                    "timing_schedule": timing_schedule,
                    "categorical": True,
                    "not_continuous_bit_depth": True,
                },
                label=f"precision recipe {v} (timing fixed: {timing_schedule})",
            )
        )
    return out


def resolution_conditions(
    linear_scales: tuple[float, ...] = DEFAULT_RESOLUTION_LINEAR,
    *,
    native_hw: tuple[int, int] = NATIVE_HW,
) -> list[ConditionSpec]:
    out: list[ConditionSpec] = []
    for scale in linear_scales:
        h, w = resolution_hw(scale, native_hw=native_hw)
        if scale == 1.0:
            cid = "resolution_native"
            params: dict[str, Any] = {
                "mode": "native",
                "linear_scale": 1.0,
                "native_hw": list(native_hw),
                "capture_hw": list(native_hw),
                "effective_policy_hw": list(native_hw),
            }
            label = f"resolution native {native_hw[0]}x{native_hw[1]}"
        else:
            cid = f"resolution_linear_{scale:g}_capture_{h}x{w}"
            params = {
                "mode": "downsample",
                "linear_scale": scale,
                "native_hw": list(native_hw),
                "capture_hw": [h, w],
                "effective_policy_hw": list(native_hw),
            }
            label = f"resolution linear {scale:g} → capture {h}x{w}"
        out.append(
            ConditionSpec(
                condition_id=cid,
                axis=ExperimentAxis.RESOLUTION.value,
                params=params,
                label=label,
            )
        )
    return out


def jpeg_conditions(
    qualities: tuple[str | int, ...] = DEFAULT_JPEG_QUALITIES,
) -> list[ConditionSpec]:
    out: list[ConditionSpec] = []
    for q in qualities:
        if q == "bypass":
            out.append(
                ConditionSpec(
                    condition_id="jpeg_bypass",
                    axis=ExperimentAxis.JPEG.value,
                    params={"mode": "native", "jpeg_quality": None, "bypass": True},
                    label="JPEG bypass (native uncompressed)",
                )
            )
            continue
        qi = int(q)
        if not 1 <= qi <= 100:
            raise ValueError("jpeg quality must be in [1, 100]")
        out.append(
            ConditionSpec(
                condition_id=f"jpeg_q{qi}",
                axis=ExperimentAxis.JPEG.value,
                params={"mode": "jpeg", "jpeg_quality": qi, "bypass": False},
                label=f"JPEG quality {qi}",
            )
        )
    return out


def h264_conditions(
    bitrates_kbps: tuple[str | float, ...] = DEFAULT_H264_BITRATES_KBPS,
    *,
    frame_rate_hz: float = 50.0,
    gop: int = 30,
    preset: str = "veryfast",
    latency_mode: str = "quality_only",
) -> list[ConditionSpec]:
    """H.264 visual-quality ladder. Quality-only excludes codec/transport delay."""
    out: list[ConditionSpec] = []
    for br in bitrates_kbps:
        if br == "bypass":
            out.append(
                ConditionSpec(
                    condition_id="h264_bypass",
                    axis=ExperimentAxis.H264.value,
                    params={
                        "mode": "native",
                        "h264_bitrate_kbps": None,
                        "bypass": True,
                        "h264_latency_mode": latency_mode,
                    },
                    label="H.264 bypass (native uncompressed)",
                )
            )
            continue
        kbps = float(br)
        if kbps <= 0:
            raise ValueError("h264 bitrate_kbps must be > 0")
        out.append(
            ConditionSpec(
                condition_id=f"h264_{int(kbps)}kbps",
                axis=ExperimentAxis.H264.value,
                params={
                    "mode": "h264",
                    "h264_bitrate_kbps": kbps,
                    "frame_rate_hz": frame_rate_hz,
                    "h264_gop": gop,
                    "h264_preset": preset,
                    "h264_latency_mode": latency_mode,
                    "h264_deployment_buffer_frames": 0,
                    "bypass": False,
                    "excludes_codec_transport_delay": latency_mode == "quality_only",
                },
                label=f"H.264 {kbps:g} kbps ({latency_mode})",
            )
        )
    return out


def make_isolated_plan(
    *,
    axis: ExperimentAxis,
    candidate: str,
    phase: PlanPhase = PlanPhase.EXPLORATORY,
    n_episodes: int | None = None,
    env_seed_start: int = DEFAULT_ENV_SEED_START,
    policy_seed_offset: int = DEFAULT_POLICY_SEED_OFFSET,
    plan_id: str | None = None,
    confirms_plan_id: str | None = None,
) -> ExperimentPlan:
    if phase is PlanPhase.SMOKE:
        n = n_episodes if n_episodes is not None else N_EPISODES_SMOKE
    else:
        n = n_episodes if n_episodes is not None else N_EPISODES_FULL
    if phase is PlanPhase.CONFIRMATION and confirms_plan_id is None:
        raise ValueError("confirmation plans require confirms_plan_id")

    seeds = build_seed_pairs(
        n,
        env_seed_start=env_seed_start,
        policy_seed_offset=policy_seed_offset,
    )
    if axis is ExperimentAxis.LATENCY_MS:
        conditions = latency_conditions()
        baseline = "latency_ms_0"
        isolation = [
            "Synthetic constant delay with precision held at onnx_fp32.",
            "Distinct from measured per-variant hardware timing replay.",
        ]
    elif axis is ExperimentAxis.PRECISION:
        conditions = precision_conditions()
        baseline = "precision_onnx_fp32"
        isolation = [
            "Precision categories under a fixed timing schedule (constant_latency_ms_0).",
            "Not a continuous bit-depth threshold; list results per recipe.",
        ]
    elif axis is ExperimentAxis.RESOLUTION:
        conditions = resolution_conditions()
        baseline = "resolution_native"
        isolation = ["Resolution axis only; latency and precision held at defaults."]
    elif axis is ExperimentAxis.JPEG:
        conditions = jpeg_conditions()
        baseline = "jpeg_bypass"
        isolation = ["JPEG visual-quality axis only; native resolution."]
    elif axis is ExperimentAxis.H264:
        conditions = h264_conditions()
        baseline = "h264_bypass"
        isolation = [
            "H.264 causal stream quality-only axis (libx264 zerolatency, no B-frames).",
            "Excludes codec/transport delay from inference latency; not a prerecorded trajectory.",
            "Unsupported without a verified FFmpeg+libx264+PyAV install.",
        ]
    else:
        raise ValueError("use make_combined_plan for COMBINED axis")

    pid = plan_id or f"{candidate}__{axis.value}__{phase.value}__n{n}"
    notes = [
        f"n_episodes={n} is explicit; 100 does not guarantee a precise boundary.",
        "Exploratory and confirmation results must remain separate.",
        "No optional stopping for a desired headline.",
    ]
    if phase is PlanPhase.SMOKE:
        notes.append("Smoke plan (5 episodes default); not for publication claims.")
    return ExperimentPlan(
        plan_id=pid,
        candidate=candidate,
        phase=phase.value,
        n_episodes=n,
        axis=axis.value,
        seed_pairs=seeds,
        conditions=conditions,
        isolation_notes=isolation,
        baseline_condition_id=baseline,
        confirms_plan_id=confirms_plan_id,
        notes=notes,
    )


def make_combined_plan(
    *,
    candidate: str,
    conditions: list[ConditionSpec],
    baseline_condition_id: str,
    n_episodes: int = N_EPISODES_FULL,
    phase: PlanPhase = PlanPhase.EXPLORATORY,
    env_seed_start: int = DEFAULT_ENV_SEED_START,
    policy_seed_offset: int = DEFAULT_POLICY_SEED_OFFSET,
    plan_id: str | None = None,
    confirms_plan_id: str | None = None,
) -> ExperimentPlan:
    """Explicit small combined grid, only after isolated sweeps."""
    if len(conditions) < 2:
        raise ValueError("combined plan needs at least two conditions")
    seeds = build_seed_pairs(
        n_episodes,
        env_seed_start=env_seed_start,
        policy_seed_offset=policy_seed_offset,
    )
    pid = plan_id or f"{candidate}__combined__{phase.value}__n{n_episodes}"
    return ExperimentPlan(
        plan_id=pid,
        candidate=candidate,
        phase=phase.value,
        n_episodes=n_episodes,
        axis=ExperimentAxis.COMBINED.value,
        seed_pairs=seeds,
        conditions=conditions,
        isolation_notes=[
            "Combined grid is explicit and small; do not imply additive axis effects.",
            "Prefer isolated sweeps before interpreting combined points.",
        ],
        baseline_condition_id=baseline_condition_id,
        confirms_plan_id=confirms_plan_id,
        notes=[
            "Combined results are not a substitute for isolated axis summaries.",
            f"n_episodes={n_episodes} is explicit.",
        ],
    )


def write_experiment_plan(plan: ExperimentPlan, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_experiment_plan(path: Path) -> ExperimentPlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    seeds = [
        EpisodeSeedPair(
            episode_index=int(s["episode_index"]),
            env_seed=int(s["env_seed"]),
            policy_rng_seed=int(s["policy_rng_seed"]),
        )
        for s in payload["seed_pairs"]
    ]
    conditions = [
        ConditionSpec(
            condition_id=c["condition_id"],
            axis=c["axis"],
            params=dict(c["params"]),
            label=c["label"],
        )
        for c in payload["conditions"]
    ]
    return ExperimentPlan(
        plan_id=payload["plan_id"],
        candidate=payload["candidate"],
        phase=payload["phase"],
        n_episodes=int(payload["n_episodes"]),
        axis=payload["axis"],
        seed_pairs=seeds,
        conditions=conditions,
        isolation_notes=list(payload.get("isolation_notes") or []),
        baseline_condition_id=payload.get("baseline_condition_id"),
        confirms_plan_id=payload.get("confirms_plan_id"),
        notes=list(payload.get("notes") or []),
    )
