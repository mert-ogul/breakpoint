"""Ordered latency traces for device-profile replay (not synthetic distributions)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

PROFILE_SCHEMA_VERSION = 1
TIMED_BOUNDARY_ORT_RUN = "ort_session_run_action_chunk"


class EndOfTraceBehavior(StrEnum):
    FAIL = "fail"
    CYCLE = "cycle"


class TraceExhaustedError(RuntimeError):
    """Raised when a non-cycling replayer runs out of samples."""


@dataclass(frozen=True)
class LatencyTraceStats:
    count: int
    median_s: float
    p95_s: float
    p99_s: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "median_s": self.median_s,
            "p95_s": self.p95_s,
            "p99_s": self.p99_s,
        }


def summarize_latency_s(values: list[float] | np.ndarray) -> LatencyTraceStats:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("latency sequence is empty")
    return LatencyTraceStats(
        count=int(arr.size),
        median_s=float(np.median(arr)),
        p95_s=float(np.percentile(arr, 95)),
        p99_s=float(np.percentile(arr, 99)),
    )


def validate_latency_sequence(values: list[float] | np.ndarray) -> list[float]:
    """Reject NaN, inf, negative, or empty sequences."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"latency sequence must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("latency sequence is empty")
    if not np.isfinite(arr).all():
        raise ValueError("latency sequence contains NaN or Inf")
    if (arr < 0).any():
        raise ValueError("latency sequence contains negative values")
    return [float(x) for x in arr.tolist()]


@dataclass
class LatencyProfile:
    """Device-specific measured inference latencies with identity metadata."""

    payload: dict[str, Any]
    latency_s: list[float]
    path: Path | None = None

    @property
    def stats(self) -> LatencyTraceStats:
        return summarize_latency_s(self.latency_s)

    def identity(self) -> dict[str, Any]:
        return {
            "schema_version": self.payload.get("schema_version"),
            "device_label": self.payload.get("device_label"),
            "candidate": self.payload.get("candidate"),
            "model_sha256": self.payload.get("model", {}).get("sha256"),
            "precision_recipe": self.payload.get("model", {}).get("precision_recipe"),
            "variant_label": self.payload.get("model", {}).get("variant_label"),
            "input_shapes": self.payload.get("input_contract", {}).get("shapes"),
            "corpus_hash": self.payload.get("input_contract", {}).get("corpus_sha256"),
            "timed_boundary": self.payload.get("timed_boundary"),
            "chunk_horizon": self.payload.get("chunk_horizon"),
            "runtime_provider": self.payload.get("runtime", {}).get("session_providers"),
            "intra_op_num_threads": self.payload.get("runtime", {}).get("intra_op_num_threads"),
            "inter_op_num_threads": self.payload.get("runtime", {}).get("inter_op_num_threads"),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.payload


def load_latency_profile(path: Path) -> LatencyProfile:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported profile schema_version={payload.get('schema_version')!r}; "
            f"expected {PROFILE_SCHEMA_VERSION}"
        )
    measurements = payload.get("measurements")
    if not isinstance(measurements, dict):
        raise ValueError("profile missing measurements object")
    raw = measurements.get("inference_latency_s")
    if raw is None:
        raise ValueError("profile missing measurements.inference_latency_s")
    latency_s = validate_latency_sequence(raw)
    expected_n = measurements.get("n_measured")
    if expected_n is not None and int(expected_n) != len(latency_s):
        raise ValueError(f"n_measured={expected_n} does not match sequence length {len(latency_s)}")
    # Prefer stored summary but recompute for integrity.
    stats = summarize_latency_s(latency_s)
    stored = measurements.get("summary") or {}
    for key, value in stats.to_dict().items():
        if key == "count":
            continue
        if key in stored and not math.isclose(
            float(stored[key]), float(value), rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError(
                f"profile summary {key}={stored[key]!r} mismatches recomputed {value!r}"
            )
    validation = payload.get("output_validation")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise ValueError(
            "profile output_validation.passed is not true; "
            "refuse to replay unvalidated device timings"
        )
    if payload.get("timed_boundary") != TIMED_BOUNDARY_ORT_RUN:
        raise ValueError(
            f"unsupported timed_boundary={payload.get('timed_boundary')!r}; "
            f"expected {TIMED_BOUNDARY_ORT_RUN!r}"
        )
    return LatencyProfile(payload=payload, latency_s=latency_s, path=path)


def assert_profile_compatible(
    profile: LatencyProfile,
    *,
    candidate: str,
    model_sha256: str | None = None,
    precision_recipe: str | None = None,
    variant_label: str | None = None,
    corpus_sha256: str | None = None,
    input_shapes: dict[str, list[int]] | None = None,
    chunk_horizon: int | None = None,
    hub_revision: str | None = None,
    timed_boundary: str | None = TIMED_BOUNDARY_ORT_RUN,
) -> None:
    """Reject trace/artifact mismatches. A device label alone is insufficient."""
    ident = profile.identity()
    errors: list[str] = []
    if ident.get("candidate") != candidate:
        errors.append(f"candidate: profile={ident.get('candidate')!r} requested={candidate!r}")
    if model_sha256 is not None and ident.get("model_sha256") != model_sha256:
        errors.append(
            f"model_sha256: profile={ident.get('model_sha256')!r} requested={model_sha256!r}"
        )
    if precision_recipe is not None and ident.get("precision_recipe") != precision_recipe:
        errors.append(
            f"precision_recipe: profile={ident.get('precision_recipe')!r} "
            f"requested={precision_recipe!r}"
        )
    if variant_label is not None and ident.get("variant_label") != variant_label:
        errors.append(
            f"variant_label: profile={ident.get('variant_label')!r} requested={variant_label!r}"
        )
    if corpus_sha256 is not None and ident.get("corpus_hash") != corpus_sha256:
        errors.append("corpus_sha256 mismatch")
    if input_shapes is not None and ident.get("input_shapes") != input_shapes:
        errors.append(
            f"input_shapes: profile={ident.get('input_shapes')!r} requested={input_shapes!r}"
        )
    if chunk_horizon is not None and int(ident.get("chunk_horizon") or -1) != int(chunk_horizon):
        errors.append(
            f"chunk_horizon: profile={ident.get('chunk_horizon')!r} requested={chunk_horizon!r}"
        )
    if hub_revision is not None:
        profile_rev = profile.payload.get("hub_revision")
        if profile_rev is not None and str(profile_rev) != str(hub_revision):
            errors.append(f"hub_revision: profile={profile_rev!r} requested={hub_revision!r}")
    if timed_boundary is not None and ident.get("timed_boundary") != timed_boundary:
        errors.append(
            f"timed_boundary: profile={ident.get('timed_boundary')!r} requested={timed_boundary!r}"
        )
    if errors:
        raise ValueError("latency profile incompatible with run artifacts: " + "; ".join(errors))


def contiguous_window_starts(
    n_trace: int,
    *,
    window_length: int,
    n_episodes: int,
) -> list[int]:
    """Evenly spaced contiguous-window starts: floor(i*(N-W)/(n_episodes-1)).

    Deterministic sampling across the measured run — not continuous full-trace
    replay. Distinct from block resampling.
    """
    if window_length < 1:
        raise ValueError("window_length must be >= 1")
    if n_episodes < 1:
        raise ValueError("n_episodes must be >= 1")
    if n_trace < window_length:
        raise ValueError(
            f"trace length {n_trace} is shorter than window_length {window_length}"
        )
    if n_episodes == 1:
        return [0]
    span = n_trace - window_length
    return [int(math.floor(i * span / (n_episodes - 1))) for i in range(n_episodes)]


def extract_contiguous_window(
    values: list[float],
    *,
    start: int,
    length: int,
) -> list[float]:
    if start < 0:
        raise ValueError(f"window start must be >= 0, got {start}")
    if length < 1:
        raise ValueError("window length must be >= 1")
    end = start + length
    if end > len(values):
        raise ValueError(f"window [{start}, {end}) exceeds trace length {len(values)}")
    return list(values[start:end])


class LatencyTraceReplayer:
    """Replay measured latencies in order to preserve jitter bursts."""

    def __init__(
        self,
        latency_s: list[float],
        *,
        end_of_trace: EndOfTraceBehavior = EndOfTraceBehavior.FAIL,
        block_length: int | None = None,
        block_resample_seed: int | None = None,
        window_start: int | None = None,
        window_length: int | None = None,
        source_path: str | None = None,
    ) -> None:
        self._base = validate_latency_sequence(latency_s)
        self.end_of_trace = EndOfTraceBehavior(end_of_trace)
        self.source_path = source_path
        self._index = 0
        self._emitted = 0
        self._consumed_trace_indices: list[int] = []
        self._consumed_delays_s: list[float] = []
        self._window_start: int | None = None
        self._window_length: int | None = None
        self._sampling_design = "full_trace"
        self._sequence = list(self._base)

        using_window = window_start is not None or window_length is not None
        using_block = block_length is not None or block_resample_seed is not None
        if using_window and using_block:
            raise ValueError(
                "contiguous window sampling and block resampling are mutually exclusive; "
                "they are not interchangeable designs"
            )

        if using_window:
            if window_start is None or window_length is None:
                raise ValueError("contiguous window requires both window_start and window_length")
            if self.end_of_trace is EndOfTraceBehavior.CYCLE:
                raise ValueError("contiguous window replay must fail on exhaustion (not cycle)")
            self._window_start = int(window_start)
            self._window_length = int(window_length)
            self._sequence = extract_contiguous_window(
                self._base, start=self._window_start, length=self._window_length
            )
            self._sampling_design = "contiguous_window"
        elif block_length is not None:
            if block_length < 1:
                raise ValueError("block_length must be >= 1")
            if block_resample_seed is None:
                raise ValueError("block resampling requires an explicit seed")
            self._sequence = _block_resample(
                self._base,
                block_length=block_length,
                seed=block_resample_seed,
            )
            self._sampling_design = "block_resample"
        elif block_resample_seed is not None:
            raise ValueError("block_resample_seed requires block_length")

    @classmethod
    def from_profile(
        cls,
        profile: LatencyProfile,
        *,
        end_of_trace: EndOfTraceBehavior = EndOfTraceBehavior.FAIL,
        block_length: int | None = None,
        block_resample_seed: int | None = None,
        window_start: int | None = None,
        window_length: int | None = None,
    ) -> LatencyTraceReplayer:
        return cls(
            profile.latency_s,
            end_of_trace=end_of_trace,
            block_length=block_length,
            block_resample_seed=block_resample_seed,
            window_start=window_start,
            window_length=window_length,
            source_path=str(profile.path) if profile.path else None,
        )

    @property
    def n_available(self) -> int:
        return len(self._sequence)

    @property
    def n_emitted(self) -> int:
        return self._emitted

    @property
    def window_bounds(self) -> tuple[int, int] | None:
        if self._window_start is None or self._window_length is None:
            return None
        return self._window_start, self._window_start + self._window_length

    def next_latency_s(self) -> float:
        if self._index >= len(self._sequence):
            if self.end_of_trace is EndOfTraceBehavior.CYCLE:
                if not self._sequence:
                    raise TraceExhaustedError("cannot cycle an empty latency trace")
                self._index = 0
            else:
                raise TraceExhaustedError(
                    f"latency trace exhausted after {len(self._sequence)} samples "
                    f"(end_of_trace={self.end_of_trace.value}, "
                    f"sampling_design={self._sampling_design})"
                )
        value = self._sequence[self._index]
        if self._window_start is not None:
            abs_idx = self._window_start + self._index
        else:
            abs_idx = self._index
        self._consumed_trace_indices.append(int(abs_idx))
        self._consumed_delays_s.append(float(value))
        self._index += 1
        self._emitted += 1
        return float(value)

    def provenance(self) -> dict[str, Any]:
        used_stats = (
            summarize_latency_s(self._consumed_delays_s).to_dict()
            if self._consumed_delays_s
            else None
        )
        full_stats = summarize_latency_s(self._base).to_dict()
        window_stats = (
            summarize_latency_s(self._sequence).to_dict()
            if self._sampling_design == "contiguous_window"
            else None
        )
        bounds = self.window_bounds
        return {
            "mode": "trace_replay",
            "source_path": self.source_path,
            "n_available": self.n_available,
            "n_emitted": self.n_emitted,
            "end_of_trace": self.end_of_trace.value,
            "preserves_order": True,
            "synthetic_distribution": False,
            "sampling_design": self._sampling_design,
            "block_resample_interchangeable_with_contiguous_window": False,
            "window_start": self._window_start,
            "window_length": self._window_length,
            "window_end_exclusive": bounds[1] if bounds else None,
            "consumed_trace_indices": list(self._consumed_trace_indices),
            "consumed_delays_s": list(self._consumed_delays_s),
            "used_sample_stats": used_stats,
            "window_sample_stats": window_stats,
            "full_profile_stats": full_stats,
            "summary": summarize_latency_s(self._sequence).to_dict(),
            "note": (
                "used_sample_stats covers delays actually consumed; "
                "full_profile_stats covers the entire measured profile sequence; "
                "contiguous_window is not continuous full-trace replay"
                if self._sampling_design == "contiguous_window"
                else None
            ),
        }


def _block_resample(
    values: list[float],
    *,
    block_length: int,
    seed: int,
) -> list[float]:
    """Optional seeded block bootstrap; preserves within-block temporal order."""
    if len(values) < block_length:
        raise ValueError(f"block_length={block_length} exceeds trace length {len(values)}")
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(len(values) / block_length))
    starts = list(range(0, len(values) - block_length + 1))
    out: list[float] = []
    for _ in range(n_blocks):
        start = int(rng.choice(starts))
        out.extend(values[start : start + block_length])
    return out[: len(values)]


def profiling_command_hint(
    *,
    device_label: str,
    model_path: str = "PATH/TO/act_chunk_fp32.onnx",
    corpus_path: str = "PATH/TO/corpus_eval.npz",
    recipe_id: str = "torch_onnx_export_fp32_opset17",
    candidate: str = "act_aloha_transfer",
    output_dir: str | None = None,
) -> str:
    out = output_dir or f"profiles/{device_label}"
    return (
        "breakpoint profile "
        f"--candidate {candidate} "
        f"--model {model_path} "
        f"--corpus {corpus_path} "
        f"--recipe-id {recipe_id} "
        f"--device-label {device_label} "
        f"--output-dir {out}"
    )


def resolve_device_profiles(
    profiles_dir: Path,
    *,
    candidate: str,
    variant_label: str | None = None,
    precision_recipe: str | None = None,
) -> list[Path]:
    """Find compatible profile JSON files under profiles/<device>/."""
    profiles_dir = Path(profiles_dir)
    if not profiles_dir.is_dir():
        return []
    matches: list[Path] = []
    for path in sorted(profiles_dir.glob("*.json")):
        try:
            profile = load_latency_profile(path)
        except (ValueError, json.JSONDecodeError, OSError):
            continue
        if profile.payload.get("candidate") != candidate:
            continue
        if variant_label is not None:
            if profile.payload.get("model", {}).get("variant_label") != variant_label:
                continue
        if precision_recipe is not None:
            if profile.payload.get("model", {}).get("precision_recipe") != precision_recipe:
                continue
        matches.append(path)
    return matches


def require_unique_device_profile(
    profiles_dir: Path,
    *,
    device_label: str,
    candidate: str,
    variant_label: str | None = None,
    precision_recipe: str | None = None,
) -> Path:
    matches = resolve_device_profiles(
        profiles_dir,
        candidate=candidate,
        variant_label=variant_label,
        precision_recipe=precision_recipe,
    )
    if not matches:
        hint = profiling_command_hint(
            device_label=device_label,
            candidate=candidate,
            recipe_id=precision_recipe or "RECIPE_ID",
            output_dir=str(profiles_dir),
        )
        raise FileNotFoundError(
            f"No compatible latency profile in {profiles_dir} for "
            f"candidate={candidate!r} variant={variant_label!r} "
            f"recipe={precision_recipe!r}. "
            f"A device label alone is insufficient. Run on the target device:\n  {hint}"
        )
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise FileNotFoundError(
            f"Ambiguous latency profiles in {profiles_dir} for "
            f"candidate={candidate!r} variant={variant_label!r}: {names}. "
            "Pass --latency-profile PATH or narrow --profile-variant/--recipe-id."
        )
    return matches[0]
