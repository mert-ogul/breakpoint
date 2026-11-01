"""Benchmark protocol helpers: planning, artifact resolution, and resume locks."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from breakpoint_eval.experiment_plan import (
    ConditionSpec,
    EpisodeSeedPair,
    ExperimentAxis,
    ExperimentPlan,
    PlanPhase,
    build_seed_pairs,
    h264_conditions,
    jpeg_conditions,
    latency_conditions,
    precision_conditions,
    write_experiment_plan,
)
from breakpoint_eval.experiment_summary import (
    DEFAULT_BASELINE_SUCCESS_FLOOR,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_BREAKPOINT_RATIO,
)
from breakpoint_eval.export_runtime import load_export_manifest, resolve_onnx_runtime
from breakpoint_eval.file_digest import sha256_file
from breakpoint_eval.latency_trace import (
    contiguous_window_starts,
    load_latency_profile,
    require_unique_device_profile,
)

LOGGER = logging.getLogger("breakpoint_eval.benchmark_protocol")

DEFAULT_CONFIG_REL = Path("configs/act-transfer.json")
PROTOCOL_NAME = "act_deployment_stress"
PROTOCOL_VERSION = "0.1"
DIFFUSION_PROTOCOL_NAME = "diffusion_pusht_stress"
DIFFUSION_PROTOCOL_VERSION = "0.1"
# Configs and locks written before the protocol was named carry "protocol": "v0.1".
LEGACY_PROTOCOL_LABEL = "v0.1"
KNOWN_PROTOCOLS = frozenset(
    {
        (PROTOCOL_NAME, PROTOCOL_VERSION),
        (DIFFUSION_PROTOCOL_NAME, DIFFUSION_PROTOCOL_VERSION),
    }
)
BACKEND_ONNXRUNTIME = "onnxruntime"
BACKEND_PYTORCH = "pytorch"
# Same label as diffusion_profile.KIND_ACTION_GENERATION_TIMING (avoid importing torch here).
_TIMING_PROFILE_KIND = "action_generation_timing"


@dataclass(frozen=True)
class ResolvedArtifacts:
    export_dir: Path
    fp32_path: Path | None = None
    fp32_sha256: str | None = None
    fp32_recipe_id: str | None = None
    int8_path: Path | None = None
    int8_sha256: str | None = None
    int8_status: str = "missing"
    int8_detail: str | None = None
    profiles: dict[str, Path] = field(default_factory=dict)
    profile_identities: dict[str, dict[str, Any]] = field(default_factory=dict)
    int4_path: Path | None = None
    int4_sha256: str | None = None
    int4_status: str = "missing"
    int4_detail: str | None = None
    backend: str = BACKEND_ONNXRUNTIME
    checkpoint_files: dict[str, str] | None = None
    upstream_agreement_path: Path | None = None
    upstream_agreement_sha256: str | None = None
    timing_profile_path: Path | None = None
    timing_profile_sha256: str | None = None


@dataclass(frozen=True)
class BenchmarkHostSettings:
    compute_device: str
    ort_providers: list[str] | None = None
    intra_op_num_threads: int | None = None
    inter_op_num_threads: int | None = None
    torch_num_threads: int | None = None


def protocol_identity(payload: dict[str, Any]) -> tuple[str, str]:
    """Return (name, version) from a config or lock, accepting the legacy "protocol" label."""
    if "protocol" in payload:
        if payload["protocol"] != LEGACY_PROTOCOL_LABEL:
            raise ValueError(f"unknown legacy protocol label {payload['protocol']!r}")
        return PROTOCOL_NAME, PROTOCOL_VERSION
    return str(payload.get("protocol_name")), str(payload.get("protocol_version"))


def config_backend(config: dict[str, Any]) -> str:
    return str(config.get("backend") or BACKEND_ONNXRUNTIME)


def native_hw_from_config(config: dict[str, Any]) -> tuple[int, int]:
    hw = config["image"]["native_hw"]
    return int(hw[0]), int(hw[1])


def baseline_variant_from_config(config: dict[str, Any]) -> str:
    fingerprint = (config.get("baseline_reuse") or {}).get("fingerprint") or {}
    return str(fingerprint.get("variant_label") or "onnx_fp32")


def backend_and_variant_from_lock(lock: dict[str, Any]) -> tuple[str, str]:
    """Return ``(backend, baseline variant_label)`` from a protocol lock.

    Both values must be present on the lock (or, for older ONNX locks that omit
    ``backend``, inferable from ONNX artifact fields). Callers pass the pair into
    completeness checks; those checks have no onnxruntime/onnx_fp32 defaults.
    """
    fingerprint = (lock.get("baseline_reuse") or {}).get("fingerprint")
    if not isinstance(fingerprint, dict) or not fingerprint.get("variant_label"):
        raise ValueError("lock requires baseline_reuse.fingerprint.variant_label")
    variant_label = str(fingerprint["variant_label"])
    raw_backend = lock.get("backend")
    if raw_backend is None:
        arts = lock.get("artifacts") or {}
        if not isinstance(arts, dict):
            raise ValueError("lock missing backend")
        if arts.get("timing_profile_path") or arts.get("checkpoint_files"):
            raw_backend = BACKEND_PYTORCH
        elif arts.get("fp32_sha256") is not None or arts.get("fp32_path") is not None:
            raw_backend = BACKEND_ONNXRUNTIME
        else:
            raise ValueError("lock missing backend")
    backend = str(raw_backend)
    if backend not in {BACKEND_ONNXRUNTIME, BACKEND_PYTORCH}:
        raise ValueError(f"unsupported lock backend {backend!r}")
    return backend, variant_label


def load_benchmark_config(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(
            f"unsupported benchmark config schema_version={payload.get('schema_version')!r}"
        )
    identity = protocol_identity(payload)
    if identity not in KNOWN_PROTOCOLS:
        raise ValueError(
            f"unknown protocol {identity[0]} {identity[1]}; "
            f"known: {sorted(f'{n} {v}' for n, v in KNOWN_PROTOCOLS)}"
        )
    h264 = payload.get("h264")
    if isinstance(h264, dict):
        latency_mode = h264.get("latency_mode")
        if latency_mode != "quality_only":
            raise ValueError(
                f"h264.latency_mode {latency_mode!r} refused; only quality_only is allowed "
                "for this benchmark (deployment mode is not part of the protocol)"
            )
    return payload


def resolve_h264_support(config: dict[str, Any]) -> dict[str, Any]:
    """Probe libx264/PyAV once; missing codec yields status unsupported."""
    from breakpoint_eval.h264_ffmpeg import (
        REQUESTED_MAX_B_FRAMES,
        REQUESTED_PIXEL_FORMAT,
        REQUESTED_RC_LOOKAHEAD,
        REQUESTED_TUNE,
        probe_ffmpeg_h264,
    )

    h264_cfg = config.get("h264")
    if not isinstance(h264_cfg, dict):
        raise ValueError(
            "config.h264 must be an object with gop, preset, latency_mode, frame_rate_hz"
        )
    for key in ("gop", "preset", "latency_mode", "frame_rate_hz"):
        if key not in h264_cfg:
            raise ValueError(f"config.h264.{key} is required")
    probe = probe_ffmpeg_h264()
    status = "ok" if probe.available else "unsupported"
    return {
        "status": status,
        "detail": probe.unsupported_reason,
        "gop": int(h264_cfg["gop"]),
        "preset": str(h264_cfg["preset"]),
        "latency_mode": str(h264_cfg["latency_mode"]),
        "frame_rate_hz": float(h264_cfg["frame_rate_hz"]),
        "tune": REQUESTED_TUNE,
        "pixel_format": REQUESTED_PIXEL_FORMAT,
        "max_b_frames": REQUESTED_MAX_B_FRAMES,
        "rc_lookahead": REQUESTED_RC_LOOKAHEAD,
        "pyav_version": probe.pyav_version,
        "libav_versions": dict(probe.libav_versions),
        "ffmpeg_version_line": probe.version_line,
        "measurements_fabricated": False,
    }


def _external_weight_files(model_path: Path) -> list[dict[str, str]]:
    """Include ONNX external data files next to the graph when present."""
    parent = Path(model_path).parent
    stem = Path(model_path).name
    extras: list[dict[str, str]] = []
    for path in sorted(parent.glob(stem + ".*")):
        if path == Path(model_path).resolve():
            continue
        if path.suffix.lower() in {".onnx", ".json", ".npz", ".md"}:
            continue
        # Common ORT external-data naming: model.onnx.data / model.onnx.external
        if path.name.startswith(stem):
            extras.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    data_sibling = parent / (stem + ".data")
    if data_sibling.is_file() and not any(e["path"] == str(data_sibling.resolve()) for e in extras):
        extras.append({"path": str(data_sibling.resolve()), "sha256": sha256_file(data_sibling)})
    return extras


def resolve_profile_by_identity(
    profiles_dir: Path,
    *,
    device_label: str,
    candidate: str,
    variant_label: str,
    expected_model_sha256: str | None = None,
) -> Path:
    """Strict identity resolution by declared digests, not newest mtime."""
    profiles_dir = Path(profiles_dir)
    matches: list[Path] = []
    for path in sorted(profiles_dir.glob("*.json")):
        try:
            profile = load_latency_profile(path)
        except (ValueError, json.JSONDecodeError, OSError):
            continue
        if profile.payload.get("candidate") != candidate:
            continue
        if profile.payload.get("device_label") != device_label:
            continue
        model = profile.payload.get("model") or {}
        if model.get("variant_label") != variant_label:
            continue
        if expected_model_sha256 is not None and model.get("sha256") != expected_model_sha256:
            continue
        matches.append(path)
    if not matches:
        # Fall back to unique candidate/variant match only when a single file exists.
        if expected_model_sha256 is None:
            try:
                return require_unique_device_profile(
                    profiles_dir,
                    device_label=device_label,
                    candidate=candidate,
                    variant_label=variant_label,
                )
            except FileNotFoundError:
                pass
        raise FileNotFoundError(
            f"no profile in {profiles_dir} matching device={device_label!r} "
            f"variant={variant_label!r} sha256={expected_model_sha256!r}"
        )
    if len(matches) > 1:
        raise FileNotFoundError(
            f"ambiguous profiles in {profiles_dir}: {[p.name for p in matches]}"
        )
    return matches[0]


def _resolve_optional_quantized_onnx(
    export: Path,
    *,
    onnx_name: str,
    candidate: str,
    expected_sha256: str | None,
    manifest_label: str,
    display_label: str,
) -> tuple[Path | None, str | None, str, str | None]:
    """Resolve an optional INT8/INT4 graph the same way: file then runtime+hash, else manifest."""
    model_path = (export / onnx_name).resolve()
    if model_path.is_file():
        try:
            runtime = resolve_onnx_runtime(
                model_path,
                expected_candidate=candidate,
                require_export_validation=True,
            )
            if expected_sha256 and runtime.model_sha256 != expected_sha256:
                raise ValueError(
                    f"{display_label} sha256 mismatch: file={runtime.model_sha256} "
                    f"expected={expected_sha256}"
                )
            return model_path, runtime.model_sha256, "ok", None
        except (ValueError, FileNotFoundError) as exc:
            detail = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("%s present but not runnable: %s", display_label, detail)
            return None, None, "unsupported", detail
    status = "missing"
    detail: str | None = None
    manifest = load_export_manifest(export / "export_manifest.json")
    for variant in manifest.get("variants") or []:
        if variant.get("label") == manifest_label:
            status = str(variant.get("status") or "missing")
            detail = variant.get("detail")
            break
    return None, None, status, detail


def _resolve_onnx_benchmark_artifacts(
    config: dict[str, Any],
    *,
    repo_root: Path,
    export: Path,
    profile_overrides: dict[str, Path] | None = None,
) -> ResolvedArtifacts:
    arts = config["artifacts"]
    fp32_path = (export / arts["fp32_onnx_name"]).resolve()
    if not fp32_path.is_file():
        raise FileNotFoundError(f"FP32 ONNX missing: {fp32_path}")
    fp32_rt = resolve_onnx_runtime(
        fp32_path,
        expected_candidate=config["candidate"],
        require_export_validation=True,
    )
    expected_fp32 = arts.get("expected_fp32_sha256")
    if expected_fp32 and fp32_rt.model_sha256 != expected_fp32:
        raise ValueError(
            f"FP32 sha256 mismatch: file={fp32_rt.model_sha256} expected={expected_fp32}"
        )

    int8_path, int8_sha, int8_status, int8_detail = _resolve_optional_quantized_onnx(
        export,
        onnx_name=arts.get("int8_onnx_name") or "act_chunk_int8_qdq.onnx",
        candidate=config["candidate"],
        expected_sha256=arts.get("expected_int8_sha256"),
        manifest_label="int8",
        display_label="INT8",
    )
    int4_path, int4_sha, int4_status, int4_detail = _resolve_optional_quantized_onnx(
        export,
        onnx_name=arts.get("int4_onnx_name") or "act_chunk_int4_matmul_nbits.onnx",
        candidate=config["candidate"],
        expected_sha256=arts.get("expected_int4_sha256"),
        manifest_label="int4_weight_only",
        display_label="INT4",
    )

    profiles: dict[str, Path] = {}
    identities: dict[str, dict[str, Any]] = {}
    for key, spec in (arts.get("profiles") or {}).items():
        override = (profile_overrides or {}).get(key)
        if override is not None:
            path = Path(override).resolve()
            if path.is_dir():
                path = resolve_profile_by_identity(
                    path,
                    device_label=spec["device_label"],
                    candidate=config["candidate"],
                    variant_label=spec.get("variant_label") or "onnx_fp32",
                    expected_model_sha256=spec.get("expected_model_sha256"),
                )
            elif not path.is_file():
                raise FileNotFoundError(f"profile override not found: {path}")
        else:
            path = resolve_profile_by_identity(
                (repo_root / spec["dir"]).resolve(),
                device_label=spec["device_label"],
                candidate=config["candidate"],
                variant_label=spec.get("variant_label") or "onnx_fp32",
                expected_model_sha256=spec.get("expected_model_sha256"),
            )
        profile = load_latency_profile(path)
        # Profile must match FP32 graph identity when expected hash unset.
        if profile.payload.get("model", {}).get("sha256") != fp32_rt.model_sha256:
            raise ValueError(
                f"profile {path} model sha256 "
                f"{profile.payload.get('model', {}).get('sha256')!r} != "
                f"export FP32 {fp32_rt.model_sha256!r}"
            )
        profiles[key] = path
        identities[key] = profile.identity()

    return ResolvedArtifacts(
        export_dir=export,
        fp32_path=fp32_path,
        fp32_sha256=fp32_rt.model_sha256,
        fp32_recipe_id=fp32_rt.recipe_id,
        int8_path=int8_path,
        int8_sha256=int8_sha,
        int8_status=int8_status,
        int8_detail=int8_detail,
        profiles=profiles,
        profile_identities=identities,
        int4_path=int4_path,
        int4_sha256=int4_sha,
        int4_status=int4_status,
        int4_detail=int4_detail,
        backend=BACKEND_ONNXRUNTIME,
    )


def resolve_pytorch_timing_profile(
    profiles_dir: Path,
    *,
    candidate: str,
    checkpoint_files: dict[str, str],
) -> Path:
    """Resolve exactly one Diffusion timing profile; refuse device profiles and wrong candidates."""
    profiles_dir = Path(profiles_dir)
    if not profiles_dir.is_dir():
        raise FileNotFoundError(f"timing profile directory not found: {profiles_dir}")
    matches: list[Path] = []
    for path in sorted(profiles_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"timing profile {path} unreadable: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"timing profile {path} is not a JSON object")
        file_candidate = payload.get("candidate")
        if file_candidate != candidate:
            raise ValueError(f"refused {path.name}: candidate {file_candidate!r} != {candidate!r}")
        if payload.get("kind") != _TIMING_PROFILE_KIND:
            raise ValueError(
                f"refused device profile or non-timing record in {profiles_dir}: {path.name}"
            )
        model = payload.get("model") or {}
        if model.get("checkpoint_files") == checkpoint_files:
            matches.append(path)
    if not matches:
        raise FileNotFoundError(
            f"no timing profile in {profiles_dir} for candidate={candidate!r} "
            "with matching checkpoint hashes"
        )
    if len(matches) > 1:
        raise FileNotFoundError(
            f"ambiguous timing profiles in {profiles_dir}: {[p.name for p in matches]}"
        )
    return matches[0]


def _resolve_pytorch_benchmark_artifacts(
    config: dict[str, Any],
    *,
    repo_root: Path,
    export: Path,
) -> ResolvedArtifacts:
    arts = config["artifacts"]
    candidate = config["candidate"]
    record_path = export / "export_manifest.json"
    if not record_path.is_file():
        raise FileNotFoundError(f"Diffusion export record missing: {record_path}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"export record {record_path} is not a JSON object")
    if record.get("candidate") != candidate:
        raise ValueError(
            f"export record candidate {record.get('candidate')!r} != config {candidate!r}"
        )
    onnx_export = record.get("onnx_export") or {}
    if onnx_export.get("status") != "unsupported":
        raise ValueError(
            f"Diffusion export onnx_export.status must be 'unsupported'; "
            f"got {onnx_export.get('status')!r}"
        )
    recorded_hashes = record.get("checkpoint_files")
    if not isinstance(recorded_hashes, dict) or not recorded_hashes:
        raise ValueError(f"export record {record_path} missing checkpoint_files")
    checkpoint_dir = record.get("checkpoint_dir")
    if not checkpoint_dir:
        raise ValueError(f"export record {record_path} missing checkpoint_dir")
    from breakpoint_eval.diffusion_export import checkpoint_file_hashes

    on_disk = checkpoint_file_hashes(Path(checkpoint_dir))
    if on_disk != recorded_hashes:
        raise ValueError(
            f"export record checkpoint_files do not match files on disk under {checkpoint_dir}"
        )

    agreement_rel = arts.get("upstream_agreement")
    if not agreement_rel:
        raise ValueError("artifacts.upstream_agreement is required for pytorch backend")
    agreement_path = (repo_root / agreement_rel).resolve()
    if not agreement_path.is_file():
        raise FileNotFoundError(f"upstream_agreement missing: {agreement_path}")
    agreement = json.loads(agreement_path.read_text(encoding="utf-8"))
    if not isinstance(agreement, dict):
        raise ValueError(f"upstream_agreement {agreement_path} is not a JSON object")
    if agreement.get("passed") is not True:
        raise ValueError(
            f"upstream_agreement failed or incomplete: passed={agreement.get('passed')!r}"
        )
    if agreement.get("checkpoint_files") != recorded_hashes:
        raise ValueError("upstream_agreement checkpoint_files differ from the export record hashes")

    timing_rel = arts.get("timing_profile_dir")
    if not timing_rel:
        raise ValueError("artifacts.timing_profile_dir is required for pytorch backend")
    timing_path = resolve_pytorch_timing_profile(
        (repo_root / timing_rel).resolve(),
        candidate=candidate,
        checkpoint_files=recorded_hashes,
    )

    return ResolvedArtifacts(
        export_dir=export,
        backend=BACKEND_PYTORCH,
        checkpoint_files=dict(recorded_hashes),
        upstream_agreement_path=agreement_path,
        upstream_agreement_sha256=sha256_file(agreement_path),
        timing_profile_path=timing_path,
        timing_profile_sha256=sha256_file(timing_path),
    )


def resolve_benchmark_artifacts(
    config: dict[str, Any],
    *,
    repo_root: Path,
    export_dir: Path | None = None,
    profile_overrides: dict[str, Path] | None = None,
) -> ResolvedArtifacts:
    repo_root = Path(repo_root).resolve()
    arts = config["artifacts"]
    export = Path(export_dir) if export_dir else (repo_root / arts["export_dir"])
    export = export.resolve()
    if not export.is_dir():
        raise FileNotFoundError(f"export directory not found: {export}")

    backend = config_backend(config)
    if backend == BACKEND_PYTORCH:
        return _resolve_pytorch_benchmark_artifacts(config, repo_root=repo_root, export=export)
    if backend != BACKEND_ONNXRUNTIME:
        raise ValueError(f"unsupported backend {backend!r}")
    return _resolve_onnx_benchmark_artifacts(
        config,
        repo_root=repo_root,
        export=export,
        profile_overrides=profile_overrides,
    )


def host_settings_from_config(config: dict[str, Any]) -> BenchmarkHostSettings:
    host = config["host"]
    if config_backend(config) == BACKEND_PYTORCH:
        return BenchmarkHostSettings(
            compute_device=str(host["compute_device"]),
            torch_num_threads=int(host["torch_num_threads"]),
        )
    return BenchmarkHostSettings(
        compute_device=str(host["compute_device"]),
        ort_providers=list(host["ort_providers"]),
        intra_op_num_threads=int(host["intra_op_num_threads"]),
        inter_op_num_threads=int(host["inter_op_num_threads"]),
    )


def _resolution_conditions_from_hw(
    levels: list[list[int]],
    *,
    native_hw: tuple[int, int],
) -> list[ConditionSpec]:
    out: list[ConditionSpec] = []
    for hw in levels:
        h, w = int(hw[0]), int(hw[1])
        if (h, w) == native_hw:
            out.append(
                ConditionSpec(
                    condition_id="resolution_native",
                    axis=ExperimentAxis.RESOLUTION.value,
                    params={
                        "mode": "native",
                        "linear_scale": 1.0,
                        "native_hw": list(native_hw),
                        "capture_hw": list(native_hw),
                        "effective_policy_hw": list(native_hw),
                    },
                    label=f"resolution native {native_hw[0]}x{native_hw[1]}",
                )
            )
        else:
            out.append(
                ConditionSpec(
                    condition_id=f"resolution_capture_{h}x{w}",
                    axis=ExperimentAxis.RESOLUTION.value,
                    params={
                        "mode": "downsample",
                        "native_hw": list(native_hw),
                        "capture_hw": [h, w],
                        "effective_policy_hw": list(native_hw),
                    },
                    label=f"resolution capture {h}x{w}",
                )
            )
    return out


def _device_profile_conditions(
    *,
    profile_keys: list[str],
    artifacts: ResolvedArtifacts,
    window_length: int,
    n_episodes: int,
) -> list[ConditionSpec]:
    out: list[ConditionSpec] = []
    for key in profile_keys:
        path = artifacts.profiles[key]
        profile = load_latency_profile(path)
        starts = contiguous_window_starts(
            len(profile.latency_s),
            window_length=window_length,
            n_episodes=n_episodes,
        )
        label = profile.payload.get("device_label") or key
        out.append(
            ConditionSpec(
                condition_id=f"device_profile_{key}_fp32",
                axis=ExperimentAxis.DEVICE_PROFILE.value,
                params={
                    "latency_profile_path": str(path),
                    "device_label": label,
                    "variant_label": "onnx_fp32",
                    "window_length": window_length,
                    "window_starts": starts,
                    "sampling_design": "contiguous_evenly_spaced_starts",
                    "not_block_resample": True,
                    "not_continuous_full_trace_replay": True,
                },
                label=f"device profile replay {label} (FP32, window={window_length})",
            )
        )
    return out


def _baseline_fingerprint(config: dict[str, Any] | None = None) -> dict[str, Any]:
    if config is not None:
        fingerprint = (config.get("baseline_reuse") or {}).get("fingerprint")
        if isinstance(fingerprint, dict):
            return {
                "latency_s": float(fingerprint.get("latency_s", 0.0)),
                "image_mode": str(fingerprint.get("image_mode") or "native"),
                "variant_label": str(fingerprint.get("variant_label") or "onnx_fp32"),
                "no_device_profile": bool(fingerprint.get("no_device_profile", True)),
            }
    return {
        "latency_s": 0.0,
        "image_mode": "native",
        "variant_label": "onnx_fp32",
        "no_device_profile": True,
    }


def condition_is_shared_baseline(
    condition: ConditionSpec,
    *,
    baseline_variant: str = "onnx_fp32",
) -> bool:
    params = condition.params
    if condition.axis == ExperimentAxis.LATENCY_MS.value:
        return float(params.get("latency_s", -1)) == 0.0
    if condition.axis == ExperimentAxis.RESOLUTION.value:
        return params.get("mode") == "native"
    if condition.axis == ExperimentAxis.JPEG.value:
        return bool(params.get("bypass"))
    if condition.axis == ExperimentAxis.H264.value:
        return bool(params.get("bypass"))
    if condition.axis == ExperimentAxis.PRECISION.value:
        return params.get("variant_label") == baseline_variant
    return False


def _h264_support_for_plans(
    config: dict[str, Any],
    h264_support: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if h264_support is not None:
        return h264_support
    if isinstance(config.get("h264"), dict):
        return resolve_h264_support(config)
    return None


def build_smoke_plan(
    config: dict[str, Any],
    artifacts: ResolvedArtifacts,
    *,
    h264_support: dict[str, Any] | None = None,
) -> ExperimentPlan:
    smoke = config["smoke"]
    seeds = build_seed_pairs(
        len(smoke["env_seeds"]),
        env_seeds=list(smoke["env_seeds"]),
        policy_seed_offset=int(smoke["policy_seed_offset"]),
    )
    window_length = int(config["trace_window"]["length"])
    support = _h264_support_for_plans(config, h264_support)
    native_hw = native_hw_from_config(config)
    conditions: list[ConditionSpec] = []
    if "latency_ms" in smoke:
        conditions.extend(latency_conditions(tuple(int(x) for x in smoke["latency_ms"])))
    if "resolution_hw" in smoke:
        conditions.extend(
            _resolution_conditions_from_hw(list(smoke["resolution_hw"]), native_hw=native_hw)
        )
    if "jpeg_qualities" in smoke:
        conditions.extend(jpeg_conditions(tuple(smoke["jpeg_qualities"])))
    if (
        support is not None
        and support.get("status") == "ok"
        and "h264_bitrates_kbps" in smoke
        and smoke.get("h264_bitrates_kbps")
    ):
        conditions.extend(
            h264_conditions(
                tuple(smoke["h264_bitrates_kbps"]),
                frame_rate_hz=float(support["frame_rate_hz"]),
                gop=int(support["gop"]),
                preset=str(support["preset"]),
                latency_mode="quality_only",
            )
        )
    notes = ["Smoke plan; not for publication claims."]
    if "precision_variants" in smoke:
        # Precision: always list FP32; INT8/INT4 only if runnable, else recorded separately.
        prec_variants = ["onnx_fp32"]
        if "int8" in smoke.get("precision_variants", []) and artifacts.int8_status == "ok":
            prec_variants.append("int8")
        if (
            "int4_weight_only" in smoke.get("precision_variants", [])
            and artifacts.int4_status == "ok"
        ):
            prec_variants.append("int4_weight_only")
        conditions.extend(precision_conditions(tuple(prec_variants)))
        notes.append(f"INT8 status at plan build: {artifacts.int8_status}")
        notes.append(f"INT4 status at plan build: {artifacts.int4_status}")
    if "device_profiles" in smoke:
        conditions.extend(
            _device_profile_conditions(
                profile_keys=list(smoke["device_profiles"]),
                artifacts=artifacts,
                window_length=window_length,
                n_episodes=len(seeds),
            )
        )
    if "h264_bitrates_kbps" in smoke:
        notes.append(f"H.264 status at plan build: {(support or {}).get('status')}")
    return ExperimentPlan(
        plan_id="v0.1_smoke",
        candidate=config["candidate"],
        phase=PlanPhase.SMOKE.value,
        n_episodes=len(seeds),
        axis=ExperimentAxis.COMBINED.value,
        seed_pairs=seeds,
        conditions=conditions,
        isolation_notes=[
            "Smoke checklist mixes axes for gatekeeping; full phase isolates axes.",
            "Confirmation with fresh seeds remains separate.",
        ],
        baseline_condition_id="latency_ms_0",
        notes=notes,
    )


def build_full_axis_plans(
    config: dict[str, Any],
    artifacts: ResolvedArtifacts,
    *,
    h264_support: dict[str, Any] | None = None,
) -> list[ExperimentPlan]:
    full = config["full"]
    n = int(full["n_episodes"])
    seeds = build_seed_pairs(
        n,
        env_seed_start=int(full["env_seed_start"]),
        policy_seed_offset=int(full["policy_seed_offset"]),
    )
    window_length = int(config["trace_window"]["length"])
    candidate = config["candidate"]
    support = _h264_support_for_plans(config, h264_support)
    native_hw = native_hw_from_config(config)
    variant = baseline_variant_from_config(config)
    plans: list[ExperimentPlan] = []

    if "latency_ms" in full:
        plans.append(
            ExperimentPlan(
                plan_id="v0.1_full_latency",
                candidate=candidate,
                phase=PlanPhase.EXPLORATORY.value,
                n_episodes=n,
                axis=ExperimentAxis.LATENCY_MS.value,
                seed_pairs=seeds,
                conditions=latency_conditions(tuple(int(x) for x in full["latency_ms"])),
                isolation_notes=[f"Synthetic constant delay; precision held at {variant}."],
                baseline_condition_id="latency_ms_0",
                notes=["Isolated latency axis."],
            )
        )

    if "resolution_hw" in full:
        plans.append(
            ExperimentPlan(
                plan_id="v0.1_full_resolution",
                candidate=candidate,
                phase=PlanPhase.EXPLORATORY.value,
                n_episodes=n,
                axis=ExperimentAxis.RESOLUTION.value,
                seed_pairs=seeds,
                conditions=_resolution_conditions_from_hw(
                    list(full["resolution_hw"]), native_hw=native_hw
                ),
                isolation_notes=["Resolution axis only; latency and precision at defaults."],
                baseline_condition_id="resolution_native",
                notes=["Isolated resolution axis; native bypass is not JPEG quality 100."],
            )
        )

    if "jpeg_qualities" in full:
        plans.append(
            ExperimentPlan(
                plan_id="v0.1_full_jpeg",
                candidate=candidate,
                phase=PlanPhase.EXPLORATORY.value,
                n_episodes=n,
                axis=ExperimentAxis.JPEG.value,
                seed_pairs=seeds,
                conditions=jpeg_conditions(tuple(full["jpeg_qualities"])),
                isolation_notes=["JPEG visual-quality axis only; native resolution."],
                baseline_condition_id="jpeg_bypass",
                notes=["Isolated JPEG axis; bypass ≠ quality 100."],
            )
        )

    if (
        support is not None
        and support.get("status") == "ok"
        and "h264_bitrates_kbps" in full
        and full.get("h264_bitrates_kbps")
    ):
        plans.append(
            ExperimentPlan(
                plan_id="v0.1_full_h264",
                candidate=candidate,
                phase=PlanPhase.EXPLORATORY.value,
                n_episodes=n,
                axis=ExperimentAxis.H264.value,
                seed_pairs=seeds,
                conditions=h264_conditions(
                    tuple(full["h264_bitrates_kbps"]),
                    frame_rate_hz=float(support["frame_rate_hz"]),
                    gop=int(support["gop"]),
                    preset=str(support["preset"]),
                    latency_mode="quality_only",
                ),
                isolation_notes=[
                    "H.264 causal stream quality-only at native resolution; FP32 and 0 ms delay.",
                    "No codec, transport, or decoding delay is simulated on this axis.",
                ],
                baseline_condition_id="h264_bypass",
                notes=["Isolated H.264 bitrate axis; visual quality only."],
            )
        )

    if "precision_variants" in full:
        prec_variants = ["onnx_fp32"]
        if "int8" in full.get("precision_variants", []) and artifacts.int8_status == "ok":
            prec_variants.append("int8")
        if (
            "int4_weight_only" in full.get("precision_variants", [])
            and artifacts.int4_status == "ok"
        ):
            prec_variants.append("int4_weight_only")
        plans.append(
            ExperimentPlan(
                plan_id="v0.1_full_precision",
                candidate=candidate,
                phase=PlanPhase.EXPLORATORY.value,
                n_episodes=n,
                axis=ExperimentAxis.PRECISION.value,
                seed_pairs=seeds,
                conditions=precision_conditions(tuple(prec_variants)),
                isolation_notes=["Precision under constant_latency_ms_0."],
                baseline_condition_id="precision_onnx_fp32",
                notes=[
                    f"INT8 status at plan build: {artifacts.int8_status}",
                    f"INT4 status at plan build: {artifacts.int4_status}",
                ],
            )
        )

    if "device_profiles" in full:
        device_conds = _device_profile_conditions(
            profile_keys=list(full["device_profiles"]),
            artifacts=artifacts,
            window_length=window_length,
            n_episodes=n,
        )
        if device_conds:
            plans.append(
                ExperimentPlan(
                    plan_id="v0.1_full_device_profile",
                    candidate=candidate,
                    phase=PlanPhase.EXPLORATORY.value,
                    n_episodes=n,
                    axis=ExperimentAxis.DEVICE_PROFILE.value,
                    seed_pairs=seeds,
                    conditions=device_conds,
                    isolation_notes=[
                        "Measured per-variant hardware timing replay; not constant-delay axis.",
                        "Contiguous 32-sample windows; not block resampling.",
                    ],
                    baseline_condition_id=None,
                    notes=["Device profile replay with predeclared windows."],
                )
            )
    return plans


def freeze_protocol_lock(
    *,
    config: dict[str, Any],
    artifacts: ResolvedArtifacts,
    source_revision: dict[str, Any],
    host: BenchmarkHostSettings,
    phase: str,
    h264_support: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stats = config["statistics"]
    shared = {
        "schema_version": 1,
        "protocol_name": protocol_identity(config)[0],
        "protocol_version": protocol_identity(config)[1],
        "phase": phase,
        "candidate": config["candidate"],
        "source_revision": source_revision,
        "trace_window": config["trace_window"],
        "scheduler": config["scheduler"],
        "statistics": {
            "breakpoint_ratio": float(stats.get("breakpoint_ratio", DEFAULT_BREAKPOINT_RATIO)),
            "baseline_success_floor": float(
                stats.get("baseline_success_floor", DEFAULT_BASELINE_SUCCESS_FLOOR)
            ),
            "n_bootstrap": int(stats.get("n_bootstrap", DEFAULT_BOOTSTRAP_SAMPLES)),
            "bootstrap_seed": int(stats.get("bootstrap_seed", DEFAULT_BOOTSTRAP_SEED)),
            "coverage": stats.get("coverage", "pointwise"),
            "confirmation_fresh_seeds_separate": True,
        },
        "baseline_reuse": config.get("baseline_reuse") or {"enabled": True},
        "phase_config": config[phase] if phase in config else config.get("smoke"),
    }
    if config_backend(config) == BACKEND_PYTORCH:
        lock: dict[str, Any] = {
            **shared,
            "backend": BACKEND_PYTORCH,
            "artifacts": {
                "export_dir": str(artifacts.export_dir),
                "checkpoint_files": dict(artifacts.checkpoint_files or {}),
                "upstream_agreement_path": (
                    str(artifacts.upstream_agreement_path)
                    if artifacts.upstream_agreement_path
                    else None
                ),
                "upstream_agreement_sha256": artifacts.upstream_agreement_sha256,
                "timing_profile_path": (
                    str(artifacts.timing_profile_path) if artifacts.timing_profile_path else None
                ),
                "timing_profile_sha256": artifacts.timing_profile_sha256,
            },
            "host": {
                "compute_device": host.compute_device,
                "torch_num_threads": host.torch_num_threads,
            },
        }
        return lock

    assert artifacts.fp32_path is not None
    lock = {
        **shared,
        "backend": BACKEND_ONNXRUNTIME,
        "artifacts": {
            "export_dir": str(artifacts.export_dir),
            "fp32_path": str(artifacts.fp32_path),
            "fp32_sha256": artifacts.fp32_sha256,
            "fp32_recipe_id": artifacts.fp32_recipe_id,
            "fp32_external_weights": _external_weight_files(artifacts.fp32_path),
            "int8_path": str(artifacts.int8_path) if artifacts.int8_path else None,
            "int8_sha256": artifacts.int8_sha256,
            "int8_status": artifacts.int8_status,
            "int8_detail": artifacts.int8_detail,
            "profiles": {k: str(v) for k, v in artifacts.profiles.items()},
            "profile_identities": artifacts.profile_identities,
            "int4_path": str(artifacts.int4_path) if artifacts.int4_path else None,
            "int4_sha256": artifacts.int4_sha256,
            "int4_status": artifacts.int4_status,
            "int4_detail": artifacts.int4_detail,
        },
        "host": {
            "compute_device": host.compute_device,
            "ort_providers": host.ort_providers,
            "intra_op_num_threads": host.intra_op_num_threads,
            "inter_op_num_threads": host.inter_op_num_threads,
        },
    }
    if h264_support is not None:
        lock["h264"] = dict(h264_support)
    return lock


# Lock fields that may differ on resume: they record where and when a run happened,
# not what it executes. Every other field is compared.
RESUME_METADATA_FIELDS = frozenset({"source_revision"})
_LOCK_IDENTITY_FIELDS = frozenset({"protocol", "protocol_name", "protocol_version"})
SUPPORTED_LOCK_SCHEMA_VERSION = 1
_COMPLETED_EPISODE_OUTCOMES = frozenset(
    {"success", "terminated_no_success", "time_limit_truncation", "policy_failed"}
)


def assert_lock_matches(existing: dict[str, Any], fresh: dict[str, Any]) -> None:
    """Refuse resume when schema, protocol identity, or any execution-defining field changed."""
    for label, lock in (("existing lock", existing), ("current configuration", fresh)):
        if lock.get("schema_version") != SUPPORTED_LOCK_SCHEMA_VERSION:
            raise ValueError(
                f"resume refused: {label} schema_version={lock.get('schema_version')!r}; "
                f"supported is {SUPPORTED_LOCK_SCHEMA_VERSION}"
            )
    if protocol_identity(existing) != protocol_identity(fresh):
        raise ValueError(
            "resume refused: protocol identity changed between lock and current configuration. "
            "Use a fresh --out directory for identity/config changes."
        )
    ignored = RESUME_METADATA_FIELDS | _LOCK_IDENTITY_FIELDS
    for key in sorted((set(existing) | set(fresh)) - ignored):
        if existing.get(key) != fresh.get(key):
            raise ValueError(
                f"resume refused: {key} changed between lock and current resolution. "
                "Use a fresh --out directory for identity/config changes."
            )


def _expected_runtime(
    condition: ConditionSpec,
    *,
    backend: str,
    variant_label: str,
) -> dict[str, Any]:
    """Runtime identity a finished report must show for this condition."""
    if backend not in {BACKEND_ONNXRUNTIME, BACKEND_PYTORCH}:
        raise ValueError(f"unsupported backend {backend!r}")
    if not variant_label:
        raise ValueError("variant_label is required")
    params = condition.params
    axis = condition.axis
    expected: dict[str, Any] = {
        "latency_mode": "trace_replay"
        if axis == ExperimentAxis.DEVICE_PROFILE.value
        else "constant",
        "latency_s": float(params["latency_s"]) if axis == ExperimentAxis.LATENCY_MS.value else 0.0,
        "image_mode": "native",
        "variant_label": variant_label,
        "policy_backend": backend,
    }
    if axis in {
        ExperimentAxis.RESOLUTION.value,
        ExperimentAxis.JPEG.value,
        ExperimentAxis.H264.value,
    }:
        expected["image_mode"] = "native" if params.get("bypass") else params["mode"]
    if axis in {ExperimentAxis.PRECISION.value, ExperimentAxis.DEVICE_PROFILE.value}:
        expected["variant_label"] = params["variant_label"]
    return expected


def _h264_summary_problem(episode: dict[str, Any], condition: ConditionSpec) -> str | None:
    """Return why one H.264 episode summary fails the quality-only evidence checks."""
    index = episode["episode_index"]
    summary = episode.get("h264_summary")
    if not isinstance(summary, dict):
        return f"episode {index} missing h264_summary"
    if summary.get("latency_mode") != "quality_only":
        return f"episode {index} h264_summary.latency_mode is not quality_only"
    requested = summary.get("requested") or {}
    params = condition.params
    if float(requested.get("bitrate_kbps", -1)) != float(params["h264_bitrate_kbps"]):
        return f"episode {index} h264 requested bitrate differs from the condition"
    if int(requested.get("gop", -1)) != int(params["h264_gop"]):
        return f"episode {index} h264 requested gop differs from the condition"
    if str(requested.get("preset")) != str(params["h264_preset"]):
        return f"episode {index} h264 requested preset differs from the condition"
    if float(requested.get("frame_rate_hz", -1)) != float(params["frame_rate_hz"]):
        return f"episode {index} h264 requested frame_rate_hz differs from the condition"
    cameras = summary.get("cameras")
    if not isinstance(cameras, dict) or not cameras:
        return f"episode {index} h264_summary has no cameras"
    n_steps = int(episode["n_control_steps"])
    for cam_key, cam in cameras.items():
        if not isinstance(cam, dict):
            return f"episode {index} camera {cam_key!r} h264 stats missing"
        if int(cam.get("n_frames_in", -1)) != n_steps:
            return (
                f"episode {index} camera {cam_key!r} n_frames_in "
                f"{cam.get('n_frames_in')!r} != n_control_steps {n_steps}"
            )
        if int(cam.get("n_frames_decoded", -1)) != int(cam.get("n_frames_in", -1)):
            return f"episode {index} camera {cam_key!r} n_frames_decoded != n_frames_in"
        if int(cam.get("n_startup_holds", -1)) != 0:
            return f"episode {index} camera {cam_key!r} n_startup_holds is not 0"
        if int(cam.get("n_last_frame_holds", -1)) != 0:
            return f"episode {index} camera {cam_key!r} n_last_frame_holds is not 0"
        if cam.get("first_frame_keyframe") is not True:
            return f"episode {index} camera {cam_key!r} first_frame_keyframe is not true"
        if int(cam.get("encoded_bytes_total", 0)) <= 0:
            return f"episode {index} camera {cam_key!r} encoded_bytes_total is not > 0"
    return None


def _shared_baseline_runtime(
    *,
    backend: str,
    variant_label: str,
) -> dict[str, Any]:
    if backend not in {BACKEND_ONNXRUNTIME, BACKEND_PYTORCH}:
        raise ValueError(f"unsupported backend {backend!r}")
    if not variant_label:
        raise ValueError("variant_label is required")
    return {
        "latency_mode": "constant",
        "latency_s": 0.0,
        "image_mode": "native",
        "variant_label": variant_label,
        "policy_backend": backend,
    }


def _load_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{path.name} unreadable: {exc}"
    if not isinstance(payload, dict):
        return None, f"{path.name} is not a JSON object"
    return payload, None


def run_report_problem(
    report_path: Path,
    *,
    plan: ExperimentPlan,
    expected_runtime: dict[str, Any],
) -> str | None:
    """Return why a run report is not a finished run of the planned episodes, else None.

    Task outcomes that end an episode (success, termination, time-limit truncation) are
    complete; runtime errors and failed environment creation are not.
    """
    report, problem = _load_json_object(Path(report_path))
    if report is None:
        return problem
    episodes = report.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        return "report has no episodes"
    provenance = report.get("provenance")
    summary = report.get("summary")
    if not isinstance(provenance, dict) or not isinstance(summary, dict):
        return "report missing provenance or summary"
    if provenance.get("env_created") is not True:
        return f"environment was not created ({provenance.get('env_error')!r})"
    if provenance.get("candidate") != plan.candidate:
        return f"candidate {provenance.get('candidate')!r} != plan {plan.candidate!r}"
    env_seeds = [pair.env_seed for pair in plan.seed_pairs]
    policy_seeds = [pair.policy_rng_seed for pair in plan.seed_pairs]
    if provenance.get("env_seeds") != env_seeds:
        return "provenance env_seeds differ from the plan"
    if provenance.get("policy_rng_seeds") != policy_seeds:
        return "provenance policy_rng_seeds differ from the plan"
    if provenance.get("latency_mode") != expected_runtime["latency_mode"]:
        return f"latency_mode {provenance.get('latency_mode')!r} differs from the condition"
    if "policy_backend" not in expected_runtime:
        raise ValueError("expected_runtime requires policy_backend")
    backend = expected_runtime["policy_backend"]
    if backend == BACKEND_PYTORCH:
        if provenance.get("policy_backend") != BACKEND_PYTORCH:
            return f"policy_backend {provenance.get('policy_backend')!r} is not 'pytorch'"
        if provenance.get("onnx_runtime") is not None:
            return "pytorch run must not include onnx_runtime provenance"
    else:
        onnx_runtime = provenance.get("onnx_runtime")
        if not isinstance(onnx_runtime, dict) or (
            onnx_runtime.get("variant_label") != expected_runtime["variant_label"]
        ):
            return f"runtime variant is not {expected_runtime['variant_label']!r}"
    if summary.get("n_episodes_total") != len(episodes):
        return "summary episode count differs from episodes"
    if summary.get("n_runtime_error"):
        return f"{summary['n_runtime_error']} runtime-error episodes"

    indices = [ep.get("episode_index") if isinstance(ep, dict) else None for ep in episodes]
    if sorted(i for i in indices if isinstance(i, int)) != list(range(plan.n_episodes)):
        return f"episode indices {indices} are not exactly 0..{plan.n_episodes - 1}"
    for ep in episodes:
        index = ep["episode_index"]
        if ep.get("seed") != env_seeds[index] or ep.get("policy_rng_seed") != policy_seeds[index]:
            return f"episode {index} seeds differ from the plan"
        if ep.get("outcome") not in _COMPLETED_EPISODE_OUTCOMES or ep.get("runtime_error"):
            return f"episode {index} outcome={ep.get('outcome')!r} runtime_error={ep.get('runtime_error')!r}"
        if ep.get("latency_mode") != expected_runtime["latency_mode"]:
            return f"episode {index} latency_mode differs from the condition"
        if expected_runtime["latency_mode"] == "constant" and (
            ep.get("latency_s") != expected_runtime["latency_s"]
        ):
            return f"episode {index} latency_s={ep.get('latency_s')!r} differs from the condition"
        if ep.get("image_mode") != expected_runtime["image_mode"]:
            return f"episode {index} image_mode={ep.get('image_mode')!r} differs from the condition"
    return None


def condition_report_problem(
    report_path: Path,
    plan: ExperimentPlan,
    condition: ConditionSpec,
    *,
    backend: str,
    variant_label: str,
) -> str | None:
    problem = run_report_problem(
        report_path,
        plan=plan,
        expected_runtime=_expected_runtime(condition, backend=backend, variant_label=variant_label),
    )
    if problem is not None:
        return problem
    if condition.axis == ExperimentAxis.DEVICE_PROFILE.value:
        starts = condition.params.get("window_starts")
        episodes = json.loads(Path(report_path).read_text(encoding="utf-8"))["episodes"]
        for ep in episodes:
            window = (ep.get("latency_trace_provenance") or {}).get("window_start")
            if starts is not None and window != starts[ep["episode_index"]]:
                return (
                    f"episode {ep['episode_index']} replay window_start differs from the condition"
                )
        return None
    if condition.axis == ExperimentAxis.H264.value and not condition.params.get("bypass"):
        episodes = json.loads(Path(report_path).read_text(encoding="utf-8"))["episodes"]
        for ep in episodes:
            h264_problem = _h264_summary_problem(ep, condition)
            if h264_problem is not None:
                return h264_problem
    return None


def shared_baseline_problem(
    shared_baseline_dir: Path,
    *,
    plan: ExperimentPlan,
    backend: str,
    variant_label: str,
) -> str | None:
    """Check the shared baseline report that reference conditions point to."""
    report_path = Path(shared_baseline_dir) / "run_report.json"
    if not report_path.is_file():
        return f"shared baseline report missing: {report_path}"
    return run_report_problem(
        report_path,
        plan=plan,
        expected_runtime=_shared_baseline_runtime(backend=backend, variant_label=variant_label),
    )


def condition_incomplete_reason(
    condition_dir: Path,
    *,
    plan: ExperimentPlan,
    condition: ConditionSpec,
    shared_baseline_dir: Path,
    backend: str,
    variant_label: str,
) -> str | None:
    """Return why a condition directory is not a finished, matching run, else None."""
    condition_dir = Path(condition_dir)
    report_path = condition_dir / "run_report.json"
    if not report_path.is_file():
        return "run_report.json missing"
    reference_path = condition_dir / "baseline_reference.json"
    if not reference_path.is_file():
        return condition_report_problem(
            report_path, plan, condition, backend=backend, variant_label=variant_label
        )

    reference, problem = _load_json_object(reference_path)
    if reference is None:
        return problem
    if reference.get("status") != "referenced_baseline":
        return f"baseline_reference status={reference.get('status')!r}"
    if reference.get("condition_id") != condition.condition_id:
        return "baseline_reference belongs to another condition"
    if not condition_is_shared_baseline(condition, baseline_variant=variant_label):
        return "condition is not eligible for baseline reuse"
    if Path(str(reference.get("reference_dir", ""))).name != Path(shared_baseline_dir).name:
        return "baseline_reference points to a different shared baseline"
    problem = shared_baseline_problem(
        shared_baseline_dir, plan=plan, backend=backend, variant_label=variant_label
    )
    if problem is not None:
        return f"referenced baseline: {problem}"
    shared_report = Path(shared_baseline_dir) / "run_report.json"
    if report_path.read_bytes() != shared_report.read_bytes():
        return "condition run_report.json differs from the referenced baseline report"
    return None


def condition_completed(
    condition_dir: Path,
    *,
    plan: ExperimentPlan,
    condition: ConditionSpec,
    shared_baseline_dir: Path,
    backend: str,
    variant_label: str,
) -> bool:
    reason = condition_incomplete_reason(
        condition_dir,
        plan=plan,
        condition=condition,
        shared_baseline_dir=shared_baseline_dir,
        backend=backend,
        variant_label=variant_label,
    )
    if reason is not None:
        LOGGER.warning("condition %s not complete: %s", condition.condition_id, reason)
    return reason is None


def write_unsupported_record(
    out_dir: Path,
    *,
    stem: str,
    status: str | None,
    detail: str | None,
    path: Path | str | None = None,
    sha256: str | None = None,
) -> Path:
    """Write {stem}_unsupported.json with status, detail, and optional path/sha256."""
    out = out_dir / f"{stem}_unsupported.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "status": status,
                "detail": detail,
                "measurements_fabricated": False,
                "path": str(path) if path else None,
                "sha256": sha256,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return out


def describe_plan_commands(plan: ExperimentPlan, *, axis_out: Path) -> list[str]:
    lines = [
        f"# plan_id={plan.plan_id} axis={plan.axis} n={plan.n_episodes} → {axis_out}",
        f"# baseline={plan.baseline_condition_id} seeds_env={[s.env_seed for s in plan.seed_pairs]}",
    ]
    for cond in plan.conditions:
        lines.append(f"#   condition {cond.condition_id}: {cond.label}")
    return lines


def seed_pairs_as_dict(pairs: list[EpisodeSeedPair]) -> list[dict[str, int]]:
    return [s.to_dict() for s in pairs]


def write_plan_bundle(plans: list[ExperimentPlan], plans_dir: Path) -> list[Path]:
    plans_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for plan in plans:
        path = write_experiment_plan(plan, plans_dir / f"{plan.plan_id}.json")
        paths.append(path)
    return paths
