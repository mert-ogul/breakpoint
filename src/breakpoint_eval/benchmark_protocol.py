"""Fixed v0.1 protocol: thin planning/resume helpers over existing plan/runner APIs."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from breakpoint_eval.experiment_plan import (
    ConditionSpec,
    EpisodeSeedPair,
    ExperimentAxis,
    ExperimentPlan,
    PlanPhase,
    build_seed_pairs,
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
from breakpoint_eval.latency_trace import (
    contiguous_window_starts,
    load_latency_profile,
    require_unique_device_profile,
)

LOGGER = logging.getLogger("breakpoint_eval.benchmark_protocol")

PROTOCOL_LOCK_NAME = "v1_protocol_lock.json"
DEFAULT_CONFIG_REL = Path("configs/v1.json")
NATIVE_HW = (480, 640)


@dataclass(frozen=True)
class ResolvedArtifacts:
    export_dir: Path
    fp32_path: Path
    fp32_sha256: str
    fp32_recipe_id: str
    int8_path: Path | None
    int8_sha256: str | None
    int8_status: str
    int8_detail: str | None
    profiles: dict[str, Path]
    profile_identities: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class V1HostSettings:
    compute_device: str
    ort_providers: list[str]
    intra_op_num_threads: int
    inter_op_num_threads: int


def load_benchmark_config(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"unsupported v1 config schema_version={payload.get('schema_version')!r}")
    if payload.get("protocol") != "v0.1":
        raise ValueError(f"expected protocol v0.1, got {payload.get('protocol')!r}")
    return payload


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    """Strict identity resolution — never pick newest file by mtime."""
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

    int8_name = arts.get("int8_onnx_name") or "act_chunk_int8_qdq.onnx"
    int8_path = (export / int8_name).resolve()
    int8_sha: str | None = None
    int8_status = "missing"
    int8_detail: str | None = None
    resolved_int8: Path | None = None
    if int8_path.is_file():
        try:
            int8_rt = resolve_onnx_runtime(
                int8_path,
                expected_candidate=config["candidate"],
                require_export_validation=True,
            )
            expected_int8 = arts.get("expected_int8_sha256")
            if expected_int8 and int8_rt.model_sha256 != expected_int8:
                raise ValueError(
                    f"INT8 sha256 mismatch: file={int8_rt.model_sha256} expected={expected_int8}"
                )
            resolved_int8 = int8_path
            int8_sha = int8_rt.model_sha256
            int8_status = "ok"
        except (ValueError, FileNotFoundError) as exc:
            int8_status = "unsupported"
            int8_detail = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("INT8 present but not runnable: %s", int8_detail)
    else:
        # Check manifest for explicit unsupported status.
        manifest = load_export_manifest(export / "export_manifest.json")
        for variant in manifest.get("variants") or []:
            if variant.get("label") == "int8":
                int8_status = str(variant.get("status") or "missing")
                int8_detail = variant.get("detail")
                break

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
        int8_path=resolved_int8,
        int8_sha256=int8_sha,
        int8_status=int8_status,
        int8_detail=int8_detail,
        profiles=profiles,
        profile_identities=identities,
    )


def host_settings_from_config(config: dict[str, Any]) -> V1HostSettings:
    host = config["host"]
    return V1HostSettings(
        compute_device=str(host["compute_device"]),
        ort_providers=list(host["ort_providers"]),
        intra_op_num_threads=int(host["intra_op_num_threads"]),
        inter_op_num_threads=int(host["inter_op_num_threads"]),
    )


def _resolution_conditions_from_hw(levels: list[list[int]]) -> list[ConditionSpec]:
    out: list[ConditionSpec] = []
    for hw in levels:
        h, w = int(hw[0]), int(hw[1])
        if (h, w) == NATIVE_HW:
            out.append(
                ConditionSpec(
                    condition_id="resolution_native",
                    axis=ExperimentAxis.RESOLUTION.value,
                    params={
                        "mode": "native",
                        "linear_scale": 1.0,
                        "native_hw": list(NATIVE_HW),
                        "capture_hw": list(NATIVE_HW),
                        "effective_policy_hw": list(NATIVE_HW),
                    },
                    label=f"resolution native {NATIVE_HW[0]}x{NATIVE_HW[1]}",
                )
            )
        else:
            out.append(
                ConditionSpec(
                    condition_id=f"resolution_capture_{h}x{w}",
                    axis=ExperimentAxis.RESOLUTION.value,
                    params={
                        "mode": "downsample",
                        "native_hw": list(NATIVE_HW),
                        "capture_hw": [h, w],
                        "effective_policy_hw": list(NATIVE_HW),
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


def _baseline_fingerprint() -> dict[str, Any]:
    return {
        "latency_s": 0.0,
        "image_mode": "native",
        "variant_label": "onnx_fp32",
        "no_device_profile": True,
    }


def condition_is_shared_baseline(condition: ConditionSpec) -> bool:
    params = condition.params
    if condition.axis == ExperimentAxis.LATENCY_MS.value:
        return float(params.get("latency_s", -1)) == 0.0
    if condition.axis == ExperimentAxis.RESOLUTION.value:
        return params.get("mode") == "native"
    if condition.axis == ExperimentAxis.JPEG.value:
        return bool(params.get("bypass"))
    if condition.axis == ExperimentAxis.PRECISION.value:
        return params.get("variant_label") == "onnx_fp32"
    return False


def build_smoke_plan(
    config: dict[str, Any],
    artifacts: ResolvedArtifacts,
) -> ExperimentPlan:
    smoke = config["smoke"]
    seeds = build_seed_pairs(
        len(smoke["env_seeds"]),
        env_seeds=list(smoke["env_seeds"]),
        policy_seed_offset=int(smoke["policy_seed_offset"]),
    )
    window_length = int(config["trace_window"]["length"])
    conditions: list[ConditionSpec] = []
    conditions.extend(latency_conditions(tuple(int(x) for x in smoke["latency_ms"])))
    conditions.extend(_resolution_conditions_from_hw(list(smoke["resolution_hw"])))
    conditions.extend(jpeg_conditions(tuple(smoke["jpeg_qualities"])))
    # Precision: always list FP32; INT8 only if runnable, else recorded separately.
    prec_variants = ["onnx_fp32"]
    if "int8" in smoke.get("precision_variants", []) and artifacts.int8_status == "ok":
        prec_variants.append("int8")
    conditions.extend(precision_conditions(tuple(prec_variants)))
    conditions.extend(
        _device_profile_conditions(
            profile_keys=list(smoke["device_profiles"]),
            artifacts=artifacts,
            window_length=window_length,
            n_episodes=len(seeds),
        )
    )
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
        notes=[
            "Smoke plan; not for publication claims.",
            f"INT8 status at plan build: {artifacts.int8_status}",
        ],
    )


def build_full_axis_plans(
    config: dict[str, Any],
    artifacts: ResolvedArtifacts,
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
    plans: list[ExperimentPlan] = []

    lat = ExperimentPlan(
        plan_id="v0.1_full_latency",
        candidate=candidate,
        phase=PlanPhase.EXPLORATORY.value,
        n_episodes=n,
        axis=ExperimentAxis.LATENCY_MS.value,
        seed_pairs=seeds,
        conditions=latency_conditions(tuple(int(x) for x in full["latency_ms"])),
        isolation_notes=["Synthetic constant delay; precision held at onnx_fp32."],
        baseline_condition_id="latency_ms_0",
        notes=["Isolated latency axis."],
    )
    plans.append(lat)

    res = ExperimentPlan(
        plan_id="v0.1_full_resolution",
        candidate=candidate,
        phase=PlanPhase.EXPLORATORY.value,
        n_episodes=n,
        axis=ExperimentAxis.RESOLUTION.value,
        seed_pairs=seeds,
        conditions=_resolution_conditions_from_hw(list(full["resolution_hw"])),
        isolation_notes=["Resolution axis only; latency and precision at defaults."],
        baseline_condition_id="resolution_native",
        notes=["Isolated resolution axis; native bypass is not JPEG quality 100."],
    )
    plans.append(res)

    jpeg = ExperimentPlan(
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
    plans.append(jpeg)

    prec_variants = ["onnx_fp32"]
    if "int8" in full.get("precision_variants", []) and artifacts.int8_status == "ok":
        prec_variants.append("int8")
    prec = ExperimentPlan(
        plan_id="v0.1_full_precision",
        candidate=candidate,
        phase=PlanPhase.EXPLORATORY.value,
        n_episodes=n,
        axis=ExperimentAxis.PRECISION.value,
        seed_pairs=seeds,
        conditions=precision_conditions(tuple(prec_variants)),
        isolation_notes=["Precision under constant_latency_ms_0."],
        baseline_condition_id="precision_onnx_fp32",
        notes=[f"INT8 status at plan build: {artifacts.int8_status}"],
    )
    plans.append(prec)

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
    host: V1HostSettings,
    phase: str,
) -> dict[str, Any]:
    stats = config["statistics"]
    return {
        "schema_version": 1,
        "protocol": config["protocol"],
        "phase": phase,
        "candidate": config["candidate"],
        "source_revision": source_revision,
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
        },
        "host": {
            "compute_device": host.compute_device,
            "ort_providers": host.ort_providers,
            "intra_op_num_threads": host.intra_op_num_threads,
            "inter_op_num_threads": host.inter_op_num_threads,
        },
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


def assert_lock_matches(existing: dict[str, Any], fresh: dict[str, Any]) -> None:
    """Refuse resume when identity or protocol config changed."""
    keys = ("protocol", "phase", "candidate", "artifacts", "host", "trace_window", "statistics")
    for key in keys:
        if existing.get(key) != fresh.get(key):
            raise ValueError(
                f"resume refused: {key} changed between lock and current resolution. "
                "Use a fresh --out directory for identity/config changes."
            )


def condition_completed(condition_dir: Path) -> bool:
    report = condition_dir / "run_report.json"
    if not report.is_file():
        return False
    ref = condition_dir / "baseline_reference.json"
    if ref.is_file():
        return True
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    summary = payload.get("summary") or {}
    if summary.get("n_runtime_error", 0) > 0:
        return False
    if (payload.get("provenance") or {}).get("env_created") is False:
        return False
    return True


def write_unsupported_int8_record(
    out_dir: Path,
    *,
    artifacts: ResolvedArtifacts,
) -> Path:
    path = out_dir / "int8_unsupported.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": artifacts.int8_status,
                "detail": artifacts.int8_detail,
                "measurements_fabricated": False,
                "path": str(artifacts.int8_path) if artifacts.int8_path else None,
                "sha256": artifacts.int8_sha256,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


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
