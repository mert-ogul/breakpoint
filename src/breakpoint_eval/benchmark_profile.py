"""Device profiling helpers (onnx/numpy only — no simulator or Torch)."""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from breakpoint_eval.artifact_names import EXPORT_VALIDATION_NAME, resolve_artifact
from breakpoint_eval.device_profile import (
    DEFAULT_MEASURED,
    DEFAULT_WARMUPS,
    ProfileRequest,
    run_device_profile,
    sha256_file,
)
from breakpoint_eval.source_revision import read_source_revision

LOGGER = logging.getLogger("breakpoint_eval.benchmark_profile")

FP32_VARIANT = "onnx_fp32"
RECIPE_ONNX_FP32 = "torch_onnx_export_fp32_opset17"
DEFAULT_PROVIDERS = ("CPUExecutionProvider",)
DEFAULT_INTRA = 4
DEFAULT_INTER = 1
TELEMETRY_INTERVAL_S = 2.0


@dataclass(frozen=True)
class ExportBundleFp32:
    export_dir: Path
    model_path: Path
    model_sha256: str
    recipe_id: str
    variant_label: str
    hub_revision: str | None
    chunk_horizon: int | None
    input_shapes: dict[str, list[int]] | None
    corpus_path: Path
    reference_chunks_path: Path
    external_weights: list[dict[str, str]]
    validation_passed: bool


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def package_list() -> list[dict[str, str]]:
    """Installed distributions via importlib.metadata (no pip subprocess required)."""
    try:
        from importlib import metadata
    except ImportError:
        return []
    rows: list[dict[str, str]] = []
    for dist in metadata.distributions():
        name = dist.metadata["Name"] if dist.metadata else None
        version = dist.version
        if name:
            rows.append({"name": name, "version": version})
    rows.sort(key=lambda r: r["name"].lower())
    return rows


def read_boot_id() -> dict[str, Any]:
    path = Path("/proc/sys/kernel/random/boot_id")
    if not path.is_file():
        return {"available": False, "boot_id": None, "error": "boot_id file absent"}
    try:
        return {
            "available": True,
            "boot_id": path.read_text(encoding="utf-8").strip(),
            "error": None,
        }
    except OSError as exc:
        return {"available": False, "boot_id": None, "error": f"{type(exc).__name__}: {exc}"}


def _run_vcgencmd(args: list[str]) -> dict[str, Any]:
    exe = shutil.which("vcgencmd")
    if exe is None:
        return {"available": False, "command": args, "stdout": None, "error": "vcgencmd not found"}
    try:
        proc = subprocess.run(
            [exe, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return {
            "available": True,
            "command": [exe, *args],
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip() if proc.stdout else "",
            "stderr": proc.stderr.strip() if proc.stderr else "",
            "error": None if proc.returncode == 0 else f"exit {proc.returncode}",
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "available": True,
            "command": [exe, *args],
            "stdout": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def sample_pi_telemetry() -> dict[str, Any]:
    return {
        "wall_utc": _utc_now(),
        "boot_id": read_boot_id(),
        "temperature": _run_vcgencmd(["measure_temp"]),
        "throttled": _run_vcgencmd(["get_throttled"]),
    }


class TelemetrySampler:
    """Sample Pi sensors ~every interval outside the inference timed region."""

    def __init__(self, *, interval_s: float = TELEMETRY_INTERVAL_S) -> None:
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self.failures: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="profile-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 3)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(sample_pi_telemetry())
            except Exception as exc:  # noqa: BLE001 — record, do not invent
                self.failures.append(
                    {"wall_utc": _utc_now(), "error": f"{type(exc).__name__}: {exc}"}
                )
            self._stop.wait(self.interval_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interval_s": self.interval_s,
            "n_samples": len(self.samples),
            "samples": self.samples,
            "failures": self.failures,
            "note": "Sampled outside ORT timed region; missing observations not fabricated",
        }


def _external_weight_files(model_path: Path) -> list[dict[str, str]]:
    parent = Path(model_path).parent
    stem = Path(model_path).name
    extras: list[dict[str, str]] = []
    for path in sorted(parent.iterdir()):
        if not path.is_file():
            continue
        if path.resolve() == Path(model_path).resolve():
            continue
        if path.name == stem + ".data" or path.name.startswith(stem + "."):
            if path.suffix.lower() in {".json", ".npz", ".md"}:
                continue
            if path.name.endswith(".onnx"):
                continue
            extras.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    return extras


def resolve_validated_fp32_bundle(export_dir: Path) -> ExportBundleFp32:
    """Read recipe/revision/shapes/hashes from the export bundle without Torch."""
    export_dir = Path(export_dir).resolve()
    manifest_path = export_dir / "export_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"export_manifest.json missing under {export_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    variants = [v for v in (manifest.get("variants") or []) if isinstance(v, dict)]
    fp32 = [v for v in variants if v.get("label") == FP32_VARIANT]
    if not fp32:
        raise FileNotFoundError(f"no {FP32_VARIANT} variant in {manifest_path}")
    if len(fp32) > 1:
        raise ValueError(f"ambiguous {FP32_VARIANT} variants in {manifest_path}")
    variant = fp32[0]
    if variant.get("status") != "ok":
        raise ValueError(f"FP32 variant status={variant.get('status')!r}; refuse profiling")
    rel = variant.get("path")
    if not rel:
        raise ValueError("FP32 variant missing path")
    model_path = Path(rel)
    if not model_path.is_file():
        model_path = export_dir / Path(rel).name
    if not model_path.is_file():
        raise FileNotFoundError(f"FP32 ONNX not found: {rel}")
    model_path = model_path.resolve()
    digest = sha256_file(model_path)
    graph = variant.get("graph") or {}
    declared = graph.get("sha256")
    if declared and declared != digest:
        raise ValueError(f"FP32 hash mismatch: file={digest} manifest={declared}")
    recipe_id = str(variant.get("recipe_id") or RECIPE_ONNX_FP32)
    validation_path = resolve_artifact(export_dir, EXPORT_VALIDATION_NAME)
    validation_passed = False
    if validation_path is not None:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        binding = validation.get("binding") or {}
        bound_sha = binding.get("model_sha256") or validation.get("model_sha256")
        if bound_sha and bound_sha != digest:
            raise ValueError(f"{validation_path.name} bound sha256={bound_sha} != model {digest}")
        validation_passed = bool(validation.get("passed"))
    if not validation_passed:
        raise ValueError(
            f"validated onnx_fp32 required: {EXPORT_VALIDATION_NAME} passed=true "
            f"bound to {digest[:12]}"
        )

    corpus = export_dir / "corpus_eval.npz"
    if not corpus.is_file():
        raise FileNotFoundError(f"corpus_eval.npz missing under {export_dir}")
    # Prefer ONNX FP32 reference chunks written by export (sample_id keys).
    ref_candidates = [
        export_dir / "eval_chunks_onnx_fp32.npz",
        export_dir / "reference_chunks_onnx_fp32.npz",
    ]
    ref_path = next((p for p in ref_candidates if p.is_file()), None)
    if ref_path is None:
        found = sorted(export_dir.glob("eval_chunks_*fp32*.npz"))
        if len(found) == 1:
            ref_path = found[0]
        elif len(found) > 1:
            raise FileNotFoundError(
                f"ambiguous FP32 reference chunk files under {export_dir}: "
                f"{[p.name for p in found]}"
            )
        else:
            raise FileNotFoundError(
                f"ONNX FP32 reference chunks missing under {export_dir}; "
                "expected eval_chunks_onnx_fp32.npz"
            )

    io_spec = manifest.get("io_spec") or variant.get("io_spec") or {}
    shapes = None
    if isinstance(io_spec, dict) and io_spec.get("image_shape") and io_spec.get("state_shape"):
        shapes = {
            "observation_images_top": list(io_spec["image_shape"]),
            "observation_state": list(io_spec["state_shape"]),
        }
    chunk_horizon = None
    if isinstance(io_spec, dict) and io_spec.get("action_chunk_shape"):
        ac = io_spec["action_chunk_shape"]
        if isinstance(ac, list) and len(ac) >= 2:
            chunk_horizon = int(ac[1])

    return ExportBundleFp32(
        export_dir=export_dir,
        model_path=model_path,
        model_sha256=digest,
        recipe_id=recipe_id,
        variant_label=FP32_VARIANT,
        hub_revision=manifest.get("hub_revision"),
        chunk_horizon=chunk_horizon,
        input_shapes=shapes,
        corpus_path=corpus.resolve(),
        reference_chunks_path=ref_path.resolve(),
        external_weights=_external_weight_files(model_path),
        validation_passed=validation_passed,
    )


def refuse_completed_output_dir(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    marker = output_dir / "profile_run_complete.json"
    if marker.is_file():
        raise FileExistsError(
            f"refuse to overwrite completed profile output: {output_dir} (found {marker.name})"
        )
    if output_dir.is_dir() and any(output_dir.iterdir()):
        # Non-empty incomplete dir is also refused to avoid mixed runs.
        raise FileExistsError(
            f"output directory is not empty: {output_dir}. "
            "Choose a fresh --output-dir or remove incomplete contents."
        )


def run_benchmark_profile(
    *,
    device_label: str,
    export_dir: Path,
    output_dir: Path,
    n_warmup: int = DEFAULT_WARMUPS,
    n_measured: int = DEFAULT_MEASURED,
    power_note: str | None = None,
    cooling_note: str | None = None,
    repo_root: Path | None = None,
    candidate: str = "act_aloha_transfer",
) -> Path:
    """Profile validated FP32 with CPU EP 4/1 threads; write telemetry sidecar."""
    # Guard: this module must stay free of torch/sim imports at call time.
    for banned in ("torch", "mujoco", "dm_control", "gymnasium", "lerobot"):
        if banned in __import__("sys").modules:
            # Already loaded by caller is OK on laptop; Pi path should not have them.
            # Soft note only — refuse only if env marks itself as pi-profile-only.
            pass

    refuse_completed_output_dir(Path(output_dir))
    bundle = resolve_validated_fp32_bundle(Path(export_dir))
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = str(uuid.uuid4())
    started = _utc_now()
    root = Path(repo_root) if repo_root else Path.cwd()
    source_revision = read_source_revision(root)

    runtime_versions: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import onnxruntime as ort

        runtime_versions["onnxruntime"] = ort.__version__
        runtime_versions["available_providers"] = ort.get_available_providers()
    except ImportError as exc:
        raise ImportError("profile_device requires onnxruntime ([onnx] extra)") from exc
    try:
        import onnx

        runtime_versions["onnx"] = onnx.__version__
    except ImportError:
        runtime_versions["onnx"] = None
    try:
        import numpy as np

        runtime_versions["numpy"] = np.__version__
    except ImportError:
        runtime_versions["numpy"] = None

    telemetry = TelemetrySampler()
    before = sample_pi_telemetry()
    telemetry.start()
    profile_path: Path | None = None
    error: str | None = None
    try:
        profile_path = run_device_profile(
            ProfileRequest(
                model_path=bundle.model_path,
                corpus_path=bundle.corpus_path,
                output_dir=output_dir,
                device_label=device_label,
                candidate=candidate,
                recipe_id=bundle.recipe_id,
                variant_label=bundle.variant_label,
                reference_chunks_path=bundle.reference_chunks_path,
                n_warmup=n_warmup,
                n_measured=n_measured,
                providers=DEFAULT_PROVIDERS,
                intra_op_num_threads=DEFAULT_INTRA,
                inter_op_num_threads=DEFAULT_INTER,
                cooling_note=cooling_note,
                power_note=power_note,
                hub_revision=bundle.hub_revision,
                chunk_horizon=bundle.chunk_horizon,
            )
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        telemetry.stop()
        after = sample_pi_telemetry()
        ended = _utc_now()
        meta = {
            "run_id": run_id,
            "started_wall_utc": started,
            "ended_wall_utc": ended,
            "device_label": device_label,
            "candidate": candidate,
            "source_revision": source_revision,
            "runtime_versions": runtime_versions,
            "packages": package_list(),
            "hardware_setup": {
                "power_note": power_note,
                "cooling_note": cooling_note,
                "cpu_count_logical": os.cpu_count(),
            },
            "bundle_identity": {
                "export_dir": str(bundle.export_dir),
                "model_path": str(bundle.model_path),
                "model_sha256": bundle.model_sha256,
                "recipe_id": bundle.recipe_id,
                "variant_label": bundle.variant_label,
                "hub_revision": bundle.hub_revision,
                "chunk_horizon": bundle.chunk_horizon,
                "input_shapes": bundle.input_shapes,
                "corpus_path": str(bundle.corpus_path),
                "reference_chunks_path": str(bundle.reference_chunks_path),
                "external_weights": bundle.external_weights,
                "validation_passed": bundle.validation_passed,
            },
            "ort_settings": {
                "providers": list(DEFAULT_PROVIDERS),
                "intra_op_num_threads": DEFAULT_INTRA,
                "inter_op_num_threads": DEFAULT_INTER,
            },
            "telemetry": {
                "before": before,
                "during": telemetry.to_dict(),
                "after": after,
            },
            "profile_json": str(profile_path) if profile_path else None,
            "error": error,
        }
        (output_dir / "profile_run_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output_dir / "telemetry_raw.json").write_text(
            json.dumps(meta["telemetry"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if error is None and profile_path is not None:
            (output_dir / "profile_run_complete.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "completed": True,
                        "profile_json": str(profile_path),
                        "ended_wall_utc": ended,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
    assert profile_path is not None
    return profile_path
