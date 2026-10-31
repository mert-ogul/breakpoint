"""Local ONNX Runtime device profiling without simulator dependencies.

Times only ``InferenceSession.run`` on batch-1 recorded inputs. Does not time
random matmuls or action-queue dequeue. Warmup is separate from measured calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from breakpoint_eval.file_digest import sha256_file
from breakpoint_eval.latency_trace import (
    PROFILE_SCHEMA_VERSION,
    TIMED_BOUNDARY_ORT_RUN,
    summarize_latency_s,
    validate_latency_sequence,
)

LOGGER = logging.getLogger("breakpoint_eval.device_profile")

INPUT_IMAGES = "observation_images_top"
INPUT_STATE = "observation_state"
OUTPUT_CHUNK = "action_chunk"

DEFAULT_WARMUPS = 50
DEFAULT_MEASURED = 1000


def require_onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "breakpoint profile requires onnxruntime "
            "(pip install 'breakpoint-eval[onnx]' / requirements-onnx.txt)"
        ) from exc
    return ort


def detect_host_info() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
    }


def read_temperature_c() -> dict[str, Any]:
    """Best-effort thermal readout. Missing sensors are recorded as unavailable."""
    readings: list[dict[str, Any]] = []
    thermal_root = Path("/sys/class/thermal")
    if thermal_root.is_dir():
        for zone in sorted(thermal_root.glob("thermal_zone*")):
            try:
                temp_raw = (zone / "temp").read_text(encoding="utf-8").strip()
                typ = (zone / "type").read_text(encoding="utf-8").strip()
                # Most Linux thermal zones report millidegrees C.
                temp_c = float(temp_raw) / 1000.0 if float(temp_raw) > 200 else float(temp_raw)
                readings.append({"source": str(zone), "type": typ, "temperature_c": temp_c})
            except (OSError, ValueError):
                continue
    return {
        "available": bool(readings),
        "readings": readings,
        "throttling": None,  # None when the OS does not expose it
        "note": None if readings else "temperature telemetry unavailable on this host",
    }


def load_profile_corpus(path: Path) -> dict[str, Any]:
    """Load representative recorded inputs (export corpus npz). No simulator."""
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    if "images_top" not in data.files or "state" not in data.files:
        raise ValueError(f"corpus {path} must contain images_top and state arrays")
    images = np.asarray(data["images_top"], dtype=np.float32)
    state = np.asarray(data["state"], dtype=np.float32)
    if images.ndim != 5 or images.shape[1] != 1:
        raise ValueError(f"expected images_top (N,1,C,H,W) batch-1 samples, got {images.shape}")
    if state.ndim != 3 or state.shape[1] != 1:
        raise ValueError(f"expected state (N,1,D) batch-1 samples, got {state.shape}")
    seeds = (
        [int(s) for s in data["seeds"].tolist()]
        if "seeds" in data.files
        else list(range(len(images)))
    )
    if "sample_ids" in data.files:
        sample_ids = [str(x) for x in data["sample_ids"].tolist()]
    else:
        sample_ids = [f"seed{seed}_t0" for seed in seeds]
    if len(sample_ids) != len(images):
        raise ValueError("corpus sample_ids length must match images_top")
    raw = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "images_top": images,
        "state": state,
        "seeds": seeds,
        "sample_ids": sample_ids,
        "n_samples": int(images.shape[0]),
        "shapes": {
            INPUT_IMAGES: list(images.shape[1:]),
            INPUT_STATE: list(state.shape[1:]),
        },
    }


def load_reference_chunks(path: Path | None) -> dict[str, np.ndarray] | None:
    if path is None:
        return None
    data = np.load(Path(path), allow_pickle=False)
    return {k: np.asarray(data[k], dtype=np.float32) for k in data.files}


@dataclass(frozen=True)
class ProfileRequest:
    model_path: Path
    corpus_path: Path
    output_dir: Path
    device_label: str
    candidate: str
    recipe_id: str
    variant_label: str = "onnx_fp32"
    reference_chunks_path: Path | None = None
    n_warmup: int = DEFAULT_WARMUPS
    n_measured: int = DEFAULT_MEASURED
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    intra_op_num_threads: int | None = None
    inter_op_num_threads: int | None = None
    cooling_note: str | None = None
    power_note: str | None = None
    load_note: str | None = None
    validation_atol: float = 1e-3
    validation_rtol: float = 1e-3
    hub_revision: str | None = None
    chunk_horizon: int | None = None


def _session_options(
    ort: Any,
    *,
    intra_op_num_threads: int | None,
    inter_op_num_threads: int | None,
) -> Any:
    opts = ort.SessionOptions()
    if intra_op_num_threads is not None:
        opts.intra_op_num_threads = int(intra_op_num_threads)
    if inter_op_num_threads is not None:
        opts.inter_op_num_threads = int(inter_op_num_threads)
    return opts


def _create_session(req: ProfileRequest) -> tuple[Any, list[str]]:
    ort = require_onnxruntime()
    available = ort.get_available_providers()
    for p in req.providers:
        if p not in available:
            raise RuntimeError(f"requested provider {p!r} not available; have {available}")
    opts = _session_options(
        ort,
        intra_op_num_threads=req.intra_op_num_threads,
        inter_op_num_threads=req.inter_op_num_threads,
    )
    session = ort.InferenceSession(
        str(req.model_path),
        sess_options=opts,
        providers=list(req.providers),
    )
    return session, list(session.get_providers())


def _run_once(session: Any, images: np.ndarray, state: np.ndarray) -> np.ndarray:
    feeds = {
        INPUT_IMAGES: np.asarray(images, dtype=np.float32),
        INPUT_STATE: np.asarray(state, dtype=np.float32),
    }
    if feeds[INPUT_IMAGES].shape[0] != 1 or feeds[INPUT_STATE].shape[0] != 1:
        raise ValueError("profiling requires batch size 1")
    out = session.run([OUTPUT_CHUNK], feeds)[0]
    return np.asarray(out, dtype=np.float32)


def _resolve_reference_chunk(
    reference_chunks: dict[str, np.ndarray],
    *,
    sample_id: str,
    seed: int,
) -> np.ndarray | None:
    """Match export sample_id keys first; accept legacy seed_{seed} as fallback."""
    if sample_id in reference_chunks:
        return reference_chunks[sample_id]
    legacy = f"seed_{seed}"
    if legacy in reference_chunks:
        return reference_chunks[legacy]
    return None


def validate_outputs_on_device(
    session: Any,
    corpus: dict[str, Any],
    *,
    reference_chunks: dict[str, np.ndarray] | None,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Validate representative outputs before trusting timings. Save this result.

    When ``reference_chunks`` is supplied, every corpus sample must have a
    matching reference entry (by ``sample_id``, or legacy ``seed_{seed}``).
    Missing matches fail validation; shape/finite-only is not enough.
    """
    samples: list[dict[str, Any]] = []
    all_ok = True
    missing_keys: list[str] = []
    sample_ids = corpus.get("sample_ids") or [f"seed{s}_t0" for s in corpus["seeds"]]
    for i in range(corpus["n_samples"]):
        images = corpus["images_top"][i]
        state = corpus["state"][i]
        out = _run_once(session, images, state)
        finite_ok = bool(np.isfinite(out).all())
        shape_ok = out.ndim == 3 and out.shape[0] == 1
        ref_ok = True
        max_abs = None
        seed = corpus["seeds"][i]
        sample_id = sample_ids[i]
        reference_compared = False
        if reference_chunks is not None:
            ref = _resolve_reference_chunk(reference_chunks, sample_id=sample_id, seed=seed)
            if ref is None:
                ref_ok = False
                missing_keys.append(sample_id)
            else:
                reference_compared = True
                if ref.shape != out.shape:
                    # reference may be stored without explicit batch if exported differently
                    if ref.ndim == 2 and out.shape[0] == 1 and ref.shape == out.shape[1:]:
                        ref = ref[None, ...]
                if ref.shape != out.shape:
                    ref_ok = False
                    shape_ok = False
                else:
                    max_abs = float(np.max(np.abs(out.astype(np.float64) - ref.astype(np.float64))))
                    ref_ok = bool(np.allclose(out, ref, atol=atol, rtol=rtol))
        sample_ok = finite_ok and shape_ok and ref_ok
        all_ok = all_ok and sample_ok
        samples.append(
            {
                "seed": seed,
                "sample_id": sample_id,
                "output_shape": list(out.shape),
                "finite": finite_ok,
                "shape_ok": shape_ok,
                "reference_compared": reference_compared,
                "max_abs_error_vs_reference": max_abs,
                "passed": sample_ok,
            }
        )
    if reference_chunks is not None and missing_keys:
        all_ok = False
    return {
        "passed": all_ok,
        "n_samples": corpus["n_samples"],
        "atol": atol,
        "rtol": rtol,
        "reference_chunks_path": None,
        "reference_required": reference_chunks is not None,
        "missing_reference_sample_ids": missing_keys,
        "samples": samples,
        "note": (
            "Timings are trusted only when passed=true. "
            "When reference chunks are supplied, every corpus sample_id must match. "
            "Simulator policy outputs may still be evaluated on the host while "
            "latency is replayed from this device profile."
        ),
    }


def _time_inference_s(
    session: Any, images: np.ndarray, state: np.ndarray
) -> tuple[float, np.ndarray]:
    """Monotonic high-resolution timing of ORT run only (no overlapping stages)."""
    feeds = {
        INPUT_IMAGES: np.asarray(images, dtype=np.float32),
        INPUT_STATE: np.asarray(state, dtype=np.float32),
    }
    # Feed construction is outside the timed boundary.
    t0 = time.perf_counter_ns()
    out = session.run([OUTPUT_CHUNK], feeds)[0]
    t1 = time.perf_counter_ns()
    return (t1 - t0) / 1e9, np.asarray(out, dtype=np.float32)


def run_device_profile(req: ProfileRequest) -> Path:
    if req.n_warmup < 0:
        raise ValueError("n_warmup must be >= 0")
    if req.n_measured < 1:
        raise ValueError("n_measured must be >= 1")

    ort = require_onnxruntime()
    model_path = Path(req.model_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")

    corpus = load_profile_corpus(Path(req.corpus_path).resolve())
    reference = load_reference_chunks(
        Path(req.reference_chunks_path).resolve() if req.reference_chunks_path else None
    )

    # Model load is recorded separately and excluded from timed samples.
    load_t0 = time.perf_counter_ns()
    session, session_providers = _create_session(req)
    load_t1 = time.perf_counter_ns()
    model_load_s = (load_t1 - load_t0) / 1e9

    outputs_meta = session.get_outputs()[0]
    out_shape = [int(d) if isinstance(d, int) else d for d in outputs_meta.shape]
    chunk_horizon = req.chunk_horizon
    if chunk_horizon is None and len(out_shape) >= 2 and isinstance(out_shape[1], int):
        chunk_horizon = int(out_shape[1])

    validation = validate_outputs_on_device(
        session,
        corpus,
        reference_chunks=reference,
        atol=req.validation_atol,
        rtol=req.validation_rtol,
    )
    if req.reference_chunks_path is not None:
        validation["reference_chunks_path"] = str(Path(req.reference_chunks_path).resolve())

    if not validation["passed"]:
        raise RuntimeError(
            "device output validation failed; refusing to record timings. "
            f"details={json.dumps(validation)}"
        )

    n_corpus = corpus["n_samples"]

    def _sample(i: int) -> tuple[np.ndarray, np.ndarray]:
        idx = i % n_corpus
        return corpus["images_top"][idx], corpus["state"][idx]

    # Warmup (not included in measured sequence).
    for i in range(req.n_warmup):
        images, state = _sample(i)
        _run_once(session, images, state)

    measured_s: list[float] = []
    for i in range(req.n_measured):
        images, state = _sample(req.n_warmup + i)
        dt_s, out = _time_inference_s(session, images, state)
        if not np.isfinite(out).all():
            raise RuntimeError(f"non-finite output during measured call {i}")
        measured_s.append(dt_s)

    measured_s = validate_latency_sequence(measured_s)
    stats = summarize_latency_s(measured_s)

    # Infer effective threads from session when possible.
    intra = req.intra_op_num_threads
    inter = req.inter_op_num_threads

    payload: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "device_label": req.device_label,
        "candidate": req.candidate,
        "hub_revision": req.hub_revision,
        "host": detect_host_info(),
        "runtime": {
            "library": "onnxruntime",
            "onnxruntime_version": ort.__version__,
            "requested_providers": list(req.providers),
            "session_providers": session_providers,
            "available_providers": ort.get_available_providers(),
            "intra_op_num_threads": intra,
            "inter_op_num_threads": inter,
        },
        "model": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
            "precision_recipe": req.recipe_id,
            "variant_label": req.variant_label,
            "bytes": model_path.stat().st_size,
        },
        "input_contract": {
            "corpus_path": corpus["path"],
            "corpus_sha256": corpus["sha256"],
            "shapes": corpus["shapes"],
            "batch_size": 1,
            "n_corpus_samples": corpus["n_samples"],
            "seeds": corpus["seeds"],
        },
        "chunk_horizon": chunk_horizon,
        "timed_boundary": TIMED_BOUNDARY_ORT_RUN,
        "stages": {
            "model_load_s": model_load_s,
            "warmup_calls": req.n_warmup,
            "measured_calls": req.n_measured,
            "inference_timed": True,
            "preprocessing_timed_separately": False,
            "postprocessing_timed_separately": False,
            "note": (
                "Only ort_session_run_action_chunk is timed per sample. "
                "Model load and warmup are excluded. Feed construction is outside "
                "the timed region. No overlapping stage sums."
            ),
        },
        "output_validation": validation,
        "measurements": {
            "n_warmup": req.n_warmup,
            "n_measured": req.n_measured,
            "clock": "time.perf_counter_ns",
            "unit": "seconds",
            "inference_latency_s": measured_s,
            "summary": stats.to_dict(),
        },
        "environment_notes": {
            "cooling": req.cooling_note,
            "power": req.power_note,
            "load": req.load_note,
            "temperature": read_temperature_c(),
        },
        "semantics": {
            "times_policy_computation": True,
            "times_random_matmul": False,
            "times_action_queue_dequeue": False,
            "constant_delay_sweep": False,
            "simulator_policy_outputs_may_run_on_host": True,
            "latency_replay_source_device": req.device_label,
        },
    }

    output_dir = Path(req.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        output_dir
        / f"{req.candidate}__{req.variant_label}__{req.device_label}__{payload['model']['sha256'][:12]}.json"
    )
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Companion raw sequence for tooling that prefers binary.
    np.savez_compressed(
        out_path.with_suffix(".npz"),
        inference_latency_s=np.asarray(measured_s, dtype=np.float64),
    )
    LOGGER.info(
        "Wrote latency profile %s (n=%s median_s=%s)", out_path, stats.count, stats.median_s
    )
    return out_path
