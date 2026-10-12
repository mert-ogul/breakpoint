"""Representative observation corpus for export calibration and parity."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("breakpoint_eval.export_corpus")


@dataclass(frozen=True)
class CorpusSample:
    seed: int
    images_top: np.ndarray  # (1,3,H,W) float32 [0,1]
    state: np.ndarray  # (1, state_dim) float32
    source: str  # simulator | synthetic_shape_matched


@dataclass
class CorpusManifest:
    role: str  # calibration | evaluation
    seeds: list[int]
    n_samples: int
    image_shape: list[int]
    state_shape: list[int]
    source: str
    path: str
    sha256: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_corpus(path: Path, samples: list[CorpusSample], *, role: str) -> CorpusManifest:
    if not samples:
        raise ValueError("corpus is empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "role": role,
        "seeds": [s.seed for s in samples],
        "source": samples[0].source,
        "images_top": np.stack([s.images_top for s in samples], axis=0),
        "state": np.stack([s.state for s in samples], axis=0),
    }
    np.savez_compressed(path, **payload)
    raw = path.read_bytes()
    sources = {s.source for s in samples}
    source = sources.pop() if len(sources) == 1 else "mixed"
    return CorpusManifest(
        role=role,
        seeds=[s.seed for s in samples],
        n_samples=len(samples),
        image_shape=list(samples[0].images_top.shape),
        state_shape=list(samples[0].state.shape),
        source=source,
        path=str(path),
        sha256=_sha256_bytes(raw),
        note=(
            "Calibration and evaluation corpora must use disjoint seeds. "
            "synthetic_shape_matched is not a claim of simulator capture."
        ),
    )


def load_corpus(path: Path) -> list[CorpusSample]:
    data = np.load(Path(path), allow_pickle=False)
    seeds = [int(s) for s in data["seeds"].tolist()]
    images = data["images_top"]
    states = data["state"]
    source = str(data["source"]) if "source" in data.files else "unknown"
    if isinstance(source, np.ndarray):
        source = str(source.item())
    samples: list[CorpusSample] = []
    for i, seed in enumerate(seeds):
        samples.append(
            CorpusSample(
                seed=seed,
                images_top=np.asarray(images[i], dtype=np.float32),
                state=np.asarray(states[i], dtype=np.float32),
                source=source,
            )
        )
    return samples


def build_synthetic_corpus(
    seeds: list[int],
    *,
    image_shape: tuple[int, int, int, int] = (1, 3, 480, 640),
    state_dim: int = 14,
) -> list[CorpusSample]:
    samples: list[CorpusSample] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        images = rng.random(image_shape, dtype=np.float32)
        state = rng.standard_normal((1, state_dim), dtype=np.float32).astype(np.float32)
        samples.append(
            CorpusSample(
                seed=seed,
                images_top=images,
                state=state,
                source="synthetic_shape_matched",
            )
        )
    return samples


def try_build_simulator_corpus(
    *,
    seeds: list[int],
    candidate: str,
    repo_root: Path,
    device: str = "cpu",
) -> tuple[list[CorpusSample], dict[str, Any]]:
    """Capture reset observations from the pinned env. Records errors without faking."""
    from breakpoint_eval.sim_env import make_vector_env, preprocess_observation
    from breakpoint_eval.stack import ensure_lerobot_on_path, resolve_candidate

    meta: dict[str, Any] = {"attempted": True, "env_created": False}
    ensure_lerobot_on_path(repo_root)
    spec = resolve_candidate(candidate)
    try:
        env = make_vector_env(spec, n_envs=1)
    except (OSError, ImportError, ModuleNotFoundError, RuntimeError) as exc:
        meta["env_error"] = f"{type(exc).__name__}: {exc}"
        LOGGER.warning("Simulator corpus unavailable: %s", meta["env_error"])
        return [], meta

    meta["env_created"] = True
    samples: list[CorpusSample] = []
    try:
        for seed in seeds:
            observation, _info = env.reset(seed=[seed])
            processed = preprocess_observation(observation)
            images = processed["observation.images.top"].detach().cpu().numpy().astype(np.float32)
            state = processed["observation.state"].detach().cpu().numpy().astype(np.float32)
            samples.append(
                CorpusSample(
                    seed=seed,
                    images_top=images,
                    state=state,
                    source="simulator",
                )
            )
    finally:
        env.close()
    meta["n_captured"] = len(samples)
    meta["device_requested"] = device
    return samples, meta


def write_corpus_sidecar(manifest: CorpusManifest, path: Path) -> None:
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
