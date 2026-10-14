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

DEFAULT_SAMPLES_PER_EPISODE = 2
SYNTHETIC_SOURCE = "synthetic_shape_matched"
SIMULATOR_SOURCE = "simulator"
SYNTHETIC_UNSUITABLE_NOTE = (
    "synthetic_shape_matched is shape-matched RNG only; unsuitable as deployment "
    "or hardware evidence. Use simulator rollout capture for calibration/evaluation."
)


@dataclass(frozen=True)
class CorpusSample:
    seed: int
    sample_id: str
    control_step_index: int
    images_top: np.ndarray  # (1,3,H,W) float32 [0,1]
    state: np.ndarray  # (1, state_dim) float32
    source: str  # simulator | synthetic_shape_matched


@dataclass
class CorpusManifest:
    role: str  # calibration | evaluation
    seeds: list[int]
    sample_ids: list[str]
    n_samples: int
    image_shape: list[int]
    state_shape: list[int]
    source: str
    path: str
    sha256: str
    note: str
    suitable_as_deployment_evidence: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_sample_id(*, seed: int, control_step_index: int) -> str:
    """Stable unique ID for one observation within an episode seed."""
    return f"seed{int(seed)}_t{int(control_step_index)}"


def sample_step_indices(n_steps: int, n_samples: int) -> list[int]:
    """Pick control-step indices spanning a rollout (includes step 0)."""
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if n_samples == 1:
        return [0]
    last = n_steps - 1
    return sorted({int(round(i * last / (n_samples - 1))) for i in range(n_samples)})


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_corpus(path: Path, samples: list[CorpusSample], *, role: str) -> CorpusManifest:
    if not samples:
        raise ValueError("corpus is empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_ids = [s.sample_id for s in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"corpus sample_id values must be unique; got {sample_ids}")
    payload = {
        "role": role,
        "seeds": np.asarray([s.seed for s in samples], dtype=np.int64),
        "sample_ids": np.asarray(sample_ids),
        "control_step_index": np.asarray([s.control_step_index for s in samples], dtype=np.int64),
        "source": samples[0].source,
        "images_top": np.stack([s.images_top for s in samples], axis=0),
        "state": np.stack([s.state for s in samples], axis=0),
    }
    np.savez_compressed(path, **payload)
    raw = path.read_bytes()
    sources = {s.source for s in samples}
    source = sources.pop() if len(sources) == 1 else "mixed"
    suitable = source == SIMULATOR_SOURCE
    note = (
        "Calibration and evaluation corpora must use disjoint episode seeds. "
        "Each sample_id is unique across multi-sample episodes."
    )
    if not suitable:
        note = f"{note} {SYNTHETIC_UNSUITABLE_NOTE}"
    return CorpusManifest(
        role=role,
        seeds=[s.seed for s in samples],
        sample_ids=sample_ids,
        n_samples=len(samples),
        image_shape=list(samples[0].images_top.shape),
        state_shape=list(samples[0].state.shape),
        source=source,
        path=str(path),
        sha256=_sha256_bytes(raw),
        note=note,
        suitable_as_deployment_evidence=suitable,
    )


def load_corpus(path: Path) -> list[CorpusSample]:
    data = np.load(Path(path), allow_pickle=False)
    seeds = [int(s) for s in data["seeds"].tolist()]
    images = data["images_top"]
    states = data["state"]
    source = str(data["source"]) if "source" in data.files else "unknown"
    if isinstance(source, np.ndarray):
        source = str(source.item())
    if "sample_ids" in data.files:
        sample_ids = [str(x) for x in data["sample_ids"].tolist()]
    else:
        sample_ids = [make_sample_id(seed=seed, control_step_index=0) for seed in seeds]
    if "control_step_index" in data.files:
        steps = [int(x) for x in data["control_step_index"].tolist()]
    else:
        steps = [0] * len(seeds)
    samples: list[CorpusSample] = []
    for i, seed in enumerate(seeds):
        samples.append(
            CorpusSample(
                seed=seed,
                sample_id=sample_ids[i],
                control_step_index=steps[i],
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
    """Shape-matched RNG observations. Explicitly unsuitable as deployment evidence."""
    samples: list[CorpusSample] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        images = rng.random(image_shape, dtype=np.float32)
        state = rng.standard_normal((1, state_dim), dtype=np.float32).astype(np.float32)
        samples.append(
            CorpusSample(
                seed=seed,
                sample_id=make_sample_id(seed=seed, control_step_index=0),
                control_step_index=0,
                images_top=images,
                state=state,
                source=SYNTHETIC_SOURCE,
            )
        )
    return samples


def _processed_arrays(processed: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    images = processed["observation.images.top"].detach().cpu().numpy().astype(np.float32)
    state = processed["observation.state"].detach().cpu().numpy().astype(np.float32)
    return images, state


def capture_simulator_rollout_corpus(
    *,
    seeds: list[int],
    candidate: str,
    repo_root: Path,
    policy: Any,
    control_period_s: float,
    device: str = "cpu",
    samples_per_episode: int = DEFAULT_SAMPLES_PER_EPISODE,
    capture_horizon_steps: int | None = None,
) -> tuple[list[CorpusSample], dict[str, Any]]:
    """Capture observations across policy rollouts (not reset-only).

    For each episode seed, runs the policy for ``capture_horizon_steps`` control
    steps and records ``samples_per_episode`` observations at distinct step
    indices, each with a unique ``sample_id``.
    """
    from breakpoint_eval.policy_adapter import PolicyAdapter
    from breakpoint_eval.sim_env import make_vector_env, preprocess_observation
    from breakpoint_eval.stack import ensure_lerobot_on_path, resolve_candidate

    if samples_per_episode < 1:
        raise ValueError("samples_per_episode must be >= 1")

    meta: dict[str, Any] = {
        "attempted": True,
        "env_created": False,
        "samples_per_episode": samples_per_episode,
        "capture_mode": "policy_rollout",
    }
    ensure_lerobot_on_path(repo_root)
    spec = resolve_candidate(candidate)
    try:
        env = make_vector_env(spec, n_envs=1)
    except (OSError, ImportError, ModuleNotFoundError, RuntimeError) as exc:
        meta["env_error"] = f"{type(exc).__name__}: {exc}"
        LOGGER.warning("Simulator corpus unavailable: %s", meta["env_error"])
        return [], meta

    meta["env_created"] = True
    n_action_steps = int(getattr(policy.config, "n_action_steps", 1))
    horizon = (
        int(capture_horizon_steps)
        if capture_horizon_steps is not None
        else max(n_action_steps, samples_per_episode)
    )
    meta["capture_horizon_steps"] = horizon
    adapter = PolicyAdapter(policy, control_period_s=control_period_s, device=device)
    samples: list[CorpusSample] = []
    per_seed: list[dict[str, Any]] = []
    try:
        for seed in seeds:
            adapter.reset()
            observation, _info = env.reset(seed=[seed])
            recorded: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            want = set(sample_step_indices(horizon, samples_per_episode))
            episode_steps = 0
            for step_index in range(horizon):
                if step_index in want and step_index not in recorded:
                    processed = preprocess_observation(observation)
                    recorded[step_index] = _processed_arrays(processed)
                obs_t = preprocess_observation(observation)
                action_step = adapter.select_action(obs_t)
                action = np.asarray(action_step.action, dtype=np.float32).reshape(1, -1)
                observation, _reward, terminated_arr, truncated_arr, _info = env.step(action)
                episode_steps += 1
                if bool(terminated_arr[0]) or bool(truncated_arr[0]):
                    break
            # If the episode ended early, still fill remaining wanted steps from
            # the last observation so sample_id count stays stable per seed.
            if len(recorded) < samples_per_episode:
                processed = preprocess_observation(observation)
                arrays = _processed_arrays(processed)
                for step_index in sorted(want):
                    if step_index not in recorded:
                        recorded[step_index] = arrays
            seed_ids: list[str] = []
            for step_index in sorted(recorded):
                images, state = recorded[step_index]
                sample_id = make_sample_id(seed=seed, control_step_index=step_index)
                seed_ids.append(sample_id)
                samples.append(
                    CorpusSample(
                        seed=seed,
                        sample_id=sample_id,
                        control_step_index=step_index,
                        images_top=images,
                        state=state,
                        source=SIMULATOR_SOURCE,
                    )
                )
            per_seed.append(
                {
                    "seed": seed,
                    "n_control_steps_run": episode_steps,
                    "sample_ids": seed_ids,
                    "step_indices": sorted(recorded),
                }
            )
    except Exception as exc:  # noqa: BLE001 — record capture failure without fabricating
        meta["capture_error"] = f"{type(exc).__name__}: {exc}"
        LOGGER.warning("Simulator rollout corpus capture failed: %s", meta["capture_error"])
        return [], meta
    finally:
        env.close()

    meta["n_captured"] = len(samples)
    meta["device_requested"] = device
    meta["episodes"] = per_seed
    return samples, meta


# Backward-compatible name used by older call sites / docs.
try_build_simulator_corpus = capture_simulator_rollout_corpus


def write_corpus_sidecar(manifest: CorpusManifest, path: Path) -> None:
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def chunk_dict_by_sample_id(
    samples: list[CorpusSample],
    chunks: list[np.ndarray],
) -> dict[str, np.ndarray]:
    if len(samples) != len(chunks):
        raise ValueError("samples and chunks lengths must match")
    return {
        s.sample_id: np.asarray(c, dtype=np.float32) for s, c in zip(samples, chunks, strict=True)
    }
