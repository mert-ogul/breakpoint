"""Unit tests for multi-candidate support, diffusion adapter, and validation gates."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from breakpoint_eval.candidate_map import CANDIDATES
from breakpoint_eval.diffusion_adapter import (
    TIMED_BOUNDARY_DIFFUSION_GENERATE,
    compute_diffusion_action_chunk,
    is_diffusion_policy,
    observe_diffusion,
)
from breakpoint_eval.diffusion_export import checkpoint_file_hashes
from breakpoint_eval.diffusion_profile import KIND_ACTION_GENERATION_TIMING
from breakpoint_eval.model_resolve import resolve_model_to_candidate
from breakpoint_eval.policy_adapter import PolicyAdapter
from breakpoint_eval.policy_validate import (
    assert_gates_passed,
    build_upstream_agreement_payload,
    compare_determinism_rerun,
    compare_seed_agreement,
    evaluate_seed_agreement,
    run_policy_validation_gates,
    validate_success_semantics,
    validate_zero_latency_equivalence,
)
from breakpoint_eval.run_episode import ControlStepRecord, EpisodeOutcome, EpisodeResult
from breakpoint_eval.support_status import SUPPORT_MATRIX, support_row


def _populate_queues(queues, batch):
    for key in batch:
        if key not in queues:
            continue
        if len(queues[key]) != queues[key].maxlen:
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            queues[key].append(batch[key])
    return queues


class _UpstreamLikePolicy:
    def __init__(self, n_action_steps: int = 3) -> None:
        self.config = SimpleNamespace(
            type="act",
            chunk_size=n_action_steps,
            n_action_steps=n_action_steps,
            n_obs_steps=1,
            temporal_ensemble_coeff=None,
        )
        self._action_queue: deque[torch.Tensor] = deque(maxlen=n_action_steps)
        self._chunk_id = 0

    def eval(self) -> _UpstreamLikePolicy:
        return self

    def reset(self) -> None:
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self._chunk_id = 0

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if len(self._action_queue) == 0:
            self._chunk_id += 1
            chunk = torch.arange(self.config.n_action_steps, dtype=torch.float32).view(-1, 1, 1)
            chunk = chunk + 10.0 * self._chunk_id
            chunk = chunk.expand(-1, 1, 2).contiguous()
            self._action_queue.extend(list(chunk))
        return self._action_queue.popleft()


class _TinyDiffusion:
    """Minimal Diffusion-like policy for offline tests (not a real UNet)."""

    name = "diffusion"

    def __init__(
        self, *, n_obs_steps: int = 2, n_action_steps: int = 3, action_dim: int = 2
    ) -> None:
        self.config = SimpleNamespace(
            type="diffusion",
            n_obs_steps=n_obs_steps,
            horizon=n_obs_steps + n_action_steps,
            n_action_steps=n_action_steps,
            noise_scheduler_type="DDPM",
            num_train_timesteps=10,
            prediction_type="epsilon",
            image_features=["observation.image"],
            temporal_ensemble_coeff=None,
        )
        self._chunk_id = 0
        self.diffusion = SimpleNamespace(num_inference_steps=4, generate_actions=self._generate)
        self._queues = None
        self.reset()

    def eval(self) -> _TinyDiffusion:
        return self

    def reset(self) -> None:
        self._chunk_id = 0
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "observation.images": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }

    def normalize_inputs(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return dict(batch)

    def unnormalize_outputs(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return batch

    def _generate(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self._chunk_id += 1
        b = batch["observation.state"].shape[0]
        t = self.config.n_action_steps
        a = 2
        base = float(self._chunk_id * 10)
        # Depend on obs history so observe() vs missing history would diverge.
        hist = batch["observation.state"].mean().item()
        actions = torch.arange(t, dtype=torch.float32).view(1, t, 1).expand(b, t, a).clone()
        actions = actions + base + hist
        return actions

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch = self.normalize_inputs(batch)
        batch = dict(batch)
        batch["observation.images"] = torch.stack(
            [batch[key] for key in self.config.image_features], dim=-4
        )
        self._queues = _populate_queues(self._queues, batch)
        if len(self._queues["action"]) == 0:
            stacked = {
                k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues
            }
            actions = self.diffusion.generate_actions(stacked)
            actions = self.unnormalize_outputs({"action": actions})["action"]
            self._queues["action"].extend(actions.transpose(0, 1))
        return self._queues["action"].popleft()


def test_candidates_and_hub_resolve() -> None:
    assert set(CANDIDATES) == {"act_aloha_transfer", "act_aloha_insertion", "diffusion_pusht"}
    key, spec = resolve_model_to_candidate("lerobot/act_aloha_sim_insertion_human")
    assert key == "act_aloha_insertion"
    assert spec.policy_type == "act"
    key, spec = resolve_model_to_candidate("lerobot/diffusion_pusht")
    assert key == "diffusion_pusht"
    assert spec.policy_type == "diffusion"


def test_support_matrix_no_cross_task_pooling_flags() -> None:
    for name, row in SUPPORT_MATRIX.items():
        assert row.candidate == name
        assert "pool" not in row.notes.lower() or "do not pool" in row.notes.lower()
    assert support_row("diffusion_pusht").onnx_fp32 == "unsupported"
    assert support_row("act_aloha_insertion").policy_type == "act"


def test_success_semantics_documented_for_all_candidates() -> None:
    for spec in CANDIDATES.values():
        gate = validate_success_semantics(spec)
        assert gate.passed, gate.detail


def test_tiny_diffusion_zero_latency_and_observe() -> None:
    policy = _TinyDiffusion()
    assert is_diffusion_policy(policy)
    adapter = PolicyAdapter(policy, control_period_s=0.1, device="cpu")
    assert adapter.chunk_info.policy_type == "diffusion"
    assert adapter.chunk_info.n_obs_steps == 2

    obs_stream = []
    for i in range(7):
        obs_stream.append(
            {
                "observation.state": torch.full((1, 2), float(i), dtype=torch.float32),
                "observation.image": torch.zeros(1, 3, 8, 8),
            }
        )
    gate = validate_zero_latency_equivalence(adapter, obs_stream, atol=0, rtol=0)
    assert gate.passed, gate.detail

    # observe advances history without filling the action queue
    adapter.reset()
    observe_diffusion(
        policy,
        {
            "observation.state": torch.ones(1, 2),
            "observation.image": torch.zeros(1, 3, 8, 8),
        },
    )
    assert len(policy._queues["action"]) == 0
    assert len(policy._queues["observation.state"]) == 2

    chunk = compute_diffusion_action_chunk(
        policy,
        {
            "observation.state": torch.full((1, 2), 3.0),
            "observation.image": torch.zeros(1, 3, 8, 8),
        },
    )
    assert chunk.shape == (3, 2)
    assert len(policy._queues["action"]) == 0


def test_act_like_adapter_still_zero_latency() -> None:
    policy = _UpstreamLikePolicy(n_action_steps=3)
    adapter = PolicyAdapter(policy, control_period_s=0.02, device="cpu")
    obs = [{"observation.state": torch.zeros(1, 2)} for _ in range(7)]
    gate = validate_zero_latency_equivalence(adapter, obs, atol=0, rtol=0)
    assert gate.passed, gate.detail


def test_run_policy_validation_gates_tiny_diffusion() -> None:
    policy = _TinyDiffusion()
    adapter = PolicyAdapter(policy, control_period_s=0.1, device="cpu")
    pol_obs = {
        "observation.state": torch.zeros(1, 2),
        "observation.image": torch.zeros(1, 3, 8, 8),
    }
    raw = {
        "pixels": np.zeros((96, 96, 3), dtype=np.uint8),
        "agent_pos": np.zeros((2,), dtype=np.float32),
    }
    results = run_policy_validation_gates(
        spec=CANDIDATES["diffusion_pusht"],
        adapter=adapter,
        policy_observation=pol_obs,
        raw_observation=raw,
        zero_latency_observations=[
            {
                "observation.state": torch.full((1, 2), float(i)),
                "observation.image": torch.zeros(1, 3, 8, 8),
            }
            for i in range(5)
        ],
    )
    assert_gates_passed(results)


def test_profile_rejects_diffusion_ort() -> None:
    from breakpoint_eval.run_profile import ProfileCliRequest, run_profile

    with pytest.raises(ValueError, match="Diffusion Policy"):
        run_profile(
            ProfileCliRequest(
                model_path=Path("/tmp/missing.onnx"),
                corpus_path=Path("/tmp/missing.npz"),
                output_dir=Path("/tmp/out"),
                device_label="rpi5",
                candidate="diffusion_pusht",
                recipe_id="torch_onnx_export_fp32_opset17",
            )
        )


def test_checkpoint_file_hashes_tmp_dir(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"type": "diffusion"}\n', encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"fake-weights")
    (tmp_path / "README.md").write_text("notes\n", encoding="utf-8")
    hashes = checkpoint_file_hashes(tmp_path)
    assert set(hashes) >= {"config.json", "model.safetensors", "README.md"}
    assert len(hashes["model.safetensors"]) == 64
    assert hashes["config.json"] != hashes["model.safetensors"]


def test_checkpoint_file_hashes_requires_weight_files(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        checkpoint_file_hashes(tmp_path)


class _ShortDiffusionEnv:
    """VectorEnv-like stub for offline upstream-agreement checks."""

    def __init__(self, *, max_steps: int = 6) -> None:
        self.max_steps = max_steps
        self._step = 0

    def reset(self, seed=None):  # noqa: ANN001
        self._step = 0
        return {
            "pixels": np.zeros((1, 16, 16, 3), dtype=np.uint8),
            "agent_pos": np.zeros((1, 2), dtype=np.float32),
        }, {}

    def step(self, action):  # noqa: ANN001
        del action
        self._step += 1
        truncated = self._step >= self.max_steps
        obs = {
            "pixels": np.zeros((1, 16, 16, 3), dtype=np.uint8),
            "agent_pos": np.full((1, 2), float(self._step), dtype=np.float32),
        }
        info: dict = {"is_success": np.array([False])}
        if truncated:
            info["final_info"] = [{"is_success": False}]
        return (
            obs,
            np.array([0.1], dtype=np.float32),
            np.array([False]),
            np.array([truncated]),
            info,
        )

    def call(self, name: str):  # noqa: ANN001
        if name == "render":
            return [None]
        raise AttributeError(name)

    def close(self) -> None:
        return None


class _ResetRaisesEnv:
    """VectorEnv-like stub whose reset fails before any control step."""

    def reset(self, seed=None):  # noqa: ANN001
        del seed
        raise RuntimeError("reset boom")

    def step(self, action):  # noqa: ANN001
        del action
        raise AssertionError("step must not run after reset failure")

    def call(self, name: str):  # noqa: ANN001
        raise AttributeError(name)

    def close(self) -> None:
        return None


def _episode_with_actions(actions: list[list[float]], *, seed: int = 0) -> EpisodeResult:
    steps = [
        ControlStepRecord(
            control_step_index=i,
            sim_time_s=float(i + 1) * 0.1,
            requested_fresh_chunk=(i % 3 == 0),
            inference_call=(i % 3 == 0),
            reward=0.1,
            terminated=False,
            truncated=i == len(actions) - 1,
            is_success=False,
            action=list(a),
        )
        for i, a in enumerate(actions)
    ]
    return EpisodeResult(
        episode_index=0,
        seed=seed,
        outcome=EpisodeOutcome.TIME_LIMIT_TRUNCATION,
        success=False,
        terminated=False,
        truncated=True,
        policy_failed=False,
        runtime_error=None,
        n_control_steps=len(steps),
        n_inference_calls=sum(1 for s in steps if s.inference_call),
        n_fresh_chunks=sum(1 for s in steps if s.requested_fresh_chunk),
        sum_reward=0.1 * len(steps),
        max_reward=0.1,
        control_period_s=0.1,
        chunk_size=3,
        n_action_steps=3,
        steps=steps,
    )


def _empty_runtime_error_episode(
    *, seed: int = 0, detail: str = "RuntimeError: boom"
) -> EpisodeResult:
    return EpisodeResult(
        episode_index=0,
        seed=seed,
        outcome=EpisodeOutcome.RUNTIME_ERROR,
        success=False,
        terminated=False,
        truncated=False,
        policy_failed=False,
        runtime_error=detail,
        n_control_steps=0,
        n_inference_calls=0,
        n_fresh_chunks=0,
        sum_reward=0.0,
        max_reward=0.0,
        control_period_s=0.1,
        chunk_size=3,
        n_action_steps=3,
        steps=[],
    )


def test_upstream_agreement_json_shape_and_pass_with_stub(monkeypatch) -> None:  # noqa: ANN001
    def fake_preprocess(observation):  # noqa: ANN001
        return {
            "observation.state": torch.from_numpy(observation["agent_pos"]).float(),
            "observation.image": torch.zeros(1, 3, 8, 8),
        }

    monkeypatch.setattr("breakpoint_eval.run_episode.preprocess_observation", fake_preprocess)
    env = _ShortDiffusionEnv(max_steps=6)
    adapter = PolicyAdapter(_TinyDiffusion(), control_period_s=0.1, device="cpu")
    seed_row, determinism = evaluate_seed_agreement(
        env=env,
        adapter=adapter,
        seed=2200,
        policy_rng_seed=12_200,
        check_determinism=True,
    )
    assert seed_row["passed"] is True, seed_row
    assert seed_row["first_mismatch_step"] is None
    assert seed_row["max_abs_action_difference"] == 0.0
    assert seed_row["runtime_error"] is None
    assert seed_row["upstream_runtime_error"] is None
    assert determinism is not None
    assert determinism["passed"] is True, determinism

    payload = build_upstream_agreement_payload(
        candidate="diffusion_pusht",
        hub_revision="deadbeef",
        lerobot_commit="abc123",
        checkpoint_files={"config.json": "aa", "model.safetensors": "bb"},
        torch_version="2.0.0",
        torch_threads=1,
        device="cpu",
        diffusion_settings={"policy_type": "diffusion", "n_obs_steps": 2},
        per_seed=[seed_row],
        determinism=determinism,
    )
    assert payload["passed"] is True
    for key in (
        "passed",
        "per_seed",
        "determinism",
        "candidate",
        "hub_revision",
        "lerobot_commit",
        "checkpoint_files",
        "torch_version",
        "torch_threads",
        "device",
        "diffusion",
    ):
        assert key in payload
    assert payload["per_seed"][0]["steps"] == 6


def test_upstream_agreement_fail_when_reset_raises(monkeypatch) -> None:  # noqa: ANN001
    def fake_preprocess(observation):  # noqa: ANN001
        return {
            "observation.state": torch.from_numpy(observation["agent_pos"]).float(),
            "observation.image": torch.zeros(1, 3, 8, 8),
        }

    monkeypatch.setattr("breakpoint_eval.run_episode.preprocess_observation", fake_preprocess)
    env = _ResetRaisesEnv()
    adapter = PolicyAdapter(_TinyDiffusion(), control_period_s=0.1, device="cpu")
    seed_row, determinism = evaluate_seed_agreement(
        env=env,
        adapter=adapter,
        seed=2200,
        policy_rng_seed=12_200,
        check_determinism=True,
    )
    assert seed_row["passed"] is False, seed_row
    assert seed_row["steps"] == 0
    assert seed_row["upstream_steps"] == 0
    assert seed_row["outcome"] == EpisodeOutcome.RUNTIME_ERROR.value
    assert seed_row["upstream_outcome"] == EpisodeOutcome.RUNTIME_ERROR.value
    assert seed_row["runtime_error"] is not None
    assert seed_row["upstream_runtime_error"] is not None
    assert "reset boom" in seed_row["runtime_error"]
    assert determinism is not None
    assert determinism["passed"] is False, determinism

    payload = build_upstream_agreement_payload(
        candidate="diffusion_pusht",
        hub_revision="deadbeef",
        lerobot_commit="abc123",
        checkpoint_files={"config.json": "aa", "model.safetensors": "bb"},
        torch_version="2.0.0",
        torch_threads=1,
        device="cpu",
        diffusion_settings={"policy_type": "diffusion"},
        per_seed=[seed_row],
        determinism=determinism,
    )
    assert payload["passed"] is False


def test_upstream_agreement_fail_on_identical_empty_runtime_errors() -> None:
    """Matching empty traces from the same reset failure must not pass."""
    left = _empty_runtime_error_episode(detail="RuntimeError: reset boom")
    right = _empty_runtime_error_episode(detail="RuntimeError: reset boom")
    row = compare_seed_agreement(seed=7, benchmark=left, upstream=right)
    assert row["passed"] is False
    assert row["outcome"] == EpisodeOutcome.RUNTIME_ERROR.value
    assert row["upstream_outcome"] == EpisodeOutcome.RUNTIME_ERROR.value
    assert row["runtime_error"] == "RuntimeError: reset boom"
    assert row["upstream_runtime_error"] == "RuntimeError: reset boom"
    assert row["max_abs_action_difference"] == 0.0

    det = compare_determinism_rerun(seed=7, first=left, second=right)
    assert det["passed"] is False
    payload = build_upstream_agreement_payload(
        candidate="diffusion_pusht",
        hub_revision="rev",
        lerobot_commit="commit",
        checkpoint_files={"config.json": "a", "model.safetensors": "b"},
        torch_version="2.0.0",
        torch_threads=1,
        device="cpu",
        diffusion_settings={"policy_type": "diffusion"},
        per_seed=[row],
        determinism=det,
    )
    assert payload["passed"] is False


def test_upstream_agreement_fail_on_action_mismatch() -> None:
    left = _episode_with_actions([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    right = _episode_with_actions([[1.0, 0.0], [2.5, 0.0], [3.0, 0.0]])
    row = compare_seed_agreement(seed=1, benchmark=left, upstream=right)
    assert row["passed"] is False
    assert row["first_mismatch_step"] == 1
    assert row["max_abs_action_difference"] == pytest.approx(0.5)

    det = compare_determinism_rerun(seed=1, first=left, second=right)
    assert det["passed"] is False
    payload = build_upstream_agreement_payload(
        candidate="diffusion_pusht",
        hub_revision="rev",
        lerobot_commit="commit",
        checkpoint_files={"config.json": "a", "model.safetensors": "b"},
        torch_version="2.0.0",
        torch_threads=2,
        device="cpu",
        diffusion_settings={"policy_type": "diffusion"},
        per_seed=[row],
        determinism=det,
    )
    assert payload["passed"] is False


def test_diffusion_profile_refuses_nonfinite_and_ort_replay(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    from breakpoint_eval import diffusion_profile as dp
    from breakpoint_eval.latency_trace import load_latency_profile

    obs = {
        "observation.state": torch.zeros(1, 2),
        "observation.image": torch.zeros(1, 3, 8, 8),
    }
    policy = _TinyDiffusion()

    monkeypatch.setattr(dp, "resolve_candidate", lambda _n: CANDIDATES["diffusion_pusht"])
    monkeypatch.setattr(dp, "ensure_lerobot_on_path", lambda _r: _r)
    monkeypatch.setattr(dp, "download_candidate_checkpoint", lambda _s, _c: tmp_path)
    monkeypatch.setattr(dp, "load_policy", lambda *_a, **_k: policy)
    monkeypatch.setattr(dp, "checkpoint_file_hashes", lambda _p: {"config.json": "a"})
    monkeypatch.setattr(dp, "_collect_native_observations", lambda **_k: [obs, obs])
    monkeypatch.setattr(
        dp,
        "compute_diffusion_action_chunk",
        lambda *_a, **_k: np.array([[np.nan, 0.0], [0.0, 1.0], [0.0, 2.0]], dtype=np.float32),
    )
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"x")

    with pytest.raises(RuntimeError, match="non-finite"):
        dp.run_diffusion_pytorch_profile(
            dp.DiffusionProfileRequest(
                output_dir=tmp_path / "out",
                n_warmup=0,
                n_measured=1,
                observation_seeds=(2300,),
                repo_root=tmp_path,
            )
        )

    # Valid-looking timing payload still refused by ORT replay loader.
    measured = [0.01, 0.02]
    from breakpoint_eval.latency_trace import summarize_latency_s

    stats = summarize_latency_s(measured)
    profile_path = tmp_path / "diffusion_timing.json"
    profile_path.write_text(
        json.dumps(
            {
                "kind": KIND_ACTION_GENERATION_TIMING,
                "timed_boundary": TIMED_BOUNDARY_DIFFUSION_GENERATE,
                "schema_version": 1,
                "output_validation": {"passed": True},
                "measurements": {
                    "n_measured": 2,
                    "inference_latency_s": measured,
                    "summary": stats.to_dict(),
                },
                "candidate": "diffusion_pusht",
                "model": {"sha256": "abc", "precision_recipe": "pytorch_fp32"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="timed_boundary"):
        load_latency_profile(profile_path)
