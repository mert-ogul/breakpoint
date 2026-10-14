"""Device profiling without simulator dependencies."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from breakpoint_eval.device_profile import (  # noqa: E402
    DEFAULT_MEASURED,
    DEFAULT_WARMUPS,
    ProfileRequest,
    load_profile_corpus,
    run_device_profile,
)
from breakpoint_eval.latency_trace import TIMED_BOUNDARY_ORT_RUN, load_latency_profile  # noqa: E402


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.img = nn.Linear(3 * 8 * 8, 8)
        self.state_proj = nn.Linear(2, 8)
        self.out = nn.Linear(8, 4 * 2)

    def forward(self, images_top: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        b = images_top.shape[0]
        h = torch.relu(self.img(images_top.reshape(b, -1)) + self.state_proj(state.reshape(b, -1)))
        return self.out(h).reshape(b, 4, 2)


def _write_tiny_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    mod = _Tiny().eval()
    imgs = torch.rand(1, 3, 8, 8)
    state = torch.randn(1, 2)
    model_path = tmp_path / "tiny.onnx"
    torch.onnx.export(
        mod,
        (imgs, state),
        str(model_path),
        input_names=["observation_images_top", "observation_state"],
        output_names=["action_chunk"],
        opset_version=17,
        dynamo=False,
    )
    # Corpus layout matches export_corpus.save_corpus stacking.
    images = np.stack(
        [np.random.default_rng(i).random((1, 3, 8, 8), dtype=np.float32) for i in range(4)]
    )
    states = np.stack(
        [np.random.default_rng(100 + i).standard_normal((1, 2), dtype=np.float32) for i in range(4)]
    )
    sample_ids = [f"seed{s}_t0" for s in (1, 2, 3, 4)]
    corpus_path = tmp_path / "corpus.npz"
    np.savez_compressed(
        corpus_path,
        role="evaluation",
        seeds=np.asarray([1, 2, 3, 4]),
        sample_ids=np.asarray(sample_ids),
        control_step_index=np.asarray([0, 0, 0, 0]),
        source="synthetic_shape_matched",
        images_top=images,
        state=states,
    )
    # Reference chunks from ORT itself after a first run (identity validation).
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    refs = {}
    for i, sample_id in enumerate(sample_ids):
        out = sess.run(
            None,
            {
                "observation_images_top": images[i],
                "observation_state": states[i],
            },
        )[0]
        refs[sample_id] = out
    ref_path = tmp_path / "refs.npz"
    np.savez_compressed(ref_path, **refs)
    return model_path, corpus_path, ref_path


def test_defaults_separate_warmup_and_measured() -> None:
    assert DEFAULT_WARMUPS == 50
    assert DEFAULT_MEASURED == 1000


def test_profile_corpus_loader_no_sim(tmp_path: Path) -> None:
    _, corpus_path, _ = _write_tiny_artifacts(tmp_path)
    corpus = load_profile_corpus(corpus_path)
    assert corpus["n_samples"] == 4
    assert corpus["shapes"]["observation_images_top"] == [1, 3, 8, 8]


def test_run_device_profile_writes_ordered_measurements(tmp_path: Path) -> None:
    model_path, corpus_path, ref_path = _write_tiny_artifacts(tmp_path)
    out_dir = tmp_path / "profiles" / "local"
    path = run_device_profile(
        ProfileRequest(
            model_path=model_path,
            corpus_path=corpus_path,
            output_dir=out_dir,
            device_label="local",
            candidate="act_aloha_transfer",
            recipe_id="torch_onnx_export_fp32_opset17",
            variant_label="onnx_fp32",
            reference_chunks_path=ref_path,
            n_warmup=2,
            n_measured=5,
            chunk_horizon=4,
        )
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["timed_boundary"] == TIMED_BOUNDARY_ORT_RUN
    assert payload["measurements"]["n_warmup"] == 2
    assert payload["measurements"]["n_measured"] == 5
    assert len(payload["measurements"]["inference_latency_s"]) == 5
    assert payload["stages"]["preprocessing_timed_separately"] is False
    assert payload["semantics"]["times_random_matmul"] is False
    assert payload["semantics"]["times_action_queue_dequeue"] is False
    assert payload["output_validation"]["passed"] is True
    summary = payload["measurements"]["summary"]
    assert summary["count"] == 5
    assert "median_s" in summary and "p95_s" in summary and "p99_s" in summary
    # Reload through strict profile loader.
    prof = load_latency_profile(path)
    assert len(prof.latency_s) == 5
    assert all(x >= 0 for x in prof.latency_s)


def test_profile_refuses_failed_validation(tmp_path: Path) -> None:
    model_path, corpus_path, _ = _write_tiny_artifacts(tmp_path)
    # Wrong-shaped reference forces validation failure.
    bad_ref = tmp_path / "bad_ref.npz"
    np.savez_compressed(bad_ref, seed1_t0=np.zeros((1, 9, 9), dtype=np.float32))
    with pytest.raises(RuntimeError, match="validation failed"):
        run_device_profile(
            ProfileRequest(
                model_path=model_path,
                corpus_path=corpus_path,
                output_dir=tmp_path / "out",
                device_label="local",
                candidate="act_aloha_transfer",
                recipe_id="torch_onnx_export_fp32_opset17",
                reference_chunks_path=bad_ref,
                n_warmup=0,
                n_measured=2,
            )
        )


def test_profile_fails_when_reference_sample_missing(tmp_path: Path) -> None:
    model_path, corpus_path, ref_path = _write_tiny_artifacts(tmp_path)
    # Drop one sample_id from the reference set.
    data = np.load(ref_path)
    keys = [k for k in data.files if k != "seed2_t0"]
    partial = tmp_path / "partial_refs.npz"
    np.savez_compressed(partial, **{k: data[k] for k in keys})
    with pytest.raises(RuntimeError, match="validation failed"):
        run_device_profile(
            ProfileRequest(
                model_path=model_path,
                corpus_path=corpus_path,
                output_dir=tmp_path / "out_missing",
                device_label="local",
                candidate="act_aloha_transfer",
                recipe_id="torch_onnx_export_fp32_opset17",
                reference_chunks_path=partial,
                n_warmup=0,
                n_measured=2,
            )
        )
