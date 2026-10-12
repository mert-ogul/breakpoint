"""CLI orchestration for ``breakpoint profile`` (no simulator imports)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from breakpoint_eval.device_profile import (
    DEFAULT_MEASURED,
    DEFAULT_WARMUPS,
    ProfileRequest,
    run_device_profile,
)


@dataclass(frozen=True)
class ProfileCliRequest:
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
    hub_revision: str | None = None
    chunk_horizon: int | None = None


def run_profile(req: ProfileCliRequest) -> Path:
    return run_device_profile(
        ProfileRequest(
            model_path=req.model_path,
            corpus_path=req.corpus_path,
            output_dir=req.output_dir,
            device_label=req.device_label,
            candidate=req.candidate,
            recipe_id=req.recipe_id,
            variant_label=req.variant_label,
            reference_chunks_path=req.reference_chunks_path,
            n_warmup=req.n_warmup,
            n_measured=req.n_measured,
            providers=req.providers,
            intra_op_num_threads=req.intra_op_num_threads,
            inter_op_num_threads=req.inter_op_num_threads,
            cooling_note=req.cooling_note,
            power_note=req.power_note,
            load_note=req.load_note,
            hub_revision=req.hub_revision,
            chunk_horizon=req.chunk_horizon,
        )
    )
