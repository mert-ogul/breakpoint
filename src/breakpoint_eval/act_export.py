"""Export the verified ACT tensor path to ONNX (queues stay outside the graph).

Export boundary (matches ACTPolicy.select_action inference compute):
  preprocess_observation  — outside (numpy → float CHW)
  normalize_inputs        — inside
  stack cameras           — inside
  ACT.model (eval, zero latent) — inside
  slice n_action_steps    — inside
  unnormalize_outputs     — inside
  _action_queue / temporal ensembler — outside
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from breakpoint_eval.policy_adapter import ActionStep, ChunkInfo, PolicyAdapter

LOGGER = logging.getLogger("breakpoint_eval.act_export")

INPUT_IMAGES = "observation_images_top"
INPUT_STATE = "observation_state"
OUTPUT_CHUNK = "action_chunk"
DEFAULT_OPSET = 17

# Pinned recipe ids — never silently replace after failure.
RECIPE_ONNX_FP32 = "torch_onnx_export_fp32_opset17"
RECIPE_ORT_INT8_STATIC_QDQ = "onnxruntime_static_qdq_uint8_act_qint8_weight_minmax_calibrate"
RECIPE_ORT_INT4_MATMUL_NBITS = "onnxruntime_matmul_nbits_weight_only_int4_rtn"
RECIPE_ORT_FP16 = "onnxruntime_transformers_float16_cast"


@dataclass(frozen=True)
class ExportIOSpec:
    image_key: str
    image_shape: tuple[int, int, int, int]  # NCHW
    state_shape: tuple[int, int]
    action_chunk_shape: tuple[int, int, int]
    opset: int = DEFAULT_OPSET


class ACTChunkModule(nn.Module):
    """Pure tensor ACT chunk: norms + model + unnormalize. No action queue."""

    def __init__(self, policy: Any) -> None:
        super().__init__()
        if getattr(policy.config, "temporal_ensemble_coeff", None) is not None:
            raise NotImplementedError(
                "ACT export requires temporal_ensemble_coeff=null (queue path)"
            )
        image_features = list(policy.config.image_features)
        if len(image_features) != 1:
            raise NotImplementedError(
                f"ACT export currently supports one camera; got {image_features}"
            )
        self.policy = policy
        self.image_key = image_features[0]
        self.n_action_steps = int(policy.config.n_action_steps)

    def forward(self, images_top: Tensor, state: Tensor) -> Tensor:
        batch: dict[str, Tensor] = {
            self.image_key: images_top,
            "observation.state": state,
        }
        batch = self.policy.normalize_inputs(batch)
        batch = dict(batch)
        batch["observation.images"] = torch.stack([batch[self.image_key]], dim=-4)
        actions = self.policy.model(batch)[0][:, : self.n_action_steps]
        return self.policy.unnormalize_outputs({"action": actions})["action"]


def build_act_chunk_module(policy: Any) -> ACTChunkModule:
    policy.eval()
    return ACTChunkModule(policy).eval()


def export_act_onnx_fp32(
    module: ACTChunkModule,
    *,
    output_path: Path,
    example_images: Tensor,
    example_state: Tensor,
    opset: int = DEFAULT_OPSET,
) -> ExportIOSpec:
    """Trace/export ACT chunk module to ONNX FP32. Raises on failure (no fallback)."""
    require_onnx()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    module.eval()
    with torch.inference_mode():
        example_out = module(example_images, example_state)
    if example_out.ndim != 3:
        raise RuntimeError(f"expected action chunk (B,T,A), got {tuple(example_out.shape)}")

    torch.onnx.export(
        module,
        (example_images, example_state),
        str(output_path),
        input_names=[INPUT_IMAGES, INPUT_STATE],
        output_names=[OUTPUT_CHUNK],
        opset_version=opset,
        dynamo=False,
        do_constant_folding=True,
    )
    return ExportIOSpec(
        image_key=module.image_key,
        image_shape=tuple(int(x) for x in example_images.shape),  # type: ignore[arg-type]
        state_shape=tuple(int(x) for x in example_state.shape),  # type: ignore[arg-type]
        action_chunk_shape=tuple(int(x) for x in example_out.shape),  # type: ignore[arg-type]
        opset=opset,
    )


def require_onnx() -> None:
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ONNX export requires optional deps: pip install 'onnx>=1.16,<1.19' "
            "'onnxruntime>=1.19,<1.23'"
        ) from exc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_onnx_graph(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path))
    op_counts: dict[str, int] = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    initializers = [init.name for init in model.graph.initializer]
    return {
        "path": str(path),
        "ir_version": int(model.ir_version),
        "opset_imports": [
            {"domain": imp.domain or "", "version": int(imp.version)} for imp in model.opset_import
        ],
        "op_counts": dict(sorted(op_counts.items())),
        "n_nodes": sum(op_counts.values()),
        "n_initializers": len(initializers),
        "inputs": [i.name for i in model.graph.input],
        "outputs": [o.name for o in model.graph.output],
        "sha256": sha256_file(path),
        "bytes": Path(path).stat().st_size,
    }


def create_ort_session(
    path: Path,
    *,
    providers: list[str] | None = None,
    intra_op_num_threads: int | None = None,
    inter_op_num_threads: int | None = None,
) -> Any:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if providers is None:
        # Prefer CPU for reproducible numeric checks; callers may override.
        providers = ["CPUExecutionProvider"]
    for p in providers:
        if p not in available:
            raise RuntimeError(
                f"requested execution provider {p!r} not available; have {available}"
            )
    opts = ort.SessionOptions()
    if intra_op_num_threads is not None:
        opts.intra_op_num_threads = int(intra_op_num_threads)
    if inter_op_num_threads is not None:
        opts.inter_op_num_threads = int(inter_op_num_threads)
    return ort.InferenceSession(str(path), sess_options=opts, providers=providers)


@dataclass
class OrtRunResult:
    action_chunk: np.ndarray
    provider: str


class OrtActChunkSession:
    """Run an exported ACT chunk graph. Precision comes from the loaded weights."""

    def __init__(
        self,
        path: Path,
        *,
        providers: list[str] | None = None,
        intra_op_num_threads: int | None = None,
        inter_op_num_threads: int | None = None,
    ) -> None:
        require_onnx()
        self.path = Path(path)
        self.intra_op_num_threads = intra_op_num_threads
        self.inter_op_num_threads = inter_op_num_threads
        self.session = create_ort_session(
            self.path,
            providers=providers,
            intra_op_num_threads=intra_op_num_threads,
            inter_op_num_threads=inter_op_num_threads,
        )
        self.providers = list(self.session.get_providers())
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        if INPUT_IMAGES not in self.input_names or INPUT_STATE not in self.input_names:
            raise RuntimeError(
                f"unexpected ONNX inputs {self.input_names}; "
                f"expected {INPUT_IMAGES!r} and {INPUT_STATE!r}"
            )

    def run(self, images_top: np.ndarray, state: np.ndarray) -> OrtRunResult:
        feeds = {
            INPUT_IMAGES: np.asarray(images_top, dtype=np.float32),
            INPUT_STATE: np.asarray(state, dtype=np.float32),
        }
        out = self.session.run([OUTPUT_CHUNK], feeds)[0]
        arr = np.asarray(out, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError("ONNX Runtime produced non-finite action chunk")
        return OrtRunResult(action_chunk=arr, provider=self.providers[0])


class OrtPolicyAdapter:
    """PolicyAdapter-compatible wrapper: ORT chunk compute + Python action queue."""

    def __init__(
        self,
        session: OrtActChunkSession,
        *,
        chunk_info: ChunkInfo,
        image_key: str = "observation.images.top",
    ) -> None:
        self._session = session
        self.chunk_info = chunk_info
        self._image_key = image_key
        self._control_step_index = 0
        self._action_queue: deque[np.ndarray] = deque(maxlen=chunk_info.n_action_steps)

    def reset(self) -> None:
        self._action_queue = deque([], maxlen=self.chunk_info.n_action_steps)
        self._control_step_index = 0

    def observe(self, observation: dict[str, Tensor]) -> None:
        """No-op for ACT (n_obs_steps=1); required by the episode scheduler API."""
        return None

    def _feeds_from_obs(self, observation: dict[str, Tensor]) -> tuple[np.ndarray, np.ndarray]:
        if self._image_key not in observation:
            raise KeyError(f"missing {self._image_key} in observation")
        if "observation.state" not in observation:
            raise KeyError("missing observation.state in observation")
        images = observation[self._image_key].detach().cpu().numpy()
        state = observation["observation.state"].detach().cpu().numpy()
        return images, state

    def compute_action_chunk(self, observation: dict[str, Tensor]) -> np.ndarray:
        images, state = self._feeds_from_obs(observation)
        result = self._session.run(images, state)
        chunk = result.action_chunk
        if chunk.ndim == 3:
            if chunk.shape[0] != 1:
                raise ValueError(f"expected batch 1, got {chunk.shape}")
            chunk = chunk[0]
        if chunk.ndim != 2:
            raise ValueError(f"expected (T,A) chunk, got {chunk.shape}")
        return chunk.astype(np.float32, copy=False)

    def select_action(self, observation: dict[str, Tensor]) -> ActionStep:
        requested = len(self._action_queue) == 0
        if requested:
            chunk = self.compute_action_chunk(observation)
            for row in chunk:
                self._action_queue.append(row.astype(np.float32, copy=False))
        action = self._action_queue.popleft()
        step = ActionStep(
            action=action,
            requested_fresh_chunk=requested,
            inference_call=requested,
            control_step_index=self._control_step_index,
        )
        self._control_step_index += 1
        return step


def pytorch_chunk(
    module: ACTChunkModule,
    images: Tensor,
    state: Tensor,
) -> np.ndarray:
    with torch.inference_mode():
        out = module(images, state)
    return out.detach().cpu().numpy().astype(np.float32, copy=False)


def observation_tensors_from_numpy(
    images_top: np.ndarray,
    state: np.ndarray,
) -> dict[str, Tensor]:
    return {
        "observation.images.top": torch.from_numpy(np.asarray(images_top, dtype=np.float32)),
        "observation.state": torch.from_numpy(np.asarray(state, dtype=np.float32)),
    }


@dataclass
class VariantArtifact:
    label: str
    recipe_id: str
    path: str | None
    status: str  # ok | unsupported | errored
    detail: str | None = None
    graph: dict[str, Any] = field(default_factory=dict)
    effective_weight_precision: str | None = None
    effective_activation_precision: str | None = None
    quantization_coverage: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def chunk_info_from_policy(policy: Any, control_period_s: float) -> ChunkInfo:
    return PolicyAdapter(policy, control_period_s=control_period_s, device="cpu").chunk_info
