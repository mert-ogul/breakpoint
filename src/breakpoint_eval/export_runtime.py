"""Resolve an exported ONNX graph against its export_manifest identity."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from breakpoint_eval.act_export import INPUT_IMAGES, INPUT_STATE, sha256_file


@dataclass(frozen=True)
class ResolvedOnnxRuntime:
    model_path: Path
    manifest_path: Path
    candidate: str
    hub_revision: str
    recipe_id: str
    variant_label: str
    model_sha256: str
    chunk_horizon: int
    input_shapes: dict[str, list[int]]
    io_spec: dict[str, Any]
    export_validation: dict[str, Any]
    corpus_eval_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model_path"] = str(self.model_path)
        payload["manifest_path"] = str(self.manifest_path)
        return payload


def find_export_manifest(model_path: Path) -> Path:
    model_path = Path(model_path).resolve()
    sibling = model_path.parent / "export_manifest.json"
    if sibling.is_file():
        return sibling
    raise FileNotFoundError(
        f"export_manifest.json not found next to ONNX model {model_path}. "
        "Pass a graph produced by `breakpoint export` so recipe/identity can be resolved."
    )


def load_export_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"export manifest must be a JSON object: {path}")
    return payload


def _variant_for_model(manifest: dict[str, Any], model_path: Path) -> dict[str, Any]:
    model_path = model_path.resolve()
    digest = sha256_file(model_path)
    variants = manifest.get("variants") or []
    by_hash: list[dict[str, Any]] = []
    by_path: list[dict[str, Any]] = []
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        graph = variant.get("graph") or {}
        if graph.get("sha256") == digest:
            by_hash.append(variant)
        path = variant.get("path")
        if path is not None and Path(path).resolve() == model_path:
            by_path.append(variant)
        elif path is not None and Path(path).name == model_path.name:
            # Same filename under the export output dir.
            if Path(path).resolve().parent == model_path.parent:
                by_path.append(variant)
    matched = by_hash or by_path
    if not matched:
        raise ValueError(
            f"ONNX model {model_path} is not listed in export manifest variants "
            f"(sha256={digest}). Refusing unlabeled/unmatched graph."
        )
    if len(matched) > 1 and by_hash:
        matched = by_hash
    if len(matched) > 1:
        raise ValueError(f"ambiguous export manifest match for {model_path}")
    variant = matched[0]
    if variant.get("status") != "ok":
        raise ValueError(
            f"matched export variant status={variant.get('status')!r}; "
            "refuse to evaluate an errored/unsupported graph as a deployment runtime"
        )
    return variant


def resolve_onnx_runtime(
    model_path: Path,
    *,
    expected_candidate: str | None = None,
    require_export_validation: bool = True,
) -> ResolvedOnnxRuntime:
    """Bind an ONNX file to its export manifest recipe/identity."""
    model_path = Path(model_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    manifest_path = find_export_manifest(model_path)
    manifest = load_export_manifest(manifest_path)
    candidate = str(manifest.get("candidate") or "")
    if expected_candidate is not None and candidate != expected_candidate:
        raise ValueError(
            f"export manifest candidate={candidate!r} does not match run "
            f"candidate={expected_candidate!r}"
        )
    variant = _variant_for_model(manifest, model_path)
    validation = dict(manifest.get("export_validation") or {})
    if require_export_validation:
        if validation.get("passed") is not True:
            raise ValueError(
                "export_validation.passed is not true; refuse to evaluate an unvalidated "
                f"export as a deployment runtime (manifest={manifest_path})"
            )
        # Defense for diagnostic/synthetic manifests that set passed incorrectly.
        if validation.get("corpus_suitable_as_deployment_evidence") is False:
            raise ValueError(
                "export corpus is not suitable as deployment evidence "
                f"(manifest={manifest_path}); diagnostic_parity_ok exports are not "
                "accepted as validated deployment runtimes"
            )
        if validation.get("closed_loop_requested") is False:
            raise ValueError(
                "export closed-loop was not requested; refuse skipped closed-loop "
                f"as deployment validation (manifest={manifest_path})"
            )
        if (
            "closed_loop_completed" in validation
            and validation.get("closed_loop_completed") is not True
        ):
            raise ValueError(
                "export closed-loop was not completed; refuse as deployment runtime "
                f"(manifest={manifest_path})"
            )
        if "closed_loop_passed" in validation and validation.get("closed_loop_passed") is not True:
            raise ValueError(
                "export closed-loop did not pass; refuse as deployment runtime "
                f"(manifest={manifest_path})"
            )
    io_spec = dict(manifest.get("io_spec") or {})
    image_shape = [int(x) for x in (io_spec.get("image_shape") or [])]
    state_shape = [int(x) for x in (io_spec.get("state_shape") or [])]
    action_shape = [int(x) for x in (io_spec.get("action_chunk_shape") or [])]
    if len(image_shape) != 4 or len(state_shape) != 2:
        raise ValueError(f"export manifest io_spec incomplete: {io_spec}")
    chunk_info = manifest.get("chunk_info") or {}
    chunk_horizon = int(
        chunk_info.get("n_action_steps") or (action_shape[1] if action_shape else 0)
    )
    if chunk_horizon < 1:
        raise ValueError("export manifest missing chunk horizon (n_action_steps)")
    corpus = manifest.get("corpus") or {}
    eval_corpus = corpus.get("evaluation") or {}
    return ResolvedOnnxRuntime(
        model_path=model_path,
        manifest_path=manifest_path,
        candidate=candidate,
        hub_revision=str(manifest.get("hub_revision") or ""),
        recipe_id=str(variant.get("recipe_id") or ""),
        variant_label=str(variant.get("label") or ""),
        model_sha256=sha256_file(model_path),
        chunk_horizon=chunk_horizon,
        input_shapes={
            INPUT_IMAGES: image_shape,
            INPUT_STATE: state_shape,
        },
        io_spec=io_spec,
        export_validation=validation,
        corpus_eval_sha256=(
            str(eval_corpus["sha256"]) if eval_corpus.get("sha256") is not None else None
        ),
    )


def apply_policy_rng_seed(seed: int) -> None:
    """Seed host RNG used by stochastic policies (e.g. Diffusion) for one episode."""
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
