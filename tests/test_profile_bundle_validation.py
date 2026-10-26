"""FP32 profiling bundle acceptance and source provenance (no Torch required)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from breakpoint_eval.benchmark_profile import resolve_validated_fp32_bundle
from breakpoint_eval.export_validation import SCHEMA_VERSION, collect_code_revision
from breakpoint_eval.file_digest import sha256_file

CANDIDATE = "act_aloha_transfer"
HUB_REVISION = "rev1"
RECIPE = "torch_onnx_export_fp32_opset17"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _validation(model_sha: str, corpus_sha: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "binding": {
            "model_sha256": model_sha,
            "corpus_eval_sha256": corpus_sha,
            "hub_revision": HUB_REVISION,
            "candidate": CANDIDATE,
            "recipe_id": RECIPE,
            "variant_label": "onnx_fp32",
        },
        "validation_settings": {"parity_atol": 1e-4},
        "numerical_parity": {"passed": True},
        "closed_loop_task_outcomes": {"passed": True, "completed": True},
        "corpus_suitable_as_deployment_evidence": True,
    }


@pytest.fixture
def export_dir(tmp_path: Path) -> Path:
    (tmp_path / "act_chunk_fp32.onnx").write_bytes(b"onnx-bytes")
    (tmp_path / "corpus_eval.npz").write_bytes(b"corpus-bytes")
    (tmp_path / "eval_chunks_onnx_fp32.npz").write_bytes(b"reference-bytes")
    model_sha = sha256_file(tmp_path / "act_chunk_fp32.onnx")
    corpus_sha = sha256_file(tmp_path / "corpus_eval.npz")
    _write_json(
        tmp_path / "export_manifest.json",
        {
            "candidate": CANDIDATE,
            "hub_revision": HUB_REVISION,
            "corpus": {"evaluation": {"sha256": corpus_sha}},
            "variants": [
                {
                    "label": "onnx_fp32",
                    "status": "ok",
                    "path": "act_chunk_fp32.onnx",
                    "recipe_id": RECIPE,
                    "graph": {"sha256": model_sha},
                }
            ],
        },
    )
    _write_json(tmp_path / "export_validation.json", _validation(model_sha, corpus_sha))
    return tmp_path


def _edit_validation(export_dir: Path, edit) -> None:
    path = export_dir / "export_validation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    edit(payload)
    _write_json(path, payload)


def test_valid_fp32_bundle_is_accepted(export_dir: Path) -> None:
    bundle = resolve_validated_fp32_bundle(export_dir, expected_candidate=CANDIDATE)
    assert bundle.validation_passed is True
    assert bundle.recipe_id == RECIPE
    assert bundle.hub_revision == HUB_REVISION


def test_legacy_validation_filename_is_read(export_dir: Path) -> None:
    (export_dir / "export_validation.json").rename(export_dir / "export_validation_v2.json")
    assert resolve_validated_fp32_bundle(export_dir).validation_passed is True


def test_sidecar_with_only_passed_flag_is_rejected(export_dir: Path) -> None:
    _write_json(export_dir / "export_validation.json", {"passed": True})
    with pytest.raises(ValueError, match="schema_version"):
        resolve_validated_fp32_bundle(export_dir)


def test_missing_validation_is_rejected(export_dir: Path) -> None:
    (export_dir / "export_validation.json").unlink()
    with pytest.raises(FileNotFoundError):
        resolve_validated_fp32_bundle(export_dir)


def test_missing_binding_is_rejected(export_dir: Path) -> None:
    _edit_validation(export_dir, lambda v: v.pop("binding"))
    with pytest.raises(ValueError, match="binding"):
        resolve_validated_fp32_bundle(export_dir)


def test_missing_binding_field_is_rejected(export_dir: Path) -> None:
    _edit_validation(export_dir, lambda v: v["binding"].pop("corpus_eval_sha256"))
    with pytest.raises(ValueError, match="corpus_eval_sha256"):
        resolve_validated_fp32_bundle(export_dir)


def test_missing_nested_check_is_rejected(export_dir: Path) -> None:
    _edit_validation(export_dir, lambda v: v.pop("numerical_parity"))
    with pytest.raises(ValueError, match="numerical_parity"):
        resolve_validated_fp32_bundle(export_dir)


@pytest.mark.parametrize("value", [False, "true", "True", 1])
def test_non_boolean_true_pass_flag_is_rejected(export_dir: Path, value: object) -> None:
    _edit_validation(export_dir, lambda v: v.update(passed=value))
    with pytest.raises(ValueError, match="passed"):
        resolve_validated_fp32_bundle(export_dir)


def test_validation_for_other_model_is_rejected(export_dir: Path) -> None:
    _edit_validation(export_dir, lambda v: v["binding"].update(model_sha256="0" * 64))
    with pytest.raises(ValueError, match="model_sha256"):
        resolve_validated_fp32_bundle(export_dir)


def test_changed_local_corpus_is_rejected(export_dir: Path) -> None:
    (export_dir / "corpus_eval.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="corpus"):
        resolve_validated_fp32_bundle(export_dir)


def test_validation_for_other_corpus_is_rejected(export_dir: Path) -> None:
    _edit_validation(export_dir, lambda v: v["binding"].update(corpus_eval_sha256="1" * 64))
    with pytest.raises(ValueError, match="corpus_eval_sha256"):
        resolve_validated_fp32_bundle(export_dir)


@pytest.mark.parametrize(
    ("field", "match"),
    [("candidate", "candidate"), ("hub_revision", "hub_revision"), ("recipe_id", "recipe_id")],
)
def test_validation_identity_mismatch_is_rejected(
    export_dir: Path, field: str, match: str
) -> None:
    _edit_validation(export_dir, lambda v: v["binding"].update({field: "other"}))
    with pytest.raises(ValueError, match=match):
        resolve_validated_fp32_bundle(export_dir)


def test_run_candidate_must_match_manifest(export_dir: Path) -> None:
    with pytest.raises(ValueError, match="candidate"):
        resolve_validated_fp32_bundle(export_dir, expected_candidate="other_candidate")


def test_wrong_variant_binding_is_rejected(export_dir: Path) -> None:
    _edit_validation(export_dir, lambda v: v["binding"].update(variant_label="onnx_fp16"))
    with pytest.raises(ValueError, match="variant_label"):
        resolve_validated_fp32_bundle(export_dir)


def test_int8_validation_presented_as_fp32_is_rejected(export_dir: Path) -> None:
    _edit_validation(export_dir, lambda v: v["binding"].update(variant_label="int8"))
    with pytest.raises(ValueError, match="INT8"):
        resolve_validated_fp32_bundle(export_dir)


def test_conflicting_validation_filenames_are_rejected(export_dir: Path) -> None:
    shutil.copy(export_dir / "export_validation.json", export_dir / "export_validation_v2.json")
    _write_json(export_dir / "export_validation_v2.json", {"passed": True})
    with pytest.raises(ValueError, match="differ"):
        resolve_validated_fp32_bundle(export_dir)


def test_ambiguous_reference_chunks_are_rejected(export_dir: Path) -> None:
    (export_dir / "reference_chunks_onnx_fp32.npz").write_bytes(b"other")
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_validated_fp32_bundle(export_dir)


def test_missing_manifest_recipe_is_rejected(export_dir: Path) -> None:
    path = export_dir / "export_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["variants"][0]["recipe_id"]
    _write_json(path, manifest)
    with pytest.raises(ValueError, match="recipe_id"):
        resolve_validated_fp32_bundle(export_dir)


def test_archive_provenance_keeps_unknown_status_unknown(tmp_path: Path) -> None:
    (tmp_path / "source_revision.txt").write_text("deadbeef\n", encoding="utf-8")
    info = collect_code_revision(tmp_path)
    assert info == {"code_revision": "deadbeef", "code_dirty": None, "uncommitted_paths": None}


def test_missing_provenance_is_not_fabricated(tmp_path: Path) -> None:
    info = collect_code_revision(tmp_path)
    assert info == {"code_revision": None, "code_dirty": None, "uncommitted_paths": None}


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_provenance_reports_uncommitted_paths(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "--allow-empty", "-m", "init")
    (tmp_path / "new.txt").write_text("x", encoding="utf-8")
    info = collect_code_revision(tmp_path)
    assert info["code_revision"]
    assert info["code_dirty"] is True
    assert info["uncommitted_paths"] == ["new.txt"]
