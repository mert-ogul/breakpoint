"""Packaging of synthetic completed runs: input resolution, validation, atomic replacement."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest
from test_benchmark_resume import _report, _write

from breakpoint_eval.experiment_plan import (
    ExperimentAxis,
    ExperimentPlan,
    build_seed_pairs,
    jpeg_conditions,
    latency_conditions,
    precision_conditions,
    write_experiment_plan,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "act_aloha_transfer"
FP32_SHA = hashlib.sha256(b"fp32-bytes").hexdigest()
INT8_SHA = hashlib.sha256(b"int8-bytes").hexdigest()


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "package_benchmark", ROOT / "scripts" / "package_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


package_benchmark = _load_script()


def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _identity(device: str) -> dict:
    return {
        "schema_version": 1,
        "device_label": device,
        "candidate": CANDIDATE,
        "model_sha256": FP32_SHA,
        "precision_recipe": "recipe",
        "variant_label": "onnx_fp32",
        "input_shapes": {"images": [1, 3, 4, 4]},
        "corpus_hash": "corpus",
        "timed_boundary": "ort_session_run_action_chunk",
        "chunk_horizon": 10,
        "runtime_provider": ["CPUExecutionProvider"],
        "intra_op_num_threads": 4,
        "inter_op_num_threads": 1,
    }


def _write_profile(directory: Path, device: str, *, name: str | None = None, sha=FP32_SHA) -> Path:
    identity = _identity(device)
    payload = {
        "schema_version": 1,
        "device_label": device,
        "candidate": CANDIDATE,
        "model": {
            "sha256": sha,
            "precision_recipe": identity["precision_recipe"],
            "variant_label": "onnx_fp32",
        },
        "input_contract": {"shapes": identity["input_shapes"], "corpus_sha256": "corpus"},
        "timed_boundary": identity["timed_boundary"],
        "chunk_horizon": 10,
        "runtime": {
            "session_providers": ["CPUExecutionProvider"],
            "intra_op_num_threads": 4,
            "inter_op_num_threads": 1,
        },
        "measurements": {"inference_latency_s": [0.01, 0.02], "n_measured": 2},
        "output_validation": {"passed": True},
    }
    path = _write(directory / (name or f"profile_{device}.json"), payload)
    path.with_suffix(".npz").write_bytes(b"samples")
    telemetry = {"n_samples": 0}
    _write(directory / "profile_run_meta.json", {"telemetry": telemetry})
    _write(directory / "telemetry_raw.json", telemetry)
    _write(directory / "profile_run_complete.json", {"completed": True})
    return path


class SyntheticRun:
    def __init__(self, root: Path, *, int8_status: str = "unsupported") -> None:
        self.root = root
        self.run = root / "run"
        self.export = root / "export"
        self.int8_status = int8_status
        self.plan = ExperimentPlan(
            plan_id="synthetic",
            candidate=CANDIDATE,
            phase="exploratory",
            n_episodes=2,
            axis=ExperimentAxis.COMBINED.value,
            seed_pairs=build_seed_pairs(2, env_seed_start=5000, policy_seed_offset=10000),
            conditions=[
                *latency_conditions((0, 40)),
                *jpeg_conditions(("bypass",)),
                *precision_conditions(("int8",)),
            ],
            baseline_condition_id="latency_ms_0",
        )
        self._write_export()
        self.profiles = {
            device: _write_profile(root / "profiles" / device, device)
            for device in ("laptop-m3", "rpi5")
        }
        self._write_run()

    def _write_export(self) -> None:
        self.export.mkdir()
        (self.export / "act_chunk_fp32.onnx").write_bytes(b"fp32-bytes")
        (self.export / "corpus_eval.npz").write_bytes(b"corpus")
        for name in ("export_validation.json", "export_manifest.json", "corpus_eval.json"):
            _write(self.export / name, {"name": name})
        if self.int8_status == "ok":
            (self.export / "act_chunk_int8_qdq.onnx").write_bytes(b"int8-bytes")
            _write(self.export / "export_validation_int8.json", {"name": "int8"})
            _write(self.export / "corpus_calibration.json", {"name": "calibration"})

    def _lock(self) -> dict:
        ok = self.int8_status == "ok"
        return {
            "schema_version": 1,
            "protocol_name": "act_deployment_stress",
            "protocol_version": "0.1",
            "artifacts": {
                "export_dir": str(self.export),
                "fp32_path": str(self.export / "act_chunk_fp32.onnx"),
                "fp32_sha256": FP32_SHA,
                "fp32_recipe_id": "recipe",
                "int8_path": str(self.export / "act_chunk_int8_qdq.onnx") if ok else None,
                "int8_sha256": INT8_SHA if ok else None,
                "int8_status": self.int8_status,
                "int8_detail": None if ok else "recorded as not runnable",
                "profiles": {k: str(v) for k, v in self.profiles.items()},
                "profile_identities": {d: _identity(d) for d in self.profiles},
            },
        }

    def _condition_dir(self, condition_id: str) -> Path:
        return self.run / "axes" / "synthetic" / "conditions" / condition_id

    def _write_run(self) -> None:
        run = self.run
        run.mkdir()
        _write(run / "protocol_lock.json", self._lock())
        _write(run / "benchmark_config.json", {"schema_version": 1})
        write_experiment_plan(self.plan, run / "plans" / "synthetic.json")
        baseline = _write(
            run / "shared_baselines" / "default_fp32_native_L0" / "run_report.json",
            _report(self.plan),
        )
        _write(baseline.parent / "baseline_fingerprint.json", {"condition_id": "latency_ms_0"})
        _write(self._condition_dir("latency_ms_0") / "run_report.json", _report(self.plan))
        _write(
            self._condition_dir("latency_ms_40") / "run_report.json",
            _report(self.plan, latency_s=0.04),
        )
        reference = self._condition_dir("jpeg_bypass")
        shutil.copyfile(baseline, _write(reference / "run_report.json", {}))
        _write(
            reference / "baseline_reference.json",
            {
                "status": "referenced_baseline",
                "reference_dir": "/original/machine/shared_baselines/default_fp32_native_L0",
                "condition_id": "jpeg_bypass",
            },
        )
        records = [
            {"condition_id": "latency_ms_0", "status": "complete"},
            {"condition_id": "latency_ms_40", "status": "complete"},
            {
                "condition_id": "jpeg_bypass",
                "status": "referenced_baseline",
                "reference_dir": "/original/machine/shared_baselines/default_fp32_native_L0",
            },
        ]
        if self.int8_status == "ok":
            _write(
                self._condition_dir("precision_int8") / "run_report.json",
                _report(self.plan, variant="int8"),
            )
            records.append({"condition_id": "precision_int8", "status": "complete"})
        else:
            records.append(
                {
                    "condition_id": "precision_int8",
                    "status": "unsupported",
                    "measurements_fabricated": False,
                }
            )
            _write(run / "int8_unsupported.json", {"status": self.int8_status})
        self.write_records(records)

    def write_records(self, records: list[dict]) -> None:
        _write(self.run / "run_records.json", [{"plan_id": "synthetic", "records": records}])

    def records(self) -> list[dict]:
        return json.loads((self.run / "run_records.json").read_text())[0]["records"]

    def set_record(self, condition_id: str, **change) -> None:
        records = self.records()
        for record in records:
            if record["condition_id"] == condition_id:
                record.update(change)
        self.write_records(records)

    def package(self, out: Path, **kwargs) -> int:
        return package_benchmark.package(self.run, out, **kwargs)


@pytest.fixture
def synthetic(tmp_path: Path) -> SyntheticRun:
    return SyntheticRun(tmp_path)


def _manifest_paths(out: Path) -> set[str]:
    return {entry["path"] for entry in json.loads((out / "manifest.json").read_text())["files"]}


def test_package_writes_manifest_with_matching_hashes(synthetic: SyntheticRun) -> None:
    out = synthetic.root / "data"
    assert synthetic.package(out) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    for entry in manifest["files"]:
        assert hashlib.sha256((out / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    paths = _manifest_paths(out)
    assert {
        "episodes/episodes.jsonl",
        "protocol/protocol_lock.json",
        "protocol/int8_unsupported.json",
        "identities/condition_resolution.json",
        "validation/export_validation.json",
        "profiles/rpi5/profile.json",
        "large_evidence.json",
    } <= paths
    assert not list(out.parent.glob(".data.*"))


def test_references_stay_references(synthetic: SyntheticRun) -> None:
    out = synthetic.root / "data"
    synthetic.package(out)
    rows = [json.loads(x) for x in (out / "episodes" / "episodes.jsonl").read_text().splitlines()]
    assert {r["condition_id"] for r in rows if r["referenced_baseline"]} == {"jpeg_bypass"}
    assert len(rows) == 6
    resolution = {
        r["condition_id"]: r
        for r in json.loads((out / "identities" / "condition_resolution.json").read_text())
    }
    assert resolution["jpeg_bypass"]["resolution"] == "referenced_baseline"
    assert resolution["jpeg_bypass"]["source_report"].endswith(
        "default_fp32_native_L0/run_report.json"
    )


def test_legacy_resumed_complete_keeps_reference_provenance(synthetic: SyntheticRun) -> None:
    synthetic.set_record("jpeg_bypass", status="resumed_complete", reference_dir=None)
    synthetic.set_record("latency_ms_40", status="resumed_complete")
    out = synthetic.root / "data"
    synthetic.package(out)
    resolution = {
        r["condition_id"]: r
        for r in json.loads((out / "identities" / "condition_resolution.json").read_text())
    }
    assert resolution["jpeg_bypass"]["resolution"] == "referenced_baseline"
    assert resolution["jpeg_bypass"]["record_status"] == "resumed_complete"
    assert resolution["latency_ms_40"]["resolution"] == "complete"


def test_failed_package_leaves_existing_destination_identical(synthetic: SyntheticRun) -> None:
    out = synthetic.root / "data"
    synthetic.package(out)
    before = _tree(out)
    (synthetic.export / "corpus_eval.json").unlink()
    with pytest.raises(FileNotFoundError):
        synthetic.package(out)
    assert _tree(out) == before
    assert not [p for p in out.parent.iterdir() if p.name.startswith(".data.")]


def test_failed_validation_leaves_existing_destination_identical(synthetic: SyntheticRun) -> None:
    out = synthetic.root / "data"
    synthetic.package(out)
    before = _tree(out)
    synthetic.set_record("latency_ms_40", status="errored", error="boom")
    with pytest.raises(ValueError, match="errored.*boom"):
        synthetic.package(out)
    assert _tree(out) == before


def test_successful_package_replaces_existing(synthetic: SyntheticRun) -> None:
    out = synthetic.root / "data"
    synthetic.package(out)
    (out / "stale.txt").write_text("old")
    _write(out / "manifest.json", {"files": []})
    synthetic.package(out)
    assert not (out / "stale.txt").exists()
    assert "episodes/episodes.jsonl" in _manifest_paths(out)
    assert not [p for p in out.parent.iterdir() if p.name.startswith(".data.")]


def test_destination_without_manifest_is_not_replaced(synthetic: SyntheticRun) -> None:
    out = synthetic.root / "data"
    out.mkdir()
    (out / "mine.txt").write_text("keep")
    with pytest.raises(ValueError, match="no manifest.json"):
        synthetic.package(out)
    assert (out / "mine.txt").read_text() == "keep"


@pytest.mark.parametrize("where", ["inside_run", "contains_run", "export"])
def test_output_overlapping_inputs_is_refused(synthetic: SyntheticRun, where: str) -> None:
    out = {
        "inside_run": synthetic.run / "data",
        "contains_run": synthetic.root,
        "export": synthetic.export,
    }[where]
    with pytest.raises(ValueError, match="overlaps input"):
        synthetic.package(out)
    assert not (synthetic.run / "data").exists()


def test_inputs_are_not_modified(synthetic: SyntheticRun) -> None:
    before = {d: _tree(synthetic.root / d) for d in ("run", "export", "profiles")}
    synthetic.package(synthetic.root / "data")
    assert {d: _tree(synthetic.root / d) for d in before} == before


def test_incomplete_and_error_records_are_rejected(synthetic: SyntheticRun) -> None:
    synthetic.set_record("latency_ms_40", status="incomplete", error="runtime-error episodes")
    with pytest.raises(ValueError, match="incomplete.*runtime-error"):
        synthetic.package(synthetic.root / "data")


def test_complete_record_with_damaged_report_is_rejected(synthetic: SyntheticRun) -> None:
    path = synthetic._condition_dir("latency_ms_40") / "run_report.json"
    report = json.loads(path.read_text())
    report["episodes"].pop()
    _write(path, report)
    with pytest.raises(ValueError, match="latency_ms_40 is incomplete"):
        synthetic.package(synthetic.root / "data")


def test_reference_to_damaged_baseline_is_rejected(synthetic: SyntheticRun) -> None:
    baseline = synthetic.run / "shared_baselines" / "default_fp32_native_L0" / "run_report.json"
    _write(baseline, {})
    with pytest.raises(ValueError, match="jpeg_bypass is incomplete"):
        synthetic.package(synthetic.root / "data")


def test_reference_record_without_reference_file_is_rejected(synthetic: SyntheticRun) -> None:
    (synthetic._condition_dir("jpeg_bypass") / "baseline_reference.json").unlink()
    with pytest.raises(ValueError, match="without baseline_reference.json"):
        synthetic.package(synthetic.root / "data")


def test_missing_planned_condition_record_is_rejected(synthetic: SyntheticRun) -> None:
    synthetic.write_records(synthetic.records()[:-1])
    with pytest.raises(ValueError, match="do not match planned conditions"):
        synthetic.package(synthetic.root / "data")


def test_unsupported_int8_is_recorded_without_int8_files(synthetic: SyntheticRun) -> None:
    out = synthetic.root / "data"
    synthetic.package(out)
    identities = json.loads((out / "identities" / "identities.json").read_text())
    assert identities["model_int8_qdq"] == {
        "sha256": None,
        "status": "unsupported",
        "detail": "recorded as not runnable",
    }
    resolution = json.loads((out / "identities" / "condition_resolution.json").read_text())
    int8 = next(r for r in resolution if r["condition_id"] == "precision_int8")
    assert int8["resolution"] == "unsupported" and int8["measurements_fabricated"] is False
    assert not any("int8" in p and p.endswith(".onnx") for p in _manifest_paths(out))
    large = json.loads((out / "large_evidence.json").read_text())["files"]
    assert not any("int8_qdq" in entry["path"] for entry in large)
    rows = (out / "episodes" / "episodes.jsonl").read_text()
    assert "precision_int8" not in rows


def test_unsupported_record_is_rejected_when_lock_says_int8_runnable(tmp_path: Path) -> None:
    run = SyntheticRun(tmp_path, int8_status="ok")
    run.set_record("precision_int8", status="unsupported", measurements_fabricated=False)
    with pytest.raises(ValueError, match="lock records INT8 as runnable"):
        run.package(tmp_path / "data")


def test_unsupported_status_is_only_for_int8(synthetic: SyntheticRun) -> None:
    synthetic.set_record("latency_ms_40", status="unsupported", measurements_fabricated=False)
    with pytest.raises(ValueError, match="not the INT8 condition"):
        synthetic.package(synthetic.root / "data")


def test_runnable_int8_is_packaged_with_its_files(tmp_path: Path) -> None:
    run = SyntheticRun(tmp_path, int8_status="ok")
    out = tmp_path / "data"
    run.package(out)
    assert "validation/export_validation_int8.json" in _manifest_paths(out)
    large = json.loads((out / "large_evidence.json").read_text())["files"]
    assert any(entry["path"].endswith("act_chunk_int8_qdq.onnx") for entry in large)
    assert "precision_int8" in (out / "episodes" / "episodes.jsonl").read_text()


def test_broken_required_int8_run_fails(tmp_path: Path) -> None:
    run = SyntheticRun(tmp_path, int8_status="ok")
    run.set_record("precision_int8", status="errored", error="runtime failure")
    with pytest.raises(ValueError, match="precision_int8.*errored"):
        run.package(tmp_path / "data")


def test_relocated_export_requires_explicit_path_and_matching_hash(
    synthetic: SyntheticRun,
) -> None:
    moved = synthetic.root / "moved_export"
    synthetic.export.rename(moved)
    with pytest.raises(FileNotFoundError, match="--export-dir"):
        synthetic.package(synthetic.root / "data")
    synthetic.package(synthetic.root / "data", export_dir=moved)
    (moved / "act_chunk_fp32.onnx").write_bytes(b"different")
    with pytest.raises(ValueError, match="does not match the locked sha256"):
        synthetic.package(synthetic.root / "other", export_dir=moved)


def test_locked_profile_missing_requires_explicit_profile(synthetic: SyntheticRun) -> None:
    moved = synthetic.root / "moved_profiles" / "rpi5"
    shutil.copytree(synthetic.root / "profiles" / "rpi5", moved)
    shutil.rmtree(synthetic.root / "profiles" / "rpi5")
    with pytest.raises(FileNotFoundError, match="--profile rpi5=PATH"):
        synthetic.package(synthetic.root / "data")
    synthetic.package(synthetic.root / "data", profile_overrides={"rpi5": moved})
    assert (synthetic.root / "data" / "profiles" / "rpi5" / "profile.json").is_file()


def test_ambiguous_profile_directory_is_rejected(synthetic: SyntheticRun) -> None:
    directory = synthetic.root / "profiles" / "rpi5"
    _write_profile(directory, "rpi5", name="second_copy.json")
    with pytest.raises(FileNotFoundError, match="ambiguous"):
        synthetic.package(synthetic.root / "data", profile_overrides={"rpi5": directory})


def test_profile_for_other_model_is_rejected(synthetic: SyntheticRun) -> None:
    other = _write_profile(synthetic.root / "other_profile", "rpi5", sha="0" * 64)
    with pytest.raises(ValueError, match="locked identity|no profile"):
        synthetic.package(synthetic.root / "data", profile_overrides={"rpi5": other})
    directory = other.parent
    with pytest.raises(FileNotFoundError, match="no profile"):
        synthetic.package(synthetic.root / "data", profile_overrides={"rpi5": directory})


def test_profile_override_for_unlocked_device_is_rejected(synthetic: SyntheticRun) -> None:
    with pytest.raises(ValueError, match="not in the run lock"):
        synthetic.package(
            synthetic.root / "data", profile_overrides={"jetson": synthetic.profiles["rpi5"]}
        )
