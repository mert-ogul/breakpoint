"""Resume checks for the deployment stress benchmark, using small saved-report fixtures."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from breakpoint_eval.benchmark_protocol import (
    BACKEND_ONNXRUNTIME,
    ResolvedArtifacts,
    assert_lock_matches,
    condition_completed,
    condition_incomplete_reason,
    freeze_protocol_lock,
    host_settings_from_config,
    load_benchmark_config,
)
from breakpoint_eval.experiment_plan import (
    ExperimentAxis,
    ExperimentPlan,
    build_seed_pairs,
    jpeg_conditions,
    latency_conditions,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "act-transfer.json"
CANDIDATE = "act_aloha_transfer"


def _plan() -> ExperimentPlan:
    return ExperimentPlan(
        plan_id="resume_fixture",
        candidate=CANDIDATE,
        phase="exploratory",
        n_episodes=2,
        axis=ExperimentAxis.COMBINED.value,
        seed_pairs=build_seed_pairs(2, env_seed_start=5000, policy_seed_offset=10000),
        conditions=[*latency_conditions((0, 40)), *jpeg_conditions(("bypass", 50))],
        baseline_condition_id="latency_ms_0",
    )


def _condition(plan: ExperimentPlan, condition_id: str):
    return next(c for c in plan.conditions if c.condition_id == condition_id)


def _episode(plan: ExperimentPlan, index: int, *, latency_s=0.0, image_mode="native", **extra):
    pair = plan.seed_pairs[index]
    episode = {
        "episode_index": index,
        "seed": pair.env_seed,
        "policy_rng_seed": pair.policy_rng_seed,
        "outcome": "success",
        "success": True,
        "runtime_error": None,
        "n_control_steps": 10,
        "n_inference_calls": 1,
        "n_fresh_chunks": 1,
        "n_underflows": 0,
        "n_missed_deadlines": 0,
        "sum_reward": 1.0,
        "max_reward": 1.0,
        "control_period_s": 0.02,
        "latency_mode": "constant",
        "latency_s": latency_s,
        "image_mode": image_mode,
        "image_degrade_summary": {},
        "latency_trace_provenance": None,
        "steps": [{"control_step_index": 9, "sim_time_s": 0.2, "is_success": True}],
    }
    episode.update(extra)
    return episode


def _report(plan: ExperimentPlan, *, latency_s=0.0, image_mode="native", variant="onnx_fp32"):
    episodes = [
        _episode(plan, i, latency_s=latency_s, image_mode=image_mode)
        for i in range(plan.n_episodes)
    ]
    return {
        "provenance": {
            "candidate": plan.candidate,
            "env_created": True,
            "env_seeds": [p.env_seed for p in plan.seed_pairs],
            "policy_rng_seeds": [p.policy_rng_seed for p in plan.seed_pairs],
            "latency_mode": "constant",
            "onnx_runtime": {"variant_label": variant},
        },
        "summary": {"n_episodes_total": len(episodes), "n_runtime_error": 0},
        "episodes": episodes,
    }


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class Run:
    """A saved run root with a finished baseline, one run condition, and one reference."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.plan = _plan()
        self.shared = root / "shared_baselines" / "default_fp32_native_L0"
        self.conditions = root / "axes" / self.plan.plan_id / "conditions"
        self.baseline_report = _write(self.shared / "run_report.json", _report(self.plan))
        _write(self.conditions / "latency_ms_0" / "run_report.json", _report(self.plan))
        _write(
            self.conditions / "latency_ms_40" / "run_report.json",
            _report(self.plan, latency_s=0.04),
        )
        reference = self.conditions / "jpeg_bypass"
        shutil.copyfile(self.baseline_report, _write(reference / "run_report.json", {}))
        _write(
            reference / "baseline_reference.json",
            {
                "status": "referenced_baseline",
                "reference_report": str(self.baseline_report),
                "reference_dir": str(self.shared),
                "condition_id": "jpeg_bypass",
                "by_explicit_reference_only": True,
                "measurements_fabricated": False,
            },
        )

    def reason(self, condition_id: str) -> str | None:
        return condition_incomplete_reason(
            self.conditions / condition_id,
            plan=self.plan,
            condition=_condition(self.plan, condition_id),
            shared_baseline_dir=self.shared,
            backend=BACKEND_ONNXRUNTIME,
            variant_label="onnx_fp32",
        )

    def edit(self, path: Path, change) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        change(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def run(tmp_path: Path) -> Run:
    return Run(tmp_path)


def test_finished_conditions_are_complete(run: Run) -> None:
    assert run.reason("latency_ms_0") is None
    assert run.reason("latency_ms_40") is None
    assert run.reason("jpeg_bypass") is None
    assert condition_completed(
        run.conditions / "latency_ms_40",
        plan=run.plan,
        condition=_condition(run.plan, "latency_ms_40"),
        shared_baseline_dir=run.shared,
        backend=BACKEND_ONNXRUNTIME,
        variant_label="onnx_fp32",
    )


def test_recorded_timeout_is_a_completed_outcome(run: Run) -> None:
    def timeout(report: dict) -> None:
        report["episodes"][0].update(outcome="time_limit_truncation", success=False)

    run.edit(run.conditions / "latency_ms_40" / "run_report.json", timeout)
    assert run.reason("latency_ms_40") is None


def test_missing_report_is_incomplete(run: Run) -> None:
    (run.conditions / "latency_ms_40" / "run_report.json").unlink()
    assert "missing" in run.reason("latency_ms_40")


def test_empty_report_is_incomplete(run: Run) -> None:
    _write(run.conditions / "latency_ms_40" / "run_report.json", {})
    assert run.reason("latency_ms_40") == "report has no episodes"
    _write(run.conditions / "latency_ms_40" / "run_report.json", {"episodes": []})
    assert run.reason("latency_ms_40") == "report has no episodes"


def test_malformed_report_is_incomplete(run: Run) -> None:
    (run.conditions / "latency_ms_40" / "run_report.json").write_text("{", encoding="utf-8")
    assert "unreadable" in run.reason("latency_ms_40")


def test_missing_episode_is_incomplete(run: Run) -> None:
    def drop(report: dict) -> None:
        report["episodes"].pop()
        report["summary"]["n_episodes_total"] = 1

    run.edit(run.conditions / "latency_ms_40" / "run_report.json", drop)
    assert "episode indices" in run.reason("latency_ms_40")


def test_duplicate_episode_is_incomplete(run: Run) -> None:
    def duplicate(report: dict) -> None:
        report["episodes"][1] = copy.deepcopy(report["episodes"][0])

    run.edit(run.conditions / "latency_ms_40" / "run_report.json", duplicate)
    assert "episode indices" in run.reason("latency_ms_40")


@pytest.mark.parametrize("field", ["seed", "policy_rng_seed"])
def test_changed_episode_seed_is_incomplete(run: Run, field: str) -> None:
    def change(report: dict) -> None:
        report["episodes"][1][field] += 1

    run.edit(run.conditions / "latency_ms_40" / "run_report.json", change)
    assert "seeds differ" in run.reason("latency_ms_40")


def test_changed_provenance_seeds_are_incomplete(run: Run) -> None:
    run.edit(
        run.conditions / "latency_ms_40" / "run_report.json",
        lambda r: r["provenance"].update(env_seeds=[1, 2]),
    )
    assert "env_seeds" in run.reason("latency_ms_40")


def test_runtime_error_episode_is_incomplete(run: Run) -> None:
    def fail(report: dict) -> None:
        report["episodes"][0].update(outcome="runtime_error", runtime_error="boom", success=False)
        report["summary"]["n_runtime_error"] = 1

    run.edit(run.conditions / "latency_ms_40" / "run_report.json", fail)
    assert "runtime-error" in run.reason("latency_ms_40")


def test_runtime_error_hidden_by_summary_is_incomplete(run: Run) -> None:
    run.edit(
        run.conditions / "latency_ms_40" / "run_report.json",
        lambda r: r["episodes"][0].update(runtime_error="boom"),
    )
    assert "episode 0" in run.reason("latency_ms_40")


def test_failed_environment_creation_is_incomplete(run: Run) -> None:
    run.edit(
        run.conditions / "latency_ms_40" / "run_report.json",
        lambda r: r["provenance"].update(env_created=False, env_error="no display"),
    )
    assert "environment was not created" in run.reason("latency_ms_40")


def test_condition_runtime_identity_must_match(run: Run) -> None:
    run.edit(
        run.conditions / "latency_ms_40" / "run_report.json",
        lambda r: r["episodes"][0].update(latency_s=0.0),
    )
    assert "latency_s" in run.reason("latency_ms_40")
    run.edit(
        run.conditions / "latency_ms_0" / "run_report.json",
        lambda r: r["provenance"].update(onnx_runtime={"variant_label": "int8"}),
    )
    assert "variant" in run.reason("latency_ms_0")


def test_reference_to_missing_baseline_is_incomplete(run: Run) -> None:
    run.baseline_report.unlink()
    assert "referenced baseline" in run.reason("jpeg_bypass")


def test_reference_to_damaged_baseline_is_incomplete(run: Run) -> None:
    run.edit(run.baseline_report, lambda r: r["episodes"].pop())
    assert "referenced baseline" in run.reason("jpeg_bypass")


def test_reference_to_baseline_with_other_identity_is_incomplete(run: Run) -> None:
    _write(run.baseline_report, _report(run.plan, latency_s=0.04))
    assert "referenced baseline" in run.reason("jpeg_bypass")


def test_reference_marker_alone_is_not_completion(run: Run) -> None:
    _write(run.conditions / "jpeg_bypass" / "run_report.json", {})
    assert run.reason("jpeg_bypass") is not None


def test_reference_copy_must_match_baseline(run: Run) -> None:
    run.edit(
        run.conditions / "jpeg_bypass" / "run_report.json",
        lambda r: r["episodes"][0].update(sum_reward=99.0),
    )
    assert "differs from the referenced baseline" in run.reason("jpeg_bypass")


def test_reference_to_other_condition_is_incomplete(run: Run) -> None:
    run.edit(
        run.conditions / "jpeg_bypass" / "baseline_reference.json",
        lambda r: r.update(condition_id="jpeg_q50"),
    )
    assert "another condition" in run.reason("jpeg_bypass")


def test_non_baseline_condition_cannot_be_a_reference(run: Run) -> None:
    shutil.copy2(
        run.conditions / "jpeg_bypass" / "baseline_reference.json",
        run.conditions / "latency_ms_40" / "baseline_reference.json",
    )
    run.edit(
        run.conditions / "latency_ms_40" / "baseline_reference.json",
        lambda r: r.update(condition_id="latency_ms_40"),
    )
    assert "not eligible" in run.reason("latency_ms_40")


def _lock(phase: str = "full") -> dict:
    config = load_benchmark_config(CONFIG_PATH)
    artifacts = ResolvedArtifacts(
        export_dir=Path("/tmp/export"),
        fp32_path=Path("/tmp/export/a.onnx"),
        fp32_sha256="aaa",
        fp32_recipe_id="r",
        int8_path=None,
        int8_sha256=None,
        int8_status="missing",
        int8_detail=None,
        profiles={},
        profile_identities={},
    )
    return freeze_protocol_lock(
        config=config,
        artifacts=artifacts,
        source_revision={"kind": "git", "revision": "abc", "dirty": False},
        host=host_settings_from_config(config),
        phase=phase,
    )


def test_identical_lock_matches() -> None:
    assert_lock_matches(_lock(), _lock())


def test_source_revision_may_differ_on_resume() -> None:
    changed = _lock()
    changed["source_revision"] = {"kind": "git", "revision": "def", "dirty": True}
    assert_lock_matches(_lock(), changed)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("scheduler", "chunk_arrival_policy", "different"),
        ("scheduler", "latency_end_of_trace", "hold_last"),
        ("phase_config", "env_seed_start", 9000),
        ("phase_config", "policy_seed_offset", 1),
        ("phase_config", "latency_ms", [0, 10]),
        ("phase_config", "jpeg_qualities", ["bypass", 90]),
        ("baseline_reuse", "enabled", False),
    ],
)
def test_changed_execution_setting_refuses_resume(section: str, key: str, value: object) -> None:
    changed = _lock()
    changed[section][key] = value
    with pytest.raises(ValueError, match=f"resume refused: {section} changed"):
        assert_lock_matches(_lock(), changed)


def test_unknown_lock_field_is_not_ignored() -> None:
    changed = _lock()
    changed["extra_setting"] = 1
    with pytest.raises(ValueError, match="extra_setting"):
        assert_lock_matches(_lock(), changed)


def test_unsupported_lock_schema_is_refused() -> None:
    changed = _lock()
    changed["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        assert_lock_matches(_lock(), changed)
    with pytest.raises(ValueError, match="schema_version"):
        assert_lock_matches(changed, _lock())


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resumed_records_reach_the_packager(run: Run) -> None:
    pytest.importorskip("torch")
    benchmark = _load_script("run_benchmark")
    package = _load_script("package_benchmark")
    records = []
    for condition_id in ("latency_ms_0", "jpeg_bypass"):
        condition = _condition(run.plan, condition_id)
        condition_dir = run.conditions / condition_id
        assert condition_completed(
            condition_dir,
            plan=run.plan,
            condition=condition,
            shared_baseline_dir=run.shared,
            backend=BACKEND_ONNXRUNTIME,
            variant_label="onnx_fp32",
        )
        records.append(
            benchmark._resumed_record(condition_dir, condition, condition_dir / "run_report.json")
        )
    complete, referenced = records
    assert complete["status"] == "complete" and complete["resumed"] is True
    assert referenced["status"] == "referenced_baseline" and referenced["resumed"] is True
    assert referenced["reference_dir"] == str(run.shared)

    complete_path = package.condition_report_path(run.root, run.plan.plan_id, complete)
    referenced_path = package.condition_report_path(run.root, run.plan.plan_id, referenced)
    assert complete_path == run.conditions / "latency_ms_0" / "run_report.json"
    assert referenced_path == run.baseline_report
    episode = json.loads(referenced_path.read_text(encoding="utf-8"))["episodes"][0]
    row = package.compact_episode_row(
        axis="jpeg",
        plan_id=run.plan.plan_id,
        condition_id="jpeg_bypass",
        referenced=True,
        source="s",
        episode=episode,
    )
    assert row["referenced_baseline"] is True and row["seed"] == 5000
