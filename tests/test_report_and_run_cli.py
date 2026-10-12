"""Report rendering from synthetic fixtures and run CLI resolution/errors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from breakpoint_eval.cli import build_parser, main
from breakpoint_eval.experiment_plan import ExperimentAxis, PlanPhase, make_isolated_plan
from breakpoint_eval.experiment_summary import summarize_experiment, write_experiment_summary
from breakpoint_eval.model_resolve import resolve_model_to_candidate
from breakpoint_eval.render_report import render_report
from breakpoint_eval.run_episode import EpisodeOutcome
from breakpoint_eval.run_orchestrate import MissingPrerequisiteError, resolve_run_plan


def _episode(seed: int, *, success: bool) -> dict:
    outcome = (
        EpisodeOutcome.SUCCESS.value if success else EpisodeOutcome.TERMINATED_NO_SUCCESS.value
    )
    return {
        "episode_index": 0,
        "seed": seed,
        "outcome": outcome,
        "success": success,
        "terminated": True,
        "truncated": False,
        "policy_failed": False,
        "runtime_error": None,
        "n_control_steps": 10,
        "n_inference_calls": 1,
        "n_fresh_chunks": 1,
        "sum_reward": 1.0,
        "max_reward": 1.0,
        "control_period_s": 0.02,
        "chunk_size": 100,
        "n_action_steps": 100,
    }


def _write_run_report(path: Path, seeds: list[int], successes: list[bool]) -> Path:
    episodes = [_episode(s, success=ok) for s, ok in zip(seeds, successes, strict=True)]
    n_ok = sum(1 for x in successes if x)
    payload = {
        "provenance": {
            "candidate": "act_aloha_transfer",
            "hub_repo": "lerobot/act_aloha_sim_transfer_cube_human",
            "hub_revision": "9e93545744656e57d20f8b2d631d5d72544002ce",
            "lerobot_commit": "3c0a209f9fac4d2a57617e686a7f2a2309144ba2",
            "env_type": "aloha",
            "env_task": "AlohaTransferCube-v0",
            "device": "cpu",
            "latency_mode": "constant",
            "control_period_s": 0.02,
            "n_action_steps": 100,
            "fps": 50,
            "chunk_arrival_policy": "discard_stale",
            "env_created": True,
        },
        "summary": {
            "n_episodes_total": len(episodes),
            "n_episodes_eligible": len(episodes),
            "n_runtime_error": 0,
            "n_success": n_ok,
            "success_rate": n_ok / len(episodes),
        },
        "episodes": episodes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _synthetic_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    plan = make_isolated_plan(
        axis=ExperimentAxis.LATENCY_MS,
        candidate="act_aloha_transfer",
        phase=PlanPhase.SMOKE,
        n_episodes=5,
    )
    seeds = [s.env_seed for s in plan.seed_pairs]
    mapping = {}
    for cond in plan.conditions:
        ms = cond.params["latency_ms"]
        ok = [True] * 5 if ms < 60 else [False] * 5
        mapping[cond.condition_id] = _write_run_report(
            run_dir / "conditions" / cond.condition_id / "run_report.json",
            seeds,
            ok,
        )
    summary = summarize_experiment(plan, mapping)
    (run_dir / "experiment_plan.json").write_text(
        json.dumps(plan.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    write_experiment_summary(summary, run_dir / "experiment_summary.json")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "candidate": "act_aloha_transfer",
                "hub_repo": "lerobot/act_aloha_sim_transfer_cube_human",
                "hub_revision": "9e93545744656e57d20f8b2d631d5d72544002ce",
                "plan_id": plan.plan_id,
                "n_episodes": 5,
                "output_dir": str(run_dir),
                "cache_dir": str(tmp_path / ".cache"),
                "latency_mode": "constant",
                "compute_device": "cpu",
                "resolved_model": "act_aloha_transfer",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_resolve_hub_repo_shorthand() -> None:
    key, spec = resolve_model_to_candidate("lerobot/act_aloha_sim_transfer_cube_human")
    assert key == "act_aloha_transfer"
    assert spec.hub_repo.startswith("lerobot/")


def test_resolve_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_model_to_candidate("not/a/real/model")


def test_render_report_from_synthetic_fixture(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    run_dir = _synthetic_run_dir(tmp_path)
    bundle = render_report(run_dir)
    assert bundle.summary_md.is_file()
    assert bundle.report_html.is_file()
    md = bundle.summary_md.read_text(encoding="utf-8")
    html = bundle.report_html.read_text(encoding="utf-8")
    assert "simulated" in md.lower() or "Simulated" in md
    assert "physical robot" in md.lower() or "physical-robot" in md
    assert "act_aloha_transfer" in md
    assert "latency_ms_40" in md
    assert "<script" not in html.lower()
    assert "cdn." not in html.lower()


def test_html_escaping(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    run_dir = _synthetic_run_dir(tmp_path)
    # Inject XSS-like text into manifest candidate display path via plan label already safe;
    # write a malicious incomplete reason through summary mutation.
    summary_path = run_dir / "experiment_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["conditions"].append(
        {
            "condition_id": "<script>alert(1)</script>",
            "label": "evil",
            "params": {},
            "report_path": None,
            "successes": 0,
            "eligible": 0,
            "planned": 5,
            "success_rate": None,
            "wilson_95": {"unresolved": True},
            "n_runtime_error": 0,
            "incomplete": True,
            "incomplete_reasons": ["<img src=x onerror=alert(1)>"],
        }
    )
    payload["axis_summary"]["points"] = payload["axis_summary"].get("points") or []
    summary_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    bundle = render_report(run_dir)
    html = bundle.report_html.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<img src=x onerror=alert(1)>" not in html


def test_report_missing_conditions_labeled(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    run_dir = _synthetic_run_dir(tmp_path)
    # Remove one condition artifact.
    victim = run_dir / "conditions" / "latency_ms_80" / "run_report.json"
    victim.unlink()
    # Rebuild summary with missing mapping for that id.
    plan = make_isolated_plan(
        axis=ExperimentAxis.LATENCY_MS,
        candidate="act_aloha_transfer",
        phase=PlanPhase.SMOKE,
        n_episodes=5,
    )
    mapping = {}
    for cond in plan.conditions:
        path = run_dir / "conditions" / cond.condition_id / "run_report.json"
        if path.is_file():
            mapping[cond.condition_id] = path
    summary = summarize_experiment(plan, mapping)
    write_experiment_summary(summary, run_dir / "experiment_summary.json")
    bundle = render_report(run_dir)
    md = bundle.summary_md.read_text(encoding="utf-8")
    assert "incomplete" in md.lower() or "missing" in md.lower()
    assert "latency_ms_80" in md


def test_cli_dry_run_and_missing_rpi5_profile(tmp_path: Path) -> None:
    out = tmp_path / "out"
    code = main(
        [
            "run",
            "lerobot/act_aloha_sim_transfer_cube_human",
            "--episodes",
            "2",
            "--out",
            str(out),
            "--device",
            "cpu",
            "--dry-run",
        ]
    )
    assert code == 0
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["candidate"] == "act_aloha_transfer"
    assert manifest["n_episodes"] == 2
    assert manifest["dry_run"] is True

    # Missing rpi5 profiles → actionable failure (no invented timings).
    code2 = main(
        [
            "run",
            "act_aloha_transfer",
            "--episodes",
            "1",
            "--out",
            str(tmp_path / "rpi"),
            "--device",
            "rpi5",
            "--dry-run",
        ]
    )
    assert code2 == 2


def test_cli_report_command(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    run_dir = _synthetic_run_dir(tmp_path)
    code = main(["report", str(run_dir)])
    assert code == 0
    assert (run_dir / "report.html").is_file()


def test_cli_report_missing_dir(tmp_path: Path) -> None:
    code = main(["report", str(tmp_path / "does_not_exist")])
    assert code == 2


def test_resolve_run_plan_explicit_profile_missing(tmp_path: Path) -> None:
    with pytest.raises(MissingPrerequisiteError, match="latency profile not found"):
        resolve_run_plan(
            model="act_aloha_transfer",
            n_episodes=1,
            output_dir=tmp_path / "o",
            device="cpu",
            latency_profile=tmp_path / "missing_profile.json",
        )


def test_parser_accepts_episodes_out_aliases() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["run", "act_aloha_transfer", "--episodes", "3", "--out", "/tmp/x", "--dry-run"]
    )
    assert args.n_episodes == 3
    assert args.output_dir == Path("/tmp/x")
    assert args.dry_run is True
