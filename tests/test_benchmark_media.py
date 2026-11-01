"""Media selection, frame capture, and composition without the simulator."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import breakpoint_eval.benchmark_media as benchmark_media
from breakpoint_eval.benchmark_media import (
    FrameTap,
    MediaVerificationError,
    PanelSpec,
    Recording,
    compare_with_saved,
    compose_frames,
    default_media_fps,
    frame_stride,
    make_layout,
    panel_event,
    require_match,
    resolve_latency_media,
    select_pair,
)

ROOT = Path(__file__).resolve().parents[1]
ACT_DATA = ROOT / "benchmarks" / "act-transfer" / "deployment-stress" / "data"
EPISODES = ACT_DATA / "episodes" / "episodes.jsonl"
PLAN_ID = "v0.1_full_latency"


def row(
    condition: str,
    seed: int,
    success: bool,
    *,
    referenced_baseline: bool = False,
) -> dict:
    return {
        "plan_id": PLAN_ID,
        "axis": "latency_ms",
        "condition_id": condition,
        "seed": seed,
        "policy_rng_seed": seed + 10000,
        "success": success,
        "referenced_baseline": referenced_baseline,
    }


def write_latency_provenance(
    data_dir: Path,
    *,
    n_action_steps: int,
    control_period_s: float,
    plan_id: str = PLAN_ID,
    condition_id: str = "latency_ms_0",
) -> Path:
    path = data_dir / "historical" / plan_id / condition_id / "provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"n_action_steps": n_action_steps, "control_period_s": control_period_s},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_select_pair_takes_lowest_seed_where_baseline_succeeded_and_stress_did_not(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    write_latency_provenance(data_dir, n_action_steps=100, control_period_s=0.02)
    rows = []
    for seed, (base, stress) in {
        10: (False, False),
        11: (True, True),
        12: (True, False),
        13: (True, False),
        14: (False, True),
    }.items():
        rows += [row("latency_ms_0", seed, base), row("latency_ms_640", seed, stress)]
    selection = select_pair(rows, data_dir=data_dir)
    assert selection["seed"] == 12
    assert selection["stress_delay_ms"] == 640
    assert selection["stress_condition"] == "latency_ms_640"
    assert selection["chunk_span_ms"] == 2000
    assert selection["seeds_baseline_succeeded_stress_did_not"] == [12, 13]
    assert selection["seeds_baseline_did_not_succeed_stress_did"] == [14]
    assert selection["policy_rng_seed"] == 10012


def test_select_pair_uses_largest_delay_strictly_below_chunk_span(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_latency_provenance(data_dir, n_action_steps=8, control_period_s=0.1)
    rows = []
    for delay in (0, 100, 400, 600, 800):
        rows += [
            row(f"latency_ms_{delay}", 20, True),
            row(f"latency_ms_{delay}", 21, True),
            row(f"latency_ms_{delay}", 22, False),
        ]
    # Chunk span is 800 ms, so stress is 600 ms. Only seed 21 fails there.
    for i, line in enumerate(rows):
        if line["condition_id"] == "latency_ms_600" and line["seed"] == 21:
            rows[i] = {**line, "success": False}
    selection = select_pair(rows, data_dir=data_dir)
    assert selection["chunk_span_ms"] == 800
    assert selection["stress_delay_ms"] == 600
    assert selection["stress_condition"] == "latency_ms_600"
    assert selection["seed"] == 21
    assert selection["control_period_s"] == pytest.approx(0.1)
    assert "strictly below one chunk span (800 ms), here 600 ms" in selection["rule"]


def test_chunk_span_from_provenance_pusht_and_act_values(tmp_path: Path) -> None:
    from breakpoint_eval.benchmark_media import chunk_span_from_provenance

    rows = [row("latency_ms_0", 1, True), row("latency_ms_600", 1, False)]
    pusht = tmp_path / "pusht"
    write_latency_provenance(pusht, n_action_steps=8, control_period_s=0.1)
    period, span = chunk_span_from_provenance(
        pusht, plan_id=PLAN_ID, baseline_condition="latency_ms_0", rows=rows
    )
    assert period == pytest.approx(0.1)
    assert span == 800

    act = tmp_path / "act"
    write_latency_provenance(act, n_action_steps=100, control_period_s=0.02)
    period, span = chunk_span_from_provenance(
        act, plan_id=PLAN_ID, baseline_condition="latency_ms_0", rows=rows
    )
    assert period == pytest.approx(0.02)
    assert span == 2000


def test_missing_latency_provenance_raises_with_path(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    rows = [row("latency_ms_0", 1, True), row("latency_ms_640", 1, False)]
    expected = data_dir / "historical" / PLAN_ID / "latency_ms_0" / "provenance.json"
    with pytest.raises(MediaVerificationError, match=str(expected)) as exc:
        select_pair(rows, data_dir=data_dir)
    assert "latency baseline provenance missing" in str(exc.value)


def test_provenance_missing_keys_raises_with_path(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    path = data_dir / "historical" / PLAN_ID / "latency_ms_0" / "provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    rows = [row("latency_ms_0", 1, True), row("latency_ms_640", 1, False)]
    with pytest.raises(MediaVerificationError, match=str(path)) as exc:
        select_pair(rows, data_dir=data_dir)
    assert "lacks n_action_steps or control_period_s" in str(exc.value)


def test_select_pair_act_chunk_span_still_picks_640(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_latency_provenance(data_dir, n_action_steps=100, control_period_s=0.02)
    rows = []
    for delay in (0, 40, 80, 160, 320, 640):
        rows += [
            row(f"latency_ms_{delay}", 5000, True),
            row(f"latency_ms_{delay}", 5001, True),
        ]
    for i, line in enumerate(rows):
        if line["condition_id"] == "latency_ms_640" and line["seed"] == 5000:
            rows[i] = {**line, "success": False}
    selection = select_pair(rows, data_dir=data_dir)
    assert selection["chunk_span_ms"] == 2000
    assert selection["stress_delay_ms"] == 640
    assert selection["stress_condition"] == "latency_ms_640"
    assert selection["seed"] == 5000
    spec = resolve_latency_media(rows, data_dir=data_dir)
    assert spec.stress_delay_ms == 640
    assert spec.chunk_span_ms == 2000


def test_select_pair_refuses_when_no_such_pair(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_latency_provenance(data_dir, n_action_steps=100, control_period_s=0.02)
    rows = [row("latency_ms_0", 1, True), row("latency_ms_640", 1, True)]
    with pytest.raises(MediaVerificationError, match="no baseline-success / stress-failure pair"):
        select_pair(rows, data_dir=data_dir)


@pytest.mark.skipif(not EPISODES.exists(), reason="packaged benchmark data not present")
def test_saved_selection_is_seed_5000() -> None:
    rows = [json.loads(line) for line in EPISODES.read_text().splitlines()]
    selection = select_pair(rows, data_dir=ACT_DATA)
    assert selection["seed"] == 5000
    assert selection["stress_delay_ms"] == 640
    assert selection["chunk_span_ms"] == 2000
    assert (selection["baseline_successes"], selection["stress_successes"]) == (25, 14)


@pytest.mark.skipif(not EPISODES.exists(), reason="packaged benchmark data not present")
def test_act_media_constants_match_today() -> None:
    rows = [json.loads(line) for line in EPISODES.read_text().splitlines()]
    spec = resolve_latency_media(rows, data_dir=ACT_DATA)
    assert spec.stress_delay_ms == 640
    assert spec.chunk_span_ms == 2000
    assert spec.control_period_s == pytest.approx(0.02)
    assert frame_stride(spec.control_period_s, 12.5) == 4


def test_frame_stride_for_pusht_period() -> None:
    assert frame_stride(0.1, 5.0) == 2
    assert default_media_fps(0.1) == pytest.approx(5.0)
    assert default_media_fps(0.02) == pytest.approx(12.5)


def _step(
    *,
    reward: float,
    action: list[float] | None = None,
    sim_time_s: float = 0.1,
) -> dict:
    return {
        "sim_time_s": sim_time_s,
        "reward": reward,
        "is_success": False,
        "request_time_s": 0.0,
        "ready_time_s": 0.0,
        "action": action if action is not None else [0.0, 0.0],
    }


def _comparison_pair(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rerun_rewards: list[float],
    saved_rewards: list[float],
    rerun_actions: list[list[float]] | None = None,
    saved_actions: list[list[float]] | None = None,
) -> dict:
    assert len(rerun_rewards) == len(saved_rewards)
    rerun_actions = rerun_actions or [[1.0, 2.0] for _ in rerun_rewards]
    saved_actions = saved_actions or [list(a) for a in rerun_actions]
    recording = Recording(
        "latency_ms_0",
        1,
        2,
        {},
        {
            "outcome": "success",
            "n_control_steps": len(rerun_rewards),
            "n_inference_calls": 1,
            "sum_reward": sum(rerun_rewards),
            "steps": [
                _step(reward=r, action=a, sim_time_s=0.1 * (i + 1))
                for i, (r, a) in enumerate(zip(rerun_rewards, rerun_actions, strict=True))
            ],
        },
        len(rerun_rewards),
    )
    saved_episode = {
        "seed": 1,
        "steps": [
            _step(reward=r, action=a, sim_time_s=0.1 * (i + 1))
            for i, (r, a) in enumerate(zip(saved_rewards, saved_actions, strict=True))
        ],
    }
    saved_row = {
        "outcome": "success",
        "n_control_steps": len(saved_rewards),
        "n_inference_calls": 1,
        "sum_reward": sum(rerun_rewards),
        "source_report": "unused.json",
    }
    monkeypatch.setattr(
        benchmark_media, "load_saved_episode", lambda source_report, seed: saved_episode
    )
    return compare_with_saved(recording, saved_row)


def test_compare_with_saved_allows_reward_ulp_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    base = 0.39217324061308
    noisy = math.nextafter(math.nextafter(base, math.inf), math.inf)
    comparison = _comparison_pair(
        monkeypatch, rerun_rewards=[noisy, 0.5], saved_rewards=[base, 0.5]
    )
    assert "reward" not in "timeline_success_request_ready_equal"
    assert comparison["timeline_success_request_ready_equal"] is True
    assert comparison["actions_bitwise_equal"] is True
    assert comparison["rewards_within_tolerance"] is True
    assert comparison["reward_rel_tol"] == 1e-12
    assert comparison["reward_abs_tol"] == 1e-12
    assert comparison["rewards_max_abs_difference"] == pytest.approx(abs(noisy - base))
    require_match(comparison)


def test_compare_with_saved_rejects_reward_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    comparison = _comparison_pair(monkeypatch, rerun_rewards=[0.5 + 1e-6], saved_rewards=[0.5])
    assert comparison["timeline_success_request_ready_equal"] is True
    assert comparison["rewards_within_tolerance"] is False
    assert comparison["rewards_max_abs_difference"] == pytest.approx(1e-6)
    with pytest.raises(MediaVerificationError):
        require_match(comparison)


def test_compare_with_saved_rejects_action_difference(monkeypatch: pytest.MonkeyPatch) -> None:
    comparison = _comparison_pair(
        monkeypatch,
        rerun_rewards=[0.5],
        saved_rewards=[0.5],
        rerun_actions=[[1.0, 2.0]],
        saved_actions=[[1.0, 2.1]],
    )
    assert comparison["timeline_success_request_ready_equal"] is True
    assert comparison["rewards_within_tolerance"] is True
    assert comparison["actions_bitwise_equal"] is False
    with pytest.raises(MediaVerificationError):
        require_match(comparison)


def test_timeline_key_excludes_reward(monkeypatch: pytest.MonkeyPatch) -> None:
    comparison = _comparison_pair(monkeypatch, rerun_rewards=[0.5 + 1e-6], saved_rewards=[0.5])
    assert "timeline_success_request_ready_equal" in comparison
    assert "timeline_reward_success_request_ready_equal" not in comparison
    assert comparison["timeline_success_request_ready_equal"] is True


class FakeInner:
    def __init__(self, terminal_at: int) -> None:
        self.terminal_at = terminal_at
        self.steps = 0
        self.resets = 0

    def reset(self, *args, **kwargs):
        self.resets += 1
        self.steps = 0
        return None

    def step(self, action):
        self.steps += 1
        return (None, 0.0, self.steps == self.terminal_at, False, {})

    def render(self):
        return np.full((2, 2, 3), self.steps if self.resets == 1 else 255, dtype=np.uint8)


def test_frame_tap_keeps_true_final_frame_not_autoreset_frame() -> None:
    inner = FakeInner(terminal_at=6)
    tap = FrameTap(stride=4)
    tap.attach(SimpleNamespace(envs=[inner]))
    inner.reset(seed=1)
    for _ in range(6):
        inner.step(None)
    inner.reset(seed=1)  # vector-env autoreset after the terminal step
    assert sorted(tap.frames) == [0, 4, 6]
    assert int(tap.frames[6][0, 0, 0]) == 6
    assert int(tap.frames[0][0, 0, 0]) == 0


def make_recording(n_steps: int, event: dict, *, hw: tuple[int, int] = (48, 64)) -> Recording:
    h, w = hw
    steps = [
        {"is_success": False, "truncated": False, "sim_time_s": 0.02 * (i + 1)}
        for i in range(n_steps)
    ]
    steps[-1].update(event)
    frames = {k: np.full((h, w, 3), k % 255, dtype=np.uint8) for k in range(0, n_steps + 1, 4)}
    frames[n_steps] = np.full((h, w, 3), 200, dtype=np.uint8)
    return Recording("c", 1, 2, frames, {"steps": steps}, n_steps)


def test_panel_event_uses_recorded_steps() -> None:
    assert panel_event(make_recording(10, {"is_success": True})) == ("Success", pytest.approx(0.2))
    assert panel_event(make_recording(12, {"truncated": True})) == ("Timeout", pytest.approx(0.24))


def test_layout_has_even_dimensions_and_four_to_three_panels() -> None:
    for width in (960, 1120, 1280):
        layout = make_layout(width)
        assert layout.height % 2 == 0 and layout.width % 2 == 0
        assert layout.panel_w * 3 // 4 == layout.panel_h


def test_layout_uses_square_panels_for_square_frames() -> None:
    layout = make_layout(1280, frame_hw=(96, 96))
    assert layout.panel_h == layout.panel_w
    assert layout.height % 2 == 0


def test_compose_frames_aligns_by_simulation_time_and_holds(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    pytest.importorskip("matplotlib")
    short = make_recording(8, {"is_success": True})
    long = make_recording(16, {"truncated": True})
    panels = [
        PanelSpec("Baseline, 0 ms delay", "a", (23, 33, 31), short, "Success", 0.16),
        PanelSpec("+640 ms delay", "b", (217, 82, 41), long, "Timeout", 0.32),
    ]
    info = compose_frames(
        panels,
        ["one", "two"],
        width=640,
        fps=12.5,
        hold_frames=3,
        out_dir=tmp_path,
        control_period_s=0.02,
    )
    assert info["control_steps_per_output_frame"] == 4
    assert info["n_motion_frames"] == 5 and info["n_frames"] == 8
    assert info["first_source_frames"] == [0, 0]
    assert info["last_source_frames"] == [8, 16]
    assert len(list(tmp_path.glob("frame_*.png"))) == 8
    with pytest.raises(ValueError):
        compose_frames(
            panels,
            [],
            width=640,
            fps=7.0,
            hold_frames=0,
            out_dir=tmp_path / "x",
            control_period_s=0.02,
        )
