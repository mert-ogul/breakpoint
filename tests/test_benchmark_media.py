"""Media selection, frame capture, and composition without the simulator."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from breakpoint_eval.benchmark_media import (
    FrameTap,
    MediaVerificationError,
    PanelSpec,
    Recording,
    compose_frames,
    make_layout,
    panel_event,
    select_pair,
)

ROOT = Path(__file__).resolve().parents[1]
EPISODES = (
    ROOT
    / "benchmarks"
    / "act-transfer"
    / "deployment-stress"
    / "data"
    / "episodes"
    / "episodes.jsonl"
)


def row(condition: str, seed: int, success: bool) -> dict:
    return {
        "plan_id": "v0.1_full_latency",
        "condition_id": condition,
        "seed": seed,
        "policy_rng_seed": seed + 10000,
        "success": success,
    }


def test_select_pair_takes_lowest_seed_where_baseline_succeeded_and_stress_did_not() -> None:
    rows = []
    for seed, (base, stress) in {
        10: (False, False),
        11: (True, True),
        12: (True, False),
        13: (True, False),
        14: (False, True),
    }.items():
        rows += [row("latency_ms_0", seed, base), row("latency_ms_640", seed, stress)]
    selection = select_pair(rows)
    assert selection["seed"] == 12
    assert selection["seeds_baseline_succeeded_stress_did_not"] == [12, 13]
    assert selection["seeds_baseline_did_not_succeed_stress_did"] == [14]
    assert selection["policy_rng_seed"] == 10012


def test_select_pair_refuses_when_no_such_pair() -> None:
    rows = [row("latency_ms_0", 1, True), row("latency_ms_640", 1, True)]
    with pytest.raises(MediaVerificationError):
        select_pair(rows)


@pytest.mark.skipif(not EPISODES.exists(), reason="packaged benchmark data not present")
def test_saved_selection_is_seed_5000() -> None:
    rows = [json.loads(line) for line in EPISODES.read_text().splitlines()]
    selection = select_pair(rows)
    assert selection["seed"] == 5000
    assert (selection["baseline_successes"], selection["stress_successes"]) == (25, 14)


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


def make_recording(n_steps: int, event: dict) -> Recording:
    steps = [
        {"is_success": False, "truncated": False, "sim_time_s": 0.02 * (i + 1)}
        for i in range(n_steps)
    ]
    steps[-1].update(event)
    frames = {k: np.full((48, 64, 3), k % 255, dtype=np.uint8) for k in range(0, n_steps + 1, 4)}
    frames[n_steps] = np.full((48, 64, 3), 200, dtype=np.uint8)
    return Recording("c", 1, 2, frames, {"steps": steps}, n_steps)


def test_panel_event_uses_recorded_steps() -> None:
    assert panel_event(make_recording(10, {"is_success": True})) == ("Success", pytest.approx(0.2))
    assert panel_event(make_recording(12, {"truncated": True})) == ("Timeout", pytest.approx(0.24))


def test_layout_has_even_dimensions_and_four_to_three_panels() -> None:
    for width in (960, 1120, 1280):
        layout = make_layout(width)
        assert layout.height % 2 == 0 and layout.width % 2 == 0
        assert layout.panel_w * 3 // 4 == layout.panel_h


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
        panels, ["one", "two"], width=640, fps=12.5, hold_frames=3, out_dir=tmp_path
    )
    assert info["control_steps_per_output_frame"] == 4
    assert info["n_motion_frames"] == 5 and info["n_frames"] == 8
    assert info["first_source_frames"] == [0, 0]
    assert info["last_source_frames"] == [8, 16]
    assert len(list(tmp_path.glob("frame_*.png"))) == 8
    with pytest.raises(ValueError):
        compose_frames(panels, [], width=640, fps=7.0, hold_frames=0, out_dir=tmp_path / "x")
