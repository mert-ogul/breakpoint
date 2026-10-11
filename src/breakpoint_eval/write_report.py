"""Persist raw episode outcomes and a small summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from breakpoint_eval.run_episode import EpisodeOutcome, EpisodeResult


def summarize_episodes(episodes: list[EpisodeResult]) -> dict[str, Any]:
    n_total = len(episodes)
    n_errored = sum(1 for ep in episodes if ep.outcome == EpisodeOutcome.RUNTIME_ERROR)
    eligible = [ep for ep in episodes if ep.outcome != EpisodeOutcome.RUNTIME_ERROR]
    n_eligible = len(eligible)
    n_success = sum(1 for ep in eligible if ep.outcome == EpisodeOutcome.SUCCESS)
    return {
        "n_episodes_total": n_total,
        "n_episodes_eligible": n_eligible,
        "n_runtime_error": n_errored,
        "n_success": n_success,
        "n_terminated_no_success": sum(
            1 for ep in eligible if ep.outcome == EpisodeOutcome.TERMINATED_NO_SUCCESS
        ),
        "n_time_limit_truncation": sum(
            1 for ep in eligible if ep.outcome == EpisodeOutcome.TIME_LIMIT_TRUNCATION
        ),
        "n_policy_failed": sum(1 for ep in eligible if ep.outcome == EpisodeOutcome.POLICY_FAILED),
        # Runtime errors excluded from the success-rate denominator.
        "success_rate": (n_success / n_eligible) if n_eligible else None,
        "n_inference_calls_total": sum(ep.n_inference_calls for ep in episodes),
        "n_control_steps_total": sum(ep.n_control_steps for ep in episodes),
        "n_fresh_chunks_total": sum(ep.n_fresh_chunks for ep in episodes),
    }


def write_run_report(
    output_dir: Path,
    *,
    provenance: dict[str, Any],
    episodes: list[EpisodeResult],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "episodes.jsonl"
    with raw_path.open("w", encoding="utf-8") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep.to_dict()) + "\n")

    summary = {
        "provenance": provenance,
        "summary": summarize_episodes(episodes),
        "episodes": [ep.to_dict() for ep in episodes],
    }
    report_path = output_dir / "run_report.json"
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    provenance_path = output_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return report_path


def write_video(path: Path, frames: list[Any], fps: int) -> None:
    from lerobot.common.utils.io_utils import write_video as lerobot_write_video

    path.parent.mkdir(parents=True, exist_ok=True)
    lerobot_write_video(str(path), frames, fps=fps)
