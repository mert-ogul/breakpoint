"""CLI seed / summary helpers."""

from __future__ import annotations

import pytest

from breakpoint_eval.run_episode import EpisodeOutcome, EpisodeResult
from breakpoint_eval.seed_list import build_seeds, parse_seed_list
from breakpoint_eval.write_report import summarize_episodes


def test_parse_and_build_seeds() -> None:
    assert parse_seed_list("1000, 1001") == [1000, 1001]
    assert build_seeds(n_episodes=3, seeds=None, start_seed=10) == [10, 11, 12]
    assert build_seeds(n_episodes=2, seeds=[7, 8], start_seed=0) == [7, 8]
    with pytest.raises(ValueError):
        build_seeds(n_episodes=2, seeds=[1], start_seed=0)


def test_success_rate_none_when_all_errored() -> None:
    ep = EpisodeResult(
        episode_index=0,
        seed=0,
        outcome=EpisodeOutcome.RUNTIME_ERROR,
        success=False,
        terminated=False,
        truncated=False,
        policy_failed=False,
        runtime_error="OSError: gl",
        n_control_steps=0,
        n_inference_calls=0,
        n_fresh_chunks=0,
        sum_reward=0.0,
        max_reward=0.0,
        control_period_s=0.02,
        chunk_size=100,
        n_action_steps=100,
    )
    summary = summarize_episodes([ep])
    assert summary["success_rate"] is None
    assert summary["n_episodes_eligible"] == 0


def test_episode_result_records_policy_rng_seed() -> None:
    ep = EpisodeResult(
        episode_index=0,
        seed=3000,
        outcome=EpisodeOutcome.SUCCESS,
        success=True,
        terminated=True,
        truncated=False,
        policy_failed=False,
        runtime_error=None,
        n_control_steps=10,
        n_inference_calls=1,
        n_fresh_chunks=1,
        sum_reward=1.0,
        max_reward=1.0,
        control_period_s=0.02,
        chunk_size=100,
        n_action_steps=100,
        policy_rng_seed=13000,
    )
    payload = ep.to_dict()
    assert payload["seed"] == 3000
    assert payload["policy_rng_seed"] == 13000
