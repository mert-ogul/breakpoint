"""Summarize multi-condition experiment run records (Wilson + paired bootstrap).

Implements the rules in docs/methodology.md. Does not invent missing runs or
interpolate exact boundaries.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from breakpoint_eval.episode_outcome import EpisodeOutcome
from breakpoint_eval.experiment_plan import ExperimentPlan, load_experiment_plan

# Project conventions (not safety criteria) — methodology.md
DEFAULT_BREAKPOINT_RATIO = 0.8
DEFAULT_BASELINE_SUCCESS_FLOOR = 0.5
DEFAULT_WILSON_Z = 1.959963984540054  # ~95%
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 20261012


@dataclass
class EpisodeBinary:
    env_seed: int
    success: bool  # True only for SUCCESS among eligible; ineligible excluded
    eligible: bool
    outcome: str
    runtime_error: str | None = None


@dataclass
class ConditionRecord:
    condition_id: str
    label: str
    params: dict[str, Any]
    ordered_value: float | None  # None for categorical axes
    episodes: list[EpisodeBinary]
    report_path: str | None
    n_planned: int
    incomplete: bool
    incomplete_reasons: list[str] = field(default_factory=list)

    @property
    def n_eligible(self) -> int:
        return sum(1 for e in self.episodes if e.eligible)

    @property
    def n_success(self) -> int:
        return sum(1 for e in self.episodes if e.eligible and e.success)

    @property
    def n_runtime_error(self) -> int:
        return sum(1 for e in self.episodes if e.outcome == EpisodeOutcome.RUNTIME_ERROR.value)

    @property
    def success_rate(self) -> float | None:
        if self.n_eligible == 0:
            return None
        return self.n_success / self.n_eligible


def wilson_interval(
    successes: int,
    n: int,
    *,
    z: float = DEFAULT_WILSON_Z,
) -> dict[str, Any]:
    """95% Wilson score interval for a binomial proportion."""
    if n < 0 or successes < 0 or successes > n:
        raise ValueError("invalid successes/n for Wilson interval")
    if n == 0:
        return {
            "low": None,
            "high": None,
            "n": 0,
            "successes": 0,
            "unresolved": True,
            "reason": "n=0",
        }
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n))
    low = 0.0 if successes == 0 else max(0.0, center - half)
    high = 1.0 if successes == n else min(1.0, center + half)
    return {
        "low": low,
        "high": high,
        "n": n,
        "successes": successes,
        "point": phat,
        "unresolved": False,
        "level": 0.95,
        "method": "wilson",
    }


def load_condition_from_report(
    path: Path,
    *,
    condition_id: str,
    label: str,
    params: dict[str, Any],
    ordered_value: float | None,
    planned_env_seeds: list[int],
) -> ConditionRecord:
    reasons: list[str] = []
    if not Path(path).is_file():
        return ConditionRecord(
            condition_id=condition_id,
            label=label,
            params=params,
            ordered_value=ordered_value,
            episodes=[],
            report_path=str(path),
            n_planned=len(planned_env_seeds),
            incomplete=True,
            incomplete_reasons=["missing_run_report"],
        )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_eps = payload.get("episodes") or []
    by_seed: dict[int, dict[str, Any]] = {}
    for ep in raw_eps:
        by_seed[int(ep["seed"])] = ep

    episodes: list[EpisodeBinary] = []
    for seed in planned_env_seeds:
        if seed not in by_seed:
            reasons.append(f"missing_episode_seed_{seed}")
            continue
        ep = by_seed[seed]
        outcome = str(ep.get("outcome"))
        eligible = outcome != EpisodeOutcome.RUNTIME_ERROR.value
        success = bool(ep.get("success")) and outcome == EpisodeOutcome.SUCCESS.value
        episodes.append(
            EpisodeBinary(
                env_seed=seed,
                success=success,
                eligible=eligible,
                outcome=outcome,
                runtime_error=ep.get("runtime_error"),
            )
        )

    if len(episodes) != len(planned_env_seeds):
        reasons.append(
            f"episode_count_mismatch: got={len(episodes)} planned={len(planned_env_seeds)}"
        )
    n_err = sum(1 for e in episodes if e.outcome == EpisodeOutcome.RUNTIME_ERROR.value)
    if n_err:
        # Do not cherry-pick successful executions for breakpoint decisions.
        reasons.append(f"runtime_errors={n_err}")
    n_eligible = sum(1 for e in episodes if e.eligible)
    if n_eligible == 0 and episodes:
        reasons.append("no_eligible_episodes")

    incomplete = bool(reasons)
    summary = payload.get("summary") or {}
    if summary.get("n_episodes_total") is not None:
        if int(summary["n_episodes_total"]) != len(planned_env_seeds):
            reasons.append(
                f"report_n_episodes_total={summary['n_episodes_total']} "
                f"!= planned={len(planned_env_seeds)}"
            )
            incomplete = True

    return ConditionRecord(
        condition_id=condition_id,
        label=label,
        params=params,
        ordered_value=ordered_value,
        episodes=episodes,
        report_path=str(path),
        n_planned=len(planned_env_seeds),
        incomplete=incomplete,
        incomplete_reasons=reasons,
    )


def ordered_value_for_condition(
    condition_id: str, params: dict[str, Any], axis: str
) -> float | None:
    if axis == "latency_ms":
        return float(params["latency_ms"])
    if axis == "resolution":
        return float(params.get("linear_scale", 1.0))
    if axis == "jpeg":
        if params.get("bypass"):
            return 100.0  # mild end of severity ordering (bypass > q95 > …)
        return float(params["jpeg_quality"])
    if axis == "h264":
        if params.get("bypass"):
            return 1.0e9  # mild end (bypass before any bitrate)
        return float(params["h264_bitrate_kbps"])
    if axis == "precision":
        return None  # categorical
    return None


def severity_sort_key(record: ConditionRecord, axis: str) -> tuple[int, float]:
    """Order mild → severe for numeric axes. Higher latency / lower scale / lower JPEG = more severe."""
    if record.ordered_value is None:
        return (0, 0.0)
    v = record.ordered_value
    if axis == "latency_ms":
        return (0, v)  # increasing delay
    if axis == "resolution":
        return (0, -v)  # decreasing linear scale
    if axis == "jpeg":
        return (0, -v)  # decreasing quality / bypass first
    if axis == "h264":
        return (0, -v)  # decreasing bitrate / bypass first
    return (0, v)


def paired_success_vectors(
    baseline: ConditionRecord,
    condition: ConditionRecord,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Align eligible-success indicators by env seed. Returns None if unpaired."""
    b_map = {e.env_seed: e for e in baseline.episodes}
    c_map = {e.env_seed: e for e in condition.episodes}
    seeds = sorted(set(b_map) & set(c_map))
    if not seeds:
        return None
    # Only pairs where BOTH eligible contribute to paired contrast rates.
    b_vals: list[float] = []
    c_vals: list[float] = []
    for s in seeds:
        be, ce = b_map[s], c_map[s]
        if not (be.eligible and ce.eligible):
            continue
        b_vals.append(1.0 if be.success else 0.0)
        c_vals.append(1.0 if ce.success else 0.0)
    if len(b_vals) < 1:
        return None
    return np.asarray(b_vals, dtype=np.float64), np.asarray(c_vals, dtype=np.float64)


def paired_bootstrap_contrast(
    baseline_success: np.ndarray,
    condition_success: np.ndarray,
    *,
    ratio: float,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Pointwise bootstrap CI for p_c - ratio * p_b on paired episode indicators."""
    n = int(baseline_success.shape[0])
    if n != condition_success.shape[0]:
        raise ValueError("paired vectors length mismatch")
    if n < 2:
        return {
            "statistic": "p_condition - ratio * p_baseline",
            "ratio": ratio,
            "n_pairs": n,
            "point": None,
            "low": None,
            "high": None,
            "unresolved": True,
            "reason": "n_pairs < 2 (degenerate)",
            "bootstrap_seed": seed,
            "n_bootstrap": n_bootstrap,
            "coverage": "pointwise",
            "simultaneous_across_sweep": False,
        }
    p_b = float(baseline_success.mean())
    p_c = float(condition_success.mean())
    point = p_c - ratio * p_b
    rng = np.random.default_rng(seed)
    stats = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        pb = float(baseline_success[idx].mean())
        pc = float(condition_success[idx].mean())
        stats[i] = pc - ratio * pb
    # Degenerate: zero variance
    if float(np.std(stats)) == 0.0 and n < 5:
        return {
            "statistic": "p_condition - ratio * p_baseline",
            "ratio": ratio,
            "n_pairs": n,
            "point": point,
            "low": None,
            "high": None,
            "unresolved": True,
            "reason": "degenerate_bootstrap_variance",
            "bootstrap_seed": seed,
            "n_bootstrap": n_bootstrap,
            "coverage": "pointwise",
            "simultaneous_across_sweep": False,
        }
    low, high = np.quantile(stats, [0.025, 0.975])
    return {
        "statistic": "p_condition - ratio * p_baseline",
        "ratio": ratio,
        "n_pairs": n,
        "point": point,
        "low": float(low),
        "high": float(high),
        "unresolved": False,
        "bootstrap_seed": seed,
        "n_bootstrap": n_bootstrap,
        "coverage": "pointwise",
        "simultaneous_across_sweep": False,
        "level": 0.95,
        "method": "paired_bootstrap",
    }


def condition_is_below_threshold(
    p_condition: float | None,
    p_baseline: float | None,
    *,
    ratio: float,
) -> bool | None:
    if p_condition is None or p_baseline is None:
        return None
    return bool(p_condition < ratio * p_baseline)


def summarize_ordered_breakpoint(
    records: list[ConditionRecord],
    *,
    axis: str,
    baseline_id: str,
    ratio: float,
    baseline_success_floor: float,
) -> dict[str, Any]:
    """First below-threshold + adjacent preceding above; no interpolation."""
    by_id = {r.condition_id: r for r in records}
    baseline = by_id.get(baseline_id)
    if baseline is None:
        return {
            "headline_suppressed": True,
            "reason": "baseline_condition_missing",
            "breakpoint": None,
        }

    p_b = baseline.success_rate
    baseline_weak = p_b is None or p_b < baseline_success_floor
    incomplete_any = baseline.incomplete or any(r.incomplete for r in records)

    # Sort mild → severe for numeric axes.
    ordered = sorted(records, key=lambda r: severity_sort_key(r, axis))
    points: list[dict[str, Any]] = []
    for rec in ordered:
        below = condition_is_below_threshold(rec.success_rate, p_b, ratio=ratio)
        points.append(
            {
                "condition_id": rec.condition_id,
                "label": rec.label,
                "ordered_value": rec.ordered_value,
                "successes": rec.n_success,
                "eligible": rec.n_eligible,
                "planned": rec.n_planned,
                "success_rate": rec.success_rate,
                "wilson_95": wilson_interval(rec.n_success, rec.n_eligible),
                "below_threshold": below,
                "incomplete": rec.incomplete,
                "incomplete_reasons": list(rec.incomplete_reasons),
                "n_runtime_error": rec.n_runtime_error,
            }
        )

    if baseline_weak:
        return {
            "headline_suppressed": True,
            "reason": "weak_baseline",
            "baseline_success_rate": p_b,
            "baseline_success_floor": baseline_success_floor,
            "ratio": ratio,
            "points": points,
            "breakpoint": None,
            "note": (
                "Baseline below eligibility floor; raw results shown, "
                "headline breakpoint suppressed."
            ),
        }

    if incomplete_any:
        # Still compute a candidate interval from complete points only, but suppress headline.
        complete_points = [p for p in points if not p["incomplete"]]
    else:
        complete_points = points

    first_below_idx = None
    for i, p in enumerate(complete_points):
        if p["below_threshold"] is True:
            first_below_idx = i
            break

    non_monotonic = False
    recovery_ids: list[str] = []
    if first_below_idx is not None:
        for p in complete_points[first_below_idx + 1 :]:
            if p["below_threshold"] is False:
                non_monotonic = True
                recovery_ids.append(p["condition_id"])

    if first_below_idx is None:
        breakpoint = {
            "status": "not_reached_within_tested_range",
            "first_below": None,
            "preceding_above": None,
            "observed_interval": None,
            "non_monotonic": False,
            "recovery_condition_ids": [],
        }
    else:
        first_below = complete_points[first_below_idx]
        preceding = complete_points[first_below_idx - 1] if first_below_idx > 0 else None
        # If the first point is already below, preceding may be None (censored left).
        interval = None
        if preceding is not None and preceding["below_threshold"] is False:
            interval = {
                "above_condition_id": preceding["condition_id"],
                "above_ordered_value": preceding["ordered_value"],
                "below_condition_id": first_below["condition_id"],
                "below_ordered_value": first_below["ordered_value"],
                "interpolated": False,
            }
        elif preceding is None:
            interval = {
                "above_condition_id": None,
                "below_condition_id": first_below["condition_id"],
                "below_ordered_value": first_below["ordered_value"],
                "censored": "left_no_preceding_above_in_tested_range",
                "interpolated": False,
            }
        else:
            interval = {
                "above_condition_id": preceding["condition_id"],
                "below_condition_id": first_below["condition_id"],
                "note": "preceding point was not strictly above threshold",
                "interpolated": False,
            }
        breakpoint = {
            "status": "observed_interval",
            "first_below": first_below["condition_id"],
            "preceding_above": None if preceding is None else preceding["condition_id"],
            "observed_interval": interval,
            "non_monotonic": non_monotonic,
            "recovery_condition_ids": recovery_ids,
        }

    suppress = incomplete_any
    return {
        "headline_suppressed": suppress,
        "reason": "incomplete_conditions" if suppress else None,
        "baseline_success_rate": p_b,
        "baseline_success_floor": baseline_success_floor,
        "ratio": ratio,
        "points": points,
        "breakpoint": None if suppress else breakpoint,
        "breakpoint_exploratory_only": breakpoint if suppress else None,
        "note": (
            "Incomplete planned conditions: breakpoint decision suppressed (not cherry-picked)."
            if suppress
            else "Observed interval only; not an interpolated exact boundary."
        ),
    }


def summarize_categorical_precision(
    records: list[ConditionRecord],
    *,
    baseline_id: str,
    ratio: float,
    baseline_success_floor: float,
) -> dict[str, Any]:
    by_id = {r.condition_id: r for r in records}
    baseline = by_id.get(baseline_id)
    p_b = baseline.success_rate if baseline else None
    baseline_weak = baseline is None or p_b is None or p_b < baseline_success_floor
    recipes = []
    for rec in records:
        recipes.append(
            {
                "condition_id": rec.condition_id,
                "variant_label": rec.params.get("variant_label"),
                "label": rec.label,
                "successes": rec.n_success,
                "eligible": rec.n_eligible,
                "planned": rec.n_planned,
                "success_rate": rec.success_rate,
                "wilson_95": wilson_interval(rec.n_success, rec.n_eligible),
                "below_threshold": condition_is_below_threshold(rec.success_rate, p_b, ratio=ratio),
                "incomplete": rec.incomplete,
                "incomplete_reasons": list(rec.incomplete_reasons),
                "categorical": True,
            }
        )
    return {
        "headline_suppressed": baseline_weak or any(r.incomplete for r in records),
        "reason": (
            "weak_baseline"
            if baseline_weak
            else ("incomplete_conditions" if any(r.incomplete for r in records) else None)
        ),
        "baseline_success_rate": p_b,
        "ratio": ratio,
        "recipes": recipes,
        "note": (
            "Precision recipes are categories; no continuous bit-depth breakpoint. "
            "Results listed per recipe."
        ),
        "breakpoint": None,
    }


def summarize_experiment(
    plan: ExperimentPlan,
    condition_reports: dict[str, Path],
    *,
    ratio: float = DEFAULT_BREAKPOINT_RATIO,
    baseline_success_floor: float = DEFAULT_BASELINE_SUCCESS_FLOOR,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    """Build a summary from a plan + mapping condition_id → run_report.json."""
    env_seeds = [s.env_seed for s in plan.seed_pairs]
    records: list[ConditionRecord] = []
    for cond in plan.conditions:
        path = condition_reports.get(cond.condition_id)
        if path is None:
            rec = ConditionRecord(
                condition_id=cond.condition_id,
                label=cond.label,
                params=cond.params,
                ordered_value=ordered_value_for_condition(
                    cond.condition_id, cond.params, plan.axis
                ),
                episodes=[],
                report_path=None,
                n_planned=plan.n_episodes,
                incomplete=True,
                incomplete_reasons=["missing_condition_mapping"],
            )
        else:
            rec = load_condition_from_report(
                path,
                condition_id=cond.condition_id,
                label=cond.label,
                params=cond.params,
                ordered_value=ordered_value_for_condition(
                    cond.condition_id, cond.params, plan.axis
                ),
                planned_env_seeds=env_seeds,
            )
        records.append(rec)

    baseline_id = plan.baseline_condition_id or records[0].condition_id
    baseline = next(r for r in records if r.condition_id == baseline_id)

    contrasts = []
    for rec in records:
        if rec.condition_id == baseline_id:
            continue
        paired = paired_success_vectors(baseline, rec)
        if paired is None:
            contrasts.append(
                {
                    "condition_id": rec.condition_id,
                    "unresolved": True,
                    "reason": "no_paired_eligible_episodes",
                    "bootstrap_seed": bootstrap_seed,
                    "coverage": "pointwise",
                    "simultaneous_across_sweep": False,
                }
            )
            continue
        b_vec, c_vec = paired
        contrasts.append(
            {
                "condition_id": rec.condition_id,
                **paired_bootstrap_contrast(
                    b_vec,
                    c_vec,
                    ratio=ratio,
                    n_bootstrap=n_bootstrap,
                    seed=bootstrap_seed,
                ),
            }
        )

    if plan.axis == "precision":
        axis_summary = summarize_categorical_precision(
            records,
            baseline_id=baseline_id,
            ratio=ratio,
            baseline_success_floor=baseline_success_floor,
        )
    else:
        axis_summary = summarize_ordered_breakpoint(
            records,
            axis=plan.axis,
            baseline_id=baseline_id,
            ratio=ratio,
            baseline_success_floor=baseline_success_floor,
        )

    return {
        "plan_id": plan.plan_id,
        "candidate": plan.candidate,
        "phase": plan.phase,
        "axis": plan.axis,
        "n_episodes_planned": plan.n_episodes,
        "env_seeds": env_seeds,
        "policy_rng_seeds": [s.policy_rng_seed for s in plan.seed_pairs],
        "ratio": ratio,
        "baseline_success_floor": baseline_success_floor,
        "baseline_condition_id": baseline_id,
        "confirms_plan_id": plan.confirms_plan_id,
        "isolation_notes": list(plan.isolation_notes),
        "conventions": {
            "not_safety_criteria": True,
            "no_interpolated_exact_boundary": True,
            "no_optional_stopping": True,
            "pointwise_bootstrap_not_simultaneous": True,
            "hundred_episodes_not_precise_boundary": True,
            "exploratory_separate_from_confirmation": True,
        },
        "conditions": [
            {
                "condition_id": r.condition_id,
                "label": r.label,
                "params": r.params,
                "report_path": r.report_path,
                "successes": r.n_success,
                "eligible": r.n_eligible,
                "planned": r.n_planned,
                "success_rate": r.success_rate,
                "wilson_95": wilson_interval(r.n_success, r.n_eligible),
                "n_runtime_error": r.n_runtime_error,
                "incomplete": r.incomplete,
                "incomplete_reasons": r.incomplete_reasons,
            }
            for r in records
        ],
        "paired_contrasts_vs_baseline": contrasts,
        "axis_summary": axis_summary,
    }


def write_experiment_summary(summary: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def summarize_from_plan_file(
    plan_path: Path,
    condition_reports: dict[str, Path],
    *,
    output_path: Path | None = None,
    ratio: float = DEFAULT_BREAKPOINT_RATIO,
    baseline_success_floor: float = DEFAULT_BASELINE_SUCCESS_FLOOR,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    plan = load_experiment_plan(plan_path)
    summary = summarize_experiment(
        plan,
        condition_reports,
        ratio=ratio,
        baseline_success_floor=baseline_success_floor,
        bootstrap_seed=bootstrap_seed,
    )
    if output_path is not None:
        write_experiment_summary(summary, output_path)
    return summary
