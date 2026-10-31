"""Render summary.md / report.html / static plots for one run directory.

Reads only that run's saved records (no model downloads, simulator access, or reruns).
Used by `breakpoint report` and after each benchmark axis. The packaged-benchmark report
is benchmark_report.py.
"""

from __future__ import annotations

import html
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from breakpoint_eval.benchmark_report_style import ACCENT, INK, LIGHT_MUTED, LIGHT_RULE

LOGGER = logging.getLogger("breakpoint_eval.render_report")

CONTROLLER_TIMING_ASSUMPTIONS = [
    "ALOHA ACT control period Δt = 20 ms (50 Hz); chunk n_action_steps = 100 when using pinned cards.",
    "Simulated latency advances control/sim time only; host wall compute is not added to L.",
    "Chunk arrival policy: discard_stale for absolute joint-target chunks.",
    "Task success is a simulated-environment outcome, not physical-robot certification.",
    "Hardware-measured timing (device profiles) is distinct from simulated constant delay.",
]


@dataclass
class ReportBundle:
    run_dir: Path
    summary_md: Path
    report_html: Path
    plots_dir: Path
    index: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_records(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    manifest = _read_json(run_dir / "run_manifest.json") or {}
    plan = _read_json(run_dir / "experiment_plan.json")
    summary = _read_json(run_dir / "experiment_summary.json")
    single = _read_json(run_dir / "run_report.json")
    conditions: dict[str, dict[str, Any]] = {}
    cond_root = run_dir / "conditions"
    if cond_root.is_dir():
        for child in sorted(cond_root.iterdir()):
            report = _read_json(child / "run_report.json")
            if report is not None:
                conditions[child.name] = report
    # Also accept flat latency_ms_* / named subdirs used by older sweeps.
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir() or child.name in {"conditions", "report", "plots"}:
            continue
        report = _read_json(child / "run_report.json")
        if report is not None and child.name not in conditions:
            conditions[child.name] = report

    latency_profile = None
    profile_path = manifest.get("latency_profile_path")
    if profile_path:
        latency_profile = _read_json(Path(profile_path))
        if latency_profile is None:
            # Try relative to run_dir / repo
            cand = run_dir / Path(profile_path).name
            latency_profile = _read_json(cand)

    partial = _read_json(run_dir / "partial_diagnostics.json")
    return {
        "run_dir": str(run_dir),
        "manifest": manifest,
        "plan": plan,
        "experiment_summary": summary,
        "single_run": single,
        "conditions": conditions,
        "latency_profile": latency_profile,
        "partial_diagnostics": partial,
    }


def _fmt_rate(successes: int | None, eligible: int | None, rate: float | None) -> str:
    if successes is None or eligible is None:
        return "n/a (missing)"
    if rate is None:
        return f"{successes}/{eligible} (rate undefined)"
    return f"{successes}/{eligible} ({rate:.3f})"


def _wilson_str(w: dict[str, Any] | None) -> str:
    if not w or w.get("unresolved"):
        return "unresolved"
    low, high = w.get("low"), w.get("high")
    if low is None or high is None:
        return "unresolved"
    return f"[{low:.3f}, {high:.3f}]"


def _provenance_bits(records: dict[str, Any]) -> dict[str, Any]:
    manifest = records.get("manifest") or {}
    single = records.get("single_run") or {}
    prov = single.get("provenance") or manifest.get("provenance") or {}
    # Prefer first condition provenance if needed.
    if not prov:
        for report in (records.get("conditions") or {}).values():
            if report.get("provenance"):
                prov = report["provenance"]
                break
    return {
        "candidate": manifest.get("candidate") or prov.get("candidate"),
        "hub_repo": prov.get("hub_repo") or manifest.get("hub_repo"),
        "hub_revision": prov.get("hub_revision") or manifest.get("hub_revision"),
        "lerobot_commit": prov.get("lerobot_commit"),
        "env_type": prov.get("env_type"),
        "env_task": prov.get("env_task"),
        "device_compute": prov.get("device") or manifest.get("compute_device"),
        "latency_mode": prov.get("latency_mode") or manifest.get("latency_mode"),
        "latency_profile_identity": prov.get("latency_profile_identity")
        or manifest.get("latency_profile_identity"),
        "chunk_arrival_policy": prov.get("chunk_arrival_policy"),
        "control_period_s": prov.get("control_period_s"),
        "n_action_steps": prov.get("n_action_steps"),
        "fps": prov.get("fps"),
        "image_degrade": prov.get("image_degrade"),
    }


def _require_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "breakpoint report requires matplotlib "
            "(pip install 'breakpoint-eval[report]' or matplotlib)"
        ) from exc
    return plt


def _plot_success_bars(
    path: Path,
    labels: list[str],
    rates: list[float | None],
    lows: list[float | None],
    highs: list[float | None],
    *,
    title: str,
) -> Path | None:
    if not labels:
        return None
    plt = _require_matplotlib()
    xs = np.arange(len(labels))
    ys = [float("nan") if r is None else float(r) for r in rates]
    yerr_low = []
    yerr_high = []
    for r, lo, hi in zip(rates, lows, highs, strict=True):
        if r is None or lo is None or hi is None:
            yerr_low.append(0.0)
            yerr_high.append(0.0)
        else:
            yerr_low.append(max(0.0, r - lo))
            yerr_high.append(max(0.0, hi - r))
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(xs, [0.0 if np.isnan(y) else y for y in ys], color=INK)
    # Only draw error bars where rates exist; missing rates stay missing.
    for i, r in enumerate(rates):
        if r is None:
            ax.text(i, 0.02, "missing", ha="center", va="bottom", fontsize=8, color=ACCENT)
        else:
            ax.errorbar(
                i,
                r,
                yerr=[[yerr_low[i]], [yerr_high[i]]],
                fmt="none",
                ecolor=INK,
                capsize=3,
            )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Success rate (eligible)")
    ax.set_title(title)
    ax.axhline(0.0, color=LIGHT_RULE, lw=0.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_latency_hist(path: Path, latency_s: list[float], *, title: str) -> Path | None:
    if not latency_s:
        return None
    plt = _require_matplotlib()
    arr = np.asarray(latency_s, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(arr * 1000.0, bins=min(40, max(5, len(arr) // 20)), color=INK, edgecolor="white")
    ax.set_xlabel("Inference latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _build_view(records: dict[str, Any], plots: dict[str, str | None]) -> dict[str, Any]:
    bits = _provenance_bits(records)
    manifest = records.get("manifest") or {}
    summary = records.get("experiment_summary")
    single = records.get("single_run")
    profile = records.get("latency_profile")
    partial = records.get("partial_diagnostics")

    isolated = None
    combined = None
    incomplete: list[dict[str, Any]] = []
    if summary:
        axis = summary.get("axis")
        block = summary.get("axis_summary") or {}
        entry = {
            "axis": axis,
            "phase": summary.get("phase"),
            "headline_suppressed": block.get("headline_suppressed"),
            "reason": block.get("reason"),
            "breakpoint": block.get("breakpoint"),
            "points": block.get("points") or block.get("recipes") or [],
            "conditions": summary.get("conditions") or [],
            "n_episodes_planned": summary.get("n_episodes_planned"),
            "ratio": summary.get("ratio"),
        }
        if axis == "combined":
            combined = entry
        else:
            isolated = entry
        for c in summary.get("conditions") or []:
            if c.get("incomplete"):
                incomplete.append(c)

    single_summary = None
    if single:
        s = single.get("summary") or {}
        single_summary = {
            "successes": s.get("n_success"),
            "eligible": s.get("n_episodes_eligible"),
            "total": s.get("n_episodes_total"),
            "runtime_errors": s.get("n_runtime_error"),
            "success_rate": s.get("success_rate"),
            "label": "simulated_task_outcome",
        }

    precision = None
    if profile:
        precision = {
            "recipe": (profile.get("model") or {}).get("precision_recipe"),
            "variant": (profile.get("model") or {}).get("variant_label"),
            "effective_weight_precision": profile.get("effective_weight_precision")
            or (profile.get("model") or {}).get("precision_recipe"),
            "effective_activation_precision": profile.get("effective_activation_precision"),
            "quantization_coverage": profile.get("quantization_coverage")
            or (profile.get("model") or {}).get("quantization_coverage"),
            "timed_boundary": profile.get("timed_boundary"),
            "device_label": profile.get("device_label"),
            "n_measured": (profile.get("measurements") or {}).get("n_measured"),
            "summary": (profile.get("measurements") or {}).get("summary"),
        }
    # Also from experiment precision axis
    if isolated and isolated.get("axis") == "precision":
        precision = precision or {}
        precision["recipe_results"] = isolated.get("points")

    return {
        "disclaimer": {
            "simulation_not_physical_robot": True,
            "hardware_timing_distinct_from_sim_task": True,
            "missing_data_not_replaced_with_zero_or_fit": True,
            "text": (
                "Simulated task outcomes are environment results. Hardware-measured "
                "timing comes only from recorded device profiles. This report does not "
                "imply the simulation ran on a physical robot, nor that every numeric "
                "output was produced on the profiled device. Missing values are shown "
                "as missing — not zeroed or curve-fitted."
            ),
        },
        "model_env": bits,
        "manifest": {
            "plan_id": manifest.get("plan_id"),
            "n_episodes": manifest.get("n_episodes"),
            "output_dir": manifest.get("output_dir"),
            "cache_dir": manifest.get("cache_dir"),
            "latency_profile_path": manifest.get("latency_profile_path"),
            "latency_mode": manifest.get("latency_mode"),
            "dry_run": manifest.get("dry_run"),
            "resolved_model": manifest.get("resolved_model"),
            "candidate": manifest.get("candidate"),
        },
        "baseline": (
            next(
                (
                    c
                    for c in (summary or {}).get("conditions", [])
                    if c.get("condition_id") == (summary or {}).get("baseline_condition_id")
                ),
                None,
            )
            if summary
            else single_summary
        ),
        "single_run": single_summary,
        "isolated_axis": isolated,
        "combined_stress": combined,
        "incomplete_conditions": incomplete,
        "device_trace": {
            "present": profile is not None,
            "path": manifest.get("latency_profile_path"),
            "identity": bits.get("latency_profile_identity"),
            "precision": precision,
        },
        "controller_timing_assumptions": CONTROLLER_TIMING_ASSUMPTIONS,
        "plots": plots,
        "partial_diagnostics": partial,
        "artifact_links": _artifact_links(Path(records["run_dir"])),
    }


def _artifact_links(run_dir: Path) -> list[dict[str, str]]:
    names = [
        "run_manifest.json",
        "experiment_plan.json",
        "experiment_summary.json",
        "run_report.json",
        "provenance.json",
        "episodes.jsonl",
        "partial_diagnostics.json",
        "summary.md",
        "report.html",
    ]
    links = []
    for name in names:
        p = run_dir / name
        if p.is_file():
            links.append({"name": name, "path": str(p)})
        r = run_dir / "report" / name
        if r.is_file():
            links.append({"name": f"report/{name}", "path": str(r)})
    cond = run_dir / "conditions"
    if cond.is_dir():
        for child in sorted(cond.iterdir()):
            rp = child / "run_report.json"
            if rp.is_file():
                links.append({"name": f"conditions/{child.name}/run_report.json", "path": str(rp)})
    return links


def _render_summary_md(view: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Breakpoint report")
    lines.append("")
    lines.append(view["disclaimer"]["text"])
    lines.append("")
    me = view["model_env"]
    lines.append("## Model and environment")
    lines.append(f"- Candidate: `{me.get('candidate')}`")
    lines.append(f"- Hub repo / revision: `{me.get('hub_repo')}` @ `{me.get('hub_revision')}`")
    lines.append(f"- LeRobot commit: `{me.get('lerobot_commit')}`")
    lines.append(f"- Env: `{me.get('env_type')}` / `{me.get('env_task')}`")
    lines.append(f"- Host policy compute device: `{me.get('device_compute')}`")
    lines.append(f"- Latency mode: `{me.get('latency_mode')}`")
    lines.append("")
    man = view["manifest"]
    lines.append("## Run resolution")
    lines.append(f"- Plan id: `{man.get('plan_id')}`")
    lines.append(f"- Episodes: `{man.get('n_episodes')}`")
    lines.append(f"- Output dir: `{man.get('output_dir')}`")
    lines.append(f"- Cache dir: `{man.get('cache_dir')}`")
    lines.append(f"- Device profile: `{man.get('latency_profile_path')}`")
    lines.append("")
    lines.append("## Baseline / episodes (simulated task outcomes)")
    base = view.get("baseline") or view.get("single_run")
    if base is None:
        lines.append("- Baseline: **missing**")
    else:
        if "successes" in base:
            lines.append(
                f"- Baseline successes/eligible: "
                f"{_fmt_rate(base.get('successes'), base.get('eligible'), base.get('success_rate'))}"
            )
            if "wilson_95" in base:
                lines.append(f"- Wilson 95%: {_wilson_str(base.get('wilson_95'))}")
        else:
            lines.append(f"- Baseline record present: `{base.get('condition_id') or 'single_run'}`")
            lines.append(
                f"- Successes/eligible: "
                f"{_fmt_rate(base.get('successes'), base.get('eligible'), base.get('success_rate'))}"
            )
            lines.append(f"- Wilson 95%: {_wilson_str(base.get('wilson_95'))}")
    lines.append("")
    lines.append("## Isolated-axis results")
    iso = view.get("isolated_axis")
    if not iso:
        lines.append("- No isolated-axis experiment_summary.json found.")
    else:
        lines.append(f"- Axis: `{iso.get('axis')}` phase=`{iso.get('phase')}`")
        lines.append(
            f"- Headline suppressed: `{iso.get('headline_suppressed')}` ({iso.get('reason')})"
        )
        bp = iso.get("breakpoint")
        lines.append(f"- Breakpoint: `{json.dumps(bp)}`")
        for p in iso.get("points") or []:
            lines.append(
                f"  - `{p.get('condition_id')}`: "
                f"{_fmt_rate(p.get('successes'), p.get('eligible'), p.get('success_rate'))} "
                f"Wilson {_wilson_str(p.get('wilson_95'))} "
                f"below={p.get('below_threshold')} incomplete={p.get('incomplete')}"
            )
    lines.append("")
    lines.append("## Combined-stress results")
    comb = view.get("combined_stress")
    if not comb:
        lines.append("- No combined-stress summary in this run directory.")
    else:
        lines.append(f"- Headline suppressed: `{comb.get('headline_suppressed')}`")
        lines.append(f"- Breakpoint: `{json.dumps(comb.get('breakpoint'))}`")
    lines.append("")
    lines.append("## Incomplete / unsupported conditions")
    inc = view.get("incomplete_conditions") or []
    if not inc:
        lines.append("- None recorded.")
    else:
        for c in inc:
            lines.append(f"- `{c.get('condition_id')}`: {c.get('incomplete_reasons')}")
    lines.append("")
    lines.append("## Device trace / effective precision")
    dt = view.get("device_trace") or {}
    if not dt.get("present"):
        lines.append("- No hardware latency profile attached (simulated timing only, if any).")
    else:
        lines.append(f"- Profile path: `{dt.get('path')}`")
        prec = dt.get("precision") or {}
        lines.append(f"- Device label: `{prec.get('device_label')}`")
        lines.append(f"- Recipe / variant: `{prec.get('recipe')}` / `{prec.get('variant')}`")
        lines.append(f"- Timed boundary: `{prec.get('timed_boundary')}`")
        lines.append(f"- Measurement summary: `{prec.get('summary')}`")
    lines.append("")
    lines.append("## Controller timing assumptions")
    for item in view.get("controller_timing_assumptions") or []:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Plots")
    for name, p in (view.get("plots") or {}).items():
        if p:
            lines.append(f"- {name}: `{p}`")
        else:
            lines.append(f"- {name}: missing (not fabricated)")
    if view.get("partial_diagnostics"):
        lines.append("")
        lines.append("## Partial diagnostics (interrupted run)")
        lines.append("```json")
        lines.append(json.dumps(view["partial_diagnostics"], indent=2))
        lines.append("```")
    lines.append("")
    lines.append("## Raw artifacts")
    for link in view.get("artifact_links") or []:
        lines.append(f"- [{link['name']}]({link['path']})")
    lines.append("")
    return "\n".join(lines)


def _html_page(view: dict[str, Any]) -> str:
    e = html.escape

    def cell(x: Any) -> str:
        if x is None:
            return "<em>missing</em>"
        return e(str(x))

    me = view["model_env"]
    man = view["manifest"]
    rows = []
    iso = view.get("isolated_axis") or {}
    for p in iso.get("points") or []:
        rows.append(
            "<tr>"
            f"<td>{cell(p.get('condition_id'))}</td>"
            f"<td>{cell(_fmt_rate(p.get('successes'), p.get('eligible'), p.get('success_rate')))}</td>"
            f"<td>{cell(_wilson_str(p.get('wilson_95')))}</td>"
            f"<td>{cell(p.get('below_threshold'))}</td>"
            f"<td>{cell(p.get('incomplete'))}</td>"
            "</tr>"
        )
    plot_imgs = []
    for name, p in (view.get("plots") or {}).items():
        if p:
            rel = Path(p).name
            plot_imgs.append(
                f'<figure><img src="plots/{e(rel)}" alt="{e(name)}" '
                f'style="max-width:100%;height:auto;"/>'
                f"<figcaption>{e(name)}</figcaption></figure>"
            )
        else:
            plot_imgs.append(f"<p><em>{e(name)}: missing (not fabricated)</em></p>")

    links = "".join(
        f'<li><a href="{e(link["name"])}">{e(link["name"])}</a></li>'
        for link in view.get("artifact_links") or []
    )

    bp = iso.get("breakpoint")
    prec = (view.get("device_trace") or {}).get("precision") or {}
    base = view.get("baseline") or view.get("single_run") or {}

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Breakpoint report</title>
<style>
  body {{ font-family: Georgia, serif; margin: 24px; color: {INK}; max-width: 960px; }}
  h1,h2 {{ font-family: Helvetica, Arial, sans-serif; }}
  .banner {{ background: #ffffff; border: 1px solid {LIGHT_RULE}; padding: 12px; margin-bottom: 16px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th, td {{ border: 1px solid {LIGHT_RULE}; padding: 6px 8px; text-align: left; }}
  th {{ background: #eef0ef; }}
  .muted {{ color: {LIGHT_MUTED}; font-size: 13px; }}
  a {{ color: {INK}; text-decoration: underline; }}
  @media print {{ body {{ margin: 12px; }} a {{ color: inherit; text-decoration: underline; }} }}
</style>
</head>
<body>
<h1>Breakpoint report</h1>
<div class="banner muted">{e(view["disclaimer"]["text"])}</div>

<h2>Model / environment</h2>
<ul>
<li>Candidate: <code>{cell(me.get("candidate"))}</code></li>
<li>Hub: <code>{cell(me.get("hub_repo"))}</code> @ <code>{cell(me.get("hub_revision"))}</code></li>
<li>Env: <code>{cell(me.get("env_type"))}</code> / <code>{cell(me.get("env_task"))}</code></li>
<li>Host compute: <code>{cell(me.get("device_compute"))}</code>; latency mode: <code>{cell(me.get("latency_mode"))}</code></li>
</ul>

<h2>Resolved run</h2>
<ul>
<li>Plan: <code>{cell(man.get("plan_id"))}</code>; episodes: <code>{cell(man.get("n_episodes"))}</code></li>
<li>Output: <code>{cell(man.get("output_dir"))}</code>; cache: <code>{cell(man.get("cache_dir"))}</code></li>
<li>Device profile: <code>{cell(man.get("latency_profile_path"))}</code></li>
</ul>

<h2>Baseline (simulated task outcomes)</h2>
<p>{cell(_fmt_rate(base.get("successes"), base.get("eligible"), base.get("success_rate")))}
; Wilson 95%: {cell(_wilson_str(base.get("wilson_95")))}</p>

<h2>Isolated-axis success</h2>
<p class="muted">Axis={cell(iso.get("axis"))}; headline_suppressed={cell(iso.get("headline_suppressed"))}; reason={cell(iso.get("reason"))}</p>
<p>Breakpoint: <code>{cell(json.dumps(bp))}</code></p>
<table>
<thead><tr><th>Condition</th><th>Successes/eligible</th><th>Wilson 95%</th><th>Below</th><th>Incomplete</th></tr></thead>
<tbody>
{"".join(rows) if rows else "<tr><td colspan='5'><em>missing isolated-axis points</em></td></tr>"}
</tbody>
</table>

<h2>Combined-stress</h2>
<p>{cell("present" if view.get("combined_stress") else "No combined-stress summary in this directory.")}</p>

<h2>Device trace / precision</h2>
<ul>
<li>Present: {cell((view.get("device_trace") or {}).get("present"))}</li>
<li>Recipe/variant: {cell(prec.get("recipe"))} / {cell(prec.get("variant"))}</li>
<li>Timed boundary: {cell(prec.get("timed_boundary"))}</li>
<li>Summary: {cell(prec.get("summary"))}</li>
</ul>

<h2>Incomplete conditions</h2>
<ul>
{"".join(f"<li><code>{cell(c.get('condition_id'))}</code>: {cell(c.get('incomplete_reasons'))}</li>" for c in (view.get("incomplete_conditions") or [])) or "<li><em>none</em></li>"}
</ul>

<h2>Controller timing assumptions</h2>
<ul>
{"".join(f"<li>{e(t)}</li>" for t in view.get("controller_timing_assumptions") or [])}
</ul>

<h2>Plots</h2>
{"".join(plot_imgs)}

<h2>Raw artifacts</h2>
<ul>{links or "<li><em>none</em></li>"}</ul>
</body>
</html>
"""


def render_report(run_dir: Path) -> ReportBundle:
    """Write report/summary.md, report/report.html, and report/plots from saved records."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    records = _discover_records(run_dir)
    out_dir = run_dir / "report"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    plots: dict[str, str | None] = {
        "success_by_condition": None,
        "latency_distribution": None,
    }

    summary = records.get("experiment_summary")
    if summary:
        points = (
            (summary.get("axis_summary") or {}).get("points")
            or (summary.get("axis_summary") or {}).get("recipes")
            or []
        )
        labels = [str(p.get("condition_id")) for p in points]
        rates = [p.get("success_rate") for p in points]
        lows = [(p.get("wilson_95") or {}).get("low") for p in points]
        highs = [(p.get("wilson_95") or {}).get("high") for p in points]
        # Missing rates stay missing; the plotting helper marks them.
        try:
            p = _plot_success_bars(
                plots_dir / "success_by_condition.png",
                labels,
                rates,
                lows,
                highs,
                title=f"Success by condition ({summary.get('axis')}) — simulated task outcomes",
            )
            plots["success_by_condition"] = str(p) if p else None
        except ImportError as exc:
            LOGGER.warning("%s", exc)
            plots["success_by_condition"] = None
    elif records.get("single_run"):
        s = records["single_run"].get("summary") or {}
        rate = s.get("success_rate")
        try:
            p = _plot_success_bars(
                plots_dir / "success_by_condition.png",
                ["single_run"],
                [rate],
                [None],
                [None],
                title="Single-run success (simulated task outcome)",
            )
            plots["success_by_condition"] = str(p) if p else None
        except ImportError as exc:
            LOGGER.warning("%s", exc)

    profile = records.get("latency_profile")
    if profile:
        seq = (profile.get("measurements") or {}).get("inference_latency_s") or []
        if seq:
            try:
                p = _plot_latency_hist(
                    plots_dir / "latency_distribution.png",
                    [float(x) for x in seq],
                    title=(
                        f"Hardware-measured ORT latency "
                        f"({profile.get('device_label')}) — not sim task outcomes"
                    ),
                )
                plots["latency_distribution"] = str(p) if p else None
            except ImportError as exc:
                LOGGER.warning("%s", exc)
                plots["latency_distribution"] = None

    view = _build_view(records, plots)
    summary_md = out_dir / "summary.md"
    report_html = out_dir / "report.html"
    # Also place copies at run_dir root for the requested filenames.
    root_md = run_dir / "summary.md"
    root_html = run_dir / "report.html"

    md_text = _render_summary_md(view)
    html_text = _html_page(view)
    summary_md.write_text(md_text, encoding="utf-8")
    report_html.write_text(html_text, encoding="utf-8")
    root_md.write_text(md_text, encoding="utf-8")
    root_html.write_text(html_text, encoding="utf-8")

    # Copy plot references stay under report/plots; root HTML uses report/plots/ via img src.
    # Root HTML currently uses plots/ relative; write a root-level html that points to report/plots.
    root_html.write_text(html_text.replace('src="plots/', 'src="report/plots/'), encoding="utf-8")

    index = {
        "run_dir": str(run_dir),
        "summary_md": str(root_md),
        "report_html": str(root_html),
        "plots": plots,
        "has_experiment_summary": summary is not None,
        "has_single_run": records.get("single_run") is not None,
        "n_condition_dirs": len(records.get("conditions") or {}),
    }
    (out_dir / "report_index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ReportBundle(
        run_dir=run_dir,
        summary_md=root_md,
        report_html=root_html,
        plots_dir=plots_dir,
        index=index,
    )
