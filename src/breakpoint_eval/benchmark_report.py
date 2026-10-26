"""Render the packaged benchmark as local HTML, Markdown, plots, and a data index."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
from pathlib import Path
from typing import Any

from breakpoint_eval import __version__
from breakpoint_eval.artifact_names import (
    EXPORT_VALIDATION_INT8_NAME,
    EXPORT_VALIDATION_NAME,
    PROTOCOL_LOCK_NAME,
    require_artifact,
)
from breakpoint_eval.benchmark_report_data import (
    AxisView,
    ConditionView,
    ProfileView,
    ReportData,
    load_report_data,
)
from breakpoint_eval.benchmark_report_plots import render_all_plots, short_label
from breakpoint_eval.benchmark_report_style import CSS

LOGGER = logging.getLogger("breakpoint_eval.benchmark_report")

GLYPH_SUCCESS = "\u25cf"
GLYPH_FAIL = "\u25cb"
GLYPH_LOST = "\u25bc"
GLYPH_GAINED = "\u25b2"


class ReportValidationError(RuntimeError):
    """Raised when displayed counts do not match the episode records."""


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def wilson_text(view: ConditionView) -> str:
    return f"{pct(view.wilson['low'])} to {pct(view.wilson['high'])}"


def ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f}"


def seed_range(data: ReportData) -> str:
    seeds = sorted({r["seed"] for r in data.rows})
    return f"{seeds[0]}\u2013{seeds[-1]}"


def episodes_per_condition(data: ReportData) -> int:
    return len(data.axes[0].conditions[0].rows)


def cell_glyph(baseline_success: bool | None, success: bool) -> tuple[str, str]:
    if baseline_success is None:
        return (GLYPH_SUCCESS if success else GLYPH_FAIL), "base"
    if baseline_success and not success:
        return GLYPH_LOST, "lost"
    if not baseline_success and success:
        return GLYPH_GAINED, "gained"
    return (GLYPH_SUCCESS if success else GLYPH_FAIL), "same"


def baseline_view(axis: AxisView) -> ConditionView:
    return next(v for v in axis.conditions if v.is_baseline)


def matrix_strings(axis: AxisView) -> list[tuple[ConditionView, list[tuple[str, str, int, str]]]]:
    base = {r["seed"]: r["success"] for r in baseline_view(axis).rows}
    out = []
    for view in axis.conditions:
        cells = []
        for row in view.rows:
            reference = None if view.is_baseline else base[row["seed"]]
            glyph, kind = cell_glyph(reference, row["success"])
            cells.append((glyph, kind, row["seed"], row["outcome"]))
        out.append((view, cells))
    return out


def matrix_html(axis: AxisView) -> str:
    rows = matrix_strings(axis)
    seeds = [c[2] for c in rows[0][1]]
    head = "".join(f'<th scope="col" title="seed {s}">{s % 100:02d}</th>' for s in seeds)
    body = []
    for view, cells in rows:
        tds = "".join(
            f'<td class="{kind}" title="seed {seed}: {esc(outcome)}">{glyph}</td>'
            for glyph, kind, seed, outcome in cells
        )
        body.append(
            f'<tr><th scope="row">{esc(short_label(view.condition_id).replace(chr(10), " "))}'
            f"</th>{tds}</tr>"
        )
    return (
        '<div class="scroll"><table class="matrix"><caption>Per-seed outcomes. Columns are the '
        f"last two digits of env seeds {seeds[0]}\u2013{seeds[-1]}.</caption>"
        f'<thead><tr><th scope="col">condition</th>{head}</tr></thead><tbody>{"".join(body)}'
        "</tbody></table></div>"
        f'<p class="legend">{GLYPH_SUCCESS} success &nbsp; {GLYPH_FAIL} no success (same as '
        f'baseline) &nbsp; <span class="lost">{GLYPH_LOST} baseline succeeded, condition did not'
        f'</span> &nbsp; <span class="gained">{GLYPH_GAINED} baseline did not succeed, condition '
        "did</span></p>"
    )


def status_text(axis: AxisView, view: ConditionView) -> str:
    if view.is_baseline:
        return "baseline"
    parts = ["below threshold" if view.below_threshold else "not below threshold"]
    if view.threshold_inside_interval:
        parts.append("interval spans threshold")
    return "; ".join(parts)


def status_glyph(view: ConditionView) -> str:
    if view.is_baseline:
        return "\u25c6"
    return GLYPH_LOST if view.below_threshold else "\u25cb"


def paired_text(view: ConditionView) -> str:
    if view.paired is None:
        return "reference"
    p = view.paired
    return (
        f"lost {p['baseline_only']} &middot; gained {p['stress_only']} &middot; "
        f"both succeeded {p['both_success']} &middot; neither {p['both_fail']}"
    )


def paired_text_plain(view: ConditionView) -> str:
    return paired_text(view).replace("&middot;", "|")


def contrast_text(view: ConditionView) -> str:
    c = view.saved_contrast
    if not c:
        return "n/a"
    return f"{c['point']:+.3f} [{c['low']:+.3f}, {c['high']:+.3f}]"


def condition_table_html(axis: AxisView) -> str:
    rows = []
    for view in axis.conditions:
        cls = "baseline" if view.is_baseline else "stress"
        rows.append(
            f'<tr class="{cls}"><th scope="row">{esc(view.label)}'
            f"<code>{esc(view.condition_id)}</code></th>"
            f'<td class="num">{view.successes}/{view.eligible}</td>'
            f'<td class="num">{pct(view.successes / view.eligible)}</td>'
            f'<td class="num">{wilson_text(view)}</td>'
            f'<td><span class="mark {cls}">{status_glyph(view)}</span> {esc(status_text(axis, view))}</td>'
            f"<td>{paired_text(view)}</td>"
            f'<td class="num">{esc(contrast_text(view))}</td></tr>'
        )
    return (
        '<div class="scroll"><table class="data"><caption>Success counts with Wilson 95% '
        'intervals</caption><thead><tr><th scope="col">Condition</th><th scope="col">Success'
        '</th><th scope="col">Rate</th><th scope="col">Wilson 95%</th><th scope="col">vs '
        f'{axis.ratio:g} &times; baseline ({pct(axis.threshold_rate)})</th><th scope="col">Paired '
        'with baseline seeds</th><th scope="col">p &minus; '
        f"{axis.ratio:g}&middot;p<sub>base</sub>, paired bootstrap 95%</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def breakpoint_sentence(axis: AxisView) -> str:
    if axis.derived:
        below = [v.condition_id for v in axis.conditions if v.below_threshold]
        return (
            "No saved experiment summary exists for this axis. Counts, intervals, and paired "
            "tables here are derived by this report from the episode records. "
            + (
                f"Point estimate below threshold: {', '.join(below)}."
                if below
                else "No condition is below the threshold."
            )
        )
    bp = axis.saved_breakpoint
    if bp is None:
        return axis.saved_note or ""
    if bp["status"] == "observed_interval":
        interval = bp["observed_interval"]
        return (
            f"Saved summary: first point below threshold is {bp['first_below']}; observed "
            f"interval is between {interval['above_condition_id']} and "
            f"{interval['below_condition_id']}. {axis.saved_note or ''}"
        )
    if bp["status"] == "not_reached_within_tested_range":
        return "Saved summary: no condition fell below the threshold within the tested range."
    return f"Saved summary breakpoint status: {bp['status']}."


def unresolved_conditions(axis: AxisView) -> list[ConditionView]:
    return [v for v in axis.conditions if v.threshold_inside_interval]


def picture_html(name: str, alt: str, plots_rel: str) -> str:
    light, dark = f"{plots_rel}/{name}.svg", f"{plots_rel}/{name}.dark.svg"
    return (
        f'<picture><source media="print" srcset="{light}">'
        f'<source media="(prefers-color-scheme: dark)" srcset="{dark}">'
        f'<img src="{light}" alt="{esc(alt)}" loading="lazy"></picture>'
    )


def evidence_links_html(data: ReportData, links: list[tuple[str, str]], out_dir: Path) -> str:
    items = "".join(
        f'<li><a href="{esc(rel_link(data, out_dir, path))}">{esc(text)}</a></li>'
        for text, path in links
    )
    return f'<ul class="evidence">{items}</ul>'


def rel_link(data: ReportData, out_dir: Path, path: str) -> str:
    return Path(os.path.relpath(data.data_dir / path, out_dir)).as_posix()


def axis_panel_html(data: ReportData, axis: AxisView, out_dir: Path, plots_rel: str) -> str:
    unresolved = unresolved_conditions(axis)
    notes = "".join(f"<li>{esc(n)}</li>" for n in axis.isolation_notes)
    unresolved_text = (
        "Conditions whose Wilson interval includes the threshold (state of the threshold "
        f"comparison is unresolved at n={episodes_per_condition(data)}): "
        + ", ".join(esc(v.condition_id) for v in unresolved)
        if unresolved
        else "No condition interval includes the threshold."
    )
    links = [
        ("Episode records (all axes, compact JSONL)", "episodes/episodes.jsonl"),
        (f"Plan: {axis.plan_id}", f"protocol/plans/{axis.plan_id}.json"),
    ]
    if not axis.derived:
        links.insert(
            1, ("Saved experiment summary", f"summaries/{axis.plan_id}.experiment_summary.json")
        )
    derived = (
        '<p class="callout derived"><strong>Derived axis.</strong> No experiment summary '
        "was saved for the device replay; this report computed its values from episode "
        "records. The baseline is the shared constant 0 ms condition.</p>"
        if axis.derived
        else ""
    )
    return (
        f'<section class="panel" id="panel-{axis.axis}" aria-labelledby="h-{axis.axis}">'
        f'<h2 id="h-{axis.axis}">{esc(axis.title)}</h2>'
        f'<p class="unit">Stress variable: <code>{esc(axis.unit)}</code>. Phase: '
        f"{esc(axis.phase)}.</p>{derived}"
        f'<ul class="plain">{notes}</ul>'
        f"{picture_html(f'success_{axis.axis}', f'Success rate by condition for {axis.title}', plots_rel)}"
        f"{condition_table_html(axis)}"
        f"<p>{esc(breakpoint_sentence(axis))}</p>"
        f"<p>{unresolved_text}</p>"
        f"<h3>Paired outcome matrix</h3>{matrix_html(axis)}"
        f"<h3>Evidence</h3>{evidence_links_html(data, links, out_dir)}</section>"
    )


def stats_row(label: str, stats: dict[str, float], note: str) -> str:
    return (
        f'<tr><th scope="row">{esc(label)}</th><td class="num">{stats["count"]}</td>'
        f'<td class="num">{ms(stats["min_s"])}</td><td class="num">{ms(stats["median_s"])}</td>'
        f'<td class="num">{ms(stats["p95_s"])}</td><td class="num">{ms(stats["p99_s"])}</td>'
        f'<td class="num">{ms(stats["max_s"])}</td><td>{esc(note)}</td></tr>'
    )


def telemetry_html(profile: ProfileView) -> str:
    t = profile.telemetry
    if not t:
        return "<p>No throttling or temperature telemetry was recorded for this laptop profile.</p>"
    return (
        '<p class="callout caution"><strong>Pi thermal state during profiling.</strong> '
        f"Of {t['n_samples']} telemetry samples, {t['throttled_now']} reported throttling active, "
        f"{t['arm_frequency_capped_now']} reported ARM frequency capped, "
        f"{t['soft_temperature_limit_now']} reported the soft temperature limit active, and "
        f"{t['under_voltage_now']} reported under-voltage. Throttle flags at start "
        f"<code>{esc(t['start_flags_hex'])}</code>, at end <code>{esc(t['end_flags_hex'])}</code>. "
        f"Temperature ranged from {t['temperature_min_c']:.1f} to {t['temperature_max_c']:.1f} "
        "&deg;C. Cooling and power-supply notes were not recorded "
        f"(<code>cooling={esc(profile.environment_notes.get('cooling'))}</code>, "
        f"<code>power={esc(profile.environment_notes.get('power'))}</code>). The replayed "
        "latencies describe this throttled state, not a cooled steady state.</p>"
    )


def timing_panel_html(data: ReportData, out_dir: Path, plots_rel: str) -> str:
    blocks = []
    for profile in data.profiles:
        label = profile.device
        blocks.append(
            f"<h3>{esc(label)}</h3>"
            f"{picture_html(f'latency_trace_{label}', f'Latency over sample index for {label}', plots_rel)}"
            '<div class="scroll"><table class="data"><caption>Latency statistics, milliseconds'
            '</caption><thead><tr><th scope="col">Set</th><th scope="col">n</th>'
            '<th scope="col">min</th><th scope="col">median</th><th scope="col">p95</th>'
            '<th scope="col">p99</th><th scope="col">max</th><th scope="col">Meaning</th></tr>'
            "</thead><tbody>"
            + stats_row("full profile", profile.full_stats, "all measured calls")
            + stats_row(
                "replay windows",
                profile.window_stats,
                "union of the 32-sample windows assigned to episodes",
            )
            + stats_row(
                "consumed",
                profile.used_stats,
                "samples actually applied to fresh chunks across 30 episodes",
            )
            + "</tbody></table></div>"
            + telemetry_html(profile)
            + evidence_links_html(
                data,
                [
                    (f"{label} profile (raw timing samples)", profile.profile_path),
                    (
                        f"{label} run metadata and telemetry",
                        f"profiles/{label}/profile_run_meta.json",
                    ),
                ],
                out_dir,
            )
        )
    return (
        '<section class="panel" id="panel-timing" aria-labelledby="h-timing">'
        '<h2 id="h-timing">Device timing</h2>'
        "<p>Each episode replays a contiguous 32-sample window of the measured trace, starting at "
        "<code>floor(i&middot;(N&minus;32)/(n&minus;1))</code>. A 400-step episode requests about "
        "four fresh chunks, so only the first few samples of each window are consumed. Replayed "
        "delays are applied on simulated time; host inference time is never added.</p>"
        + "".join(blocks)
        + "</section>"
    )


def honest_notes(data: ReportData) -> list[str]:
    n = len(data.rows)
    outcomes: dict[str, int] = {}
    for row in data.rows:
        outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
    errors = outcomes.get("runtime_error", 0) + outcomes.get("policy_failed", 0)
    first = data.axes[0]
    int8 = data.lock["artifacts"]["int8_status"]
    notes = [
        f"Errors: {errors} runtime errors or policy failures in {n} episode records. Outcomes: "
        + ", ".join(f"{v} {k}" for k, v in sorted(outcomes.items()))
        + ".",
        f"Unsupported conditions: none skipped. INT8 export status is '{int8}' and the INT8 "
        "graph ran in every planned episode. The INT8 graph has no FP32 numerical-parity "
        "requirement; see the provenance section for its recorded chunk error.",
        f"Baseline strength: {baseline_view(first).successes}/{baseline_view(first).eligible} "
        f"({pct(first.baseline_rate)}) against the project floor of {first.baseline_floor:g}. "
        + (
            "The baseline is not weak by that floor."
            if first.baseline_rate >= first.baseline_floor
            else "The baseline is below the floor; breakpoint claims are not supported."
        ),
        f"Sample size: {episodes_per_condition(data)} paired episodes per condition. Intervals "
        "are wide; the run is exploratory and does not locate an exact boundary.",
    ]
    unresolved = [
        f"{a.axis}: {', '.join(v.condition_id for v in unresolved_conditions(a))}"
        for a in data.axes
        if unresolved_conditions(a)
    ]
    notes.append(
        "Unresolved threshold comparisons (Wilson interval includes the threshold): "
        + "; ".join(unresolved)
        + "."
    )
    resolution = next(a for a in data.axes if a.axis == "resolution")
    if len({v.ordered_value for v in resolution.conditions[1:]}) == 1:
        notes.append(
            "Resolution ordering: the saved summary stored the same ordered value for "
            "every resolution, so conditions are ordered by plan order (native, "
            "240x320, 120x160, 60x80), which is mild to severe."
        )
    rpi = next(p for p in data.profiles if p.device == "rpi5")
    notes.append(
        f"Pi 5 timings were captured while throttling was reported in "
        f"{rpi.telemetry['throttled_now']} of {rpi.telemetry['n_samples']} telemetry samples; "
        "cooling and power were not recorded."
    )
    return notes


def observations(data: ReportData) -> list[str]:
    base = baseline_view(data.axes[0])
    lines = [
        f"Baseline (ONNX FP32, native image, 0 ms delay): {base.successes}/{base.eligible} "
        f"({pct(base.successes / base.eligible)}; Wilson 95% {wilson_text(base)})."
    ]
    for axis in data.axes:
        below = [v for v in axis.conditions if v.below_threshold]
        if below:
            first = below[0]
            tag = " (interval includes threshold)" if first.threshold_inside_interval else ""
            lines.append(
                f"{axis.title}: first point below {axis.ratio:g} x baseline is "
                f"{short_label(first.condition_id).replace(chr(10), ' ')} at "
                f"{first.successes}/{first.eligible}{tag}."
            )
        else:
            lines.append(f"{axis.title}: no condition fell below {axis.ratio:g} x baseline.")
    return lines


def provenance_html(data: ReportData, out_dir: Path) -> str:
    art = data.lock["artifacts"]
    lock = data.lock
    lock_rel = f"protocol/{require_artifact(data.data_dir / 'protocol', PROTOCOL_LOCK_NAME).name}"
    int8_validation = data.validation_files.get(EXPORT_VALIDATION_INT8_NAME, {})
    fp32_validation = data.validation_files.get(EXPORT_VALIDATION_NAME, {})
    runtime = int8_validation.get("runtime_checks", {})
    checks = "".join(
        f'<tr><td class="num">{"pass" if c.passed else "FAIL"}</td><td>{esc(c.name)}</td>'
        f"<td>{esc(c.detail)}</td></tr>"
        for c in data.checks
    )
    large = "".join(
        f'<tr><td><code>{esc(e["path"])}</code></td><td class="num">{e["bytes"]:,}</td>'
        f"<td>{esc(e['role'])}</td><td><code>{e['sha256'][:16]}&hellip;</code></td></tr>"
        for e in data.large_evidence
    )
    return (
        '<h2 id="provenance">Provenance</h2>'
        "<details><summary>Model, runtime, and protocol identities</summary><dl>"
        f"<dt>Policy</dt><dd><code>{esc(lock['candidate'])}</code></dd>"
        f"<dt>FP32 graph sha256</dt><dd><code>{esc(art['fp32_sha256'])}</code></dd>"
        f"<dt>INT8 QDQ graph sha256</dt><dd><code>{esc(art['int8_sha256'])}</code></dd>"
        f"<dt>FP32 export validation</dt><dd>schema {esc(fp32_validation.get('schema_version'))}, "
        f"passed={esc(fp32_validation.get('passed'))}</dd>"
        f"<dt>INT8 recorded chunk error vs FP32</dt><dd>max abs {runtime.get('max_abs_error', 0):.3f}, "
        f"mean abs {runtime.get('mean_abs_error', 0):.4f} (no FP32 allclose required; "
        "full-model INT8 not claimed)</dd>"
        f"<dt>ONNX Runtime</dt><dd>{esc(', '.join(lock['host']['ort_providers']))}; intra-op "
        f"{lock['host']['intra_op_num_threads']}, inter-op {lock['host']['inter_op_num_threads']}; "
        f"simulation host device <code>{esc(lock['host']['compute_device'])}</code></dd>"
        f"<dt>Code revision at run time</dt><dd><code>{esc(lock['source_revision']['revision'])}"
        f"</code>, working tree dirty={esc(lock['source_revision']['dirty'])} (untracked files "
        "only: <code>.cursor/</code>, <code>profiles/</code>)</dd>"
        f"<dt>Chunk arrival</dt><dd><code>{esc(lock['scheduler']['chunk_arrival_policy'])}</code>"
        "; one inference in flight; delay applied once per fresh chunk on simulated time</dd>"
        f"<dt>Seeds</dt><dd>env {seed_range(data)}; policy RNG = env seed + "
        f"{lock['phase_config']['policy_seed_offset']}</dd>"
        f"<dt>Statistics</dt><dd>Wilson 95%; paired bootstrap seed "
        f"{lock['statistics']['bootstrap_seed']}, {lock['statistics']['n_bootstrap']} resamples, "
        "pointwise coverage; threshold ratio "
        f"{lock['statistics']['breakpoint_ratio']:g}, baseline floor "
        f"{lock['statistics']['baseline_success_floor']:g}</dd></dl></details>"
        "<details><summary>Checks run by this report</summary>"
        '<div class="scroll"><table class="data"><thead><tr><th scope="col">Result</th>'
        '<th scope="col">Check</th><th scope="col">Detail</th></tr></thead>'
        f"<tbody>{checks}</tbody></table></div></details>"
        "<details><summary>Large evidence kept outside the repository</summary>"
        "<p>Not packaged. The raw control-step logs stay in the local run directory; the "
        "SHA-256 values identify them.</p>"
        '<div class="scroll"><table class="data"><thead><tr><th scope="col">Path</th>'
        '<th scope="col">Bytes</th><th scope="col">Role</th><th scope="col">sha256</th></tr>'
        f"</thead><tbody>{large}</tbody></table></div></details>"
        "<details><summary>Historical paths</summary><p>The verbatim protocol, summary, and "
        "provenance files keep the absolute paths of the machine that produced them. Links in "
        "this report are relative.</p>"
        f'<ul class="evidence"><li><a href="{esc(rel_link(data, out_dir, lock_rel))}">'
        "Protocol lock</a></li>"
        f'<li><a href="{esc(rel_link(data, out_dir, "manifest.json"))}">Data manifest</a></li>'
        f'<li><a href="{esc(rel_link(data, out_dir, "large_evidence.json"))}">Large evidence index'
        "</a></li></ul></details>"
    )


def tab_css(axes_ids: list[str]) -> str:
    rules = []
    for tab in axes_ids:
        rules.append(
            f"#tab-{tab}:checked ~ #panel-{tab}{{display:block}}"
            f"#tab-{tab}:checked ~ .tablist label[for=tab-{tab}]{{border-bottom-color:var(--fg);"
            "font-weight:600}"
            f"#tab-{tab}:focus-visible ~ .tablist label[for=tab-{tab}]{{outline:2px solid var(--fg);"
            "outline-offset:2px}"
        )
    return "".join(rules)


def render_html(data: ReportData, out_dir: Path, plots_rel: str) -> str:
    tabs = [(a.axis, a.title) for a in data.axes] + [("timing", "Device timing")]
    radios = "".join(
        f'<input type="radio" name="view" id="tab-{key}"{" checked" if i == 0 else ""}>'
        for i, (key, _) in enumerate(tabs)
    )
    labels = "".join(f'<label for="tab-{key}">{esc(title)}</label>' for key, title in tabs)
    panels = "".join(axis_panel_html(data, a, out_dir, plots_rel) for a in data.axes)
    panels += timing_panel_html(data, out_dir, plots_rel)
    obs = "".join(f"<li>{esc(line)}</li>" for line in observations(data))
    notes = "".join(f"<li>{esc(line)}</li>" for line in honest_notes(data))
    summary = (
        f"{len(data.rows)} episode records &middot; {data.n_runs} simulated runs "
        f"({sum(len(a.conditions) for a in data.axes)} condition rows; the baseline is reused by "
        f"reference) &middot; {episodes_per_condition(data)} paired episodes per condition "
        f"&middot; env seeds {seed_range(data)}"
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="light dark">'
        "<title>ACT deployment stress benchmark: cube transfer</title>"
        f"<style>{CSS}{tab_css([k for k, _ in tabs])}</style></head><body><main>"
        '<header><p class="eyebrow">Breakpoint &middot; ACT deployment stress benchmark &middot; exploratory</p>'
        "<h1>ACT cube transfer under isolated stress</h1>"
        '<p class="lead">One pretrained ACT policy (<code>lerobot/act_aloha_sim_transfer_cube_human'
        "</code>) in the ALOHA TransferCube simulation, run as ONNX Runtime on CPU. Four isolated "
        "stress axes and a device latency replay, with paired seeds. Simulation only: nothing here "
        "describes a physical robot.</p>"
        f'<p class="facts">{summary}</p></header>'
        '<section aria-labelledby="h-obs"><h2 id="h-obs">What was measured</h2>'
        f'<ul class="plain">{obs}</ul></section>'
        '<section aria-labelledby="h-caveats"><h2 id="h-caveats">Read before the numbers</h2>'
        f'<ul class="plain caveats">{notes}</ul></section>'
        f'<div class="tabs">{radios}<div class="tablist" role="presentation">{labels}</div>'
        f"{panels}</div>{provenance_html(data, out_dir)}"
        f"<footer><p>Generated by <code>scripts/render_benchmark.py</code> (breakpoint-eval {__version__}) "
        f"from packaged data with manifest sha256 <code>{data.manifest_sha256[:16]}&hellip;</code>."
        "</p></footer></main></body></html>"
    )


def render_markdown(data: ReportData, out_dir: Path, plots_rel: str) -> str:
    lines = [
        "# ACT deployment stress benchmark: cube transfer under isolated stress",
        "",
        "Simulation only. One pretrained ACT policy, ONNX Runtime on CPU, paired seeds "
        f"{seed_range(data)}, {episodes_per_condition(data)} episodes per condition. "
        "Exploratory run.",
        "",
        "## What was measured",
        "",
        *[f"- {line}" for line in observations(data)],
        "",
        "## Read before the numbers",
        "",
        *[f"- {line}" for line in honest_notes(data)],
        "",
    ]
    for axis in data.axes:
        lines += [
            f"## {axis.title}",
            "",
            f"Stress variable: `{axis.unit}`.",
            "",
            f"![{axis.title}]({plots_rel}/success_{axis.axis}.png)",
            "",
            "| Condition | Success | Rate | Wilson 95% | Status | Paired with baseline |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
        for v in axis.conditions:
            lines.append(
                f"| `{v.condition_id}` | {v.successes}/{v.eligible} | "
                f"{pct(v.successes / v.eligible)} | {wilson_text(v)} | "
                f"{status_glyph(v)} {status_text(axis, v)} | {paired_text_plain(v)} |"
            )
        lines += [
            "",
            breakpoint_sentence(axis),
            "",
            "Per-seed outcomes "
            f"({GLYPH_SUCCESS} success, {GLYPH_FAIL} none, {GLYPH_LOST} lost, "
            f"{GLYPH_GAINED} gained):",
            "",
            "```",
        ]
        for view, cells in matrix_strings(axis):
            lines.append(f"{view.condition_id:<32}" + "".join(c[0] for c in cells))
        lines += ["```", ""]
    lines += ["## Device timing", ""]
    for p in data.profiles:
        lines += [
            f"### {p.device}",
            "",
            f"![{p.device} latency]({plots_rel}/latency_trace_{p.device}.png)",
            "",
            "| Set | n | median ms | p95 ms | p99 ms |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name, s in (
            ("full profile", p.full_stats),
            ("replay windows", p.window_stats),
            ("consumed", p.used_stats),
        ):
            lines.append(
                f"| {name} | {s['count']} | {ms(s['median_s'])} | {ms(s['p95_s'])} | "
                f"{ms(s['p99_s'])} |"
            )
        lines.append("")
    lines += [
        "## Checks",
        "",
        *[f"- {'pass' if c.passed else 'FAIL'}: {c.name} ({c.detail})" for c in data.checks],
        "",
    ]
    return "\n".join(lines)


def displayed_counts(data: ReportData) -> list[dict[str, Any]]:
    return [
        {
            "axis": a.axis,
            "plan_id": a.plan_id,
            "condition_id": v.condition_id,
            "successes": v.successes,
            "eligible": v.eligible,
            "runtime_errors": v.n_runtime_error,
            "planned": len(v.rows),
            "wilson_low": v.wilson["low"],
            "wilson_high": v.wilson["high"],
            "derived_by_report": a.derived,
        }
        for a in data.axes
        for v in a.conditions
    ]


def write_data_index(data: ReportData, out_dir: Path, written: list[Path]) -> Path:
    files = [
        {
            "path": p.relative_to(out_dir).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in sorted(written)
    ]
    payload = {
        "schema_version": 1,
        "generator": "scripts/render_benchmark.py",
        "breakpoint_eval_version": __version__,
        "input": {
            "manifest_sha256": data.manifest_sha256,
            "episode_rows": len(data.rows),
            "simulated_runs": data.n_runs,
        },
        "validation": [
            {"name": c.name, "passed": c.passed, "detail": c.detail} for c in data.checks
        ],
        "displayed_counts": displayed_counts(data),
        "derived_by_report": [
            "device_profile axis counts and paired tables",
            "threshold-inside-interval flags",
            "timing statistics",
            "Pi telemetry summary",
        ],
        "files": files,
    }
    path = out_dir / "data_index.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def render_report(input_dir: Path, out_dir: Path) -> list[Path]:
    data = load_report_data(input_dir)
    failed = [c for c in data.checks if not c.passed]
    if failed:
        raise ReportValidationError("; ".join(f"{c.name}: {c.detail}" for c in failed))
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    render_all_plots(data.axes, data.profiles, plots_dir)
    written = sorted(plots_dir.glob("*"))
    html_path = out_dir / "index.html"
    md_path = out_dir / "report.md"
    html_path.write_text(render_html(data, out_dir, "plots"), encoding="utf-8")
    md_path.write_text(render_markdown(data, out_dir, "plots"), encoding="utf-8")
    written += [html_path, md_path]
    written.append(write_data_index(data, out_dir, written))
    LOGGER.info("wrote %d files to %s", len(written), out_dir)
    return written
