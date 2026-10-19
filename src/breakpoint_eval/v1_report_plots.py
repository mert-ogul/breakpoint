"""Matplotlib figures for the v0.1 report (light and dark variants; SVG and PNG)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from breakpoint_eval.v1_report_data import AxisView, ProfileView  # noqa: E402

LOGGER = logging.getLogger("breakpoint_eval.v1_report_plots")

CHARCOAL = "#111827"
OFF_WHITE = "#F8FAFC"
BASELINE_BLUE = "#2563EB"
STRESS_AMBER = "#B45309"
SANS_FAMILIES = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


@dataclass(frozen=True)
class Theme:
    name: str
    background: str
    foreground: str
    muted: str
    rule: str


THEMES = (
    Theme("light", OFF_WHITE, CHARCOAL, "#4B5563", "#CBD5E1"),
    Theme("dark", CHARCOAL, OFF_WHITE, "#9CA3AF", "#374151"),
)


def short_label(condition_id: str) -> str:
    if condition_id.startswith("latency_ms_"):
        return f"{condition_id.rsplit('_', 1)[1]} ms"
    if condition_id == "resolution_native":
        return "480x640\nnative"
    if condition_id.startswith("resolution_capture_"):
        return condition_id.rsplit("_", 1)[1]
    if condition_id == "jpeg_bypass":
        return "bypass"
    if condition_id.startswith("jpeg_q"):
        return f"q{condition_id[6:]}"
    if condition_id == "precision_onnx_fp32":
        return "FP32"
    if condition_id == "precision_int8":
        return "INT8 QDQ"
    if condition_id.startswith("device_profile_"):
        return condition_id.removeprefix("device_profile_").removesuffix("_fp32") + "\nreplay"
    return condition_id


def apply_theme(theme: Theme) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": SANS_FAMILIES,
        "svg.fonttype": "none",
        "svg.hashsalt": "breakpoint-v1",
        "figure.facecolor": theme.background,
        "axes.facecolor": theme.background,
        "savefig.facecolor": theme.background,
        "axes.edgecolor": theme.rule,
        "axes.labelcolor": theme.foreground,
        "text.color": theme.foreground,
        "xtick.color": theme.muted,
        "ytick.color": theme.muted,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })


def style_axes(ax: plt.Axes, theme: Theme) -> None:
    ax.grid(axis="y", color=theme.rule, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)


def success_figure(axis: AxisView, theme: Theme) -> plt.Figure:
    apply_theme(theme)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    style_axes(ax, theme)
    n = len(axis.conditions)
    ax.axhline(axis.threshold_rate, color=theme.muted, linestyle=(0, (4, 3)), linewidth=1.0)
    for x, view in enumerate(axis.conditions):
        rate = view.successes / view.eligible
        yerr = [[rate - view.wilson["low"]], [view.wilson["high"] - rate]]
        color = BASELINE_BLUE if view.is_baseline else STRESS_AMBER
        ax.errorbar(x, rate, yerr=yerr, fmt="none", ecolor=color, elinewidth=1.4, capsize=4)
        ax.plot(x, rate, marker="o" if view.is_baseline else "s", markersize=9, color=color,
                markerfacecolor=color if (view.is_baseline or view.below_threshold)
                else theme.background, markeredgewidth=1.8, linestyle="none")
        ax.annotate(f"{view.successes}/{view.eligible}", (x, view.wilson["high"]),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9,
                    color=theme.foreground)
    ax.set_xticks(range(n), [short_label(v.condition_id) for v in axis.conditions])
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("episodes succeeded (Wilson 95%)")
    ax.set_xlabel(axis.unit, fontsize=8.5, color=theme.muted)
    handles = [
        plt.Line2D([], [], marker="o", color=BASELINE_BLUE, linestyle="none", markersize=8,
                   label="baseline"),
        plt.Line2D([], [], marker="s", color=STRESS_AMBER, markerfacecolor=theme.background,
                   markeredgewidth=1.6, linestyle="none", markersize=8,
                   label="stress, estimate not below threshold"),
        plt.Line2D([], [], marker="s", color=STRESS_AMBER, linestyle="none", markersize=8,
                   label="stress, estimate below threshold"),
        plt.Line2D([], [], color=theme.muted, linestyle=(0, (4, 3)),
                   label=f"threshold: {axis.ratio:g} x baseline = {100 * axis.threshold_rate:.1f}%"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
              frameon=False, fontsize=8.5)
    fig.tight_layout()
    return fig


def trace_figure(profile: ProfileView, theme: Theme) -> plt.Figure:
    apply_theme(theme)
    fig, (ax, strip) = plt.subplots(
        2, 1, figsize=(7.2, 5.2), sharex=True, gridspec_kw={"height_ratios": [7, 1], "hspace": 0.08}
    )
    style_axes(ax, theme)
    ms = profile.samples_s * 1000.0
    length = len(profile.samples_s)
    ax.plot(np.arange(length), ms, color=theme.muted, linewidth=0.7)
    consumed = np.asarray(profile.consumed_indices)
    ax.plot(consumed, ms[consumed], linestyle="none", marker="s", markersize=3.6,
            color=STRESS_AMBER)
    median_ms = profile.full_stats["median_s"] * 1000.0
    ax.axhline(median_ms, color=theme.foreground, linestyle=(0, (4, 3)), linewidth=0.9)
    ax.set_ylabel("chunk latency (ms)")
    ax.tick_params(labelbottom=False)
    for k, start in enumerate(profile.window_starts):
        strip.add_patch(plt.Rectangle((start, 0), 32, 1, color=STRESS_AMBER,
                                      alpha=0.28 if k % 2 == 0 else 0.5, linewidth=0))
    for index in consumed:
        strip.add_patch(plt.Rectangle((index, 0), 1, 1, color=STRESS_AMBER, linewidth=0))
    strip.set_ylim(0, 1)
    strip.set_yticks([])
    strip.spines["left"].set_visible(False)
    strip.set_xlim(0, length)
    strip.set_xlabel("profile sample index (after 50 discarded warmups)")
    handles = [
        plt.Line2D([], [], color=theme.muted, linewidth=1.2, label="all 1000 profile samples"),
        plt.Line2D([], [], color=theme.foreground, linestyle=(0, (4, 3)),
                   label=f"full-profile median {median_ms:.1f} ms"),
        plt.Line2D([], [], color=STRESS_AMBER, marker="s", linestyle="none", markersize=4,
                   label="samples consumed by episodes (also solid in the strip)"),
        plt.Rectangle((0, 0), 1, 1, color=STRESS_AMBER, alpha=0.4,
                      label="strip: one 32-sample window per episode"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.34)
    return fig


def save_figure(fig: plt.Figure, out_dir: Path, name: str, theme: Theme) -> list[Path]:
    suffix = "" if theme.name == "light" else ".dark"
    paths = []
    for ext in ("svg", "png") if theme.name == "light" else ("svg",):
        path = out_dir / f"{name}{suffix}.{ext}"
        kwargs = {"metadata": {"Date": None}} if ext == "svg" else {"dpi": 160}
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def render_all_plots(
    axes: list[AxisView], profiles: list[ProfileView], out_dir: Path
) -> dict[str, str]:
    """Write every figure in both themes; returns figure name -> base file stem."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stems: dict[str, str] = {}
    for theme in THEMES:
        for axis in axes:
            name = f"success_{axis.axis}"
            save_figure(success_figure(axis, theme), out_dir, name, theme)
            stems[name] = name
        for profile in profiles:
            name = f"latency_trace_{profile.device}"
            save_figure(trace_figure(profile, theme), out_dir, name, theme)
            stems[name] = name
    LOGGER.info("wrote %d figures (light svg+png, dark svg)", len(stems))
    return stems
