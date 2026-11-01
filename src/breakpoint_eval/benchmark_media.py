"""Select, re-record, verify, and compose the comparison media from saved results.

Selection reads packaged episode records for the seed pair and the latency plan's 0 ms provenance
for chunk span. Recording re-runs exactly one baseline and one stress episode through
``run_baseline`` with the saved seeds and effective settings, then checks them against the saved
run before any media is written.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from breakpoint_eval.artifact_names import PROTOCOL_LOCK_NAME, require_artifact
from breakpoint_eval.benchmark_protocol import BACKEND_ONNXRUNTIME, backend_and_variant_from_lock
from breakpoint_eval.benchmark_report_style import (
    ACCENT_RGB,
    DARK_BG,
    DARK_FG_RGB,
    INK_RGB,
    MUTED_RGB,
    hex_to_rgb,
)
from breakpoint_eval.stack import repo_root_from_package

LOGGER = logging.getLogger("breakpoint_eval.benchmark_media")

REPO_ROOT = repo_root_from_package()
CHARCOAL = hex_to_rgb(DARK_BG)
OFF_WHITE = DARK_FG_RGB
BASELINE_BLUE = INK_RGB
STRESS_AMBER = ACCENT_RGB
MUTED = MUTED_RGB
REWARD_REL_TOL = 1e-12
REWARD_ABS_TOL = 1e-12


class MediaVerificationError(RuntimeError):
    """Raised when a re-recorded episode does not match the saved result."""


@dataclass(frozen=True)
class LatencyMediaSpec:
    plan_id: str
    baseline_condition: str
    stress_condition: str
    stress_delay_ms: int
    control_period_s: float
    chunk_span_ms: int
    selection_rule: str


@dataclass
class Recording:
    condition_id: str
    seed: int
    policy_rng_seed: int
    frames: dict[int, np.ndarray]
    episode: dict[str, Any]
    n_steps: int


@dataclass
class PanelSpec:
    title: str
    detail: str
    color: tuple[int, int, int]
    recording: Recording
    event_label: str
    event_time_s: float


def latency_ms_from_condition_id(condition_id: str) -> int:
    prefix = "latency_ms_"
    if not condition_id.startswith(prefix):
        raise ValueError(f"not a latency condition id: {condition_id!r}")
    return int(condition_id[len(prefix) :])


def frame_stride(control_period_s: float, fps: float) -> int:
    steps = round(1.0 / fps / control_period_s)
    if abs(steps * control_period_s * fps - 1.0) > 1e-9:
        raise ValueError(f"fps {fps} does not align with the {control_period_s} s control period")
    return steps


def default_media_fps(control_period_s: float) -> float:
    """Real-time playback fps.

    Prefer stride 4 when that still yields at least 5 fps (ACT 20 ms → 12.5 fps). Otherwise
    stride 2 (PushT 100 ms → 5 fps).
    """
    for stride in (4, 2, 1):
        fps = 1.0 / (stride * control_period_s)
        if fps >= 5.0 - 1e-9:
            return fps
    return 1.0 / control_period_s


def selection_rule_text(stress_delay_ms: int, chunk_span_ms: int) -> str:
    return (
        f"Take the largest tested constant delay strictly below one chunk span "
        f"({chunk_span_ms} ms), here {stress_delay_ms} ms, against the 0 ms baseline of the same "
        "latency plan. Among seeds where the baseline succeeded and the stress episode did not, "
        "use the lowest env seed. No other seeds were run or compared."
    )


def latency_rows_from_package(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latency_rows = [r for r in rows if r.get("axis") == "latency_ms"]
    if not latency_rows:
        latency_rows = [r for r in rows if str(r["condition_id"]).startswith("latency_ms_")]
    return latency_rows


def _latency_plan_delays(latency_rows: list[dict[str, Any]]) -> tuple[str, dict[int, str]]:
    plan_ids = {r["plan_id"] for r in latency_rows}
    if len(plan_ids) != 1:
        raise MediaVerificationError(
            f"expected one latency_ms plan in saved results, found {sorted(plan_ids)}"
        )
    plan_id = next(iter(plan_ids))
    delays = {
        latency_ms_from_condition_id(r["condition_id"]): r["condition_id"]
        for r in latency_rows
        if r["plan_id"] == plan_id
    }
    if 0 not in delays:
        raise MediaVerificationError("latency plan has no 0 ms baseline condition")
    return plan_id, delays


def stress_delay_below_chunk_span(delays: dict[int, str], chunk_span_ms: int) -> int:
    below = [delay for delay in delays if 0 < delay < chunk_span_ms]
    if not below:
        raise MediaVerificationError(
            f"no tested delay is strictly below the {chunk_span_ms} ms chunk span"
        )
    return max(below)


def _baseline_rows(
    rows: list[dict[str, Any]], *, plan_id: str, baseline_condition: str
) -> list[dict[str, Any]]:
    return [r for r in rows if r["plan_id"] == plan_id and r["condition_id"] == baseline_condition]


def _shared_baseline_provenance_path(data_dir: Path) -> Path:
    from breakpoint_eval.benchmark_report_data import find_shared_baseline, load_json

    lock = load_json(require_artifact(data_dir / "protocol", PROTOCOL_LOCK_NAME))
    plans_dir = data_dir / "protocol" / "plans"
    plans = [load_json(path) for path in sorted(plans_dir.glob("*.json"))]
    shared = find_shared_baseline(plans, lock)
    return data_dir / "historical" / shared.plan_id / shared.condition_id / "provenance.json"


def latency_baseline_provenance_path(
    data_dir: Path,
    *,
    plan_id: str,
    baseline_condition: str,
    rows: list[dict[str, Any]],
) -> Path:
    """Provenance file that supplies n_action_steps and control_period_s for chunk span."""
    primary = data_dir / "historical" / plan_id / baseline_condition / "provenance.json"
    if primary.is_file():
        return primary
    baseline = _baseline_rows(rows, plan_id=plan_id, baseline_condition=baseline_condition)
    if baseline and all(bool(r.get("referenced_baseline")) for r in baseline):
        shared = _shared_baseline_provenance_path(data_dir)
        if shared.is_file():
            return shared
        raise MediaVerificationError(f"latency baseline provenance missing: {shared}")
    raise MediaVerificationError(f"latency baseline provenance missing: {primary}")


def chunk_span_from_provenance(
    data_dir: Path,
    *,
    plan_id: str,
    baseline_condition: str,
    rows: list[dict[str, Any]],
) -> tuple[float, int]:
    path = latency_baseline_provenance_path(
        data_dir, plan_id=plan_id, baseline_condition=baseline_condition, rows=rows
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("n_action_steps") is None or payload.get("control_period_s") is None:
        raise MediaVerificationError(
            f"latency baseline provenance lacks n_action_steps or control_period_s: {path}"
        )
    control_period_s = float(payload["control_period_s"])
    n_action_steps = int(payload["n_action_steps"])
    chunk_span_ms = int(round(n_action_steps * control_period_s * 1000.0))
    return control_period_s, chunk_span_ms


def resolve_latency_media(rows: list[dict[str, Any]], *, data_dir: Path) -> LatencyMediaSpec:
    latency_rows = latency_rows_from_package(rows)
    plan_id, delays = _latency_plan_delays(latency_rows)
    control_period_s, chunk_span_ms = chunk_span_from_provenance(
        data_dir, plan_id=plan_id, baseline_condition=delays[0], rows=rows
    )
    stress_delay_ms = stress_delay_below_chunk_span(delays, chunk_span_ms)
    return LatencyMediaSpec(
        plan_id=plan_id,
        baseline_condition=delays[0],
        stress_condition=delays[stress_delay_ms],
        stress_delay_ms=stress_delay_ms,
        control_period_s=control_period_s,
        chunk_span_ms=chunk_span_ms,
        selection_rule=selection_rule_text(stress_delay_ms, chunk_span_ms),
    )


def select_pair(rows: list[dict[str, Any]], *, data_dir: Path) -> dict[str, Any]:
    latency_rows = latency_rows_from_package(rows)
    plan_id, delays = _latency_plan_delays(latency_rows)
    control_period_s, chunk_span_ms = chunk_span_from_provenance(
        data_dir, plan_id=plan_id, baseline_condition=delays[0], rows=rows
    )
    stress_delay_ms = stress_delay_below_chunk_span(delays, chunk_span_ms)
    baseline_condition = delays[0]
    stress_condition = delays[stress_delay_ms]

    def condition_rows(condition: str) -> dict[int, dict[str, Any]]:
        return {
            r["seed"]: r for r in rows if r["plan_id"] == plan_id and r["condition_id"] == condition
        }

    base, stress = condition_rows(baseline_condition), condition_rows(stress_condition)
    lost = sorted(s for s in base if base[s]["success"] and not stress[s]["success"])
    gained = sorted(s for s in base if not base[s]["success"] and stress[s]["success"])
    if not lost:
        raise MediaVerificationError("no baseline-success / stress-failure pair in saved results")
    seed = lost[0]
    return {
        "rule": selection_rule_text(stress_delay_ms, chunk_span_ms),
        "disclosure": "Selected example, not a typical episode. Aggregate results are shown "
        "alongside.",
        "seed": seed,
        "policy_rng_seed": base[seed]["policy_rng_seed"],
        "baseline": base[seed],
        "stress": stress[seed],
        "n_seeds": len(base),
        "seeds_baseline_succeeded_stress_did_not": lost,
        "seeds_baseline_did_not_succeed_stress_did": gained,
        "baseline_successes": sum(r["success"] for r in base.values()),
        "stress_successes": sum(r["success"] for r in stress.values()),
        "plan_id": plan_id,
        "baseline_condition": baseline_condition,
        "stress_condition": stress_condition,
        "stress_delay_ms": stress_delay_ms,
        "chunk_span_ms": chunk_span_ms,
        "control_period_s": control_period_s,
    }


class FrameTap:
    """Collect real frames from the sub-environment, before vector-env autoreset replaces them."""

    def __init__(self, stride: int) -> None:
        self.stride = stride
        self.frames: dict[int, np.ndarray] = {}
        self.step_index = 0
        self.finished = False

    def attach(self, vector_env: Any) -> None:
        inner = vector_env.envs[0]
        original_reset, original_step = inner.reset, inner.step

        def reset(*args: Any, **kwargs: Any) -> Any:
            result = original_reset(*args, **kwargs)
            if not self.finished and self.step_index == 0:
                self.frames[0] = np.asarray(inner.render())
            return result

        def step(action: Any) -> Any:
            result = original_step(action)
            if self.finished:
                return result
            self.step_index += 1
            terminal = bool(result[2] or result[3])
            if terminal or self.step_index % self.stride == 0:
                self.frames[self.step_index] = np.asarray(inner.render())
            self.finished = terminal
            return result

        inner.reset, inner.step = reset, step


@contextmanager
def frame_tap(tap: FrameTap):
    from breakpoint_eval import run_baseline

    original = run_baseline.make_vector_env

    def patched(spec: Any, *, n_envs: int = 1) -> Any:
        env = original(spec, n_envs=n_envs)
        tap.attach(env)
        return env

    run_baseline.make_vector_env = patched
    try:
        yield
    finally:
        run_baseline.make_vector_env = original


def record_episode(
    *,
    plan_path: Path,
    condition_id: str,
    seed: int,
    policy_rng_seed: int,
    host: dict[str, Any],
    work_dir: Path,
    stride: int,
    export_dir: Path,
    backend: str,
) -> Recording:
    from breakpoint_eval.experiment_plan import load_experiment_plan
    from breakpoint_eval.run_baseline import run_baseline
    from breakpoint_eval.run_plan_execute import condition_run_request

    plan = load_experiment_plan(plan_path)
    condition = next(c for c in plan.conditions if c.condition_id == condition_id)
    onnx_model = (export_dir / "act_chunk_fp32.onnx") if backend == BACKEND_ONNXRUNTIME else None
    request = condition_run_request(
        plan=plan,
        condition=condition,
        condition_dir=work_dir / condition_id,
        repo_root=REPO_ROOT,
        device=host["compute_device"],
        onnx_model=onnx_model,
        latency_profile=None,
        save_video=False,
        ort_providers=host.get("ort_providers"),
        ort_intra_op_num_threads=host.get("intra_op_num_threads"),
        ort_inter_op_num_threads=host.get("inter_op_num_threads"),
    )
    request = dataclasses.replace(request, seeds=[seed], policy_rng_seeds=[policy_rng_seed])
    if backend != BACKEND_ONNXRUNTIME:
        torch_threads = host.get("torch_num_threads")
        if torch_threads is not None:
            request = dataclasses.replace(request, torch_threads=int(torch_threads))
    tap = FrameTap(stride)
    with frame_tap(tap):
        report_path = run_baseline(request)
    episode = json.loads(Path(report_path).read_text(encoding="utf-8"))["episodes"][0]
    return Recording(
        condition_id, seed, policy_rng_seed, tap.frames, episode, episode["n_control_steps"]
    )


def load_saved_episode(source_report: str, seed: int) -> dict[str, Any] | None:
    path = REPO_ROOT / source_report
    if not path.exists():
        return None
    return next(
        e for e in json.loads(path.read_text(encoding="utf-8"))["episodes"] if e["seed"] == seed
    )


def compare_with_saved(recording: Recording, saved_row: dict[str, Any]) -> dict[str, Any]:
    ep = recording.episode
    result: dict[str, Any] = {
        "condition_id": recording.condition_id,
        "seed": recording.seed,
        "policy_rng_seed": recording.policy_rng_seed,
        "rerun_outcome": ep["outcome"],
        "saved_outcome": saved_row["outcome"],
        "rerun_n_control_steps": ep["n_control_steps"],
        "saved_n_control_steps": saved_row["n_control_steps"],
        "rerun_n_inference_calls": ep["n_inference_calls"],
        "saved_n_inference_calls": saved_row["n_inference_calls"],
        "rerun_sum_reward": ep["sum_reward"],
        "saved_sum_reward": saved_row["sum_reward"],
    }
    result["matches_compact_record"] = all(
        result[f"rerun_{k}"] == result[f"saved_{k}"]
        for k in ("outcome", "n_control_steps", "n_inference_calls", "sum_reward")
    )
    raw = load_saved_episode(saved_row["source_report"], recording.seed)
    if raw is None:
        result["raw_step_comparison"] = "saved control-step log not available locally"
        return result
    new_steps, old_steps = ep["steps"], raw["steps"]
    same_timeline = len(new_steps) == len(old_steps) and all(
        (a["sim_time_s"], a["is_success"], a["request_time_s"], a["ready_time_s"])
        == (b["sim_time_s"], b["is_success"], b["request_time_s"], b["ready_time_s"])
        for a, b in zip(new_steps, old_steps, strict=False)
    )
    reward_diffs = [
        abs(float(a["reward"]) - float(b["reward"]))
        for a, b in zip(new_steps, old_steps, strict=False)
    ]
    rewards_within_tolerance = len(new_steps) == len(old_steps) and all(
        math.isclose(
            float(a["reward"]),
            float(b["reward"]),
            rel_tol=REWARD_REL_TOL,
            abs_tol=REWARD_ABS_TOL,
        )
        for a, b in zip(new_steps, old_steps, strict=False)
    )
    diffs = [
        np.max(np.abs(np.asarray(a["action"]) - np.asarray(b["action"])))
        for a, b in zip(new_steps, old_steps, strict=False)
    ]
    result.update(
        {
            "raw_step_comparison": "compared against saved control-step log",
            "timeline_success_request_ready_equal": same_timeline,
            "rewards_max_abs_difference": float(max(reward_diffs)) if reward_diffs else None,
            "reward_rel_tol": REWARD_REL_TOL,
            "reward_abs_tol": REWARD_ABS_TOL,
            "rewards_within_tolerance": rewards_within_tolerance,
            "max_abs_action_difference": float(max(diffs)) if diffs else None,
            "actions_bitwise_equal": bool(diffs and max(diffs) == 0.0),
        }
    )
    return result


def require_match(comparison: dict[str, Any]) -> None:
    ok = (
        comparison["matches_compact_record"]
        and comparison.get("timeline_success_request_ready_equal", True)
        and comparison.get("rewards_within_tolerance", True)
        and comparison.get("actions_bitwise_equal", True)
    )
    if not ok:
        raise MediaVerificationError(f"re-recorded episode differs from saved: {comparison}")


def panel_event(recording: Recording) -> tuple[str, float]:
    steps = recording.episode["steps"]
    success = next((s for s in steps if s["is_success"]), None)
    if success is not None:
        return "Success", float(success["sim_time_s"])
    if steps[-1]["truncated"]:
        return "Timeout", float(steps[-1]["sim_time_s"])
    return "Terminated", float(steps[-1]["sim_time_s"])


def font_dir() -> Path:
    import matplotlib

    return Path(matplotlib.get_data_path()) / "fonts" / "ttf"


@dataclass
class Layout:
    width: int
    margin: int
    panel_w: int
    panel_h: int
    header_h: int
    status_h: int
    footer_h: int
    height: int
    scale: float


def make_layout(width: int, frame_hw: tuple[int, int] | None = None) -> Layout:
    scale = width / 1280
    margin = round(16 * scale)
    panel_w = (width - 3 * margin) // 2
    if frame_hw is None:
        panel_h = panel_w * 3 // 4
    else:
        frame_h, frame_w = frame_hw
        panel_h = max(1, panel_w * frame_h // frame_w)
    header_h, status_h, footer_h = round(44 * scale), round(40 * scale), round(62 * scale)
    height = margin + header_h + panel_h + status_h + footer_h + margin
    return Layout(
        width, margin, panel_w, panel_h, header_h, status_h, footer_h, height + height % 2, scale
    )


def fit_frame(frame: np.ndarray, layout: Layout):
    from PIL import Image

    image = Image.fromarray(frame)
    scale = min(layout.panel_w / image.width, layout.panel_h / image.height)
    new_w = max(1, round(image.width * scale))
    new_h = max(1, round(image.height * scale))
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (layout.panel_w, layout.panel_h), CHARCOAL)
    canvas.paste(resized, ((layout.panel_w - new_w) // 2, (layout.panel_h - new_h) // 2))
    return canvas


def compose_frames(
    panels: list[PanelSpec],
    footer_lines: list[str],
    *,
    width: int,
    fps: float,
    hold_frames: int,
    out_dir: Path,
    control_period_s: float,
) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont

    sample = next(iter(panels[0].recording.frames.values()))
    layout = make_layout(width, frame_hw=(int(sample.shape[0]), int(sample.shape[1])))
    fonts = font_dir()

    def px(base: int) -> int:
        return max(15, round(base * layout.scale))

    sans_bold = ImageFont.truetype(str(fonts / "DejaVuSans-Bold.ttf"), px(21))
    sans = ImageFont.truetype(str(fonts / "DejaVuSans.ttf"), px(17))
    mono = ImageFont.truetype(str(fonts / "DejaVuSansMono.ttf"), px(19))
    mono_bold = ImageFont.truetype(str(fonts / "DejaVuSansMono-Bold.ttf"), px(19))
    steps_per_frame = frame_stride(control_period_s, fps)
    last_step = max(p.recording.n_steps for p in panels)
    n_motion = -(-last_step // steps_per_frame) + 1
    xs = [layout.margin + i * (layout.panel_w + layout.margin) for i in range(2)]
    out_dir.mkdir(parents=True, exist_ok=True)
    resized: dict[tuple[int, int], Any] = {}
    used: list[list[int]] = [[], []]
    for j in range(n_motion + hold_frames):
        held = j >= n_motion
        step_index = min(j, n_motion - 1) * steps_per_frame
        canvas = Image.new("RGB", (layout.width, layout.height), CHARCOAL)
        draw = ImageDraw.Draw(canvas)
        for i, panel in enumerate(panels):
            k = min(step_index, panel.recording.n_steps)
            used[i].append(k)
            key = (i, k)
            if key not in resized:
                resized[key] = fit_frame(panel.recording.frames[k], layout)
            x, y = xs[i], layout.margin
            draw.rectangle([x, y, x + layout.panel_w, y + layout.header_h], fill=panel.color)
            draw.text(
                (x + 12, y + layout.header_h // 2),
                panel.title,
                font=sans_bold,
                fill=OFF_WHITE,
                anchor="lm",
            )
            title_w = draw.textlength(panel.title, font=sans_bold)
            draw.text(
                (x + 12 + title_w + 10, y + layout.header_h // 2),
                panel.detail,
                font=sans,
                fill=OFF_WHITE,
                anchor="lm",
            )
            canvas.paste(resized[key], (x, y + layout.header_h))
            sy = y + layout.header_h + layout.panel_h
            clock = f"t = {k * control_period_s:5.2f} s"
            draw.text((x, sy + layout.status_h // 2), clock, font=mono, fill=MUTED, anchor="lm")
            reached = k * control_period_s >= panel.event_time_s - 1e-9
            if reached:
                label = f"{panel.event_label.upper()} {panel.event_time_s:.2f} s"
                draw.text(
                    (x + layout.panel_w, sy + layout.status_h // 2),
                    label,
                    font=mono_bold,
                    fill=OFF_WHITE,
                    anchor="rm",
                )
                tw = draw.textlength(label, font=mono_bold)
                draw.rectangle(
                    [
                        x + layout.panel_w - tw,
                        sy + layout.status_h - 5,
                        x + layout.panel_w,
                        sy + layout.status_h - 2,
                    ],
                    fill=panel.color,
                )
            else:
                draw.text(
                    (x + layout.panel_w, sy + layout.status_h // 2),
                    "running",
                    font=mono,
                    fill=MUTED,
                    anchor="rm",
                )
        fy = layout.margin + layout.header_h + layout.panel_h + layout.status_h + 6
        for n, line in enumerate(footer_lines):
            draw.text(
                (layout.margin, fy + n * px(24)),
                line + (" | final hold" if held and n == 0 else ""),
                font=sans,
                fill=MUTED,
            )
        canvas.save(out_dir / f"frame_{j:04d}.png")
    return {
        "width": layout.width,
        "height": layout.height,
        "n_frames": n_motion + hold_frames,
        "n_motion_frames": n_motion,
        "n_hold_frames": hold_frames,
        "fps": fps,
        "control_steps_per_output_frame": steps_per_frame,
        "source_frame_indices": used,
        "first_source_frames": [u[0] for u in used],
        "last_source_frames": [u[-1] for u in used],
    }


def run_command(command: list[str], log: list[str]) -> None:
    log.append(" ".join(command))
    subprocess.run(command, check=True, capture_output=True)


def encode_media(frames_dir: Path, out_dir: Path, fps: float) -> list[str]:
    commands: list[str] = []
    rate = f"{int(fps * 2)}/2" if fps * 2 == int(fps * 2) else str(fps)
    pattern = str(frames_dir / "frame_%04d.png")
    palette = frames_dir / "palette.png"
    run_command(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            rate,
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(out_dir / "comparison.mp4"),
        ],
        commands,
    )
    run_command(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            rate,
            "-i",
            pattern,
            "-vf",
            "palettegen=stats_mode=diff",
            str(palette),
        ],
        commands,
    )
    run_command(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            rate,
            "-i",
            pattern,
            "-i",
            str(palette),
            "-lavfi",
            "paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
            "-loop",
            "0",
            str(out_dir / "comparison.gif"),
        ],
        commands,
    )
    return commands


def probe_media(out_dir: Path) -> dict[str, Any]:
    from PIL import Image

    probe = json.loads(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,nb_read_frames,codec_name,pix_fmt:format=duration",
                "-of",
                "json",
                str(out_dir / "comparison.mp4"),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    with Image.open(out_dir / "comparison.gif") as gif:
        durations = []
        for index in range(gif.n_frames):
            gif.seek(index)
            durations.append(gif.info["duration"])
        size = gif.size
    return {
        "mp4": {
            **probe["streams"][0],
            "duration_s": float(probe["format"]["duration"]),
            "bytes": (out_dir / "comparison.mp4").stat().st_size,
        },
        "gif": {
            "frames": len(durations),
            "frame_durations_ms": sorted(set(durations)),
            "duration_s": sum(durations) / 1000,
            "size": list(size),
            "bytes": (out_dir / "comparison.gif").stat().st_size,
        },
    }


def social_preview(panels: list[PanelSpec], footer_lines: list[str], out_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    fonts = font_dir()
    title = ImageFont.truetype(str(fonts / "DejaVuSans-Bold.ttf"), 64)
    sub = ImageFont.truetype(str(fonts / "DejaVuSans.ttf"), 24)
    chip = ImageFont.truetype(str(fonts / "DejaVuSans-Bold.ttf"), 20)
    small = ImageFont.truetype(str(fonts / "DejaVuSans.ttf"), 19)
    canvas = Image.new("RGB", (1280, 640), CHARCOAL)
    draw = ImageDraw.Draw(canvas)
    draw.text((90, 22), "Breakpoint", font=title, fill=OFF_WHITE)
    draw.text(
        (90, 96), "Stress-test a pretrained robot policy in simulation.", font=sub, fill=MUTED
    )
    panel_w, panel_h, top = 540, 405, 168
    for i, panel in enumerate(panels):
        x = 90 + i * (panel_w + 20)
        draw.rectangle([x, top - 34, x + panel_w, top], fill=panel.color)
        label = (
            f"{panel.title}  {panel.detail.replace(' inference', '')}  |  "
            f"{panel.event_label} {panel.event_time_s:.2f} s"
        )
        draw.text((x + 10, top - 17), label, font=chip, fill=OFF_WHITE, anchor="lm")
        frame = Image.fromarray(panel.recording.frames[panel.recording.n_steps])
        fitted = Image.new("RGB", (panel_w, panel_h), CHARCOAL)
        scale = min(panel_w / frame.width, panel_h / frame.height)
        new_w, new_h = max(1, round(frame.width * scale)), max(1, round(frame.height * scale))
        fitted.paste(
            frame.resize((new_w, new_h), Image.LANCZOS),
            ((panel_w - new_w) // 2, (panel_h - new_h) // 2),
        )
        canvas.paste(fitted, (x, top))
    for n, line in enumerate(footer_lines):
        draw.text((90, top + panel_h + 10 + n * 26), line, font=small, fill=MUTED)
    canvas.save(out_path)


def build_panels(recordings: dict[str, Recording], *, stress_delay_ms: int) -> list[PanelSpec]:
    baseline_id = min(recordings, key=latency_ms_from_condition_id)
    stress_id = max(recordings, key=latency_ms_from_condition_id)
    specs = [
        (baseline_id, "Baseline, 0 ms delay", "", BASELINE_BLUE),
        (stress_id, f"+{stress_delay_ms} ms delay", "", STRESS_AMBER),
    ]
    panels = []
    for condition, title, detail, color in specs:
        label, event_time = panel_event(recordings[condition])
        panels.append(PanelSpec(title, detail, color, recordings[condition], label, event_time))
    return panels


def footer_lines_for_package(
    *,
    lock: dict[str, Any],
    selection: dict[str, Any],
) -> list[str]:
    delay = f"+{selection['stress_delay_ms']} ms"
    n = selection["n_seeds"]
    aggregate = (
        f"Selected example, not typical. All {n} seeds: baseline "
        f"{selection['baseline_successes']}/{n}, {delay} "
        f"{selection['stress_successes']}/{n} succeeded."
    )
    backend, _ = backend_and_variant_from_lock(lock)
    if backend == BACKEND_ONNXRUNTIME:
        return [
            f"ACT ONNX FP32 | ALOHA TransferCube sim | env seed {selection['seed']} | policy RNG "
            f"{selection['policy_rng_seed']} | aligned by simulation time, real-time speed",
            aggregate,
        ]
    return [f"{aggregate} Simulation footage, not a physical robot."]


def make_media(
    *,
    data_dir: Path,
    out_dir: Path,
    work_dir: Path,
    export_dir: Path,
    width: int,
    fps: float | None,
    hold_frames: int,
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (data_dir / "episodes" / "episodes.jsonl").read_text().splitlines()
        if line
    ]
    lock = json.loads(require_artifact(data_dir / "protocol", PROTOCOL_LOCK_NAME).read_text())
    backend, _ = backend_and_variant_from_lock(lock)
    selection = select_pair(rows, data_dir=data_dir)
    control_period_s = float(selection["control_period_s"])
    if fps is None:
        fps = default_media_fps(control_period_s)
    stride = frame_stride(control_period_s, fps)
    plan_path = data_dir / "protocol" / "plans" / f"{selection['plan_id']}.json"
    recordings, comparisons = {}, []
    for condition, key in (
        (selection["baseline_condition"], "baseline"),
        (selection["stress_condition"], "stress"),
    ):
        LOGGER.info("recording %s seed %s", condition, selection["seed"])
        recordings[condition] = record_episode(
            plan_path=plan_path,
            condition_id=condition,
            seed=selection["seed"],
            policy_rng_seed=selection["policy_rng_seed"],
            host=lock["host"],
            work_dir=work_dir,
            stride=stride,
            export_dir=export_dir,
            backend=backend,
        )
        comparison = compare_with_saved(recordings[condition], selection[key])
        require_match(comparison)
        comparisons.append(comparison)
    panels = build_panels(recordings, stress_delay_ms=selection["stress_delay_ms"])
    footer = footer_lines_for_package(lock=lock, selection=selection)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = work_dir / "composed_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    composed = compose_frames(
        panels,
        footer,
        width=width,
        fps=fps,
        hold_frames=hold_frames,
        out_dir=frames_dir,
        control_period_s=control_period_s,
    )
    commands = encode_media(frames_dir, out_dir, fps)
    if backend == BACKEND_ONNXRUNTIME:
        social_preview(
            panels,
            [footer[1], "Simulation footage, not a physical robot."],
            out_dir / "social-preview.png",
        )
    selection_out = {
        k: v
        for k, v in selection.items()
        if k
        not in (
            "baseline",
            "stress",
            "plan_id",
            "baseline_condition",
            "stress_condition",
            "stress_delay_ms",
            "control_period_s",
        )
    }
    selection_out["baseline_record"] = selection["baseline"]["source_report"]
    selection_out["stress_record"] = selection["stress"]["source_report"]
    verification = {
        "rerun_vs_saved": comparisons,
        "composition": composed,
        "media": probe_media(out_dir),
        "commands": commands,
        "presentation_choices": [
            "Footage is subsampled to every control_steps_per_output_frame-th simulation frame.",
            "A panel holds its real final frame after its episode ends; the baseline holds from "
            "its success time until the stress episode ends.",
            f"{hold_frames} extra frames repeat the last frames at the end (final hold).",
        ],
    }
    (out_dir / "selection.json").write_text(json.dumps(selection_out, indent=2) + "\n")
    (out_dir / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    return verification
