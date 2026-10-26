"""Select, re-record, verify, and compose the comparison media from saved results.

Selection reads only packaged episode records. Recording re-runs exactly one baseline and one
stress episode through ``run_baseline`` with the saved seeds and effective settings, then checks
them against the saved run before any media is written.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from breakpoint_eval.artifact_names import PROTOCOL_LOCK_NAME, require_artifact
from breakpoint_eval.stack import repo_root_from_package

LOGGER = logging.getLogger("breakpoint_eval.benchmark_media")

REPO_ROOT = repo_root_from_package()
EXPORT_DIR = REPO_ROOT / "artifacts" / "export" / "act-v1"
BASELINE_CONDITION = "latency_ms_0"
STRESS_CONDITION = "latency_ms_640"
LATENCY_PLAN = "v0.1_full_latency"
SELECTION_RULE = (
    "Take the most severe tested constant delay (640 ms) against the 0 ms baseline of the same "
    "latency plan. Among seeds where the baseline succeeded and the stress episode did not, "
    "use the lowest env seed. No other seeds were run or compared."
)
CHARCOAL, OFF_WHITE = (17, 24, 39), (248, 250, 252)
BASELINE_BLUE, STRESS_AMBER = (37, 99, 235), (180, 83, 9)
MUTED = (156, 163, 175)
CONTROL_PERIOD_S = 0.02


class MediaVerificationError(RuntimeError):
    """Raised when a re-recorded episode does not match the saved result."""


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


def select_pair(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def condition_rows(condition: str) -> dict[int, dict[str, Any]]:
        return {
            r["seed"]: r
            for r in rows
            if r["plan_id"] == LATENCY_PLAN and r["condition_id"] == condition
        }

    base, stress = condition_rows(BASELINE_CONDITION), condition_rows(STRESS_CONDITION)
    lost = sorted(s for s in base if base[s]["success"] and not stress[s]["success"])
    gained = sorted(s for s in base if not base[s]["success"] and stress[s]["success"])
    if not lost:
        raise MediaVerificationError("no baseline-success / stress-failure pair in saved results")
    seed = lost[0]
    return {
        "rule": SELECTION_RULE,
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
) -> Recording:
    from breakpoint_eval.experiment_plan import load_experiment_plan
    from breakpoint_eval.run_baseline import run_baseline
    from breakpoint_eval.run_plan_execute import _condition_run_request

    plan = load_experiment_plan(plan_path)
    condition = next(c for c in plan.conditions if c.condition_id == condition_id)
    request = _condition_run_request(
        plan=plan,
        condition=condition,
        condition_dir=work_dir / condition_id,
        repo_root=REPO_ROOT,
        device=host["compute_device"],
        onnx_model=EXPORT_DIR / "act_chunk_fp32.onnx",
        latency_profile=None,
        save_video=False,
        ort_providers=host["ort_providers"],
        ort_intra_op_num_threads=host["intra_op_num_threads"],
        ort_inter_op_num_threads=host["inter_op_num_threads"],
    )
    request = dataclasses.replace(request, seeds=[seed], policy_rng_seeds=[policy_rng_seed])
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
        (a["sim_time_s"], a["reward"], a["is_success"], a["request_time_s"], a["ready_time_s"])
        == (b["sim_time_s"], b["reward"], b["is_success"], b["request_time_s"], b["ready_time_s"])
        for a, b in zip(new_steps, old_steps, strict=False)
    )
    diffs = [
        np.max(np.abs(np.asarray(a["action"]) - np.asarray(b["action"])))
        for a, b in zip(new_steps, old_steps, strict=False)
    ]
    result.update(
        {
            "raw_step_comparison": "compared against saved control-step log",
            "timeline_reward_success_request_ready_equal": same_timeline,
            "max_abs_action_difference": float(max(diffs)) if diffs else None,
            "actions_bitwise_equal": bool(diffs and max(diffs) == 0.0),
        }
    )
    return result


def require_match(comparison: dict[str, Any]) -> None:
    ok = comparison["matches_compact_record"] and comparison.get(
        "timeline_reward_success_request_ready_equal", True
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


def make_layout(width: int) -> Layout:
    scale = width / 1280
    margin = round(16 * scale)
    panel_w = (width - 3 * margin) // 2
    panel_h = panel_w * 3 // 4
    header_h, status_h, footer_h = round(44 * scale), round(40 * scale), round(62 * scale)
    height = margin + header_h + panel_h + status_h + footer_h + margin
    return Layout(
        width, margin, panel_w, panel_h, header_h, status_h, footer_h, height + height % 2, scale
    )


def fit_frame(frame: np.ndarray, layout: Layout):
    from PIL import Image

    return Image.fromarray(frame).resize((layout.panel_w, layout.panel_h), Image.LANCZOS)


def compose_frames(
    panels: list[PanelSpec],
    footer_lines: list[str],
    *,
    width: int,
    fps: float,
    hold_frames: int,
    out_dir: Path,
) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont

    layout = make_layout(width)
    fonts = font_dir()

    def px(base: int) -> int:
        return max(15, round(base * layout.scale))

    sans_bold = ImageFont.truetype(str(fonts / "DejaVuSans-Bold.ttf"), px(21))
    sans = ImageFont.truetype(str(fonts / "DejaVuSans.ttf"), px(17))
    mono = ImageFont.truetype(str(fonts / "DejaVuSansMono.ttf"), px(19))
    mono_bold = ImageFont.truetype(str(fonts / "DejaVuSansMono-Bold.ttf"), px(19))
    steps_per_frame = round(1.0 / fps / CONTROL_PERIOD_S)
    if abs(steps_per_frame * CONTROL_PERIOD_S * fps - 1.0) > 1e-9:
        raise ValueError(f"fps {fps} does not align with the {CONTROL_PERIOD_S} s control period")
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
            clock = f"t = {k * CONTROL_PERIOD_S:5.2f} s"
            draw.text((x, sy + layout.status_h // 2), clock, font=mono, fill=MUTED, anchor="lm")
            reached = k * CONTROL_PERIOD_S >= panel.event_time_s - 1e-9
            if reached:
                glyph = "\u2713" if panel.event_label == "Success" else "\u2715"
                label = f"{glyph} {panel.event_label.upper()} {panel.event_time_s:.2f} s"
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
        glyph = "\u2713" if panel.event_label == "Success" else "\u2715"
        label = (
            f"{panel.title}  {panel.detail.replace(' inference', '')}  |  {glyph} "
            f"{panel.event_label} {panel.event_time_s:.2f} s"
        )
        draw.text((x + 10, top - 17), label, font=chip, fill=OFF_WHITE, anchor="lm")
        frame = Image.fromarray(panel.recording.frames[panel.recording.n_steps])
        canvas.paste(frame.resize((panel_w, panel_h), Image.LANCZOS), (x, top))
    for n, line in enumerate(footer_lines):
        draw.text((90, top + panel_h + 10 + n * 26), line, font=small, fill=MUTED)
    canvas.save(out_path)


def build_panels(recordings: dict[str, Recording]) -> list[PanelSpec]:
    specs = [
        (BASELINE_CONDITION, "BASELINE", "0 ms delay", BASELINE_BLUE),
        (STRESS_CONDITION, "STRESS", "+640 ms inference delay", STRESS_AMBER),
    ]
    panels = []
    for condition, title, detail, color in specs:
        label, event_time = panel_event(recordings[condition])
        panels.append(PanelSpec(title, detail, color, recordings[condition], label, event_time))
    return panels


def make_media(
    *,
    data_dir: Path,
    out_dir: Path,
    work_dir: Path,
    width: int,
    fps: float,
    hold_frames: int,
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (data_dir / "episodes" / "episodes.jsonl").read_text().splitlines()
        if line
    ]
    lock = json.loads(require_artifact(data_dir / "protocol", PROTOCOL_LOCK_NAME).read_text())
    selection = select_pair(rows)
    stride = round(1.0 / fps / CONTROL_PERIOD_S)
    plan_path = data_dir / "protocol" / "plans" / f"{LATENCY_PLAN}.json"
    recordings, comparisons = {}, []
    for condition, key in ((BASELINE_CONDITION, "baseline"), (STRESS_CONDITION, "stress")):
        LOGGER.info("recording %s seed %s", condition, selection["seed"])
        recordings[condition] = record_episode(
            plan_path=plan_path,
            condition_id=condition,
            seed=selection["seed"],
            policy_rng_seed=selection["policy_rng_seed"],
            host=lock["host"],
            work_dir=work_dir,
            stride=stride,
        )
        comparison = compare_with_saved(recordings[condition], selection[key])
        require_match(comparison)
        comparisons.append(comparison)
    panels = build_panels(recordings)
    footer = [
        f"ACT ONNX FP32 | ALOHA TransferCube sim | env seed {selection['seed']} | policy RNG "
        f"{selection['policy_rng_seed']} | aligned by simulation time, real-time speed",
        f"Selected example, not typical. All {selection['n_seeds']} seeds: baseline "
        f"{selection['baseline_successes']}/{selection['n_seeds']}, +640 ms "
        f"{selection['stress_successes']}/{selection['n_seeds']} succeeded.",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = work_dir / "composed_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    composed = compose_frames(
        panels, footer, width=width, fps=fps, hold_frames=hold_frames, out_dir=frames_dir
    )
    commands = encode_media(frames_dir, out_dir, fps)
    social_preview(
        panels,
        [footer[1], "Simulation footage, not a physical robot."],
        out_dir / "social-preview.png",
    )
    selection_out = {k: v for k, v in selection.items() if k not in ("baseline", "stress")}
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
