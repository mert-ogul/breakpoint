"""FFmpeg availability, flag validation, and install guidance for optional H.264.

Kept separate from the lightweight core: importing this module does not require
FFmpeg. Call ``probe_ffmpeg_h264`` / ``require_ffmpeg_h264`` before streaming.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any


class H264UnsupportedError(RuntimeError):
    """Raised when H.264 streaming cannot run on this environment."""


# Flags we request for causal streaming. Validated against `ffmpeg -h encoder=libx264`
# when the CLI is present; missing flags raise.
REQUESTED_ENCODER = "libx264"
REQUESTED_PIXEL_FORMAT = "yuv420p"
REQUESTED_TUNE = "zerolatency"
REQUESTED_PRESET_DEFAULT = "veryfast"
# No B-frames / zero lookahead, required to avoid future-frame references.
REQUESTED_MAX_B_FRAMES = 0
REQUESTED_RC_LOOKAHEAD = 0


INSTALL_GUIDANCE = (
    "H.264 streaming requires a verified FFmpeg install with libx264 and the "
    "optional Python binding PyAV. Core Breakpoint does not bundle these.\n"
    "  macOS:  brew install ffmpeg && pip install 'breakpoint-eval[h264]'\n"
    "  Debian: sudo apt-get install -y ffmpeg && pip install 'breakpoint-eval[h264]'\n"
    "Then re-run; unsupported environments are marked, not faked."
)


@dataclass(frozen=True)
class FFmpegProbe:
    available: bool
    ffmpeg_path: str | None
    version_line: str | None
    version: str | None
    has_libx264: bool
    has_h264_decoder: bool
    pyav_available: bool
    pyav_version: str | None
    libav_versions: dict[str, Any] = field(default_factory=dict)
    validated_flags: dict[str, bool] = field(default_factory=dict)
    unsupported_reason: str | None = None
    install_guidance: str = INSTALL_GUIDANCE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run_ffmpeg(args: list[str], *, timeout_s: float = 15.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def _parse_version(version_text: str) -> tuple[str | None, str | None]:
    line = version_text.strip().splitlines()[0] if version_text.strip() else None
    if not line:
        return None, None
    match = re.search(r"ffmpeg version\s+(\S+)", line)
    return line, (match.group(1) if match else None)


def _encoder_help_text(ffmpeg_path: str) -> str:
    proc = _run_ffmpeg([ffmpeg_path, "-hide_banner", "-h", f"encoder={REQUESTED_ENCODER}"])
    # ffmpeg prints help to stdout or stderr depending on build.
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def _generic_video_help(ffmpeg_path: str) -> str:
    """Combine libx264 AVOptions with top-level help (GOP/bitrate are often generic)."""
    specific = _encoder_help_text(ffmpeg_path)
    common = _run_ffmpeg([ffmpeg_path, "-hide_banner", "-h"])
    return specific + "\n" + (common.stdout or "") + "\n" + (common.stderr or "")


def _encoders_list(ffmpeg_path: str) -> str:
    proc = _run_ffmpeg([ffmpeg_path, "-hide_banner", "-encoders"])
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def _decoders_list(ffmpeg_path: str) -> str:
    proc = _run_ffmpeg([ffmpeg_path, "-hide_banner", "-decoders"])
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def _validate_libx264_flags(help_text: str) -> dict[str, bool]:
    lower = help_text.lower()
    # Match option names as FFmpeg documents them for libx264 + generic video.
    return {
        "tune_zerolatency": "zerolatency" in lower or "-tune" in lower,
        "preset": "-preset" in lower or "preset" in lower,
        "bf_or_bframes": (
            "-bf" in lower
            or "b-frames" in lower
            or "bframes" in lower
            or "b_strategy" in lower
            or "weightb" in lower
        ),
        "g_gop": (
            re.search(r"(^|\s)-g\s", help_text, re.M) is not None
            or "group of picture" in lower
            or "gop" in lower
        ),
        "bitrate_b": (
            re.search(r"(^|\s)-b:v\b", help_text, re.M) is not None
            or re.search(r"(^|\s)-b\s", help_text, re.M) is not None
            or "bitrate" in lower
            or "bit rate" in lower
        ),
        "pix_fmt": "pix_fmt" in lower or "pixel format" in lower or "yuv420p" in lower,
        "x264opts_or_x264-params": "x264opts" in lower or "x264-params" in lower,
        "rc_lookahead": "rc-lookahead" in lower,
    }


def _probe_pyav() -> tuple[bool, str | None, dict[str, Any], bool, bool]:
    try:
        import av
    except ImportError:
        return False, None, {}, False, False
    versions = {}
    try:
        versions = dict(av.library_versions)
    except Exception:  # noqa: BLE001
        versions = {}
    has_enc = False
    has_dec = False
    try:
        av.CodecContext.create(REQUESTED_ENCODER, "w")
        has_enc = True
    except Exception:  # noqa: BLE001
        has_enc = False
    try:
        av.CodecContext.create("h264", "r")
        has_dec = True
    except Exception:  # noqa: BLE001
        has_dec = False
    return True, getattr(av, "__version__", None), versions, has_enc, has_dec


def probe_ffmpeg_h264() -> FFmpegProbe:
    """Probe FFmpeg CLI + PyAV without raising. Safe to call from core CLI help paths."""
    pyav_ok, pyav_ver, libav, pyav_enc, pyav_dec = _probe_pyav()
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        reason = "ffmpeg binary not found on PATH"
        if not pyav_ok:
            reason += "; PyAV not installed"
        elif not (pyav_enc and pyav_dec):
            reason += "; PyAV present but libx264/h264 codec unavailable"
        return FFmpegProbe(
            available=False,
            ffmpeg_path=None,
            version_line=None,
            version=None,
            has_libx264=False,
            has_h264_decoder=False,
            pyav_available=pyav_ok,
            pyav_version=pyav_ver,
            libav_versions=libav,
            validated_flags={},
            unsupported_reason=reason,
        )

    try:
        ver_proc = _run_ffmpeg([ffmpeg_path, "-version"])
    except (OSError, subprocess.TimeoutExpired) as exc:
        return FFmpegProbe(
            available=False,
            ffmpeg_path=ffmpeg_path,
            version_line=None,
            version=None,
            has_libx264=False,
            has_h264_decoder=False,
            pyav_available=pyav_ok,
            pyav_version=pyav_ver,
            libav_versions=libav,
            validated_flags={},
            unsupported_reason=f"ffmpeg -version failed: {exc}",
        )

    version_line, version = _parse_version(ver_proc.stdout or ver_proc.stderr or "")
    encoders = _encoders_list(ffmpeg_path)
    decoders = _decoders_list(ffmpeg_path)
    has_libx264 = bool(re.search(r"\bV[\w\.]*\s+libx264\b", encoders)) or "libx264" in encoders
    has_h264_dec = bool(re.search(r"\bV[\w\.]*\s+h264\b", decoders)) or "h264" in decoders
    help_text = _generic_video_help(ffmpeg_path) if has_libx264 else ""
    flags = _validate_libx264_flags(help_text) if help_text else {}

    missing: list[str] = []
    if not has_libx264:
        missing.append("libx264 encoder missing from this FFmpeg build")
    if not has_h264_dec:
        missing.append("h264 decoder missing from this FFmpeg build")
    if not pyav_ok:
        missing.append("PyAV not installed (pip install 'breakpoint-eval[h264]')")
    elif not (pyav_enc and pyav_dec):
        missing.append("PyAV cannot open libx264 encoder and/or h264 decoder")
    # Encoder-specific AVOptions that must appear on libx264 help pages.
    required_flag_keys = ("tune_zerolatency", "preset", "x264opts_or_x264-params", "rc_lookahead")
    if has_libx264 and flags:
        for key in required_flag_keys:
            if not flags.get(key, False):
                missing.append(f"libx264 help missing expected option support: {key}")
    # GOP / bitrate are AVCodecContext properties; confirm via generic help or PyAV.
    if has_libx264 and flags and not flags.get("g_gop", False) and not (pyav_ok and pyav_enc):
        missing.append("GOP (-g) support not found in FFmpeg help and PyAV unavailable")
    if has_libx264 and flags and not flags.get("bitrate_b", False) and not (pyav_ok and pyav_enc):
        missing.append("bitrate support not found in FFmpeg help and PyAV unavailable")

    ok = not missing
    return FFmpegProbe(
        available=ok,
        ffmpeg_path=ffmpeg_path,
        version_line=version_line,
        version=version,
        has_libx264=has_libx264,
        has_h264_decoder=has_h264_dec,
        pyav_available=pyav_ok and pyav_enc and pyav_dec,
        pyav_version=pyav_ver,
        libav_versions=libav,
        validated_flags=flags,
        unsupported_reason=None if ok else "; ".join(missing),
    )


def require_ffmpeg_h264() -> FFmpegProbe:
    probe = probe_ffmpeg_h264()
    if not probe.available:
        detail = probe.unsupported_reason or "H.264 unsupported"
        raise H264UnsupportedError(f"{detail}\n{INSTALL_GUIDANCE}")
    return probe
