"""Camera degradation at the raw rendered-image boundary (before policy norms).

Operates on uint8 HWC RGB frames as produced by gym envs. Does not add
encoding/network delay. JPEG uses a real codec round-trip (OpenCV).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

# Documented resize kernels (OpenCV).
DOWNSAMPLE_INTERPOLATION = "INTER_AREA"
UPSAMPLE_INTERPOLATION = "INTER_LINEAR"


class ImageMode(StrEnum):
    NATIVE = "native"  # uncompressed identity bypass (no codec, no resize)
    DOWNSAMPLE = "downsample"  # capture res → resize back to policy native
    JPEG = "jpeg"  # encode/decode at native resolution


@dataclass(frozen=True)
class ImageDegradeConfig:
    mode: ImageMode = ImageMode.NATIVE
    # Capture resolution for downsample mode (height, width).
    capture_height: int | None = None
    capture_width: int | None = None
    # JPEG quality in [1, 100]. Quality 100 is still encoded (not identity).
    jpeg_quality: int | None = None
    # Control / camera frame rate used only for optional bandwidth estimates.
    frame_rate_hz: float | None = None

    def __post_init__(self) -> None:
        if self.mode is ImageMode.DOWNSAMPLE:
            if self.capture_height is None or self.capture_width is None:
                raise ValueError("downsample mode requires capture_height and capture_width")
            if self.capture_height < 1 or self.capture_width < 1:
                raise ValueError("capture dimensions must be >= 1")
        if self.mode is ImageMode.JPEG:
            if self.jpeg_quality is None:
                raise ValueError("jpeg mode requires jpeg_quality")
            if not 1 <= self.jpeg_quality <= 100:
                raise ValueError("jpeg_quality must be in [1, 100]")
        if self.frame_rate_hz is not None and self.frame_rate_hz <= 0:
            raise ValueError("frame_rate_hz must be > 0 when set")


@dataclass
class FrameDegradeStats:
    camera_key: str
    native_height: int
    native_width: int
    capture_height: int
    capture_width: int
    effective_height: int
    effective_width: int
    compressed_bytes: int | None
    mode: str


@dataclass
class ImageDegradeResult:
    observation: dict[str, Any]
    frames: list[FrameDegradeStats] = field(default_factory=list)
    total_compressed_bytes: int = 0


def codec_provenance() -> dict[str, Any]:
    import cv2

    return {
        "image_codec_library": "opencv",
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "jpeg_api": "cv2.imencode(.jpg)/cv2.imdecode",
        "color_convention_env": "RGB",
        "color_convention_jpeg_codec": "BGR (OpenCV)",
        "downsample_interpolation": DOWNSAMPLE_INTERPOLATION,
        "upsample_interpolation": UPSAMPLE_INTERPOLATION,
        "quantization_note": (
            "JPEG quality is a codec quality parameter, not a bitrate. "
            "Optional bandwidth estimates use measured JPEG payload bytes only."
        ),
        "transport_simulation": False,
        "isolates_visual_quality_only": True,
    }


def config_provenance(cfg: ImageDegradeConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    payload["mode"] = cfg.mode.value
    payload.update(codec_provenance())
    return payload


def estimate_bandwidth_kbps(
    *,
    mean_compressed_bytes: float,
    frame_rate_hz: float,
) -> dict[str, Any]:
    """Estimate kbps from measured JPEG bytes. Excludes container/RTP/TCP overhead."""
    bitrate_kbps = (mean_compressed_bytes * 8.0 * frame_rate_hz) / 1000.0
    return {
        "bitrate_kbps": bitrate_kbps,
        "mean_compressed_bytes": mean_compressed_bytes,
        "frame_rate_hz": frame_rate_hz,
        "framing": "JPEG payload bytes only; no RTP/TCP/ISO-BMFF container overhead",
        "exclusions": ["network headers", "muxing", "retransmits", "audio"],
    }


def _validate_raw_frame(frame: np.ndarray, *, camera_key: str) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"{camera_key}: expected ndarray, got {type(frame).__name__}")
    if frame.ndim == 4:
        if frame.shape[0] != 1:
            raise ValueError(f"{camera_key}: expected batch size 1, got shape {frame.shape}")
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"{camera_key}: expected HWC RGB with 3 channels, got shape {frame.shape}")
    if frame.dtype != np.uint8:
        raise ValueError(f"{camera_key}: expected dtype uint8, got {frame.dtype}")
    if frame.shape[0] < 1 or frame.shape[1] < 1:
        raise ValueError(f"{camera_key}: empty spatial dimensions {frame.shape}")
    return frame


def _resize_rgb(frame: np.ndarray, *, height: int, width: int, interpolation: int) -> np.ndarray:
    import cv2

    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    # cv2.resize takes (width, height)
    out = cv2.resize(frame, (width, height), interpolation=interpolation)
    return np.ascontiguousarray(out)


def _jpeg_roundtrip_rgb(frame: np.ndarray, *, quality: int) -> tuple[np.ndarray, int]:
    import cv2

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError("cv2.imencode(.jpg) failed")
    compressed_bytes = int(encoded.size)
    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise RuntimeError("cv2.imdecode failed")
    if decoded_bgr.ndim != 3 or decoded_bgr.shape[2] != 3:
        raise RuntimeError(f"unexpected decoded shape {decoded_bgr.shape}")
    rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb), compressed_bytes


def degrade_frame(
    frame: np.ndarray,
    cfg: ImageDegradeConfig,
    *,
    camera_key: str,
) -> tuple[np.ndarray, FrameDegradeStats]:
    """Degrade one RGB uint8 HWC frame; output matches native HxWx3 uint8."""
    import cv2

    src = _validate_raw_frame(frame, camera_key=camera_key)
    native_h, native_w = int(src.shape[0]), int(src.shape[1])
    compressed_bytes: int | None = None

    if cfg.mode is ImageMode.NATIVE:
        out = np.ascontiguousarray(src.copy())
        capture_h, capture_w = native_h, native_w
    elif cfg.mode is ImageMode.DOWNSAMPLE:
        assert cfg.capture_height is not None and cfg.capture_width is not None
        capture_h, capture_w = int(cfg.capture_height), int(cfg.capture_width)
        low = _resize_rgb(
            src,
            height=capture_h,
            width=capture_w,
            interpolation=cv2.INTER_AREA,
        )
        out = _resize_rgb(
            low,
            height=native_h,
            width=native_w,
            interpolation=cv2.INTER_LINEAR,
        )
    elif cfg.mode is ImageMode.JPEG:
        assert cfg.jpeg_quality is not None
        capture_h, capture_w = native_h, native_w
        out, compressed_bytes = _jpeg_roundtrip_rgb(src, quality=cfg.jpeg_quality)
        if out.shape[0] != native_h or out.shape[1] != native_w:
            # Some codecs can nudge size; force policy-native geometry.
            out = _resize_rgb(
                out,
                height=native_h,
                width=native_w,
                interpolation=cv2.INTER_LINEAR,
            )
    else:
        raise ValueError(f"unsupported image mode {cfg.mode!r}")

    if out.dtype != np.uint8 or out.shape != (native_h, native_w, 3):
        raise RuntimeError(
            f"{camera_key}: degrade must preserve native uint8 HxWx3; "
            f"got dtype={out.dtype} shape={out.shape} native={(native_h, native_w, 3)}"
        )

    stats = FrameDegradeStats(
        camera_key=camera_key,
        native_height=native_h,
        native_width=native_w,
        capture_height=capture_h,
        capture_width=capture_w,
        effective_height=int(out.shape[0]),
        effective_width=int(out.shape[1]),
        compressed_bytes=compressed_bytes,
        mode=cfg.mode.value,
    )
    return out, stats


def _degrade_pixels_value(
    pixels: Any,
    cfg: ImageDegradeConfig,
    *,
    camera_key: str,
) -> tuple[Any, list[FrameDegradeStats]]:
    arr = np.asarray(pixels)
    if arr.ndim == 4:
        # Batched vector-env frames (B,H,W,C)
        frames_out = []
        stats_all: list[FrameDegradeStats] = []
        for i in range(arr.shape[0]):
            out_i, st = degrade_frame(arr[i], cfg, camera_key=f"{camera_key}[{i}]")
            frames_out.append(out_i)
            stats_all.append(st)
        return np.stack(frames_out, axis=0), stats_all
    out, st = degrade_frame(arr, cfg, camera_key=camera_key)
    return out, [st]


def degrade_observation(
    observation: dict[str, Any],
    cfg: ImageDegradeConfig,
) -> ImageDegradeResult:
    """Apply image degradation to raw env observation pixels; leave other keys intact."""
    if "pixels" not in observation:
        raise ValueError("observation missing 'pixels' at raw image boundary")

    obs_out = dict(observation)
    pixels = observation["pixels"]
    stats: list[FrameDegradeStats] = []
    total_bytes = 0

    if isinstance(pixels, dict):
        # Preserve camera key ordering from the env dict.
        new_pixels: dict[str, Any] = {}
        for key in pixels:
            degraded, st = _degrade_pixels_value(pixels[key], cfg, camera_key=str(key))
            new_pixels[key] = degraded
            stats.extend(st)
            for s in st:
                if s.compressed_bytes is not None:
                    total_bytes += s.compressed_bytes
        obs_out["pixels"] = new_pixels
    else:
        degraded, st = _degrade_pixels_value(pixels, cfg, camera_key="pixels")
        obs_out["pixels"] = degraded
        stats.extend(st)
        for s in st:
            if s.compressed_bytes is not None:
                total_bytes += s.compressed_bytes

    # Pass through non-image keys unchanged (e.g. agent_pos).
    return ImageDegradeResult(
        observation=obs_out,
        frames=stats,
        total_compressed_bytes=total_bytes,
    )


def resolution_smoke_configs(
    *,
    native_hw: tuple[int, int] | None = None,
) -> list[tuple[str, ImageDegradeConfig]]:
    """Isolated resolution smoke: native + downsample capture sizes.

    Default sizes target ALOHA 480x640. Pass ``native_hw`` for other envs (e.g.
    PushT rendered pixels) so capture sizes stay below the native frame.
    """
    if native_hw is None:
        sizes = [(240, 320), (120, 160), (60, 80)]
    else:
        nh, nw = native_hw
        sizes = [(max(1, nh // 2), max(1, nw // 2)), (max(1, nh // 4), max(1, nw // 4))]
    out: list[tuple[str, ImageDegradeConfig]] = [
        ("native", ImageDegradeConfig(mode=ImageMode.NATIVE)),
    ]
    for h, w in sizes:
        out.append(
            (
                f"downsample_{h}x{w}",
                ImageDegradeConfig(
                    mode=ImageMode.DOWNSAMPLE,
                    capture_height=h,
                    capture_width=w,
                ),
            )
        )
    return out


def jpeg_smoke_configs(
    qualities: tuple[int, ...] = (100, 75, 50, 25, 10),
) -> list[tuple[str, ImageDegradeConfig]]:
    """Isolated JPEG smoke at native resolution (quality 100 is still encoded)."""
    return [
        (f"jpeg_q{q}", ImageDegradeConfig(mode=ImageMode.JPEG, jpeg_quality=q)) for q in qualities
    ]


def summarize_frame_stats(
    frames: list[FrameDegradeStats],
    cfg: ImageDegradeConfig,
) -> dict[str, Any]:
    if not frames:
        return {"n_frames": 0}
    bytes_list = [f.compressed_bytes for f in frames if f.compressed_bytes is not None]
    summary: dict[str, Any] = {
        "n_frames": len(frames),
        "mode": cfg.mode.value,
        # Preserve first-seen camera order (env dict order), not alphabetical.
        "camera_keys": list(dict.fromkeys(f.camera_key.split("[", 1)[0] for f in frames)),
        "native_hw": [frames[0].native_height, frames[0].native_width],
        "capture_hw": [frames[0].capture_height, frames[0].capture_width],
        "effective_hw": [frames[0].effective_height, frames[0].effective_width],
        "jpeg_quality": cfg.jpeg_quality,
        "requested_capture_hw": (
            [cfg.capture_height, cfg.capture_width] if cfg.mode is ImageMode.DOWNSAMPLE else None
        ),
    }
    if bytes_list:
        mean_b = float(np.mean(bytes_list))
        summary["compressed_bytes_total"] = int(np.sum(bytes_list))
        summary["compressed_bytes_mean"] = mean_b
        summary["compressed_bytes_min"] = int(np.min(bytes_list))
        summary["compressed_bytes_max"] = int(np.max(bytes_list))
        if cfg.frame_rate_hz is not None:
            summary["bandwidth_estimate"] = estimate_bandwidth_kbps(
                mean_compressed_bytes=mean_b,
                frame_rate_hz=float(cfg.frame_rate_hz),
            )
    return summary
