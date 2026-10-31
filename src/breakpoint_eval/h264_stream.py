"""Persistent causal H.264 encode/decode streams (per camera, per episode).

Frames are fed in render order through a live encoder→decoder pair. This is not
per-frame JPEG-style still compression. Codec state resets between episodes.
Future-frame leakage is prevented via zerolatency + no B-frames. In quality_only
mode, a frame that is not decoded in its own push is an error. Startup and
last-frame holds exist only in deployment mode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from breakpoint_eval.h264_ffmpeg import (
    REQUESTED_ENCODER,
    REQUESTED_MAX_B_FRAMES,
    REQUESTED_PIXEL_FORMAT,
    REQUESTED_PRESET_DEFAULT,
    REQUESTED_RC_LOOKAHEAD,
    REQUESTED_TUNE,
    FFmpegProbe,
    require_ffmpeg_h264,
)

# yuv420p / libx264 commonly require even dimensions; we pad to multiples of 2
# (and document 16-alignment preference for macroblocks).
H264_DIMENSION_MULTIPLE = 2
H264_MACROBLOCK_HINT = 16
# Single-threaded encode + slice/low-delay decode so each control step is
# deterministic and returns its own decoded frame under quality_only.
ENCODER_THREAD_COUNT = 1
DECODER_THREAD_COUNT = 1
DECODER_THREAD_TYPE = "SLICE"


class H264LatencyMode(StrEnum):
    """How codec buffering interacts with the rest of the timing model."""

    QUALITY_ONLY = "quality_only"
    # Measure/model codec buffering delay; do not add it to inference latency_s.
    DEPLOYMENT = "deployment"


class DecodeAvailability(StrEnum):
    DECODED = "decoded"
    STARTUP_HOLD = "startup_hold"  # no prior decoded frame yet
    LAST_FRAME_HOLD = "last_frame_hold"  # decoder not ready; reuse last causal frame


@dataclass(frozen=True)
class H264StreamConfig:
    bitrate_kbps: float
    frame_rate_hz: float
    gop: int = 30
    preset: str = REQUESTED_PRESET_DEFAULT
    tune: str = REQUESTED_TUNE
    pixel_format: str = REQUESTED_PIXEL_FORMAT
    max_b_frames: int = REQUESTED_MAX_B_FRAMES
    rc_lookahead: int = REQUESTED_RC_LOOKAHEAD
    latency_mode: H264LatencyMode = H264LatencyMode.QUALITY_ONLY
    # Optional modeled decoder buffer delay (control periods) for deployment mode.
    deployment_buffer_frames: int = 0

    def __post_init__(self) -> None:
        if self.bitrate_kbps <= 0:
            raise ValueError("bitrate_kbps must be > 0")
        if self.frame_rate_hz <= 0:
            raise ValueError("frame_rate_hz must be > 0")
        if self.gop < 1:
            raise ValueError("gop must be >= 1")
        if self.max_b_frames != 0:
            raise ValueError("causal H.264 requires max_b_frames=0 (no B-frames)")
        if self.rc_lookahead != 0:
            raise ValueError("causal H.264 requires rc_lookahead=0")
        if self.tune != REQUESTED_TUNE:
            raise ValueError(f"causal H.264 requires tune={REQUESTED_TUNE!r}")
        if self.deployment_buffer_frames < 0:
            raise ValueError("deployment_buffer_frames must be >= 0")
        if self.latency_mode is H264LatencyMode.QUALITY_ONLY and self.deployment_buffer_frames != 0:
            raise ValueError("quality_only mode must keep deployment_buffer_frames=0")


@dataclass
class H264FrameRecord:
    camera_key: str
    frame_index: int
    capture_timestamp_s: float
    encode_timestamp_s: float
    decode_timestamp_s: float | None
    availability: str
    native_hw: tuple[int, int]
    codec_hw: tuple[int, int]
    pad_hw: tuple[int, int]
    effective_hw: tuple[int, int]
    packet_bytes: int
    used_hold: bool
    decoder_buffer_depth: int
    codec_buffering_delay_s: float | None


@dataclass
class H264StreamStats:
    camera_key: str
    n_frames_in: int = 0
    n_frames_decoded: int = 0
    n_startup_holds: int = 0
    n_last_frame_holds: int = 0
    n_keyframes: int = 0
    first_frame_keyframe: bool | None = None
    encoded_bytes_total: int = 0
    requested_bitrate_kbps: float = 0.0
    frame_rate_hz: float = 0.0
    elapsed_stream_s: float = 0.0
    actual_bitrate_kbps: float | None = None
    bitrate_error_kbps: float | None = None
    short_episode_note: str | None = None
    codec_hw: tuple[int, int] | None = None
    frame_records: list[H264FrameRecord] = field(default_factory=list)

    def finalize(self) -> None:
        if self.n_frames_in < 1 or self.frame_rate_hz <= 0:
            self.elapsed_stream_s = 0.0
            self.actual_bitrate_kbps = None
            self.bitrate_error_kbps = None
            self.short_episode_note = "no frames encoded"
            return
        # Duration from ordered capture timestamps when present; else n/fps.
        if len(self.frame_records) >= 2:
            t0 = self.frame_records[0].capture_timestamp_s
            t1 = self.frame_records[-1].capture_timestamp_s
            self.elapsed_stream_s = max(0.0, float(t1 - t0) + (1.0 / self.frame_rate_hz))
        else:
            self.elapsed_stream_s = float(self.n_frames_in) / self.frame_rate_hz
        bits = self.encoded_bytes_total * 8.0
        if self.elapsed_stream_s > 0:
            self.actual_bitrate_kbps = bits / self.elapsed_stream_s / 1000.0
            self.bitrate_error_kbps = self.actual_bitrate_kbps - self.requested_bitrate_kbps
        else:
            self.actual_bitrate_kbps = None
            self.bitrate_error_kbps = None
        if self.n_frames_in < max(2, self.frame_rate_hz):
            self.short_episode_note = (
                "short episode: header/keyframe overhead can dominate actual bitrate; "
                "requested vs actual difference is retained and must not be hidden"
            )

    def to_dict(self) -> dict[str, Any]:
        self.finalize()
        payload = asdict(self)
        return payload


def pad_to_codec_dimensions(
    height: int, width: int, *, multiple: int = H264_DIMENSION_MULTIPLE
) -> tuple[int, int, int, int]:
    """Return (codec_h, codec_w, pad_bottom, pad_right) for H.264-friendly sizes."""
    if height < 1 or width < 1:
        raise ValueError(f"invalid frame size {(height, width)}")
    codec_h = int(np.ceil(height / multiple) * multiple)
    codec_w = int(np.ceil(width / multiple) * multiple)
    return codec_h, codec_w, codec_h - height, codec_w - width


def x264_encoder_options(config: H264StreamConfig) -> dict[str, str]:
    """libx264 option dict applied to the encoder (preset, tune, x264-params)."""
    return {
        "preset": config.preset,
        "tune": config.tune,
        "x264-params": f"rc-lookahead={config.rc_lookahead}:bframes=0:sliced_threads=1",
    }


def _pad_frame(frame: np.ndarray, codec_h: int, codec_w: int) -> np.ndarray:
    h, w, _ = frame.shape
    if h == codec_h and w == codec_w:
        return np.ascontiguousarray(frame)
    out = np.zeros((codec_h, codec_w, 3), dtype=np.uint8)
    out[:h, :w] = frame
    return out


def _crop_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    return np.ascontiguousarray(frame[:height, :width])


class CausalH264CameraStream:
    """One live libx264 encoder + h264 decoder for a single camera stream."""

    def __init__(
        self,
        *,
        camera_key: str,
        native_height: int,
        native_width: int,
        config: H264StreamConfig,
        ffmpeg_probe: FFmpegProbe | None = None,
    ) -> None:
        self.camera_key = camera_key
        self.native_height = int(native_height)
        self.native_width = int(native_width)
        self.config = config
        self.probe = ffmpeg_probe or require_ffmpeg_h264()
        self.codec_h, self.codec_w, self.pad_bottom, self.pad_right = pad_to_codec_dimensions(
            self.native_height, self.native_width
        )
        self._last_decoded: np.ndarray | None = None
        self._pending_deployment: list[tuple[float, np.ndarray]] = []
        self.stats = H264StreamStats(
            camera_key=camera_key,
            requested_bitrate_kbps=float(config.bitrate_kbps),
            frame_rate_hz=float(config.frame_rate_hz),
            codec_hw=(self.codec_h, self.codec_w),
        )
        self._frame_index = 0
        self._encoder_options = x264_encoder_options(config)
        self._open_codecs()

    @property
    def runtime_settings(self) -> dict[str, Any]:
        """Applied encoder/decoder settings after open (read-only snapshot)."""
        return {
            "options": dict(self._encoder_options),
            "encoder_thread_count": int(self._encoder_thread_count),
            "decoder_thread_count": int(self._decoder_thread_count),
            "decoder_thread_type": str(self._decoder_thread_type),
            "decoder_low_delay": bool(self._decoder_low_delay),
        }

    def _open_codecs(self) -> None:
        from fractions import Fraction

        import av

        rate = Fraction(int(round(self.config.frame_rate_hz * 1000)), 1000)
        enc = av.CodecContext.create(REQUESTED_ENCODER, "w")
        enc.width = self.codec_w
        enc.height = self.codec_h
        enc.pix_fmt = self.config.pixel_format
        enc.framerate = rate
        enc.time_base = Fraction(rate.denominator, rate.numerator)
        enc.gop_size = int(self.config.gop)
        enc.max_b_frames = int(self.config.max_b_frames)
        enc.bit_rate = int(self.config.bitrate_kbps * 1000)
        enc.thread_count = ENCODER_THREAD_COUNT
        self._encoder_options = x264_encoder_options(self.config)
        enc.options = dict(self._encoder_options)
        # Force open so misconfiguration fails at episode start, not mid-rollout.
        enc.open()

        dec = av.CodecContext.create("h264", "r")
        dec.thread_type = av.codec.context.ThreadType.SLICE
        dec.thread_count = DECODER_THREAD_COUNT
        dec.flags |= av.codec.context.Flags.low_delay
        dec.open()

        self._encoder = enc
        self._decoder = dec
        self._rate = rate
        self._encoder_thread_count = int(enc.thread_count)
        self._decoder_thread_count = int(dec.thread_count)
        self._decoder_thread_type = DECODER_THREAD_TYPE
        self._decoder_low_delay = bool(dec.flags & av.codec.context.Flags.low_delay)

    def reset(self) -> None:
        """Reset codec state between episodes (new encoder/decoder pair)."""
        self.close()
        self._last_decoded = None
        self._pending_deployment.clear()
        self._frame_index = 0
        self.stats = H264StreamStats(
            camera_key=self.camera_key,
            requested_bitrate_kbps=float(self.config.bitrate_kbps),
            frame_rate_hz=float(self.config.frame_rate_hz),
            codec_hw=(self.codec_h, self.codec_w),
        )
        self._open_codecs()

    def close(self) -> None:
        for attr in ("_encoder", "_decoder"):
            ctx = getattr(self, attr, None)
            if ctx is None:
                continue
            try:
                ctx.close()
            except Exception:  # noqa: BLE001
                pass
            setattr(self, attr, None)

    def push_frame(
        self,
        frame_rgb: np.ndarray,
        *,
        capture_timestamp_s: float,
    ) -> tuple[np.ndarray, H264FrameRecord]:
        """Encode the next rendered frame in order; return policy-native RGB."""
        import time as time_mod

        import av

        if frame_rgb.shape != (self.native_height, self.native_width, 3):
            raise ValueError(
                f"{self.camera_key}: expected native "
                f"{(self.native_height, self.native_width, 3)}, got {frame_rgb.shape}"
            )
        if frame_rgb.dtype != np.uint8:
            raise ValueError(f"{self.camera_key}: expected uint8, got {frame_rgb.dtype}")

        padded = _pad_frame(frame_rgb, self.codec_h, self.codec_w)
        video_frame = av.VideoFrame.from_ndarray(padded, format="rgb24")
        video_frame = video_frame.reformat(format=self.config.pixel_format)
        video_frame.pts = self._frame_index
        video_frame.time_base = self._encoder.time_base

        encode_t = time_mod.perf_counter()
        packets = list(self._encoder.encode(video_frame))
        packet_bytes = int(sum(int(p.size) for p in packets))
        self.stats.encoded_bytes_total += packet_bytes
        frame_has_keyframe = any(bool(p.is_keyframe) for p in packets)
        self.stats.n_keyframes += sum(1 for p in packets if p.is_keyframe)
        if self._frame_index == 0:
            self.stats.first_frame_keyframe = frame_has_keyframe

        decoded_rgb: np.ndarray | None = None
        decode_t: float | None = None
        buffer_depth = 0
        for packet in packets:
            frames = self._decoder.decode(packet)
            buffer_depth = len(frames)
            for dec_frame in frames:
                arr = dec_frame.to_ndarray(format="rgb24")
                decoded_rgb = _crop_frame(arr, self.native_height, self.native_width)
                decode_t = time_mod.perf_counter()
                self.stats.n_frames_decoded += 1

        availability: DecodeAvailability
        used_hold = False
        out: np.ndarray
        codec_delay_s: float | None = None

        if self.config.latency_mode is H264LatencyMode.DEPLOYMENT:
            out, availability, used_hold, codec_delay_s, decode_t = self._deployment_select(
                decoded_rgb=decoded_rgb,
                capture_timestamp_s=capture_timestamp_s,
                decode_t=decode_t,
            )
        elif decoded_rgb is not None:
            out = decoded_rgb
            availability = DecodeAvailability.DECODED
            self._last_decoded = out.copy()
        else:
            raise RuntimeError(
                f"{self.camera_key}: frame {self._frame_index} was not decoded "
                "in its own control step"
            )

        if out.shape != (self.native_height, self.native_width, 3) or out.dtype != np.uint8:
            raise RuntimeError(
                f"{self.camera_key}: H.264 path must preserve policy-native uint8 HxWx3; "
                f"got {out.dtype} {out.shape}"
            )

        record = H264FrameRecord(
            camera_key=self.camera_key,
            frame_index=self._frame_index,
            capture_timestamp_s=float(capture_timestamp_s),
            encode_timestamp_s=float(encode_t),
            decode_timestamp_s=float(decode_t) if decode_t is not None else None,
            availability=availability.value,
            native_hw=(self.native_height, self.native_width),
            codec_hw=(self.codec_h, self.codec_w),
            pad_hw=(self.pad_bottom, self.pad_right),
            effective_hw=(int(out.shape[0]), int(out.shape[1])),
            packet_bytes=packet_bytes,
            used_hold=used_hold,
            decoder_buffer_depth=buffer_depth,
            codec_buffering_delay_s=codec_delay_s,
        )
        self.stats.n_frames_in += 1
        self.stats.frame_records.append(record)
        self._frame_index += 1
        return out, record

    def _deployment_select(
        self,
        *,
        decoded_rgb: np.ndarray | None,
        capture_timestamp_s: float,
        decode_t: float | None,
    ) -> tuple[np.ndarray, DecodeAvailability, bool, float | None, float | None]:
        """Apply optional buffer delay without touching inference latency_s."""
        period = 1.0 / self.config.frame_rate_hz
        delay_s = float(self.config.deployment_buffer_frames) * period
        if decoded_rgb is not None:
            available_at = capture_timestamp_s + delay_s
            self._pending_deployment.append((available_at, decoded_rgb.copy()))
            self._last_decoded = decoded_rgb.copy()

        # Release frames whose availability time has been reached (causal only).
        ready: np.ndarray | None = None
        while (
            self._pending_deployment
            and self._pending_deployment[0][0] <= capture_timestamp_s + 1e-12
        ):
            _t, ready = self._pending_deployment.pop(0)

        if ready is not None:
            return ready, DecodeAvailability.DECODED, False, delay_s, decode_t
        if self._last_decoded is not None and delay_s > 0 and not self._pending_deployment:
            # Should not happen often; treat as last-frame hold.
            self.stats.n_last_frame_holds += 1
            return (
                self._last_decoded.copy(),
                DecodeAvailability.LAST_FRAME_HOLD,
                True,
                delay_s,
                decode_t,
            )
        if self._last_decoded is not None and self._pending_deployment:
            self.stats.n_last_frame_holds += 1
            return (
                self._last_decoded.copy(),
                DecodeAvailability.LAST_FRAME_HOLD,
                True,
                delay_s,
                decode_t,
            )
        self.stats.n_startup_holds += 1
        zeros = np.zeros((self.native_height, self.native_width, 3), dtype=np.uint8)
        return zeros, DecodeAvailability.STARTUP_HOLD, True, delay_s, decode_t


class EpisodeH264Session:
    """Owns per-camera streams for one episode; create fresh or reset each episode."""

    def __init__(
        self, config: H264StreamConfig, *, ffmpeg_probe: FFmpegProbe | None = None
    ) -> None:
        self.config = config
        self.probe = ffmpeg_probe or require_ffmpeg_h264()
        self._streams: dict[str, CausalH264CameraStream] = {}
        self.sim_time_s = 0.0

    def reset_episode(self) -> None:
        for stream in self._streams.values():
            stream.reset()
        self.sim_time_s = 0.0

    def close(self) -> None:
        for stream in self._streams.values():
            stream.close()
        self._streams.clear()

    def _stream_for(self, camera_key: str, frame: np.ndarray) -> CausalH264CameraStream:
        key = str(camera_key)
        if key not in self._streams:
            h, w = int(frame.shape[0]), int(frame.shape[1])
            self._streams[key] = CausalH264CameraStream(
                camera_key=key,
                native_height=h,
                native_width=w,
                config=self.config,
                ffmpeg_probe=self.probe,
            )
        return self._streams[key]

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        camera_key: str,
        capture_timestamp_s: float | None = None,
    ) -> tuple[np.ndarray, H264FrameRecord]:
        ts = self.sim_time_s if capture_timestamp_s is None else float(capture_timestamp_s)
        stream = self._stream_for(camera_key, frame)
        return stream.push_frame(frame, capture_timestamp_s=ts)

    def advance_time(self, dt_s: float) -> None:
        self.sim_time_s += float(dt_s)

    def _encoder_evidence(self) -> dict[str, Any]:
        evidence: dict[str, Any] = {
            "options": x264_encoder_options(self.config),
            "encoder_thread_count": ENCODER_THREAD_COUNT,
            "decoder_thread_count": DECODER_THREAD_COUNT,
            "decoder_thread_type": DECODER_THREAD_TYPE,
            "decoder_low_delay": True,
            "pixel_format": self.config.pixel_format,
            "gop": self.config.gop,
        }
        if self._streams:
            evidence["codec_hw"] = {
                key: [stream.codec_h, stream.codec_w] for key, stream in self._streams.items()
            }
            live = next(iter(self._streams.values())).runtime_settings
            evidence["options"] = live["options"]
            evidence["encoder_thread_count"] = live["encoder_thread_count"]
            evidence["decoder_thread_count"] = live["decoder_thread_count"]
            evidence["decoder_thread_type"] = live["decoder_thread_type"]
            evidence["decoder_low_delay"] = live["decoder_low_delay"]
        return evidence

    def summary(self) -> dict[str, Any]:
        cameras = {}
        for key, stream in self._streams.items():
            cameras[key] = stream.stats.to_dict()
        encoder = self._encoder_evidence()
        return {
            "h264": True,
            "latency_mode": self.config.latency_mode.value,
            "requested": {
                "bitrate_kbps": self.config.bitrate_kbps,
                "frame_rate_hz": self.config.frame_rate_hz,
                "gop": self.config.gop,
                "preset": self.config.preset,
                "tune": self.config.tune,
                "pixel_format": self.config.pixel_format,
                "max_b_frames": self.config.max_b_frames,
                "rc_lookahead": self.config.rc_lookahead,
                "deployment_buffer_frames": self.config.deployment_buffer_frames,
                "dimension_multiple": H264_DIMENSION_MULTIPLE,
                "macroblock_hint": H264_MACROBLOCK_HINT,
                "encoder_options": encoder["options"],
                "encoder_thread_count": encoder["encoder_thread_count"],
                "decoder_thread_count": encoder["decoder_thread_count"],
                "decoder_thread_type": encoder["decoder_thread_type"],
                "decoder_low_delay": encoder["decoder_low_delay"],
            },
            "encoder": encoder,
            "ffmpeg": self.probe.to_dict(),
            "cameras": cameras,
            "exclusions_quality_only": (
                [
                    "codec_encode_wall_time",
                    "codec_decode_wall_time",
                    "network_transport",
                    "jitter_buffer",
                    "inference_latency",
                ]
                if self.config.latency_mode is H264LatencyMode.QUALITY_ONLY
                else None
            ),
            "deployment_notes": (
                {
                    "codec_buffering_modeled_separately": True,
                    "added_to_inference_latency_s": False,
                    "deployment_buffer_frames": self.config.deployment_buffer_frames,
                }
                if self.config.latency_mode is H264LatencyMode.DEPLOYMENT
                else None
            ),
            "not_prerecorded_trajectory": True,
            "persistent_stream_per_camera": True,
        }
