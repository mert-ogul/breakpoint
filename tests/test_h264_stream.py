"""H.264 causal stream tests: synthetic clip, missing codec, shapes, bitrate."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from breakpoint_eval.h264_ffmpeg import (
    INSTALL_GUIDANCE,
    FFmpegProbe,
    H264UnsupportedError,
    probe_ffmpeg_h264,
    require_ffmpeg_h264,
)
from breakpoint_eval.h264_stream import (
    CausalH264CameraStream,
    EpisodeH264Session,
    H264LatencyMode,
    H264StreamConfig,
    pad_to_codec_dimensions,
)


def _moving_clip(n: int = 12, h: int = 48, w: int = 64) -> list[np.ndarray]:
    """Short generated clip with a moving bright square (order-sensitive)."""
    frames: list[np.ndarray] = []
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (20, 30, 40)
        x0 = (i * 3) % max(1, w - 8)
        frame[10:18, x0 : x0 + 8, 0] = 255
        frame[10:18, x0 : x0 + 8, 1] = 200
        frames.append(frame)
    return frames


def test_pad_to_codec_dimensions_explicit() -> None:
    ch, cw, pb, pr = pad_to_codec_dimensions(480, 640)
    assert (ch, cw, pb, pr) == (480, 640, 0, 0)
    ch, cw, pb, pr = pad_to_codec_dimensions(481, 639)
    assert ch == 482 and cw == 640
    assert pb == 1 and pr == 1


def test_h264_config_rejects_b_frames_and_lookahead() -> None:
    with pytest.raises(ValueError, match="max_b_frames"):
        H264StreamConfig(
            bitrate_kbps=1000,
            frame_rate_hz=30,
            max_b_frames=1,
        )
    with pytest.raises(ValueError, match="rc_lookahead"):
        H264StreamConfig(
            bitrate_kbps=1000,
            frame_rate_hz=30,
            rc_lookahead=5,
        )


def test_missing_ffmpeg_raises_with_install_guidance() -> None:
    fake = FFmpegProbe(
        available=False,
        ffmpeg_path=None,
        version_line=None,
        version=None,
        has_libx264=False,
        has_h264_decoder=False,
        pyav_available=False,
        pyav_version=None,
        unsupported_reason="ffmpeg binary not found on PATH",
    )
    with patch("breakpoint_eval.h264_ffmpeg.probe_ffmpeg_h264", return_value=fake):
        with pytest.raises(H264UnsupportedError) as excinfo:
            require_ffmpeg_h264()
    msg = str(excinfo.value)
    assert "ffmpeg" in msg.lower()
    assert "brew install ffmpeg" in msg or INSTALL_GUIDANCE[:20] in msg


def test_probe_does_not_raise() -> None:
    probe = probe_ffmpeg_h264()
    assert isinstance(probe.available, bool)
    assert probe.install_guidance


def _pyav_libx264_works() -> bool:
    try:
        import av

        av.CodecContext.create("libx264", "w")
        av.CodecContext.create("h264", "r")
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="module")
def h264_ready() -> FFmpegProbe:
    """Prefer verified FFmpeg CLI; fall back to PyAV libs for offline stream tests."""
    try:
        return require_ffmpeg_h264()
    except H264UnsupportedError:
        if not _pyav_libx264_works():
            pytest.skip("Neither FFmpeg CLI nor PyAV libx264 available")
        import av

        return FFmpegProbe(
            available=True,
            ffmpeg_path=None,
            version_line="pyav-embedded-ffmpeg-libs",
            version=getattr(av, "__version__", None),
            has_libx264=True,
            has_h264_decoder=True,
            pyav_available=True,
            pyav_version=getattr(av, "__version__", None),
            libav_versions=dict(getattr(av, "library_versions", {}) or {}),
            validated_flags={
                "tune_zerolatency": True,
                "preset": True,
                "bf_or_bframes": True,
                "g_gop": True,
            },
            unsupported_reason=None,
        )


def test_stream_order_shapes_bitrate_and_reset(h264_ready: FFmpegProbe) -> None:
    cfg = H264StreamConfig(
        bitrate_kbps=800.0,
        frame_rate_hz=30.0,
        gop=8,
        preset="veryfast",
        latency_mode=H264LatencyMode.QUALITY_ONLY,
    )
    clip = _moving_clip(16, h=45, w=63)  # odd dims → padding
    session = EpisodeH264Session(cfg, ffmpeg_probe=h264_ready)
    outs: list[np.ndarray] = []
    for i, frame in enumerate(clip):
        out, rec = session.process_frame(frame, camera_key="top", capture_timestamp_s=i / 30.0)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8
        assert rec.pad_hw != (0, 0) or frame.shape[0] % 2 == 0
        assert rec.frame_index == i
        assert rec.capture_timestamp_s == pytest.approx(i / 30.0)
        outs.append(out)
        session.advance_time(1.0 / 30.0)

    summary = session.summary()
    cam = summary["cameras"]["top"]
    assert cam["n_frames_in"] == 16
    assert cam["encoded_bytes_total"] > 0
    assert cam["requested_bitrate_kbps"] == 800.0
    assert cam["actual_bitrate_kbps"] is not None
    assert cam["bitrate_error_kbps"] is not None
    assert cam["short_episode_note"]  # 16 frames < 30 fps → short-episode note
    assert summary["exclusions_quality_only"]
    assert summary["not_prerecorded_trajectory"] is True
    # Order: early frames differ from late frames (moving square survived codec).
    assert not np.array_equal(outs[0], outs[-1])

    # Reset between episodes clears codec state / indices.
    session.reset_episode()
    out0, rec0 = session.process_frame(clip[0], camera_key="top", capture_timestamp_s=0.0)
    assert rec0.frame_index == 0
    assert out0.shape == clip[0].shape
    session.close()


def test_deployment_mode_does_not_use_future_frames(h264_ready: FFmpegProbe) -> None:
    cfg = H264StreamConfig(
        bitrate_kbps=1200.0,
        frame_rate_hz=10.0,
        gop=5,
        latency_mode=H264LatencyMode.DEPLOYMENT,
        deployment_buffer_frames=1,
    )
    stream = CausalH264CameraStream(
        camera_key="cam",
        native_height=32,
        native_width=40,
        config=cfg,
        ffmpeg_probe=h264_ready,
    )
    f0 = np.full((32, 40, 3), 10, dtype=np.uint8)
    f1 = np.full((32, 40, 3), 200, dtype=np.uint8)
    out0, r0 = stream.push_frame(f0, capture_timestamp_s=0.0)
    # With 1-frame buffer, first output is startup/last hold — never f1.
    assert r0.availability in {"startup_hold", "last_frame_hold", "decoded"}
    if r0.availability != "decoded":
        assert out0.mean() < 50 or r0.used_hold
    out1, r1 = stream.push_frame(f1, capture_timestamp_s=0.1)
    assert r1.codec_buffering_delay_s == pytest.approx(0.1)
    # Inference latency is not part of this path.
    assert cfg.latency_mode is H264LatencyMode.DEPLOYMENT
    stream.close()


def test_degrade_observation_h264_requires_session() -> None:
    from breakpoint_eval.camera_degrade import ImageDegradeConfig, ImageMode, degrade_observation

    cfg = ImageDegradeConfig(
        mode=ImageMode.H264,
        frame_rate_hz=30.0,
        h264_bitrate_kbps=1000.0,
    )
    obs = {"pixels": np.zeros((32, 40, 3), dtype=np.uint8), "agent_pos": np.zeros(2)}
    with pytest.raises(RuntimeError, match="h264_session"):
        degrade_observation(obs, cfg)


def test_isolated_still_h264_frame_rejected() -> None:
    from breakpoint_eval.camera_degrade import ImageDegradeConfig, ImageMode, degrade_frame

    cfg = ImageDegradeConfig(
        mode=ImageMode.H264,
        frame_rate_hz=30.0,
        h264_bitrate_kbps=1000.0,
    )
    with pytest.raises(RuntimeError, match="persistent"):
        degrade_frame(np.zeros((32, 40, 3), dtype=np.uint8), cfg, camera_key="top")


def _benchmark_cfg(bitrate_kbps: float = 1000.0) -> H264StreamConfig:
    return H264StreamConfig(
        bitrate_kbps=bitrate_kbps,
        frame_rate_hz=50.0,
        gop=30,
        preset="veryfast",
        tune="zerolatency",
        latency_mode=H264LatencyMode.QUALITY_ONLY,
    )


def _run_clip(
    session: EpisodeH264Session,
    clip: list[np.ndarray],
    *,
    camera_key: str = "top",
    fps: float = 50.0,
) -> tuple[list[np.ndarray], list[int]]:
    outs: list[np.ndarray] = []
    packet_sizes: list[int] = []
    for i, frame in enumerate(clip):
        out, rec = session.process_frame(frame, camera_key=camera_key, capture_timestamp_s=i / fps)
        outs.append(out.copy())
        packet_sizes.append(int(rec.packet_bytes))
        session.advance_time(1.0 / fps)
    return outs, packet_sizes


def test_causality_prefix_identical(h264_ready: FFmpegProbe) -> None:
    k = 6
    n = 12
    base = _moving_clip(n, h=48, w=64)
    clip_a = [f.copy() for f in base]
    clip_b = [f.copy() for f in base]
    for i in range(k, n):
        clip_b[i] = np.full_like(clip_b[i], 180)

    cfg = _benchmark_cfg(2000.0)
    sa = EpisodeH264Session(cfg, ffmpeg_probe=h264_ready)
    sb = EpisodeH264Session(cfg, ffmpeg_probe=h264_ready)
    outs_a, sizes_a = _run_clip(sa, clip_a)
    outs_b, sizes_b = _run_clip(sb, clip_b)
    for i in range(k):
        assert sizes_a[i] == sizes_b[i]
        assert np.array_equal(outs_a[i], outs_b[i])
    assert not np.array_equal(outs_a[k], outs_b[k])
    sa.close()
    sb.close()


def test_reset_episode_and_fresh_session_match(h264_ready: FFmpegProbe) -> None:
    cfg = _benchmark_cfg(1000.0)
    clip = _moving_clip(10, h=48, w=64)
    session = EpisodeH264Session(cfg, ffmpeg_probe=h264_ready)
    outs1, sizes1 = _run_clip(session, clip)
    cam1 = session.summary()["cameras"]["top"]
    assert cam1["first_frame_keyframe"] is True
    assert cam1["frame_records"][0]["frame_index"] == 0

    session.reset_episode()
    outs2, sizes2 = _run_clip(session, clip)
    cam2 = session.summary()["cameras"]["top"]
    assert cam2["first_frame_keyframe"] is True
    assert cam2["frame_records"][0]["frame_index"] == 0
    assert sizes1 == sizes2
    for a, b in zip(outs1, outs2, strict=True):
        assert np.array_equal(a, b)
    session.close()

    fresh = EpisodeH264Session(cfg, ffmpeg_probe=h264_ready)
    outs3, sizes3 = _run_clip(fresh, clip)
    cam3 = fresh.summary()["cameras"]["top"]
    assert cam3["first_frame_keyframe"] is True
    assert sizes1 == sizes3
    for a, b in zip(outs1, outs3, strict=True):
        assert np.array_equal(a, b)
    fresh.close()


def test_quality_only_decodes_every_frame(h264_ready: FFmpegProbe) -> None:
    cfg = _benchmark_cfg(500.0)
    clip = _moving_clip(8, h=48, w=64)
    session = EpisodeH264Session(cfg, ffmpeg_probe=h264_ready)
    outs, _ = _run_clip(session, clip)
    cam = session.summary()["cameras"]["top"]
    assert all(r["availability"] == "decoded" for r in cam["frame_records"])
    assert cam["n_startup_holds"] == 0
    assert cam["n_last_frame_holds"] == 0
    assert cam["n_frames_decoded"] == cam["n_frames_in"] == len(outs)
    session.close()


def test_quality_only_raises_when_decoder_returns_nothing(
    h264_ready: FFmpegProbe,
) -> None:
    cfg = _benchmark_cfg(1000.0)
    stream = CausalH264CameraStream(
        camera_key="cam",
        native_height=32,
        native_width=40,
        config=cfg,
        ffmpeg_probe=h264_ready,
    )
    frame = np.full((32, 40, 3), 40, dtype=np.uint8)

    class _EmptyDecoder:
        def decode(self, _packet: object) -> list:
            return []

    stream._decoder = _EmptyDecoder()
    with pytest.raises(RuntimeError, match="not decoded in its own control step"):
        stream.push_frame(frame, capture_timestamp_s=0.0)
    stream.close()


def test_encoded_bytes_match_packet_sum_and_flush_empty(
    h264_ready: FFmpegProbe,
) -> None:
    cfg = _benchmark_cfg(4000.0)
    clip = _moving_clip(15, h=48, w=64)
    session = EpisodeH264Session(cfg, ffmpeg_probe=h264_ready)
    _run_clip(session, clip)
    stream = session._streams["top"]
    flush_packets = list(stream._encoder.encode(None))
    assert flush_packets == []
    cam = session.summary()["cameras"]["top"]
    packet_sum = sum(int(r["packet_bytes"]) for r in cam["frame_records"])
    assert cam["encoded_bytes_total"] == packet_sum
    assert cam["n_keyframes"] >= 1
    assert cam["first_frame_keyframe"] is True
    session.close()


def test_summary_records_encoder_settings(h264_ready: FFmpegProbe) -> None:
    cfg = _benchmark_cfg(2000.0)
    session = EpisodeH264Session(cfg, ffmpeg_probe=h264_ready)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    session.process_frame(frame, camera_key="top", capture_timestamp_s=0.0)
    summary = session.summary()
    requested = summary["requested"]
    encoder = summary["encoder"]
    expected_options = {
        "preset": "veryfast",
        "tune": "zerolatency",
        "x264-params": "rc-lookahead=0:bframes=0:sliced_threads=1",
    }
    assert requested["gop"] == 30
    assert requested["preset"] == "veryfast"
    assert requested["tune"] == "zerolatency"
    assert requested["pixel_format"] == cfg.pixel_format
    assert requested["encoder_options"] == expected_options
    assert requested["encoder_thread_count"] == 1
    assert requested["decoder_thread_count"] == 1
    assert requested["decoder_thread_type"] == "SLICE"
    assert requested["decoder_low_delay"] is True
    assert encoder["options"] == expected_options
    assert encoder["encoder_thread_count"] == 1
    assert encoder["decoder_thread_count"] == 1
    assert encoder["decoder_thread_type"] == "SLICE"
    assert encoder["decoder_low_delay"] is True
    assert encoder["pixel_format"] == cfg.pixel_format
    assert encoder["gop"] == 30
    assert encoder["codec_hw"]["top"] == [48, 64]
    assert summary["ffmpeg"]["pyav_version"]
    session.close()
