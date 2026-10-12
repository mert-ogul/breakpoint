"""Unit tests for raw-pixel camera degradation (offline, no sim)."""

from __future__ import annotations

import numpy as np
import pytest

from breakpoint_eval.camera_degrade import (
    DOWNSAMPLE_INTERPOLATION,
    UPSAMPLE_INTERPOLATION,
    ImageDegradeConfig,
    ImageMode,
    codec_provenance,
    degrade_frame,
    degrade_observation,
    estimate_bandwidth_kbps,
    jpeg_smoke_configs,
    resolution_smoke_configs,
    summarize_frame_stats,
)
from breakpoint_eval.cli import parse_capture_hw

pytest.importorskip("cv2")


def _rgb_ramp(h: int = 48, w: int = 64) -> np.ndarray:
    """Deterministic uint8 RGB with distinct channels (not grayscale)."""
    yy, xx = np.mgrid[0:h, 0:w]
    r = (xx * 3 + 10).astype(np.uint8)
    g = (yy * 5 + 40).astype(np.uint8)
    b = ((xx + yy) * 2 + 80).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def test_identity_bypass_native() -> None:
    frame = _rgb_ramp()
    cfg = ImageDegradeConfig(mode=ImageMode.NATIVE)
    out, stats = degrade_frame(frame, cfg, camera_key="top")
    assert out.dtype == np.uint8
    assert out.shape == frame.shape
    np.testing.assert_array_equal(out, frame)
    assert out is not frame
    assert stats.compressed_bytes is None
    assert stats.mode == "native"
    assert stats.capture_height == frame.shape[0]
    assert stats.effective_height == frame.shape[0]


def test_channel_order_preserved_through_jpeg() -> None:
    """RGB env convention survives OpenCV BGR codec path (red channel stays red)."""
    h, w = 32, 40
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[..., 0] = 220  # R
    frame[..., 1] = 10
    frame[..., 2] = 10
    cfg = ImageDegradeConfig(mode=ImageMode.JPEG, jpeg_quality=95)
    out, _ = degrade_frame(frame, cfg, camera_key="cam")
    # Mean red should remain clearly dominant over green/blue.
    assert float(out[..., 0].mean()) > float(out[..., 1].mean()) + 80
    assert float(out[..., 0].mean()) > float(out[..., 2].mean()) + 80


def test_jpeg_quality_100_is_not_identity() -> None:
    frame = _rgb_ramp(64, 80)
    cfg = ImageDegradeConfig(mode=ImageMode.JPEG, jpeg_quality=100)
    out, stats = degrade_frame(frame, cfg, camera_key="top")
    assert stats.compressed_bytes is not None and stats.compressed_bytes > 0
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    # Real codec round-trip: not bit-identical to the source.
    assert not np.array_equal(out, frame)


def test_jpeg_deterministic() -> None:
    frame = _rgb_ramp()
    cfg = ImageDegradeConfig(mode=ImageMode.JPEG, jpeg_quality=50)
    a, sa = degrade_frame(frame, cfg, camera_key="top")
    b, sb = degrade_frame(frame, cfg, camera_key="top")
    np.testing.assert_array_equal(a, b)
    assert sa.compressed_bytes == sb.compressed_bytes


def test_downsample_preserves_policy_shape() -> None:
    frame = _rgb_ramp(48, 64)
    cfg = ImageDegradeConfig(
        mode=ImageMode.DOWNSAMPLE,
        capture_height=12,
        capture_width=16,
    )
    out, stats = degrade_frame(frame, cfg, camera_key="top")
    assert out.shape == (48, 64, 3)
    assert out.dtype == np.uint8
    assert stats.capture_height == 12
    assert stats.capture_width == 16
    assert stats.effective_height == 48
    assert stats.effective_width == 64
    assert DOWNSAMPLE_INTERPOLATION == "INTER_AREA"
    assert UPSAMPLE_INTERPOLATION == "INTER_LINEAR"
    # Downsample+upsample changes content for a non-constant image.
    assert not np.array_equal(out, frame)


def test_downsample_deterministic() -> None:
    frame = _rgb_ramp()
    cfg = ImageDegradeConfig(mode=ImageMode.DOWNSAMPLE, capture_height=12, capture_width=16)
    a, _ = degrade_frame(frame, cfg, camera_key="top")
    b, _ = degrade_frame(frame, cfg, camera_key="top")
    np.testing.assert_array_equal(a, b)


def test_multi_camera_order_and_agent_pos_passthrough() -> None:
    top = _rgb_ramp(24, 32)
    wrist = (top // 2).astype(np.uint8)
    obs = {
        "pixels": {"top": top, "wrist": wrist},
        "agent_pos": np.arange(14, dtype=np.float32),
    }
    cfg = ImageDegradeConfig(mode=ImageMode.JPEG, jpeg_quality=80)
    result = degrade_observation(obs, cfg)
    assert list(result.observation["pixels"].keys()) == ["top", "wrist"]
    assert result.observation["pixels"]["top"].shape == top.shape
    assert result.observation["pixels"]["wrist"].shape == wrist.shape
    np.testing.assert_array_equal(result.observation["agent_pos"], obs["agent_pos"])
    assert len(result.frames) == 2
    assert [f.camera_key for f in result.frames] == ["top", "wrist"]
    assert result.total_compressed_bytes > 0
    summary = summarize_frame_stats(result.frames, cfg)
    assert summary["camera_keys"] == ["top", "wrist"]
    assert summary["jpeg_quality"] == 80


def test_batched_pixels() -> None:
    batch = np.stack([_rgb_ramp(), _rgb_ramp() + 1], axis=0)
    cfg = ImageDegradeConfig(mode=ImageMode.NATIVE)
    result = degrade_observation({"pixels": batch}, cfg)
    assert result.observation["pixels"].shape == batch.shape
    assert len(result.frames) == 2


def test_malformed_inputs() -> None:
    cfg = ImageDegradeConfig(mode=ImageMode.NATIVE)
    with pytest.raises(TypeError):
        degrade_frame([1, 2, 3], cfg, camera_key="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dtype"):
        degrade_frame(np.zeros((8, 8, 3), dtype=np.float32), cfg, camera_key="x")
    with pytest.raises(ValueError, match="channels"):
        degrade_frame(np.zeros((8, 8, 1), dtype=np.uint8), cfg, camera_key="x")
    with pytest.raises(ValueError, match="missing 'pixels'"):
        degrade_observation({"agent_pos": np.zeros(2)}, cfg)
    with pytest.raises(ValueError, match="capture"):
        ImageDegradeConfig(mode=ImageMode.DOWNSAMPLE)
    with pytest.raises(ValueError, match="jpeg_quality"):
        ImageDegradeConfig(mode=ImageMode.JPEG)
    with pytest.raises(ValueError, match="jpeg_quality"):
        ImageDegradeConfig(mode=ImageMode.JPEG, jpeg_quality=0)
    with pytest.raises(ValueError, match="jpeg_quality"):
        ImageDegradeConfig(mode=ImageMode.JPEG, jpeg_quality=101)


def test_bandwidth_estimate_from_bytes_not_quality() -> None:
    est = estimate_bandwidth_kbps(mean_compressed_bytes=1000.0, frame_rate_hz=50.0)
    assert est["bitrate_kbps"] == pytest.approx((1000.0 * 8.0 * 50.0) / 1000.0)
    assert "JPEG payload" in est["framing"]
    assert "network headers" in est["exclusions"]
    prov = codec_provenance()
    assert prov["transport_simulation"] is False
    assert "not a bitrate" in prov["quantization_note"].lower()
    assert prov["opencv_version"]


def test_resolution_smoke_sweep() -> None:
    """Offline resolution smoke: native + capture sizes, policy HxW preserved."""
    native_h, native_w = 48, 64
    frame = _rgb_ramp(native_h, native_w)
    jobs = resolution_smoke_configs()
    assert jobs[0][0] == "native"
    assert jobs[0][1].mode is ImageMode.NATIVE
    for name, cfg in jobs:
        if cfg.mode is ImageMode.NATIVE:
            out, _stats = degrade_frame(frame, cfg, camera_key="top")
        else:
            # Sweep sizes are ALOHA-scale; scale capture relative to this fixture.
            assert cfg.capture_height and cfg.capture_width
            scale_h = max(1, native_h * cfg.capture_height // 480)
            scale_w = max(1, native_w * cfg.capture_width // 640)
            local = ImageDegradeConfig(
                mode=ImageMode.DOWNSAMPLE,
                capture_height=scale_h,
                capture_width=scale_w,
            )
            out, stats = degrade_frame(frame, local, camera_key="top")
            assert stats.capture_height == scale_h
        assert out.shape == (native_h, native_w, 3), name
        assert out.dtype == np.uint8, name


def test_jpeg_smoke_sweep() -> None:
    """Offline JPEG smoke at native resolution; bytes generally rise with quality."""
    frame = _rgb_ramp(64, 80)
    jobs = jpeg_smoke_configs()
    assert [cfg.jpeg_quality for _, cfg in jobs] == [100, 75, 50, 25, 10]
    byte_by_q: dict[int, int] = {}
    for name, cfg in jobs:
        out, stats = degrade_frame(frame, cfg, camera_key="top")
        assert out.shape == frame.shape, name
        assert stats.compressed_bytes is not None and stats.compressed_bytes > 0
        assert not np.array_equal(out, frame), name
        assert cfg.jpeg_quality is not None
        byte_by_q[cfg.jpeg_quality] = stats.compressed_bytes
    # Coarse monotonicity: high quality should not be smaller than low quality.
    assert byte_by_q[100] >= byte_by_q[10]
    assert byte_by_q[75] >= byte_by_q[25]


def test_parse_capture_hw() -> None:
    assert parse_capture_hw("240x320") == (240, 320)
    assert parse_capture_hw("120,160") == (120, 160)
    with pytest.raises(ValueError):
        parse_capture_hw("240")
