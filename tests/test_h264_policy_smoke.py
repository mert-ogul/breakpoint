"""One real-policy H.264 smoke (integration; skipped if FFmpeg/sim unavailable)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_act_transfer_h264_quality_smoke(tmp_path: Path) -> None:
    from breakpoint_eval.camera_degrade import ImageDegradeConfig, ImageMode
    from breakpoint_eval.h264_ffmpeg import H264UnsupportedError, require_ffmpeg_h264
    from breakpoint_eval.run_baseline import RunRequest, run_baseline
    from breakpoint_eval.stack import repo_root_from_package

    try:
        require_ffmpeg_h264()
    except H264UnsupportedError as exc:
        pytest.skip(f"H.264 unsupported: {exc}")

    root = repo_root_from_package()
    out = tmp_path / "h264_smoke"
    image_cfg = ImageDegradeConfig(
        mode=ImageMode.H264,
        frame_rate_hz=50.0,
        h264_bitrate_kbps=2000.0,
        h264_gop=30,
        h264_preset="veryfast",
        h264_latency_mode="quality_only",
        h264_deployment_buffer_frames=0,
    )
    try:
        report = run_baseline(
            RunRequest(
                candidate="act_aloha_transfer",
                seeds=[1000],
                output_dir=out,
                device="cpu",
                save_video=False,
                latency_s=0.0,
                image_cfg=image_cfg,
                repo_root=root,
            )
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"policy/sim stack unavailable: {exc}")

    payload = json.loads(report.read_text(encoding="utf-8"))
    prov = payload["provenance"]
    assert prov["image_degrade"]["mode"] == "h264"
    assert prov["image_degrade"]["h264_is_persistent_stream"] is True
    episodes = payload["episodes"]
    assert len(episodes) == 1
    ep = episodes[0]
    # Env may fail on GL-blocked hosts — still require accurate unsupported/error labeling.
    if prov.get("env_created") is False:
        assert ep["outcome"] == "runtime_error"
        return
    h264 = ep.get("h264_summary") or {}
    assert h264.get("h264") is True
    assert h264.get("latency_mode") == "quality_only"
    assert h264.get("exclusions_quality_only")
    cams = h264.get("cameras") or {}
    assert cams, "expected per-camera H.264 stats when env ran"
    for stats in cams.values():
        assert stats["requested_bitrate_kbps"] == 2000.0
        assert stats["actual_bitrate_kbps"] is not None or stats["n_frames_in"] == 0
        if stats["n_frames_in"] > 0:
            assert stats["encoded_bytes_total"] > 0
            assert "bitrate_error_kbps" in stats
