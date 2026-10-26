"""The CLI reports only the documented H.264 failure; other errors are not swallowed."""

from __future__ import annotations

import logging

import pytest

from breakpoint_eval import cli, h264_ffmpeg


def test_h264_unavailable_logs_guidance_and_returns_false(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def unavailable() -> None:
        raise h264_ffmpeg.H264UnsupportedError("no libx264")

    monkeypatch.setattr(h264_ffmpeg, "require_ffmpeg_h264", unavailable)
    with caplog.at_level(logging.ERROR):
        assert cli._h264_available() is False
    assert "no libx264" in caplog.text


def test_h264_guard_does_not_hide_unexpected_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> None:
        raise ImportError("broken internal import")

    monkeypatch.setattr(h264_ffmpeg, "require_ffmpeg_h264", broken)
    with pytest.raises(ImportError, match="broken internal import"):
        cli._h264_available()


def test_h264_available_when_probe_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(h264_ffmpeg, "require_ffmpeg_h264", lambda: None)
    assert cli._h264_available() is True
