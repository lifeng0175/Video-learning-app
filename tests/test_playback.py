"""Unit tests for playback compatibility inspection and proxy generation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from src import playback


VIDEO_ID = "a" * 64
AV1_WEBM = playback.MediaInfo(
    format_names=("matroska", "webm"),
    duration_sec=87.82,
    video_codec="av1",
    video_pixel_format="yuv420p",
    audio_codec="opus",
)
H264_AAC_MP4 = playback.MediaInfo(
    format_names=("mov", "mp4", "m4a", "3gp", "3g2", "mj2"),
    duration_sec=87.82,
    video_codec="h264",
    video_pixel_format="yuv420p",
    audio_codec="aac",
)


def test_probe_media_reads_actual_container_and_default_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "misleading.mp4"
    video_path.write_bytes(b"media")
    payload = {
        "format": {"format_name": "matroska,webm", "duration": "12.5"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "vp9",
                "pix_fmt": "yuv420p",
                "disposition": {"default": 0},
            },
            {
                "codec_type": "video",
                "codec_name": "av1",
                "pix_fmt": "yuv420p10le",
                "disposition": {"default": 1},
            },
            {"codec_type": "audio", "codec_name": "opus"},
        ],
    }
    monkeypatch.setattr(playback.shutil, "which", lambda command: f"C:/{command}.exe")
    monkeypatch.setattr(
        playback.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout=json.dumps(payload), stderr=""
        ),
    )

    result = playback.probe_media(video_path)

    assert result == playback.MediaInfo(
        format_names=("matroska", "webm"),
        duration_sec=12.5,
        video_codec="av1",
        video_pixel_format="yuv420p10le",
        audio_codec="opus",
    )


def test_probe_media_reports_missing_ffprobe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "video.webm"
    video_path.write_bytes(b"media")
    monkeypatch.setattr(playback.shutil, "which", lambda command: None)

    with pytest.raises(playback.PlaybackToolNotFoundError, match="ffprobe"):
        playback.probe_media(video_path)


def test_assess_playback_uses_compatible_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_path = tmp_path / f"{VIDEO_ID}.mp4"
    original_path.write_bytes(b"original")
    monkeypatch.setattr(playback, "probe_media", lambda *args, **kwargs: H264_AAC_MP4)

    result = playback.assess_playback(original_path, VIDEO_ID, tmp_path / "cache")

    assert result.status == "original_compatible"
    assert result.original_path == original_path
    assert result.playback_path == original_path


def test_assess_playback_requests_proxy_for_av1_webm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_path = tmp_path / f"{VIDEO_ID}.webm"
    original_path.write_bytes(b"original")
    monkeypatch.setattr(playback, "probe_media", lambda *args, **kwargs: AV1_WEBM)

    result = playback.assess_playback(original_path, VIDEO_ID, tmp_path / "cache")

    assert result.status == "proxy_required"
    assert result.playback_path == original_path


def test_assess_playback_reuses_only_a_valid_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_path = tmp_path / f"{VIDEO_ID}.webm"
    original_path.write_bytes(b"original")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_path = playback.playback_cache_path(VIDEO_ID, cache_dir)
    cache_path.write_bytes(b"proxy")
    results = iter((AV1_WEBM, H264_AAC_MP4))
    monkeypatch.setattr(
        playback,
        "probe_media",
        lambda *args, **kwargs: next(results),
    )

    result = playback.assess_playback(original_path, VIDEO_ID, cache_dir)

    assert result.status == "proxy_cached"
    assert result.playback_path == cache_path


def test_assess_playback_rejects_cache_with_wrong_duration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_path = tmp_path / f"{VIDEO_ID}.webm"
    original_path.write_bytes(b"original")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    playback.playback_cache_path(VIDEO_ID, cache_dir).write_bytes(b"proxy")
    wrong_duration = playback.MediaInfo(
        format_names=H264_AAC_MP4.format_names,
        duration_sec=50.0,
        video_codec="h264",
        video_pixel_format="yuv420p",
        audio_codec="aac",
    )
    results = iter((AV1_WEBM, wrong_duration))
    monkeypatch.setattr(
        playback,
        "probe_media",
        lambda *args, **kwargs: next(results),
    )

    result = playback.assess_playback(original_path, VIDEO_ID, cache_dir)

    assert result.status == "proxy_required"
    assert result.playback_path == original_path


def test_ensure_playback_proxy_transcodes_validates_and_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_path = tmp_path / f"{VIDEO_ID}.webm"
    original_bytes = b"original AV1 WebM"
    original_path.write_bytes(original_bytes)
    cache_dir = tmp_path / "cache"
    probed_paths: list[Path] = []

    def fake_probe(video_path: str | Path, **kwargs: object) -> playback.MediaInfo:
        path = Path(video_path)
        probed_paths.append(path)
        return AV1_WEBM if path == original_path else H264_AAC_MP4

    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        Path(command[-1]).write_bytes(b"valid proxy")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(playback, "probe_media", fake_probe)
    monkeypatch.setattr(playback, "_resolve_tool", lambda command: f"C:/{command}.exe")
    monkeypatch.setattr(playback.subprocess, "run", fake_run)
    monkeypatch.setattr(
        playback.shutil,
        "disk_usage",
        lambda path: playback.shutil._ntuple_diskusage(1000, 0, 1000),
    )

    result = playback.ensure_playback_proxy(original_path, VIDEO_ID, cache_dir)

    expected = cache_dir / f"{VIDEO_ID}.h264-aac.v1.mp4"
    assert result == expected
    assert expected.read_bytes() == b"valid proxy"
    assert original_path.read_bytes() == original_bytes
    assert probed_paths[0] == original_path
    assert any(path.suffixes[-2:] == [".partial", ".mp4"] for path in probed_paths)
    assert commands and commands[0][0] == "C:/ffmpeg.exe"
    assert "libx264" in commands[0]
    assert "aac" in commands[0]
    assert not list(cache_dir.glob("*.partial.mp4"))


def test_ensure_playback_proxy_reuses_valid_cache_without_ffmpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_path = tmp_path / f"{VIDEO_ID}.webm"
    original_path.write_bytes(b"original")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_path = playback.playback_cache_path(VIDEO_ID, cache_dir)
    cache_path.write_bytes(b"proxy")
    results = iter((AV1_WEBM, H264_AAC_MP4))
    monkeypatch.setattr(
        playback,
        "probe_media",
        lambda *args, **kwargs: next(results),
    )
    monkeypatch.setattr(
        playback.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("FFmpeg must not run for a valid cache"),
    )

    assert playback.ensure_playback_proxy(original_path, VIDEO_ID, cache_dir) == cache_path


def test_ensure_playback_proxy_reports_missing_ffmpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_path = tmp_path / f"{VIDEO_ID}.webm"
    original_bytes = b"original"
    original_path.write_bytes(original_bytes)
    monkeypatch.setattr(playback, "probe_media", lambda *args, **kwargs: AV1_WEBM)
    monkeypatch.setattr(playback.shutil, "which", lambda command: None)
    monkeypatch.setattr(
        playback.shutil,
        "disk_usage",
        lambda path: playback.shutil._ntuple_diskusage(1000, 0, 1000),
    )

    with pytest.raises(playback.PlaybackToolNotFoundError, match="ffmpeg"):
        playback.ensure_playback_proxy(
            original_path,
            VIDEO_ID,
            tmp_path / "cache",
        )

    assert original_path.read_bytes() == original_bytes


def test_ensure_playback_proxy_cleans_partial_file_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_path = tmp_path / f"{VIDEO_ID}.webm"
    original_bytes = b"original"
    original_path.write_bytes(original_bytes)
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(playback, "probe_media", lambda *args, **kwargs: AV1_WEBM)
    monkeypatch.setattr(playback, "_resolve_tool", lambda command: command)
    monkeypatch.setattr(
        playback.shutil,
        "disk_usage",
        lambda path: playback.shutil._ntuple_diskusage(1000, 0, 1000),
    )

    def fail_transcode(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        Path(command[-1]).write_bytes(b"incomplete")
        return subprocess.CompletedProcess(command, 1, "", "encoding failed")

    monkeypatch.setattr(playback.subprocess, "run", fail_transcode)

    with pytest.raises(playback.PlaybackTranscodeError, match="encoding failed"):
        playback.ensure_playback_proxy(original_path, VIDEO_ID, cache_dir)

    assert original_path.read_bytes() == original_bytes
    assert not playback.playback_cache_path(VIDEO_ID, cache_dir).exists()
    assert not list(cache_dir.glob("*.partial.mp4"))


def test_ensure_playback_proxy_rejects_media_outside_mvp_rule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_path = tmp_path / f"{VIDEO_ID}.mp4"
    original_path.write_bytes(b"original")
    monkeypatch.setattr(playback, "probe_media", lambda *args, **kwargs: H264_AAC_MP4)

    with pytest.raises(playback.UnsupportedPlaybackConversionError, match="AV1 WebM"):
        playback.ensure_playback_proxy(original_path, VIDEO_ID, tmp_path / "cache")


def test_playback_cache_path_requires_sha256_video_id(tmp_path: Path) -> None:
    expected = tmp_path / f"{VIDEO_ID}.h264-aac.v1.mp4"
    assert playback.playback_cache_path(VIDEO_ID, tmp_path) == expected

    with pytest.raises(ValueError, match="SHA-256"):
        playback.playback_cache_path("legacy-name", tmp_path)
