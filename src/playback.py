"""Browser playback compatibility checks and derived proxy generation."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


PLAYBACK_PROFILE = "h264-aac.v1"
COMPATIBLE_PIXEL_FORMATS = frozenset({"yuv420p", "yuvj420p"})


class PlaybackError(RuntimeError):
    """Base error for playback inspection and proxy generation."""


class PlaybackToolNotFoundError(PlaybackError):
    """Raised when FFmpeg or ffprobe cannot be found."""


class MediaProbeError(PlaybackError):
    """Raised when ffprobe cannot return usable media information."""


class UnsupportedPlaybackConversionError(PlaybackError):
    """Raised when no MVP conversion rule exists for the input media."""


class PlaybackTranscodeError(PlaybackError):
    """Raised when FFmpeg cannot produce a valid playback proxy."""


class InvalidPlaybackCacheError(PlaybackError):
    """Raised when a playback cache file fails validation."""


@dataclass(frozen=True, slots=True)
class MediaInfo:
    """Media properties required for browser compatibility decisions."""

    format_names: tuple[str, ...]
    duration_sec: float | None
    video_codec: str
    video_pixel_format: str | None
    audio_codec: str | None


@dataclass(frozen=True, slots=True)
class PlaybackAssessment:
    """Playback path and action required for an original video."""

    original_path: Path
    playback_path: Path
    media_info: MediaInfo
    status: Literal[
        "original_compatible",
        "proxy_cached",
        "proxy_required",
        "compatibility_unknown",
    ]
    reason: str


def probe_media(
    video_path: str | Path,
    *,
    ffprobe_command: str = "ffprobe",
    timeout_sec: float = 30.0,
) -> MediaInfo:
    """Inspect actual container and stream codecs with ffprobe."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ffprobe_path = _resolve_tool(ffprobe_command)
    command = [
        ffprobe_path,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            check=False,
            creationflags=_subprocess_creation_flags(),
        )
    except subprocess.TimeoutExpired as error:
        raise MediaProbeError(f"ffprobe timed out for: {video_path}") from error
    except OSError as error:
        raise MediaProbeError(f"ffprobe could not inspect: {video_path}") from error

    if completed.returncode != 0:
        details = completed.stderr.strip() or "unknown ffprobe error"
        raise MediaProbeError(f"ffprobe failed for {video_path}: {details}")

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise MediaProbeError("ffprobe returned invalid JSON") from error

    return _media_info_from_probe(payload)


def assess_playback(
    original_path: str | Path,
    video_id: str,
    playback_cache_dir: str | Path,
    *,
    ffprobe_command: str = "ffprobe",
) -> PlaybackAssessment:
    """Choose the original, a valid proxy, or an explicit conversion action."""
    original_path = Path(original_path)
    _validate_video_id(video_id)
    original_info = probe_media(original_path, ffprobe_command=ffprobe_command)

    if _is_known_compatible(original_info):
        return PlaybackAssessment(
            original_path=original_path,
            playback_path=original_path,
            media_info=original_info,
            status="original_compatible",
            reason="The original is a browser-compatible H.264/AAC MP4.",
        )

    cache_path = playback_cache_path(video_id, playback_cache_dir)
    if cache_path.exists():
        try:
            _validate_proxy(cache_path, original_info, ffprobe_command)
        except (InvalidPlaybackCacheError, MediaProbeError):
            pass
        else:
            return PlaybackAssessment(
                original_path=original_path,
                playback_path=cache_path,
                media_info=original_info,
                status="proxy_cached",
                reason="A validated H.264/AAC playback copy is available.",
            )

    if _is_av1_webm(original_info):
        return PlaybackAssessment(
            original_path=original_path,
            playback_path=original_path,
            media_info=original_info,
            status="proxy_required",
            reason="AV1 WebM requires a compatible H.264/AAC playback copy.",
        )

    return PlaybackAssessment(
        original_path=original_path,
        playback_path=original_path,
        media_info=original_info,
        status="compatibility_unknown",
        reason=(
            "Browser compatibility is not known for "
            f"{original_info.video_codec}/{original_info.audio_codec or 'no audio'}."
        ),
    )


def ensure_playback_proxy(
    original_path: str | Path,
    video_id: str,
    playback_cache_dir: str | Path,
    *,
    ffmpeg_command: str = "ffmpeg",
    ffprobe_command: str = "ffprobe",
    timeout_sec: float | None = None,
) -> Path:
    """Create and validate an H.264/AAC MP4 proxy without changing the original."""
    original_path = Path(original_path)
    _validate_video_id(video_id)
    original_info = probe_media(original_path, ffprobe_command=ffprobe_command)
    if not _is_av1_webm(original_info):
        raise UnsupportedPlaybackConversionError(
            "MVP playback conversion is limited to AV1 WebM videos"
        )

    playback_cache_dir = Path(playback_cache_dir)
    playback_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = playback_cache_path(video_id, playback_cache_dir)

    if cache_path.exists():
        try:
            _validate_proxy(cache_path, original_info, ffprobe_command)
        except (InvalidPlaybackCacheError, MediaProbeError):
            pass
        else:
            return cache_path

    free_bytes = shutil.disk_usage(playback_cache_dir).free
    if free_bytes <= original_path.stat().st_size:
        raise PlaybackTranscodeError(
            "Insufficient free disk space to create a playback copy"
        )

    ffmpeg_path = _resolve_tool(ffmpeg_command)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=playback_cache_dir,
            prefix=f".{video_id}.",
            suffix=".partial.mp4",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        command = _build_transcode_command(
            ffmpeg_path,
            original_path,
            temporary_path,
        )
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
                check=False,
                creationflags=_subprocess_creation_flags(),
            )
        except subprocess.TimeoutExpired as error:
            raise PlaybackTranscodeError("FFmpeg transcoding timed out") from error
        except OSError as error:
            raise PlaybackTranscodeError("FFmpeg could not be started") from error

        if completed.returncode != 0:
            details = (completed.stderr.strip() or "unknown FFmpeg error")[-4000:]
            raise PlaybackTranscodeError(f"FFmpeg transcoding failed: {details}")

        _validate_proxy(temporary_path, original_info, ffprobe_command)

        if cache_path.exists():
            try:
                _validate_proxy(cache_path, original_info, ffprobe_command)
            except (InvalidPlaybackCacheError, MediaProbeError):
                pass
            else:
                return cache_path

        os.replace(temporary_path, cache_path)
        temporary_path = None
        return cache_path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def playback_cache_path(video_id: str, playback_cache_dir: str | Path) -> Path:
    """Return the versioned playback cache path for an original video ID."""
    _validate_video_id(video_id)
    return Path(playback_cache_dir) / f"{video_id}.{PLAYBACK_PROFILE}.mp4"


def _media_info_from_probe(payload: object) -> MediaInfo:
    if not isinstance(payload, dict):
        raise MediaProbeError("ffprobe JSON root must be an object")

    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise MediaProbeError("ffprobe output does not contain streams")

    video_stream = _select_stream(streams, "video")
    if video_stream is None:
        raise MediaProbeError("media does not contain a video stream")
    audio_stream = _select_stream(streams, "audio")

    format_data = payload.get("format")
    if not isinstance(format_data, dict):
        format_data = {}
    format_name = format_data.get("format_name")
    format_names = tuple(
        part.strip().lower()
        for part in str(format_name or "").split(",")
        if part.strip()
    )

    video_codec = str(video_stream.get("codec_name") or "").lower()
    if not video_codec:
        raise MediaProbeError("video stream does not report a codec")

    pixel_format = video_stream.get("pix_fmt")
    audio_codec = audio_stream.get("codec_name") if audio_stream else None
    return MediaInfo(
        format_names=format_names,
        duration_sec=_probe_duration(format_data, streams),
        video_codec=video_codec,
        video_pixel_format=str(pixel_format).lower() if pixel_format else None,
        audio_codec=str(audio_codec).lower() if audio_codec else None,
    )


def _select_stream(streams: list[object], codec_type: str) -> dict[str, object] | None:
    candidates = [
        stream
        for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == codec_type
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda stream: int(
            isinstance(stream.get("disposition"), dict)
            and stream["disposition"].get("default") == 1
        ),
    )


def _probe_duration(
    format_data: dict[str, object],
    streams: list[object],
) -> float | None:
    values = [format_data.get("duration")]
    values.extend(
        stream.get("duration") for stream in streams if isinstance(stream, dict)
    )
    for value in values:
        try:
            duration = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if math.isfinite(duration) and duration > 0:
            return duration
    return None


def _is_known_compatible(media_info: MediaInfo) -> bool:
    return (
        "mp4" in media_info.format_names
        and media_info.video_codec == "h264"
        and media_info.video_pixel_format in COMPATIBLE_PIXEL_FORMATS
        and media_info.audio_codec in {None, "aac"}
    )


def _is_av1_webm(media_info: MediaInfo) -> bool:
    return "webm" in media_info.format_names and media_info.video_codec == "av1"


def _validate_proxy(
    proxy_path: Path,
    original_info: MediaInfo,
    ffprobe_command: str,
) -> MediaInfo:
    if not proxy_path.is_file() or proxy_path.stat().st_size <= 0:
        raise InvalidPlaybackCacheError(f"Playback cache is empty: {proxy_path}")

    proxy_info = probe_media(proxy_path, ffprobe_command=ffprobe_command)
    if not _is_known_compatible(proxy_info):
        raise InvalidPlaybackCacheError(
            f"Playback cache is not an H.264/AAC MP4: {proxy_path}"
        )
    if original_info.duration_sec is None or proxy_info.duration_sec is None:
        raise InvalidPlaybackCacheError("Playback cache duration cannot be validated")

    tolerance = max(1.0, original_info.duration_sec * 0.01)
    if abs(proxy_info.duration_sec - original_info.duration_sec) > tolerance:
        raise InvalidPlaybackCacheError(
            f"Playback cache duration does not match the original: {proxy_path}"
        )
    return proxy_info


def _build_transcode_command(
    ffmpeg_path: str,
    original_path: Path,
    temporary_path: Path,
) -> list[str]:
    return [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(original_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-sn",
        "-dn",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        str(temporary_path),
    ]


def _resolve_tool(command: str) -> str:
    resolved = shutil.which(command)
    if resolved is None:
        raise PlaybackToolNotFoundError(
            f"Required playback tool '{command}' was not found on PATH"
        )
    return resolved


def _validate_video_id(video_id: str) -> None:
    if len(video_id) != 64 or any(
        character not in "0123456789abcdef" for character in video_id
    ):
        raise ValueError("video_id must be a lowercase SHA-256 hex digest")


def _subprocess_creation_flags() -> int:
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
