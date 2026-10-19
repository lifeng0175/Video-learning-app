"""Data loading helpers for the local video and transcript UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.transcript import load_transcript
from src.video_loader import SUPPORTED_VIDEO_SUFFIXES


class TranscriptFormatError(ValueError):
    """Raised when transcript JSON lacks data required by the UI."""


def discover_transcript_files(data_dir: str | Path) -> list[Path]:
    """Find transcript JSON files under the project data directory."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return []
    return sorted(data_dir.rglob("*.transcript.json"))


def discover_video_files(data_dir: str | Path) -> list[Path]:
    """Find supported videos in the project's unified video directory."""
    videos_dir = Path(data_dir) / "videos"
    if not videos_dir.is_dir():
        return []
    return sorted(
        path
        for path in videos_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES
    )


def find_transcript_for_video(
    video_path: str | Path,
    data_dir: str | Path,
) -> Path | None:
    """Find the transcript associated with a stored video, if one exists."""
    video_path = Path(video_path)
    transcript_paths = discover_transcript_files(data_dir)

    expected_name = f"{video_path.stem}.transcript.json"
    for transcript_path in transcript_paths:
        if transcript_path.name == expected_name:
            return transcript_path

    for transcript_path in transcript_paths:
        try:
            transcript_data = load_transcript(transcript_path)
        except (json.JSONDecodeError, OSError, UnicodeError):
            continue

        video = transcript_data.get("video")
        if isinstance(video, dict) and video.get("file_name") == video_path.name:
            return transcript_path

    return None


def load_video_transcript(
    transcript_path: str | Path,
    data_dir: str | Path,
) -> tuple[dict[str, Any], Path]:
    """Load a transcript and resolve its video within the project data tree."""
    transcript_data = load_transcript(transcript_path)
    file_name = _validate_transcript_data(transcript_data)
    video_path = _find_video_path(data_dir, file_name)
    return transcript_data, video_path


def _validate_transcript_data(transcript_data: object) -> str:
    if not isinstance(transcript_data, dict):
        raise TranscriptFormatError("the JSON root must be an object")

    video = transcript_data.get("video")
    if not isinstance(video, dict):
        raise TranscriptFormatError("video must be an object")

    file_name = video.get("file_name")
    if not isinstance(file_name, str) or not file_name.strip():
        raise TranscriptFormatError("video.file_name must be a non-empty string")
    if Path(file_name).name != file_name:
        raise TranscriptFormatError("video.file_name must not contain a path")

    segments = transcript_data.get("segments")
    if not isinstance(segments, list):
        raise TranscriptFormatError("segments must be an array")

    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise TranscriptFormatError(f"segments[{index}] must be an object")

        for field in ("start", "end"):
            value = segment.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TranscriptFormatError(
                    f"segments[{index}].{field} must be a number"
                )

        if not isinstance(segment.get("text"), str):
            raise TranscriptFormatError(f"segments[{index}].text must be a string")

    return file_name


def _find_video_path(data_dir: str | Path, file_name: str) -> Path:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Project data directory not found: {data_dir}")

    matches = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.name == file_name
    )
    if not matches:
        raise FileNotFoundError(
            f"Video '{file_name}' could not be located under: {data_dir}"
        )
    return matches[0]
