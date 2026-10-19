"""Adapter between uploaded file objects and the local Video Loader."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import BinaryIO

from src.video_loader import (
    DEFAULT_CHUNK_SIZE,
    SUPPORTED_VIDEO_SUFFIXES,
    UnsupportedVideoFormatError,
    VideoImportResult,
    import_video,
)


def import_uploaded_video(
    uploaded_file: BinaryIO,
    original_file_name: str,
    videos_dir: str | Path,
    staging_dir: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> VideoImportResult:
    """Stage an uploaded file, then delegate its import to the Video Loader."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    safe_file_name = _safe_file_name(original_file_name)
    suffix = Path(safe_file_name).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_SUFFIXES:
        raise UnsupportedVideoFormatError(
            f"Unsupported video format '{Path(safe_file_name).suffix}'; "
            "expected MP4 or WebM"
        )

    videos_dir = Path(videos_dir)
    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        uploaded_file.seek(0)
    except (AttributeError, OSError):
        pass

    with tempfile.TemporaryDirectory(prefix="upload-", dir=staging_dir) as temp_dir:
        staged_path = Path(temp_dir) / safe_file_name
        with staged_path.open("xb") as staged_file:
            while chunk := uploaded_file.read(chunk_size):
                staged_file.write(chunk)
            staged_file.flush()
            os.fsync(staged_file.fileno())

        return import_video(staged_path, videos_dir, chunk_size=chunk_size)


def _safe_file_name(file_name: str) -> str:
    normalized_name = file_name.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not normalized_name or normalized_name in {".", ".."}:
        raise ValueError("uploaded file must have a valid file name")
    return normalized_name
