"""Tests for the Streamlit upload-to-Video-Loader adapter."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from src.video_loader import UnsupportedVideoFormatError
from src.video_upload import import_uploaded_video


class TrackingUpload(io.BytesIO):
    """BytesIO that records requested read sizes."""

    def __init__(self, content: bytes) -> None:
        super().__init__(content)
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return super().read(size)


def test_uploaded_video_is_staged_in_chunks_and_imported(tmp_path: Path) -> None:
    upload = TrackingUpload(b"uploaded-video-content")
    videos_dir = tmp_path / "data" / "videos"
    staging_dir = tmp_path / "data" / "cache" / "uploads"

    result = import_uploaded_video(
        upload,
        "lesson.webm",
        videos_dir,
        staging_dir,
        chunk_size=5,
    )

    assert result.original_file_name == "lesson.webm"
    assert result.stored_path.parent == videos_dir
    assert result.stored_path.read_bytes() == b"uploaded-video-content"
    assert upload.read_sizes and set(upload.read_sizes) == {5}
    assert list(staging_dir.iterdir()) == []


def test_duplicate_upload_reuses_existing_video(tmp_path: Path) -> None:
    videos_dir = tmp_path / "videos"
    staging_dir = tmp_path / "uploads"

    first = import_uploaded_video(
        io.BytesIO(b"same-content"),
        "first.mp4",
        videos_dir,
        staging_dir,
    )
    second = import_uploaded_video(
        io.BytesIO(b"same-content"),
        "renamed.mp4",
        videos_dir,
        staging_dir,
    )

    assert first.is_duplicate is False
    assert second.is_duplicate is True
    assert second.stored_path == first.stored_path
    assert second.original_file_name == "renamed.mp4"
    assert len(list(videos_dir.glob("*.mp4"))) == 1


def test_upload_adapter_sanitizes_name_and_rejects_unsupported_type(
    tmp_path: Path,
) -> None:
    result = import_uploaded_video(
        io.BytesIO(b"content"),
        r"C:\user\lesson.webm",
        tmp_path / "videos",
        tmp_path / "uploads",
    )
    assert result.original_file_name == "lesson.webm"

    with pytest.raises(UnsupportedVideoFormatError):
        import_uploaded_video(
            io.BytesIO(b"content"),
            "lesson.mov",
            tmp_path / "videos",
            tmp_path / "uploads",
        )
