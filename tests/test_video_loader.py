"""Pytest tests for content-addressed local video imports."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from src import video_loader
from src.video_loader import (
    InvalidVideoFileError,
    UnsupportedVideoFormatError,
    import_video,
    resolve_video_path,
)
from src.video_transcript import load_video_transcript


@pytest.mark.parametrize("suffix", [".mp4", ".MP4", ".webm", ".WEBM"])
def test_imports_supported_video_with_streaming_hash(
    tmp_path: Path,
    suffix: str,
) -> None:
    content = b"video-content-" * 17
    source = tmp_path / f"source{suffix}"
    source.write_bytes(content)

    result = import_video(source, tmp_path / "library", chunk_size=7)

    expected_id = hashlib.sha256(content).hexdigest()
    assert result.video_id == expected_id
    assert result.original_file_name == source.name
    assert result.stored_file_name == f"{expected_id}{suffix.lower()}"
    assert result.stored_path.read_bytes() == content
    assert result.size_bytes == len(content)
    assert result.is_duplicate is False
    assert list((tmp_path / "library").glob("*.partial")) == []


def test_same_content_with_different_names_is_deduplicated(tmp_path: Path) -> None:
    content = b"same video bytes"
    first_source = tmp_path / "first.mp4"
    second_source = tmp_path / "renamed.mp4"
    first_source.write_bytes(content)
    second_source.write_bytes(content)
    videos_dir = tmp_path / "videos"

    first = import_video(first_source, videos_dir, chunk_size=3)
    second = import_video(second_source, videos_dir, chunk_size=3)

    assert first.video_id == second.video_id
    assert second.stored_path == first.stored_path
    assert second.original_file_name == "renamed.mp4"
    assert first.is_duplicate is False
    assert second.is_duplicate is True
    assert len(list(videos_dir.glob("*.mp4"))) == 1


def test_same_name_with_different_content_does_not_overwrite(tmp_path: Path) -> None:
    first_source = tmp_path / "one" / "video.webm"
    second_source = tmp_path / "two" / "video.webm"
    first_source.parent.mkdir()
    second_source.parent.mkdir()
    first_source.write_bytes(b"first")
    second_source.write_bytes(b"second")
    videos_dir = tmp_path / "videos"

    first = import_video(first_source, videos_dir)
    second = import_video(second_source, videos_dir)

    assert first.video_id != second.video_id
    assert first.stored_path != second.stored_path
    assert first.stored_path.read_bytes() == b"first"
    assert second.stored_path.read_bytes() == b"second"


def test_concurrent_winner_is_validated_and_returned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"concurrent video"
    source = tmp_path / "source.mp4"
    source.write_bytes(content)
    videos_dir = tmp_path / "videos"

    def simulate_competing_publish(source_path: Path, destination_path: Path) -> None:
        destination_path.write_bytes(Path(source_path).read_bytes())
        raise FileExistsError(destination_path)

    monkeypatch.setattr(video_loader.os, "link", simulate_competing_publish)

    result = import_video(source, videos_dir, chunk_size=4)

    assert result.is_duplicate is True
    assert result.stored_path.read_bytes() == content
    assert list(videos_dir.glob("*.partial")) == []


def test_existing_corrupt_target_is_not_overwritten(tmp_path: Path) -> None:
    content = b"expected content"
    source = tmp_path / "source.mp4"
    source.write_bytes(content)
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    video_id = hashlib.sha256(content).hexdigest()
    destination = videos_dir / f"{video_id}.mp4"
    destination.write_bytes(b"corrupt")

    with pytest.raises(InvalidVideoFileError, match="content ID"):
        import_video(source, videos_dir)

    assert destination.read_bytes() == b"corrupt"
    assert list(videos_dir.glob("*.partial")) == []


def test_source_change_aborts_without_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.webm"
    source.write_bytes(b"content")
    videos_dir = tmp_path / "videos"
    monkeypatch.setattr(video_loader, "_source_is_unchanged", lambda *args: False)

    with pytest.raises(InvalidVideoFileError, match="changed"):
        import_video(source, videos_dir)

    assert list(videos_dir.glob("*.webm")) == []
    assert list(videos_dir.glob("*.partial")) == []


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_rejects_invalid_chunk_size(tmp_path: Path, chunk_size: int) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"content")

    with pytest.raises(ValueError, match="chunk_size"):
        import_video(source, tmp_path / "videos", chunk_size=chunk_size)


def test_rejects_missing_directory_unsupported_and_empty_sources(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        import_video(tmp_path / "missing.mp4", tmp_path / "videos")

    source_directory = tmp_path / "directory.mp4"
    source_directory.mkdir()
    with pytest.raises(IsADirectoryError):
        import_video(source_directory, tmp_path / "videos")

    unsupported = tmp_path / "video.mov"
    unsupported.write_bytes(b"content")
    with pytest.raises(UnsupportedVideoFormatError):
        import_video(unsupported, tmp_path / "videos")

    empty = tmp_path / "empty.webm"
    empty.touch()
    with pytest.raises(InvalidVideoFileError, match="empty"):
        import_video(empty, tmp_path / "videos")


def test_resolve_video_path_and_invalid_library_states(tmp_path: Path) -> None:
    content = b"resolve me"
    source = tmp_path / "source.webm"
    source.write_bytes(content)
    videos_dir = tmp_path / "videos"
    result = import_video(source, videos_dir)

    assert resolve_video_path(result.video_id, videos_dir) == result.stored_path

    with pytest.raises(ValueError, match="SHA-256"):
        resolve_video_path("not-a-video-id", videos_dir)
    with pytest.raises(FileNotFoundError):
        resolve_video_path("0" * 64, videos_dir)

    duplicate_format = videos_dir / f"{result.video_id}.mp4"
    duplicate_format.write_bytes(content)
    with pytest.raises(InvalidVideoFileError, match="Multiple"):
        resolve_video_path(result.video_id, videos_dir)


def test_imported_file_is_compatible_with_current_video_locator(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    source = tmp_path / "source.webm"
    source.write_bytes(b"webm bytes")
    result = import_video(source, data_dir / "videos")
    transcript_path = data_dir / "metadata" / f"{result.video_id}.transcript.json"
    transcript_path.parent.mkdir()
    transcript_path.write_text(
        json.dumps(
            {
                "video": {"file_name": result.stored_file_name, "duration": 1.0},
                "segments": [{"start": 0.0, "end": 1.0, "text": "Hello."}],
            }
        ),
        encoding="utf-8",
    )

    _, resolved_video = load_video_transcript(transcript_path, data_dir)

    assert resolved_video == result.stored_path


def test_publish_failure_does_not_leave_partial_or_final_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"content")
    videos_dir = tmp_path / "videos"

    def fail_publish(source_path: Path, destination_path: Path) -> None:
        raise OSError("hard links unavailable")

    monkeypatch.setattr(video_loader.os, "link", fail_publish)

    with pytest.raises(OSError, match="without overwriting"):
        import_video(source, videos_dir)

    assert list(videos_dir.iterdir()) == []
