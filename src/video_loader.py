"""Safe, content-addressed imports for local MP4 and WebM videos."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024
SUPPORTED_VIDEO_SUFFIXES = frozenset({".mp4", ".webm"})


class UnsupportedVideoFormatError(ValueError):
    """Raised when an import source is not a supported video type."""


class InvalidVideoFileError(ValueError):
    """Raised when video contents or library state fail integrity checks."""


@dataclass(frozen=True, slots=True)
class VideoImportResult:
    """Result of importing or deduplicating a local video."""

    video_id: str
    original_file_name: str
    stored_file_name: str
    stored_path: Path
    size_bytes: int
    is_duplicate: bool


def import_video(
    source_path: str | Path,
    videos_dir: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> VideoImportResult:
    """Copy a local video into content-addressed storage without overwriting."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    source_path = Path(source_path)
    videos_dir = Path(videos_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Video source not found: {source_path}")
    if not source_path.is_file():
        raise IsADirectoryError(f"Video source is not a file: {source_path}")

    suffix = source_path.suffix.lower()
    if suffix not in SUPPORTED_VIDEO_SUFFIXES:
        raise UnsupportedVideoFormatError(
            f"Unsupported video format '{source_path.suffix}'; "
            "expected MP4 or WebM"
        )

    source_stat_before = source_path.stat()
    if source_stat_before.st_size == 0:
        raise InvalidVideoFileError(f"Video source is empty: {source_path}")

    videos_dir.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=videos_dir,
            prefix=".import-",
            suffix=".partial",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            digest = hashlib.sha256()
            copied_size = 0

            with source_path.open("rb") as source_file:
                while chunk := source_file.read(chunk_size):
                    temporary_file.write(chunk)
                    digest.update(chunk)
                    copied_size += len(chunk)

            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        source_stat_after = source_path.stat()
        if not _source_is_unchanged(
            source_stat_before,
            source_stat_after,
            copied_size,
        ):
            raise InvalidVideoFileError(
                f"Video source changed while it was being imported: {source_path}"
            )

        video_id = digest.hexdigest()
        existing_path = _validated_existing_path(
            videos_dir,
            video_id,
            copied_size,
            chunk_size,
        )
        if existing_path is not None:
            return _build_result(
                video_id=video_id,
                source_path=source_path,
                stored_path=existing_path,
                size_bytes=copied_size,
                is_duplicate=True,
            )

        destination_path = videos_dir / f"{video_id}{suffix}"
        try:
            os.link(temporary_path, destination_path)
        except FileExistsError:
            # Another importer won the atomic create race. Trust it only after
            # validating that the published file really has the expected ID.
            existing_path = _validated_existing_path(
                videos_dir,
                video_id,
                copied_size,
                chunk_size,
            )
            if existing_path is None:
                raise InvalidVideoFileError(
                    f"Concurrent import created an invalid target: {destination_path}"
                )
            return _build_result(
                video_id=video_id,
                source_path=source_path,
                stored_path=existing_path,
                size_bytes=copied_size,
                is_duplicate=True,
            )
        except OSError as error:
            raise OSError(
                "Could not publish the imported video without overwriting; "
                "the destination filesystem must support hard links"
            ) from error

        return _build_result(
            video_id=video_id,
            source_path=source_path,
            stored_path=destination_path,
            size_bytes=copied_size,
            is_duplicate=False,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def resolve_video_path(video_id: str, videos_dir: str | Path) -> Path:
    """Resolve a content-addressed video ID to its stored local path."""
    if len(video_id) != 64 or any(
        character not in "0123456789abcdef" for character in video_id
    ):
        raise ValueError("video_id must be a lowercase SHA-256 hex digest")

    candidates = _video_id_candidates(Path(videos_dir), video_id)
    if not candidates:
        raise FileNotFoundError(f"No imported video found for ID: {video_id}")
    if len(candidates) > 1:
        raise InvalidVideoFileError(
            f"Multiple stored videos found for ID {video_id}: {candidates}"
        )
    return candidates[0]


def _source_is_unchanged(
    before: os.stat_result,
    after: os.stat_result,
    copied_size: int,
) -> bool:
    return (
        copied_size == before.st_size
        and after.st_size == before.st_size
        and after.st_mtime_ns == before.st_mtime_ns
    )


def _validated_existing_path(
    videos_dir: Path,
    video_id: str,
    expected_size: int,
    chunk_size: int,
) -> Path | None:
    candidates = _video_id_candidates(videos_dir, video_id)
    if not candidates:
        return None
    if len(candidates) > 1:
        raise InvalidVideoFileError(
            f"Multiple stored videos found for ID {video_id}: {candidates}"
        )

    existing_path = candidates[0]
    if existing_path.stat().st_size != expected_size:
        raise InvalidVideoFileError(
            f"Stored video does not match its content ID: {existing_path}"
        )
    if _hash_file(existing_path, chunk_size) != video_id:
        raise InvalidVideoFileError(
            f"Stored video does not match its content ID: {existing_path}"
        )
    return existing_path


def _video_id_candidates(videos_dir: Path, video_id: str) -> list[Path]:
    return sorted(
        candidate
        for suffix in SUPPORTED_VIDEO_SUFFIXES
        if (candidate := videos_dir / f"{video_id}{suffix}").is_file()
    )


def _hash_file(path: Path, chunk_size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _build_result(
    *,
    video_id: str,
    source_path: Path,
    stored_path: Path,
    size_bytes: int,
    is_duplicate: bool,
) -> VideoImportResult:
    return VideoImportResult(
        video_id=video_id,
        original_file_name=source_path.name,
        stored_file_name=stored_path.name,
        stored_path=stored_path,
        size_bytes=size_bytes,
        is_duplicate=is_duplicate,
    )
