"""Application workflow for ASR transcription and transcript persistence."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from src.asr import load_asr_model, transcribe_video
from src.transcript import build_transcript_data, load_transcript, save_transcript
from src.video_loader import SUPPORTED_VIDEO_SUFFIXES
from src.video_transcript import (
    TranscriptFormatError,
    find_transcript_for_video,
    load_video_transcript,
)


MODEL_NAME = "turbo"
LANGUAGE = "en"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
BEAM_SIZE = 1
WORD_TIMESTAMPS = True

TranscriptionStage = Literal[
    "preparing",
    "loading_model",
    "transcribing",
    "saving",
    "completed",
]
TranscriptionStatus = Literal["created", "existing"]
ModelLoader = Callable[[], Any]
ProgressCallback = Callable[[TranscriptionStage], None]

_TRANSCRIPTION_LOCK = threading.Lock()


class TranscriptionServiceError(RuntimeError):
    """Base error for the application transcription workflow."""


class InvalidTranscriptionVideoError(TranscriptionServiceError):
    """Raised when the original video cannot be used for transcription."""


class ExistingTranscriptInvalidError(TranscriptionServiceError):
    """Raised when an existing transcript is invalid or belongs to another video."""


class TranscriptionBusyError(TranscriptionServiceError):
    """Raised when another transcription is already using the shared model."""


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Outcome of creating or reusing a transcript."""

    status: TranscriptionStatus
    transcript_path: Path
    transcript_data: dict[str, Any]


def transcribe_video_to_transcript(
    video_path: str | Path,
    metadata_dir: str | Path,
    *,
    model_loader: ModelLoader | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TranscriptionResult:
    """Transcribe an original video and safely persist its transcript JSON."""
    video_path = _validate_video_path(video_path)
    metadata_dir = Path(metadata_dir)
    data_dir = metadata_dir.parent
    target_path = transcript_path_for_video(video_path, metadata_dir)
    _notify(progress_callback, "preparing")

    existing_path = find_transcript_for_video(video_path, data_dir)
    if existing_path is not None:
        return _load_existing_result(existing_path, video_path, data_dir)

    if not _TRANSCRIPTION_LOCK.acquire(blocking=False):
        raise TranscriptionBusyError("another transcription is already running")

    try:
        existing_path = find_transcript_for_video(video_path, data_dir)
        if existing_path is not None:
            return _load_existing_result(existing_path, video_path, data_dir)

        _notify(progress_callback, "loading_model")
        loader = model_loader or load_asr_model
        model = loader()

        _notify(progress_callback, "transcribing")
        segments, info, elapsed_time = transcribe_video(
            model,
            video_path,
            language=LANGUAGE,
            beam_size=BEAM_SIZE,
            word_timestamps=WORD_TIMESTAMPS,
        )
        transcript_data = build_transcript_data(
            video_path=video_path,
            model_name=MODEL_NAME,
            language=LANGUAGE,
            info=info,
            segments=segments,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            beam_size=BEAM_SIZE,
            word_timestamps=WORD_TIMESTAMPS,
            processing_time_sec=elapsed_time,
        )

        _notify(progress_callback, "saving")
        published = _publish_transcript_without_overwrite(
            transcript_data,
            target_path,
        )
        if not published:
            return _load_existing_result(target_path, video_path, data_dir)

        _notify(progress_callback, "completed")
        return TranscriptionResult(
            status="created",
            transcript_path=target_path,
            transcript_data=transcript_data,
        )
    finally:
        _TRANSCRIPTION_LOCK.release()


def transcript_path_for_video(
    video_path: str | Path,
    metadata_dir: str | Path,
) -> Path:
    """Return the stable transcript path for an imported or legacy video."""
    video_path = Path(video_path)
    return Path(metadata_dir) / f"{video_path.stem}.transcript.json"


def _validate_video_path(video_path: str | Path) -> Path:
    video_path = Path(video_path)
    if not video_path.exists():
        raise InvalidTranscriptionVideoError(f"video not found: {video_path}")
    if not video_path.is_file():
        raise InvalidTranscriptionVideoError(f"video is not a file: {video_path}")
    if video_path.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        raise InvalidTranscriptionVideoError(
            f"unsupported video format: {video_path.suffix}"
        )
    try:
        with video_path.open("rb") as video_file:
            if not video_file.read(1):
                raise InvalidTranscriptionVideoError(f"video is empty: {video_path}")
    except OSError as error:
        raise InvalidTranscriptionVideoError(
            f"video is not readable: {video_path}"
        ) from error
    return video_path


def _load_existing_result(
    transcript_path: Path,
    video_path: Path,
    data_dir: Path,
) -> TranscriptionResult:
    try:
        transcript_data, resolved_video_path = load_video_transcript(
            transcript_path,
            data_dir,
        )
    except (
        json.JSONDecodeError,
        TranscriptFormatError,
        FileNotFoundError,
        OSError,
        UnicodeError,
    ) as error:
        raise ExistingTranscriptInvalidError(
            f"existing transcript is invalid: {transcript_path}"
        ) from error

    if resolved_video_path.resolve() != video_path.resolve():
        raise ExistingTranscriptInvalidError(
            f"existing transcript belongs to another video: {transcript_path}"
        )

    return TranscriptionResult(
        status="existing",
        transcript_path=transcript_path,
        transcript_data=transcript_data,
    )


def _publish_transcript_without_overwrite(
    transcript_data: Mapping[str, Any],
    target_path: Path,
) -> bool:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target_path.parent,
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        save_transcript(transcript_data, temporary_path)
        if load_transcript(temporary_path) != dict(transcript_data):
            raise TranscriptionServiceError("temporary transcript validation failed")

        try:
            os.link(temporary_path, target_path)
        except FileExistsError:
            return False
        temporary_path.unlink()
        temporary_path = None
        return True
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _notify(
    progress_callback: ProgressCallback | None,
    stage: TranscriptionStage,
) -> None:
    if progress_callback is not None:
        progress_callback(stage)
