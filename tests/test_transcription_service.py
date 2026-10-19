"""Unit tests for the Streamlit-independent transcription workflow."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src import transcription_service
from src.transcript import load_transcript


VIDEO_ID = "b" * 64


def _video_path(tmp_path: Path, name: str = f"{VIDEO_ID}.webm") -> Path:
    path = tmp_path / "data" / "videos" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"original video")
    return path


def _asr_result() -> tuple[list[SimpleNamespace], SimpleNamespace, float]:
    word = SimpleNamespace(
        word=" hello ",
        start=0.1,
        end=0.5,
        probability=0.99,
    )
    segment = SimpleNamespace(
        start=0.0,
        end=1.0,
        text=" Hello world. ",
        words=[word],
    )
    info = SimpleNamespace(duration=10.0, language="en")
    return [segment], info, 2.0


def test_first_transcription_uses_baseline_and_safely_saves_json(
    tmp_path: Path,
) -> None:
    video_path = _video_path(tmp_path)
    metadata_dir = tmp_path / "data" / "metadata"
    model = object()
    model_loader = Mock(return_value=model)
    stages: list[str] = []

    with patch.object(
        transcription_service,
        "transcribe_video",
        return_value=_asr_result(),
    ) as transcribe_mock:
        result = transcription_service.transcribe_video_to_transcript(
            video_path,
            metadata_dir,
            model_loader=model_loader,
            progress_callback=stages.append,
        )

    expected_path = metadata_dir / f"{VIDEO_ID}.transcript.json"
    assert result.status == "created"
    assert result.transcript_path == expected_path
    assert load_transcript(expected_path) == result.transcript_data
    assert result.transcript_data["schema_version"] == "1.0"
    assert result.transcript_data["video"]["file_name"] == video_path.name
    assert result.transcript_data["segments"][0]["words"][0]["word"] == "hello"
    assert result.transcript_data["asr"]["model"] == "turbo"
    assert result.transcript_data["asr"]["device"] == "cuda"
    assert result.transcript_data["asr"]["compute_type"] == "float16"
    assert result.transcript_data["asr"]["beam_size"] == 1
    assert result.transcript_data["asr"]["word_timestamps"] is True
    model_loader.assert_called_once_with()
    transcribe_mock.assert_called_once_with(
        model,
        video_path,
        language="en",
        beam_size=1,
        word_timestamps=True,
    )
    assert stages == [
        "preparing",
        "loading_model",
        "transcribing",
        "saving",
        "completed",
    ]
    assert not list(metadata_dir.glob("*.tmp"))


def test_existing_valid_transcript_skips_model_and_asr(tmp_path: Path) -> None:
    video_path = _video_path(tmp_path)
    metadata_dir = tmp_path / "data" / "metadata"

    with patch.object(
        transcription_service,
        "transcribe_video",
        return_value=_asr_result(),
    ):
        created = transcription_service.transcribe_video_to_transcript(
            video_path,
            metadata_dir,
            model_loader=Mock(return_value=object()),
        )

    model_loader = Mock()
    with patch.object(transcription_service, "transcribe_video") as transcribe_mock:
        existing = transcription_service.transcribe_video_to_transcript(
            video_path,
            metadata_dir,
            model_loader=model_loader,
        )

    assert existing.status == "existing"
    assert existing.transcript_path == created.transcript_path
    assert existing.transcript_data == created.transcript_data
    model_loader.assert_not_called()
    transcribe_mock.assert_not_called()


def test_legacy_video_uses_stem_based_transcript_name(tmp_path: Path) -> None:
    video_path = _video_path(tmp_path, "legacy-lesson.mp4")
    metadata_dir = tmp_path / "data" / "metadata"

    with patch.object(
        transcription_service,
        "transcribe_video",
        return_value=_asr_result(),
    ):
        result = transcription_service.transcribe_video_to_transcript(
            video_path,
            metadata_dir,
            model_loader=Mock(return_value=object()),
        )

    assert result.transcript_path == metadata_dir / "legacy-lesson.transcript.json"
    assert result.transcript_data["video"]["file_name"] == "legacy-lesson.mp4"


def test_corrupt_existing_transcript_is_not_overwritten(tmp_path: Path) -> None:
    video_path = _video_path(tmp_path)
    metadata_dir = tmp_path / "data" / "metadata"
    metadata_dir.mkdir(parents=True)
    transcript_path = metadata_dir / f"{VIDEO_ID}.transcript.json"
    corrupt_content = "{not valid json"
    transcript_path.write_text(corrupt_content, encoding="utf-8")
    model_loader = Mock()

    with (
        pytest.raises(transcription_service.ExistingTranscriptInvalidError),
        patch.object(transcription_service, "transcribe_video") as transcribe_mock,
    ):
        transcription_service.transcribe_video_to_transcript(
            video_path,
            metadata_dir,
            model_loader=model_loader,
        )

    assert transcript_path.read_text(encoding="utf-8") == corrupt_content
    model_loader.assert_not_called()
    transcribe_mock.assert_not_called()


def test_write_failure_leaves_no_formal_or_temporary_json(tmp_path: Path) -> None:
    video_path = _video_path(tmp_path)
    metadata_dir = tmp_path / "data" / "metadata"
    target_path = metadata_dir / f"{VIDEO_ID}.transcript.json"

    def fail_after_partial_write(data: object, path: str | Path) -> None:
        Path(path).write_text("partial", encoding="utf-8")
        raise OSError("disk full")

    with (
        patch.object(
            transcription_service,
            "transcribe_video",
            return_value=_asr_result(),
        ),
        patch.object(
            transcription_service,
            "save_transcript",
            side_effect=fail_after_partial_write,
        ),
        pytest.raises(OSError, match="disk full"),
    ):
        transcription_service.transcribe_video_to_transcript(
            video_path,
            metadata_dir,
            model_loader=Mock(return_value=object()),
        )

    assert not target_path.exists()
    assert not list(metadata_dir.glob("*.tmp"))


def test_playback_proxy_is_never_substituted_for_original(tmp_path: Path) -> None:
    original_path = _video_path(tmp_path)
    proxy_path = (
        tmp_path
        / "data"
        / "cache"
        / "playback"
        / f"{VIDEO_ID}.h264-aac.v1.mp4"
    )
    proxy_path.parent.mkdir(parents=True)
    proxy_path.write_bytes(b"playback proxy")

    with patch.object(
        transcription_service,
        "transcribe_video",
        return_value=_asr_result(),
    ) as transcribe_mock:
        transcription_service.transcribe_video_to_transcript(
            original_path,
            tmp_path / "data" / "metadata",
            model_loader=Mock(return_value=object()),
        )

    assert transcribe_mock.call_args.args[1] == original_path
    assert transcribe_mock.call_args.args[1] != proxy_path


def test_model_load_failure_preserves_video_and_releases_lock(tmp_path: Path) -> None:
    video_path = _video_path(tmp_path)
    metadata_dir = tmp_path / "data" / "metadata"
    original_content = video_path.read_bytes()

    with pytest.raises(RuntimeError, match="CUDA initialization failed"):
        transcription_service.transcribe_video_to_transcript(
            video_path,
            metadata_dir,
            model_loader=Mock(side_effect=RuntimeError("CUDA initialization failed")),
        )

    assert video_path.read_bytes() == original_content
    assert not transcription_service.transcript_path_for_video(
        video_path,
        metadata_dir,
    ).exists()
    acquired = transcription_service._TRANSCRIPTION_LOCK.acquire(blocking=False)
    assert acquired
    transcription_service._TRANSCRIPTION_LOCK.release()


def test_busy_service_rejects_a_second_gpu_job(tmp_path: Path) -> None:
    video_path = _video_path(tmp_path)
    acquired = transcription_service._TRANSCRIPTION_LOCK.acquire(blocking=False)
    assert acquired
    try:
        with pytest.raises(transcription_service.TranscriptionBusyError):
            transcription_service.transcribe_video_to_transcript(
                video_path,
                tmp_path / "data" / "metadata",
                model_loader=Mock(),
            )
    finally:
        transcription_service._TRANSCRIPTION_LOCK.release()


def test_missing_or_empty_video_is_rejected_before_model_load(tmp_path: Path) -> None:
    model_loader = Mock()
    with pytest.raises(transcription_service.InvalidTranscriptionVideoError):
        transcription_service.transcribe_video_to_transcript(
            tmp_path / "missing.webm",
            tmp_path / "data" / "metadata",
            model_loader=model_loader,
        )

    empty_video = tmp_path / "empty.webm"
    empty_video.touch()
    with pytest.raises(transcription_service.InvalidTranscriptionVideoError):
        transcription_service.transcribe_video_to_transcript(
            empty_video,
            tmp_path / "data" / "metadata",
            model_loader=model_loader,
        )

    model_loader.assert_not_called()
