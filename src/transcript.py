"""Transcript serialization helpers for faster-whisper results."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence, cast

if TYPE_CHECKING:
    from faster_whisper.transcribe import Segment, TranscriptionInfo


def build_transcript_data(
    video_path: str | Path,
    model_name: str,
    language: str,
    info: TranscriptionInfo,
    segments: Sequence[Segment],
    *,
    device: str = "cuda",
    compute_type: str = "float16",
    beam_size: int = 1,
    word_timestamps: bool = True,
    processing_time_sec: float,
    schema_version: str = "1.0",
) -> dict[str, Any]:
    """Convert faster-whisper results to the application's transcript schema."""
    transcript_segments: list[dict[str, Any]] = []

    for segment_id, segment in enumerate(segments):
        words: list[dict[str, Any]] = []
        if segment.words:
            for word_id, word in enumerate(segment.words):
                words.append(
                    {
                        "id": word_id,
                        "word": word.word.strip(),
                        "start": round(word.start, 3),
                        "end": round(word.end, 3),
                        "probability": round(word.probability, 4),
                    }
                )

        transcript_segments.append(
            {
                "id": segment_id,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.text.strip(),
                "words": words,
            }
        )

    video_path = Path(video_path)
    realtime_factor = (
        round(processing_time_sec / info.duration, 3)
        if info.duration > 0
        else None
    )
    realtime_speedup = (
        round(info.duration / processing_time_sec, 2)
        if processing_time_sec > 0
        else None
    )

    return {
        "schema_version": schema_version,
        "video": {
            "file_name": video_path.name,
            "duration": round(info.duration, 3),
        },
        "asr": {
            "model": model_name,
            "library": "faster-whisper",
            "library_version": version("faster-whisper"),
            "requested_language": language,
            "detected_language": info.language,
            "device": device,
            "compute_type": compute_type,
            "beam_size": beam_size,
            "word_timestamps": word_timestamps,
            "processing_time_sec": round(processing_time_sec, 3),
            "realtime_factor": realtime_factor,
            "realtime_speedup": realtime_speedup,
        },
        "segments": transcript_segments,
    }


def save_transcript(
    transcript_data: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Save transcript data as readable UTF-8 JSON and return its path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(transcript_data, file, ensure_ascii=False, indent=2)

    return output_path


def load_transcript(input_path: str | Path) -> dict[str, Any]:
    """Load transcript JSON from disk."""
    with Path(input_path).open("r", encoding="utf-8") as file:
        return cast(dict[str, Any], json.load(file))
