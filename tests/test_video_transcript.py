"""Unit tests for Phase 1B video/transcript data loading."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.video_transcript import (
    TranscriptFormatError,
    discover_transcript_files,
    discover_video_files,
    find_transcript_for_video,
    load_video_transcript,
)


class VideoTranscriptTests(unittest.TestCase):
    def test_discovers_supported_videos_in_unified_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            videos_dir = data_dir / "videos"
            videos_dir.mkdir()
            mp4_path = videos_dir / "first.mp4"
            webm_path = videos_dir / "second.WEBM"
            mp4_path.write_bytes(b"mp4")
            webm_path.write_bytes(b"webm")
            (videos_dir / "ignored.mov").write_bytes(b"mov")

            self.assertEqual(
                discover_video_files(data_dir),
                sorted([mp4_path, webm_path]),
            )

    def test_discovers_nested_transcript_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            transcript_path = data_dir / "metadata" / "sample.transcript.json"
            transcript_path.parent.mkdir()
            transcript_path.write_text("{}", encoding="utf-8")

            self.assertEqual(
                discover_transcript_files(data_dir),
                [transcript_path],
            )

    def test_loads_transcript_and_resolves_video_by_file_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            transcript_path = data_dir / "metadata" / "sample.transcript.json"
            video_path = data_dir / "videos" / "sample.webm"
            transcript_path.parent.mkdir()
            video_path.parent.mkdir()
            video_path.write_bytes(b"video")
            transcript_path.write_text(
                json.dumps(
                    {
                        "video": {
                            "file_name": "sample.webm",
                            "file_path": r"D:\stale\sample.webm",
                            "duration": 10.0,
                        },
                        "segments": [
                            {"start": 0.0, "end": 1.0, "text": "Hello."}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            transcript_data, resolved_video = load_video_transcript(
                transcript_path,
                data_dir,
            )

            self.assertEqual(resolved_video, video_path)
            self.assertEqual(transcript_data["segments"][0]["text"], "Hello.")

    def test_finds_transcript_for_video_by_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            video_path = data_dir / "videos" / "content-id.webm"
            transcript_path = data_dir / "metadata" / "custom-name.transcript.json"
            video_path.parent.mkdir()
            transcript_path.parent.mkdir()
            video_path.write_bytes(b"video")
            transcript_path.write_text(
                json.dumps(
                    {
                        "video": {"file_name": video_path.name},
                        "segments": [],
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                find_transcript_for_video(video_path, data_dir),
                transcript_path,
            )

    def test_returns_none_when_video_has_no_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            video_path = data_dir / "videos" / "content-id.mp4"
            video_path.parent.mkdir()
            video_path.write_bytes(b"video")

            self.assertIsNone(find_transcript_for_video(video_path, data_dir))

    def test_rejects_invalid_json_structure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            transcript_path = data_dir / "sample.transcript.json"
            transcript_path.write_text(
                json.dumps({"video": {"file_name": "sample.webm"}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(TranscriptFormatError, "segments"):
                load_video_transcript(transcript_path, data_dir)

    def test_reports_invalid_json_syntax(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            transcript_path = data_dir / "broken.transcript.json"
            transcript_path.write_text("{broken", encoding="utf-8")

            with self.assertRaises(json.JSONDecodeError):
                load_video_transcript(transcript_path, data_dir)

    def test_reports_missing_transcript_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)

            with self.assertRaises(FileNotFoundError):
                load_video_transcript(
                    data_dir / "missing.transcript.json",
                    data_dir,
                )

    def test_reports_missing_video(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            transcript_path = data_dir / "sample.transcript.json"
            transcript_path.write_text(
                json.dumps(
                    {
                        "video": {"file_name": "missing.webm"},
                        "segments": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "could not be located"):
                load_video_transcript(transcript_path, data_dir)


if __name__ == "__main__":
    unittest.main()
