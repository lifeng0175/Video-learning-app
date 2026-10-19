"""Unit tests for transcript construction and JSON persistence."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.transcript import (
    build_transcript_data,
    load_transcript,
    save_transcript,
)


class TranscriptUnitTests(unittest.TestCase):
    @patch("src.transcript.version", return_value="9.9.9")
    def test_build_transcript_data_uses_schema_1_0(self, version: object) -> None:
        word = SimpleNamespace(
            word=" hello ",
            start=0.1234,
            end=0.5678,
            probability=0.98765,
        )
        segment = SimpleNamespace(
            start=0.1234,
            end=1.2345,
            text=" Hello world. ",
            words=[word],
        )
        info = SimpleNamespace(duration=87.82, language="en")

        data = build_transcript_data(
            Path("videos") / "sample.webm",
            "turbo",
            "en",
            info,
            [segment],
            processing_time_sec=1.94,
        )

        self.assertEqual(data["schema_version"], "1.0")
        self.assertEqual(
            data["video"],
            {"file_name": "sample.webm", "duration": 87.82},
        )
        self.assertEqual(data["asr"]["library"], "faster-whisper")
        self.assertEqual(data["asr"]["library_version"], "9.9.9")
        self.assertEqual(data["asr"]["processing_time_sec"], 1.94)
        self.assertEqual(data["asr"]["realtime_factor"], 0.022)
        self.assertEqual(data["asr"]["realtime_speedup"], 45.27)
        self.assertEqual(
            data["segments"][0]["words"][0],
            {
                "id": 0,
                "word": "hello",
                "start": 0.123,
                "end": 0.568,
                "probability": 0.9877,
            },
        )

    @patch("src.transcript.version", return_value="9.9.9")
    def test_zero_durations_do_not_divide_by_zero(self, version: object) -> None:
        data = build_transcript_data(
            "sample.webm",
            "turbo",
            "en",
            SimpleNamespace(duration=0.0, language="en"),
            [],
            processing_time_sec=0.0,
        )

        self.assertIsNone(data["asr"]["realtime_factor"])
        self.assertIsNone(data["asr"]["realtime_speedup"])

    @patch("src.transcript.version", return_value="9.9.9")
    def test_save_and_load_transcript_round_trip(self, version: object) -> None:
        data = build_transcript_data(
            "sample.webm",
            "turbo",
            "en",
            SimpleNamespace(duration=10.0, language="en"),
            [],
            processing_time_sec=2.0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "metadata" / "transcript.json"
            saved_path = save_transcript(data, output_path)
            loaded_data = load_transcript(saved_path)

        self.assertEqual(saved_path, output_path)
        self.assertEqual(loaded_data, data)


if __name__ == "__main__":
    unittest.main()
