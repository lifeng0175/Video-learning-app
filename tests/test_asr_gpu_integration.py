"""Offline GPU integration test for the complete ASR-to-JSON pipeline.

Run explicitly with RUN_GPU_INTEGRATION=1. The test requires the existing
turbo model cache and a WebM file under data/videos; it never downloads them.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE

from src.asr import load_asr_model, transcribe_video
from src.transcript import (
    build_transcript_data,
    load_transcript,
    save_transcript,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TURBO_CACHE_DIR = (
    Path(HF_HUB_CACHE)
    / "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo"
    / "snapshots"
)


def find_cached_turbo_snapshot() -> Path | None:
    """Return a complete local turbo snapshot without network access."""
    if not TURBO_CACHE_DIR.is_dir():
        return None

    for snapshot_path in sorted(TURBO_CACHE_DIR.iterdir()):
        if snapshot_path.is_dir() and (snapshot_path / "model.bin").is_file():
            return snapshot_path
    return None


@unittest.skipUnless(
    os.environ.get("RUN_GPU_INTEGRATION") == "1",
    "set RUN_GPU_INTEGRATION=1 to run the GPU integration test",
)
class AsrGpuIntegrationTests(unittest.TestCase):
    def test_real_video_transcript_json_round_trip(self) -> None:
        video_paths = sorted((PROJECT_ROOT / "data" / "videos").glob("*.webm"))
        self.assertTrue(video_paths, "no local WebM video found in data/videos")

        model_path = find_cached_turbo_snapshot()
        self.assertIsNotNone(model_path, "cached turbo model snapshot not found")

        model = load_asr_model(model_name=str(model_path))
        segments, info, elapsed_time = transcribe_video(
            model,
            video_paths[0],
        )
        transcript_data = build_transcript_data(
            video_path=video_paths[0],
            model_name="turbo",
            language="en",
            info=info,
            segments=segments,
            device="cuda",
            compute_type="float16",
            beam_size=1,
            word_timestamps=True,
            processing_time_sec=elapsed_time,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = save_transcript(
                transcript_data,
                Path(temp_dir) / "transcript.json",
            )
            loaded_transcript = load_transcript(json_path)

        self.assertEqual(loaded_transcript, transcript_data)
        self.assertEqual(loaded_transcript["schema_version"], "1.0")
        self.assertNotIn("file_path", loaded_transcript["video"])
        self.assertEqual(loaded_transcript["asr"]["library"], "faster-whisper")
        self.assertEqual(
            loaded_transcript["asr"]["library_version"],
            version("faster-whisper"),
        )
        self.assertEqual(loaded_transcript["asr"]["detected_language"], "en")
        self.assertGreater(len(loaded_transcript["segments"]), 0)
        self.assertTrue(
            any(segment["words"] for segment in loaded_transcript["segments"])
        )
        self.assertGreater(loaded_transcript["asr"]["processing_time_sec"], 0)
        self.assertGreater(loaded_transcript["asr"]["realtime_factor"], 0)
        self.assertGreater(loaded_transcript["asr"]["realtime_speedup"], 0)


if __name__ == "__main__":
    unittest.main()
