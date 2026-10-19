"""Streamlit smoke test for the Phase 1B static UI."""

from __future__ import annotations

import json
import os
import unittest
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.playback import MediaInfo, PlaybackAssessment
from src.transcription_service import TranscriptionResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_fake_transcript(
    video_path: str | Path,
    metadata_dir: str | Path,
    **kwargs: object,
) -> TranscriptionResult:
    """Write a minimal valid transcript without loading an ASR model."""
    video_path = Path(video_path)
    transcript_path = Path(metadata_dir) / f"{video_path.stem}.transcript.json"
    transcript_data = {
        "schema_version": "1.0",
        "video": {"file_name": video_path.name, "duration": 1.0},
        "asr": {
            "model": "turbo",
            "device": "cuda",
            "compute_type": "float16",
            "beam_size": 1,
            "word_timestamps": True,
        },
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "Hello from ASR.",
                "words": [
                    {
                        "id": 0,
                        "word": "Hello",
                        "start": 0.0,
                        "end": 0.5,
                        "probability": 0.99,
                    }
                ],
            }
        ],
    }
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        json.dumps(transcript_data),
        encoding="utf-8",
    )
    return TranscriptionResult(
        status="created",
        transcript_path=transcript_path,
        transcript_data=transcript_data,
    )


class AppSmokeTests(unittest.TestCase):
    def test_app_renders_local_video_and_transcript(self) -> None:
        with TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            video_path = data_dir / "videos" / "lesson.webm"
            transcript_path = data_dir / "metadata" / "lesson.transcript.json"
            video_path.parent.mkdir(parents=True)
            transcript_path.parent.mkdir(parents=True)
            video_path.write_bytes(b"video")
            transcript_path.write_text(
                json.dumps(
                    {
                        "video": {"file_name": video_path.name, "duration": 1.0},
                        "segments": [
                            {"start": 0.0, "end": 1.0, "text": "Hello."}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch.dict(
                    os.environ,
                    {"VIDEO_LEARNING_DATA_DIR": str(data_dir)},
                ),
                patch(
                    "src.transcription_service.transcribe_video_to_transcript"
                ) as transcribe_mock,
            ):
                app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=20)

            self.assertEqual(list(app.exception), [])
            self.assertEqual(list(app.error), [])
            self.assertEqual(
                [title.value for title in app.title],
                ["動画・文字起こし"],
            )
            self.assertEqual(
                [subheader.value for subheader in app.subheader],
                ["動画", "文字起こし"],
            )
            self.assertEqual(len(app.get("video")), 1)
            self.assertEqual(len(app.get("markdown")), 1)
            transcribe_mock.assert_not_called()

    def test_auto_transcription_setting_persists_across_app_restart(self) -> None:
        with TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"

            with patch.dict(
                os.environ,
                {"VIDEO_LEARNING_DATA_DIR": str(data_dir)},
            ):
                first_session = AppTest.from_file(PROJECT_ROOT / "app.py").run(
                    timeout=20
                )
                self.assertEqual(
                    first_session.checkbox[0].label,
                    "動画追加後に自動で文字起こしする",
                )
                self.assertTrue(first_session.checkbox[0].value)
                first_session.checkbox[0].set_value(False).run(timeout=20)

                settings_path = data_dir / "config" / "app_settings.json"
                self.assertTrue(settings_path.is_file())
                self.assertFalse(
                    json.loads(settings_path.read_text(encoding="utf-8"))[
                        "auto_transcribe_after_import"
                    ]
                )

                restarted_session = AppTest.from_file(
                    PROJECT_ROOT / "app.py"
                ).run(timeout=20)

            self.assertFalse(restarted_session.checkbox[0].value)

    def test_new_import_auto_transcribes_once_and_duplicate_does_not_repeat(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            content = b"new uploaded webm"

            with (
                patch.dict(
                    os.environ,
                    {"VIDEO_LEARNING_DATA_DIR": str(data_dir)},
                ),
                patch(
                    "src.transcription_service.transcribe_video_to_transcript",
                    side_effect=create_fake_transcript,
                ) as transcribe_mock,
            ):
                app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=20)
                self.assertTrue(app.checkbox[0].value)
                app.file_uploader[0].set_value(
                    ("new.webm", content, "video/webm")
                ).run(timeout=20)
                app.button[0].click().run(timeout=20)

                self.assertEqual(list(app.exception), [])
                self.assertEqual(list(app.error), [])
                self.assertEqual(len(app.get("video")), 1)
                self.assertTrue(
                    any("Hello from ASR." in item.value for item in app.markdown)
                )
                self.assertTrue(
                    any("インポートしました" in success.value for success in app.success)
                )
                self.assertEqual(transcribe_mock.call_count, 1)

                expected_id = sha256(content).hexdigest()
                stored_videos = list((data_dir / "videos").glob("*.webm"))
                self.assertEqual(len(stored_videos), 1)
                self.assertEqual(stored_videos[0].stem, expected_id)

                app.run(timeout=20)
                self.assertEqual(transcribe_mock.call_count, 1)

                app.button[0].click().run(timeout=20)
                self.assertEqual(len(list((data_dir / "videos").glob("*.webm"))), 1)
                self.assertEqual(transcribe_mock.call_count, 1)
                self.assertTrue(
                    any("すでに登録されています" in info.value for info in app.info)
                )

    def test_auto_on_still_allows_manual_transcription_of_existing_video(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            video_path = data_dir / "videos" / "existing.webm"
            video_path.parent.mkdir(parents=True)
            video_path.write_bytes(b"existing video")

            with (
                patch.dict(
                    os.environ,
                    {"VIDEO_LEARNING_DATA_DIR": str(data_dir)},
                ),
                patch(
                    "src.transcription_service.transcribe_video_to_transcript",
                    side_effect=create_fake_transcript,
                ) as transcribe_mock,
            ):
                app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=20)

                self.assertTrue(app.checkbox[0].value)
                transcribe_mock.assert_not_called()
                start_button = next(
                    button
                    for button in app.button
                    if button.label == "文字起こしを開始"
                )
                start_button.click().run(timeout=20)

                self.assertEqual(transcribe_mock.call_count, 1)
                self.assertTrue(
                    any("Hello from ASR." in item.value for item in app.markdown)
                )

    def test_video_switch_does_not_trigger_auto_transcription(self) -> None:
        with TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            videos_dir = data_dir / "videos"
            videos_dir.mkdir(parents=True)
            first_video = videos_dir / "first.webm"
            second_video = videos_dir / "second.webm"
            first_video.write_bytes(b"first")
            second_video.write_bytes(b"second")

            with (
                patch.dict(
                    os.environ,
                    {"VIDEO_LEARNING_DATA_DIR": str(data_dir)},
                ),
                patch(
                    "src.transcription_service.transcribe_video_to_transcript"
                ) as transcribe_mock,
            ):
                app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=20)
                app.selectbox[0].select(second_video).run(timeout=20)

            transcribe_mock.assert_not_called()

    def test_disabling_auto_transcription_requires_manual_button(self) -> None:
        with TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            content = b"manual transcription video"

            with (
                patch.dict(
                    os.environ,
                    {"VIDEO_LEARNING_DATA_DIR": str(data_dir)},
                ),
                patch(
                    "src.transcription_service.transcribe_video_to_transcript",
                    side_effect=create_fake_transcript,
                ) as transcribe_mock,
            ):
                app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=20)
                app.checkbox[0].set_value(False).run(timeout=20)
                app.file_uploader[0].set_value(
                    ("manual.webm", content, "video/webm")
                ).run(timeout=20)
                app.button[0].click().run(timeout=20)

                transcribe_mock.assert_not_called()
                self.assertTrue(
                    any(
                        button.label == "文字起こしを開始"
                        for button in app.button
                    )
                )
                start_button = next(
                    button
                    for button in app.button
                    if button.label == "文字起こしを開始"
                )
                start_button.click().run(timeout=20)

                self.assertEqual(transcribe_mock.call_count, 1)
                self.assertEqual(list(app.error), [])
                self.assertTrue(
                    any("Hello from ASR." in item.value for item in app.markdown)
                )

    def test_failed_auto_transcription_preserves_video_and_allows_retry(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            content = b"ASR failure video"
            attempts = 0

            def fail_once(
                video_path: str | Path,
                metadata_dir: str | Path,
                **kwargs: object,
            ) -> TranscriptionResult:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("CUDA initialization failed")
                return create_fake_transcript(video_path, metadata_dir, **kwargs)

            with (
                patch.dict(
                    os.environ,
                    {"VIDEO_LEARNING_DATA_DIR": str(data_dir)},
                ),
                patch(
                    "src.transcription_service.transcribe_video_to_transcript",
                    side_effect=fail_once,
                ),
            ):
                app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=20)
                app.file_uploader[0].set_value(
                    ("failure.webm", content, "video/webm")
                ).run(timeout=20)
                app.button[0].click().run(timeout=20)

                stored_videos = list((data_dir / "videos").glob("*.webm"))
                self.assertEqual(len(stored_videos), 1)
                self.assertEqual(attempts, 1)
                self.assertTrue(
                    any("文字起こしに失敗しました" in error.value for error in app.error)
                )
                retry_button = next(
                    button
                    for button in app.button
                    if button.label == "文字起こしを再試行"
                )
                retry_button.click().run(timeout=20)

                self.assertEqual(attempts, 2)
                self.assertTrue(stored_videos[0].is_file())
                self.assertEqual(list(app.error), [])
                self.assertTrue(
                    any("Hello from ASR." in item.value for item in app.markdown)
                )

    def test_av1_proxy_is_created_only_after_explicit_button_click(self) -> None:
        with TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            video_id = "a" * 64
            original_path = data_dir / "videos" / f"{video_id}.webm"
            proxy_path = (
                data_dir
                / "cache"
                / "playback"
                / f"{video_id}.h264-aac.v1.mp4"
            )
            original_path.parent.mkdir(parents=True)
            original_path.write_bytes(b"AV1 WebM")
            media_info = MediaInfo(
                format_names=("matroska", "webm"),
                duration_sec=10.0,
                video_codec="av1",
                video_pixel_format="yuv420p",
                audio_codec="opus",
            )
            proxy_required = PlaybackAssessment(
                original_path=original_path,
                playback_path=original_path,
                media_info=media_info,
                status="proxy_required",
                reason="A compatible copy is required.",
            )
            proxy_cached = PlaybackAssessment(
                original_path=original_path,
                playback_path=proxy_path,
                media_info=media_info,
                status="proxy_cached",
                reason="A validated playback copy is available.",
            )

            def create_proxy(*args: object, **kwargs: object) -> Path:
                proxy_path.parent.mkdir(parents=True, exist_ok=True)
                proxy_path.write_bytes(b"H.264/AAC MP4")
                return proxy_path

            with (
                patch.dict(
                    os.environ,
                    {"VIDEO_LEARNING_DATA_DIR": str(data_dir)},
                ),
                patch(
                    "src.playback.assess_playback",
                    side_effect=[proxy_required, proxy_required, proxy_cached],
                ) as assess_mock,
                patch(
                    "src.playback.ensure_playback_proxy",
                    side_effect=create_proxy,
                ) as ensure_mock,
            ):
                app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=20)

                ensure_mock.assert_not_called()
                create_button = next(
                    button
                    for button in app.button
                    if button.label == "互換再生用コピーを作成"
                )
                create_button.click().run(timeout=20)

            self.assertEqual(list(app.exception), [])
            self.assertEqual(list(app.error), [])
            ensure_mock.assert_called_once_with(
                original_path,
                video_id,
                data_dir / "cache" / "playback",
            )
            self.assertEqual(assess_mock.call_count, 3)
            self.assertFalse(
                any(
                    button.label == "互換再生用コピーを作成"
                    for button in app.button
                )
            )
            self.assertTrue(
                any("互換再生用コピーを使用" in caption.value for caption in app.caption)
            )


if __name__ == "__main__":
    unittest.main()
