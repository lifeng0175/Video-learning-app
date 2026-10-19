"""Unit tests for ASR helpers that do not require a GPU."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src import asr


class AsrUnitTests(unittest.TestCase):
    def tearDown(self) -> None:
        asr._CUDA_DLL_HANDLES.clear()

    def test_setup_cuda_dlls_registers_existing_paths_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            site_packages = Path(temp_dir) / "Lib" / "site-packages"
            expected_paths = (
                site_packages / "nvidia" / "cublas" / "bin",
                site_packages / "nvidia" / "cudnn" / "bin",
            )
            for dll_path in expected_paths:
                dll_path.mkdir(parents=True)

            add_dll_directory = Mock(side_effect=[object(), object()])
            with (
                patch.object(asr.sys, "prefix", temp_dir),
                patch.dict(asr.os.environ, {"PATH": ""}),
                patch.object(
                    asr.os,
                    "add_dll_directory",
                    add_dll_directory,
                    create=True,
                ),
            ):
                self.assertEqual(asr.setup_cuda_dlls(), expected_paths)
                self.assertEqual(asr.setup_cuda_dlls(), expected_paths)
                self.assertEqual(
                    asr.os.environ["PATH"].split(asr.os.pathsep)[:2],
                    [str(path) for path in expected_paths],
                )

            self.assertEqual(add_dll_directory.call_count, 2)
            add_dll_directory.assert_any_call(str(expected_paths[0]))
            add_dll_directory.assert_any_call(str(expected_paths[1]))

    @patch.object(asr, "WhisperModel")
    @patch.object(asr, "setup_cuda_dlls")
    def test_load_asr_model_uses_verified_defaults(
        self,
        setup_cuda_dlls: Mock,
        whisper_model: Mock,
    ) -> None:
        loaded_model = asr.load_asr_model()

        setup_cuda_dlls.assert_called_once_with()
        whisper_model.assert_called_once_with(
            "turbo",
            device="cuda",
            compute_type="float16",
        )
        self.assertIs(loaded_model, whisper_model.return_value)

    def test_transcribe_video_materializes_generator_and_returns_runtime(self) -> None:
        segments = [object(), object()]
        info = SimpleNamespace(duration=10.0, language="en")
        model = Mock()
        model.transcribe.return_value = (iter(segments), info)

        with patch.object(asr.time, "time", side_effect=[100.0, 102.5]):
            result_segments, result_info, elapsed_time = asr.transcribe_video(
                model,
                Path("sample.webm"),
            )

        model.transcribe.assert_called_once_with(
            "sample.webm",
            language="en",
            beam_size=1,
            word_timestamps=True,
            log_progress=True,
        )
        self.assertEqual(result_segments, segments)
        self.assertIs(result_info, info)
        self.assertEqual(elapsed_time, 2.5)


if __name__ == "__main__":
    unittest.main()
