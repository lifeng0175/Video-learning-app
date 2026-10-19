"""Core faster-whisper model setup and transcription helpers."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo


_CUDA_DLL_HANDLES: dict[Path, object] = {}


def setup_cuda_dlls() -> tuple[Path, ...]:
    """Add pip-installed NVIDIA DLL directories to the Windows search path."""
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        return ()

    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    cuda_dll_paths = (
        site_packages / "nvidia" / "cublas" / "bin",
        site_packages / "nvidia" / "cudnn" / "bin",
    )

    available_paths: list[Path] = []
    for dll_path in cuda_dll_paths:
        if not dll_path.exists():
            continue

        available_paths.append(dll_path)
        if dll_path not in _CUDA_DLL_HANDLES:
            _CUDA_DLL_HANDLES[dll_path] = add_dll_directory(str(dll_path))

    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    normalized_entries = {entry.casefold() for entry in path_entries if entry}
    missing_entries = [
        str(dll_path)
        for dll_path in available_paths
        if str(dll_path).casefold() not in normalized_entries
    ]
    if missing_entries:
        os.environ["PATH"] = os.pathsep.join([*missing_entries, *path_entries])

    return tuple(available_paths)


def load_asr_model(
    model_name: str = "turbo",
    device: str = "cuda",
    compute_type: str = "float16",
) -> WhisperModel:
    """Set up CUDA DLLs and load a faster-whisper model."""
    setup_cuda_dlls()
    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
    )


def transcribe_video(
    model: WhisperModel,
    video_path: str | Path,
    language: str = "en",
    beam_size: int = 1,
    word_timestamps: bool = True,
    log_progress: bool = True,
) -> tuple[list[Segment], TranscriptionInfo, float]:
    """Transcribe a video and return materialized segments, info, and runtime."""
    start_time = time.time()

    segments_generator, info = model.transcribe(
        str(Path(video_path)),
        language=language,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
        log_progress=log_progress,
    )
    segments = list(segments_generator)

    elapsed_time = time.time() - start_time
    return segments, info, elapsed_time
