"""Persistent settings for the local application prototype."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SETTINGS_SCHEMA_VERSION = 1


class AppSettingsFormatError(ValueError):
    """Raised when the persisted settings file has an invalid structure."""


@dataclass(frozen=True, slots=True)
class AppSettings:
    """User-configurable application preferences."""

    auto_transcribe_after_import: bool = True


def load_app_settings(settings_path: str | Path) -> AppSettings:
    """Load settings, returning first-use defaults when no file exists."""
    settings_path = Path(settings_path)
    if not settings_path.exists():
        return AppSettings()
    if not settings_path.is_file():
        raise AppSettingsFormatError(f"settings path is not a file: {settings_path}")

    try:
        with settings_path.open("r", encoding="utf-8") as settings_file:
            payload = json.load(settings_file)
    except json.JSONDecodeError as error:
        raise AppSettingsFormatError("settings file contains invalid JSON") from error
    return _parse_settings(payload)


def save_app_settings(
    settings: AppSettings,
    settings_path: str | Path,
) -> Path:
    """Atomically persist application settings as UTF-8 JSON."""
    settings_path = Path(settings_path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SETTINGS_SCHEMA_VERSION,
        **asdict(settings),
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=settings_path.parent,
            prefix=f".{settings_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")

        os.replace(temporary_path, settings_path)
        temporary_path = None
        return settings_path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _parse_settings(payload: Any) -> AppSettings:
    if not isinstance(payload, dict):
        raise AppSettingsFormatError("settings JSON root must be an object")
    if payload.get("schema_version") != SETTINGS_SCHEMA_VERSION:
        raise AppSettingsFormatError("unsupported settings schema version")

    auto_transcribe = payload.get("auto_transcribe_after_import")
    if not isinstance(auto_transcribe, bool):
        raise AppSettingsFormatError(
            "auto_transcribe_after_import must be a boolean"
        )
    return AppSettings(auto_transcribe_after_import=auto_transcribe)
