"""Tests for persistent local application settings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.app_settings import (
    AppSettings,
    AppSettingsFormatError,
    load_app_settings,
    save_app_settings,
)


def test_first_use_defaults_auto_transcription_to_on(tmp_path: Path) -> None:
    settings = load_app_settings(tmp_path / "config" / "app_settings.json")

    assert settings.auto_transcribe_after_import is True


def test_off_preference_is_persisted_and_restored(tmp_path: Path) -> None:
    settings_path = tmp_path / "config" / "app_settings.json"

    saved_path = save_app_settings(
        AppSettings(auto_transcribe_after_import=False),
        settings_path,
    )
    restored = load_app_settings(settings_path)

    assert saved_path == settings_path
    assert restored.auto_transcribe_after_import is False
    assert json.loads(settings_path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "auto_transcribe_after_import": False,
    }
    assert not list(settings_path.parent.glob("*.tmp"))


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"schema_version": 99, "auto_transcribe_after_import": True},
        {"schema_version": 1, "auto_transcribe_after_import": "false"},
    ],
)
def test_invalid_settings_are_rejected(tmp_path: Path, payload: object) -> None:
    settings_path = tmp_path / "app_settings.json"
    settings_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AppSettingsFormatError):
        load_app_settings(settings_path)


def test_invalid_settings_json_is_rejected(tmp_path: Path) -> None:
    settings_path = tmp_path / "app_settings.json"
    settings_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(AppSettingsFormatError, match="invalid JSON"):
        load_app_settings(settings_path)
