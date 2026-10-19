"""Project-level pytest configuration."""

from __future__ import annotations

import getpass
import os
import re
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Keep pytest temporary files project-local and isolated by OS user."""
    if config.option.basetemp is not None:
        return

    try:
        identity = os.getlogin()
    except OSError:
        identity = getpass.getuser()

    safe_identity = re.sub(r"[^A-Za-z0-9_.-]+", "-", identity).strip("-._")
    if not safe_identity:
        safe_identity = "user"

    config.option.basetemp = Path(config.rootpath) / f".pytest_tmp-{safe_identity}"
