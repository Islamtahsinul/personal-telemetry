"""Shared fixtures: an isolated Settings + Database per test."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import Settings  # noqa: E402
from core.database import Database  # noqa: E402


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Settings pointing at a temporary data directory (ignores the user's .env)."""
    s = Settings(_env_file=None, data_dir=tmp_path / "data", timezone="Europe/London",
                 weather_latitude=51.5, weather_longitude=-0.12)
    s.ensure_directories()
    return s


@pytest.fixture()
def db(settings: Settings) -> Database:
    """Initialised temporary database."""
    database = Database(settings=settings)
    database.init_schema()
    return database
