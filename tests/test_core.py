"""Tests for configuration, schema and upserts."""

from __future__ import annotations

from datetime import UTC, date

import duckdb
import pandas as pd
import pytest

from core.config import Settings
from core.database import TABLES, Database


def test_settings_derive_paths(settings: Settings) -> None:
    assert settings.duckdb_path == settings.data_dir / "telemetry.duckdb"
    assert settings.android_export_dir.exists()
    assert settings.mfp_export_dir.exists()


def test_invalid_timezone_rejected(tmp_path) -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, data_dir=tmp_path, timezone="Mars/Olympus")


def test_schema_creates_all_tables(db: Database) -> None:
    with db.connect() as con:
        names = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    assert set(TABLES) <= names


def test_upsert_merges_partial_columns(db: Database) -> None:
    db.upsert("s24_health", pd.DataFrame({"date": [date(2026, 1, 1)], "steps": [100], "weight_kg": [80.0]}))
    db.upsert("s24_health", pd.DataFrame({"date": ["2026-01-01"], "steps": [250], "weight_kg": [None]}))
    row = db.read_table("s24_health").iloc[0]
    assert row["steps"] == 250
    assert row["weight_kg"] == 80.0  # null did not overwrite


def test_upsert_overwrite_nulls(db: Database) -> None:
    db.upsert("s24_health", pd.DataFrame({"date": [date(2026, 1, 1)], "weight_kg": [80.0]}))
    db.upsert("s24_health", pd.DataFrame({"date": [date(2026, 1, 1)], "weight_kg": [None]}), overwrite_nulls=True)
    assert pd.isna(db.read_table("s24_health").iloc[0]["weight_kg"])


def test_upsert_dedupes_keys_and_drops_unknown_columns(db: Database) -> None:
    frame = pd.DataFrame({"date": [date(2026, 1, 1)] * 2, "category": ["groceries"] * 2,
                          "amount_gbp": [1.0, 2.0], "bogus": [1, 2]})
    assert db.upsert("monzo_spending", frame) == 1
    assert db.read_table("monzo_spending")["amount_gbp"].tolist() == [2.0]


def test_upsert_requires_primary_key(db: Database) -> None:
    with pytest.raises(ValueError):
        db.upsert("monzo_spending", pd.DataFrame({"date": [date(2026, 1, 1)], "amount_gbp": [1.0]}))


def test_subjective_scores_are_constrained(db: Database) -> None:
    with pytest.raises(duckdb.ConstraintException):
        db.upsert("subjective_state", pd.DataFrame({"date": [date(2026, 1, 1)], "mood": [6]}))


def test_token_roundtrip(db: Database) -> None:
    from datetime import datetime

    expiry = datetime(2030, 1, 1, tzinfo=UTC)
    db.save_token("whoop", "access", "refresh", expiry, "offline")
    token = db.get_token("whoop")
    assert token is not None and token["refresh_token"] == "refresh"
