"""Tests for UI-managed settings and the Monzo CSV import."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from connectors.monzo_sheets import MonzoSheetsConnector
from core.config import Settings, get_settings, reload_settings
from core.settings_store import SettingsStore


def test_store_roundtrip_without_keyring(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "t.duckdb", use_keyring=False)
    store.save({"weather_latitude": 53.48, "whoop_client_secret": "shh", "aw_coding_app_patterns": ["code"]})
    loaded = store.load()
    assert loaded == {"weather_latitude": 53.48, "whoop_client_secret": "shh", "aw_coding_app_patterns": ["code"]}
    store.save({"weather_latitude": None})
    store.delete("whoop_client_secret")
    assert store.load() == {"aw_coding_app_patterns": ["code"]}


def test_load_missing_file_is_empty(tmp_path: Path) -> None:
    assert SettingsStore(tmp_path / "missing.duckdb", use_keyring=False).load() == {}


def test_saved_settings_override_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("WEATHER_LATITUDE", "1.0")
    base = Settings(_env_file=None)
    SettingsStore(base.duckdb_path, use_keyring=False).save({"weather_latitude": 53.5, "timezone": "America/New_York"})
    try:
        settings = reload_settings()
        assert settings.weather_latitude == 53.5
        assert settings.timezone == "America/New_York"
        assert settings.data_dir == data_dir.resolve()
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


def test_invalid_saved_settings_are_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    base = Settings(_env_file=None)
    SettingsStore(base.duckdb_path, use_keyring=False).save({"timezone": "Not/AZone"})
    try:
        assert reload_settings().timezone == base.timezone
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


def test_monzo_csv_import_dedupes_overlapping_exports(settings, db) -> None:
    rows = {
        "Transaction ID": ["tx_1", "tx_2"], "Date": ["01/03/2026", "02/03/2026"], "Time": ["10:00:00", "12:00:00"],
        "Type": ["Card payment", "Card payment"], "Name": ["Tesco", "Pret"], "Category": ["Groceries", "Eating out"],
        "Amount": ["-10.00", "-6.50"], "Currency": ["GBP", "GBP"],
    }
    folder = settings.monzo_import_dir
    pd.DataFrame(rows).to_csv(folder / "export_a.csv", index=False)
    pd.DataFrame({k: v[:1] for k, v in rows.items()}).to_csv(folder / "export_b.csv", index=False)  # overlaps tx_1
    connector = MonzoSheetsConnector(settings, db)
    assert connector.missing_config() == []
    result = connector.run(date(2026, 3, 1), date(2026, 3, 2))
    assert result.status == "success"
    spend = db.read_table("monzo_spending")
    assert spend.loc[spend["category"] == "groceries", "amount_gbp"].item() == 10.0
