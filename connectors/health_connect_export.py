"""Parser for Android Health Connect's scheduled export (``health_connect_export.db``).

Health Connect -> Export and import -> Scheduled export writes a zip containing an
unencrypted SQLite database to Google Drive (or another location). Point the
Samsung section's synced folder at it and every sync picks up the newest export.

Storage conventions in the database (Health Connect internal units):

* ``local_date`` - days since 1970-01-01 in the record's local time
* energy - small calories (cal; /1000 -> kcal)
* mass (weight, macros, sodium) - grams
* distance - metres; hydration volume - litres

Several apps can write the same type (e.g. phone and WHOOP both write steps).
Additive totals take the largest single-app daily total, mirroring Health
Connect's own de-duplication by app priority rather than double counting.
"""

from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

EXPORT_DB_NAME = "health_connect_export.db"
EPOCH = date(1970, 1, 1)
MFP_PACKAGE = "com.myfitnesspal.android"


def is_health_connect_db(name: str) -> bool:
    """True for the database file inside a Health Connect export."""
    return Path(name.replace("\\", "/")).name == EXPORT_DB_NAME


@contextmanager
def _open(content: bytes) -> Iterator[sqlite3.Connection]:
    """Open database bytes read-only via a temporary file."""
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    try:
        handle.write(content)
        handle.close()
        con = sqlite3.connect(f"file:{Path(handle.name).as_posix()}?mode=ro", uri=True)
        try:
            yield con
        finally:
            con.close()
    finally:
        Path(handle.name).unlink(missing_ok=True)


def _query(con: sqlite3.Connection, sql: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(sql, con)
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        return pd.DataFrame()  # table absent in this Health Connect version


def _with_dates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["date"] = [EPOCH + timedelta(days=int(d)) for d in frame["local_date"]]
    return frame.drop(columns="local_date")


def _max_app_total(con: sqlite3.Connection, table: str, column: str, name: str, scale: float = 1.0) -> pd.Series:
    """Per-day total of ``column``, taking the largest single-app total."""
    frame = _with_dates(_query(
        con, f"SELECT local_date, app_info_id, SUM({column}) AS v FROM {table} GROUP BY local_date, app_info_id"))
    if frame.empty:
        return pd.Series(dtype="float64", name=name)
    return (frame.groupby("date")["v"].max() * scale).rename(name)


def _latest(con: sqlite3.Connection, table: str, column: str, name: str, scale: float = 1.0) -> pd.Series:
    """Last instantaneous reading per day (weight, body fat...)."""
    frame = _with_dates(_query(con, f"SELECT local_date, time, {column} AS v FROM {table} ORDER BY time"))
    if frame.empty:
        return pd.Series(dtype="float64", name=name)
    return (frame.groupby("date")["v"].last() * scale).rename(name)


def _mean(con: sqlite3.Connection, table: str, column: str, name: str) -> pd.Series:
    frame = _with_dates(_query(con, f"SELECT local_date, {column} AS v FROM {table}"))
    if frame.empty:
        return pd.Series(dtype="float64", name=name)
    return frame.groupby("date")["v"].mean().rename(name)


def parse_health_connect_db(content: bytes) -> dict[str, pd.DataFrame]:
    """Turn a Health Connect export database into daily ``s24_health`` / ``mfp_nutrition`` frames."""
    with _open(content) as con:
        health = pd.concat(
            [
                _max_app_total(con, "steps_record_table", "count", "steps"),
                _max_app_total(con, "distance_record_table", "distance", "distance_m"),
                _max_app_total(con, "active_calories_burned_record_table", "energy", "active_calories_kcal", 1e-3),
                _max_app_total(con, "total_calories_burned_record_table", "energy", "total_calories_kcal", 1e-3),
                _max_app_total(con, "floors_climbed_record_table", "floors", "floors_climbed"),
                _mean(con, "resting_heart_rate_record_table", "beats_per_minute", "resting_heart_rate_bpm"),
                _latest(con, "weight_record_table", "weight", "weight_kg", 1e-3),
                _latest(con, "body_fat_record_table", "percentage", "body_fat_pct"),
                _latest(con, "lean_body_mass_record_table", "mass", "skeletal_muscle_mass_kg", 1e-3),
                _latest(con, "basal_metabolic_rate_record_table", "basal_metabolic_rate", "bmr_kcal"),
            ],
            axis=1,
        )

        apps = _query(con, "SELECT row_id, package_name FROM application_info_table")
        mfp_ids = apps.loc[apps["package_name"] == MFP_PACKAGE, "row_id"].tolist() if not apps.empty else []
        source_filter = f"WHERE app_info_id IN ({','.join(map(str, mfp_ids))})" if mfp_ids else ""
        food = _with_dates(_query(con, f"""
            SELECT local_date, energy, protein, total_carbohydrate, total_fat, dietary_fiber, sugar, sodium
            FROM nutrition_record_table {source_filter}"""))
        if food.empty and source_filter:  # MFP not writing yet - fall back to any app's food logs
            food = _with_dates(_query(con, """
                SELECT local_date, energy, protein, total_carbohydrate, total_fat, dietary_fiber, sugar, sodium
                FROM nutrition_record_table"""))
        water = _max_app_total(con, "hydration_record_table", "volume", "water_ml", 1e3)

    frames: dict[str, pd.DataFrame] = {}
    health = health.dropna(how="all")
    if not health.empty:
        health.index.name = "date"
        health = health.reset_index().sort_values("date")
        if "steps" in health:
            health["steps"] = health["steps"].round().astype("Int64")
        frames["s24_health"] = health

    nutrition: pd.DataFrame | None = None
    if not food.empty:
        nutrition = food.groupby("date").sum(min_count=1)
        nutrition = pd.DataFrame({
            "calories_kcal": nutrition["energy"] / 1000,
            "protein_g": nutrition["protein"],
            "carbs_g": nutrition["total_carbohydrate"],
            "fat_g": nutrition["total_fat"],
            "fiber_g": nutrition["dietary_fiber"],
            "sugar_g": nutrition["sugar"],
            "sodium_mg": nutrition["sodium"] * 1000,
        })
    if not water.empty:
        nutrition = water.to_frame() if nutrition is None else nutrition.join(water, how="outer")
    if nutrition is not None and not nutrition.empty:
        nutrition.index.name = "date"
        nutrition = nutrition.reset_index().sort_values("date")
        nutrition["source"] = "health_connect"
        frames["mfp_nutrition"] = nutrition
    return frames
