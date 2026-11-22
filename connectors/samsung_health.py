"""Parser for Samsung Health's "Download personal data" export.

Samsung Health -> Settings -> Download personal data produces a folder (usually
zipped) of CSVs named ``com.samsung.<datatype>.<timestamp>.csv``. Each file has
a metadata line (``<datatype>,<version>,<n>``) before the real header, and some
columns are prefixed with the datatype name (``com.samsung.shealth.calories_burned.active_calorie``).

Daily totals used:

* ``com.samsung.shealth.step_daily_trend`` - steps, distance. Rows with
  ``source_type == -2`` are Samsung's merged all-device total and are preferred.
* ``com.samsung.shealth.calories_burned.details`` - active, resting and
  thermic-effect calories per day.
* ``com.samsung.shealth.tracker.floors_day_summary`` - floors.
* ``com.samsung.health.weight`` - weight, body fat, skeletal muscle, BMR.
* ``com.samsung.health.nutrition`` / ``com.samsung.health.water_intake`` - food
  logged in Samsung Health or synced from MyFitnessPal via Health Connect.

The weight, nutrition and water mappings follow Samsung's published field names
but have not yet been checked against a real export containing those files.
"""

from __future__ import annotations

import io
import re
from zoneinfo import ZoneInfo

import pandas as pd

SAMSUNG_PREFIX = "com.samsung."
_OFFSET = re.compile(r"UTC([+-])(\d{2}):?(\d{2})")


def is_samsung_csv(name: str, head: bytes) -> bool:
    """True if a file looks like a Samsung Health export CSV."""
    return _basename(name).startswith(SAMSUNG_PREFIX) or head.lstrip(b"\xef\xbb\xbf").startswith(b"com.samsung.")


def _basename(name: str) -> str:
    return name.replace("\\", "/").rsplit("/", 1)[-1]


def datatype_of(name: str) -> str:
    """``com.samsung.shealth.step_daily_trend.20261020183425.csv`` -> ``com.samsung.shealth.step_daily_trend``."""
    stem = _basename(name).removesuffix(".csv")
    return re.sub(r"\.\d{8,}$", "", stem)


def read_samsung_csv(content: bytes) -> pd.DataFrame:
    """Read one export CSV: skip the metadata line and strip datatype prefixes from column names."""
    frame = pd.read_csv(io.BytesIO(content), skiprows=1, index_col=False, encoding="utf-8-sig", low_memory=False)
    frame.columns = [str(c).rsplit(".", 1)[-1] if str(c).startswith(SAMSUNG_PREFIX) else str(c) for c in frame.columns]
    return frame.loc[:, ~frame.columns.str.startswith("Unnamed")]


def _day(values: pd.Series) -> pd.Series:
    """Samsung ``day_time`` (local midnight, as text or epoch ms) -> date."""
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_datetime(values, unit="ms", utc=True).dt.date
    return pd.to_datetime(values, errors="coerce").dt.date


def _local_day(frame: pd.DataFrame, tz: ZoneInfo, column: str = "start_time") -> pd.Series:
    """Event timestamps (UTC, with optional ``time_offset`` like ``UTC+0100``) -> local date."""
    moments = pd.to_datetime(frame[column], errors="coerce", utc=True)
    if "time_offset" in frame:
        def shift(offset: object) -> pd.Timedelta:
            match = _OFFSET.match(str(offset))
            if not match:
                return pd.NaT
            sign = 1 if match.group(1) == "+" else -1
            return sign * pd.Timedelta(hours=int(match.group(2)), minutes=int(match.group(3)))

        offsets = frame["time_offset"].map(shift)
        local = (moments.dt.tz_localize(None) + offsets).dt.date
        fallback = moments.dt.tz_convert(tz).dt.date
        return local.where(offsets.notna(), fallback)
    return moments.dt.tz_convert(tz).dt.date


def parse_samsung_export(files: dict[str, bytes], tz: ZoneInfo) -> dict[str, pd.DataFrame]:
    """Turn export CSVs into ``{"s24_health": ..., "mfp_nutrition": ...}`` daily frames.

    Args:
        files: ``{filename: raw bytes}`` for every CSV in the export.
        tz: Local timezone (for event-based records such as weigh-ins and meals).
    """
    by_type: dict[str, pd.DataFrame] = {}
    for name, content in files.items():
        try:
            by_type[datatype_of(name)] = read_samsung_csv(content)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
            continue

    health: list[pd.DataFrame] = []

    steps = by_type.get("com.samsung.shealth.step_daily_trend")
    if steps is not None and {"day_time", "count"} <= set(steps.columns):
        steps = steps.assign(date=_day(steps["day_time"]))
        merged = steps.loc[steps.get("source_type", pd.Series(dtype=int)) == -2]
        # Days without a merged row: take the largest single-device total.
        rest = steps.loc[~steps["date"].isin(merged["date"])].sort_values("count").drop_duplicates("date", keep="last")
        daily = pd.concat([merged, rest]).drop_duplicates("date", keep="first")
        health.append(pd.DataFrame({"date": daily["date"], "steps": daily["count"].round().astype("Int64"),
                                    "distance_m": pd.to_numeric(daily.get("distance"), errors="coerce")}))

    calories = by_type.get("com.samsung.shealth.calories_burned.details")
    if calories is not None and "day_time" in calories:
        cal = calories.assign(date=_day(calories["day_time"]))
        active = pd.to_numeric(cal.get("active_calorie"), errors="coerce")
        resting = pd.to_numeric(cal.get("rest_calorie"), errors="coerce")
        tef = pd.to_numeric(cal.get("tef_calorie"), errors="coerce").fillna(0)
        frame = pd.DataFrame({"date": cal["date"], "active_calories_kcal": active,
                              "total_calories_kcal": resting + active.fillna(0) + tef,
                              "bmr_kcal": resting})
        health.append(frame.groupby("date", as_index=False).last())

    floors = by_type.get("com.samsung.shealth.tracker.floors_day_summary")
    if floors is not None and {"day_time", "floor_count"} <= set(floors.columns):
        fl = pd.DataFrame({"date": _day(floors["day_time"]),
                           "floors_climbed": pd.to_numeric(floors["floor_count"], errors="coerce")})
        health.append(fl.groupby("date", as_index=False).max())

    weight = by_type.get("com.samsung.health.weight")
    if weight is not None and "start_time" in weight:
        w = pd.DataFrame({
            "date": _local_day(weight, tz),
            "weight_kg": pd.to_numeric(weight.get("weight"), errors="coerce"),
            "body_fat_pct": pd.to_numeric(weight.get("body_fat"), errors="coerce"),
            "skeletal_muscle_mass_kg": pd.to_numeric(weight.get("skeletal_muscle_mass"), errors="coerce"),
        }).dropna(subset=["date"])
        health.append(w.groupby("date", as_index=False).last())

    frames: dict[str, pd.DataFrame] = {}
    if health:
        combined = health[0]
        for part in health[1:]:
            combined = combined.merge(part, on="date", how="outer", suffixes=("", "_dup"))
            for col in [c for c in combined.columns if c.endswith("_dup")]:
                base = col.removesuffix("_dup")
                combined[base] = combined[base].fillna(combined.pop(col))
        frames["s24_health"] = combined.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    nutrition_parts: list[pd.DataFrame] = []
    food = by_type.get("com.samsung.health.nutrition")
    if food is not None and "start_time" in food:
        n = pd.DataFrame({
            "date": _local_day(food, tz),
            "calories_kcal": pd.to_numeric(food.get("calorie"), errors="coerce"),
            "protein_g": pd.to_numeric(food.get("protein"), errors="coerce"),
            "carbs_g": pd.to_numeric(food.get("carbohydrate"), errors="coerce"),
            "fat_g": pd.to_numeric(food.get("total_fat"), errors="coerce"),
            "fiber_g": pd.to_numeric(food.get("dietary_fiber"), errors="coerce"),
            "sugar_g": pd.to_numeric(food.get("sugar"), errors="coerce"),
            "sodium_mg": pd.to_numeric(food.get("sodium"), errors="coerce"),
        }).dropna(subset=["date"])
        nutrition_parts.append(n.groupby("date", as_index=False).sum(min_count=1))
    water = by_type.get("com.samsung.health.water_intake")
    if water is not None and "start_time" in water:
        wt = pd.DataFrame({"date": _local_day(water, tz),
                           "water_ml": pd.to_numeric(water.get("amount"), errors="coerce")}).dropna(subset=["date"])
        nutrition_parts.append(wt.groupby("date", as_index=False).sum(min_count=1))
    if nutrition_parts:
        nutrition = nutrition_parts[0]
        for part in nutrition_parts[1:]:
            nutrition = nutrition.merge(part, on="date", how="outer")
        nutrition["source"] = "samsung_health"
        frames["mfp_nutrition"] = nutrition
    return frames
