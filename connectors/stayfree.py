"""Parser for StayFree's "Total Usage" Excel report (phone screen time).

StayFree -> Settings -> Export data writes ``StayFree Export - Total Usage - <date>.xls``
with three tabs, each "wide" (one row per app, one column per day):

* **Usage Time** - ``App | Device | June 1, 2026 | ... | Total Usage`` with durations
  such as ``1h 4m 12s``.
* **Usage Count** - the same grid with app launch counts.
* **Device Unlocks** - a single ``Device Unlocks`` row per day.

Footer rows ("Created by StayFree", "Creation date: ...") and the trailing
``Total Usage`` column are ignored.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from connectors.utils import categorize_app, parse_duration_to_minutes, slugify

S24_MODEL = re.compile(r"SM-S928", re.IGNORECASE)


def is_stayfree_report(path: Path) -> bool:
    """True for a StayFree Excel export (by name, or by its tab names)."""
    if path.suffix.lower() not in {".xls", ".xlsx"}:
        return False
    if "stayfree" in path.name.lower():
        return True
    try:
        return "Usage Time" in pd.ExcelFile(path).sheet_names
    except Exception:  # noqa: BLE001 - unreadable workbook is simply "not StayFree"
        return False


def _device_name(raw: object) -> str:
    text = str(raw)
    return "s24_ultra" if S24_MODEL.search(text) else slugify(text)


def _melt(grid: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Wide app x day grid -> long ``app, device, date, <value_name>`` rows."""
    header = grid.iloc[0]
    body = grid.iloc[1:].copy()
    body.columns = ["app", "device", *header.iloc[2:].tolist()]
    body = body.loc[body["app"].notna() & body["device"].notna()]
    day_cols = [c for c in body.columns[2:] if pd.notna(pd.to_datetime(str(c), errors="coerce", format="%B %d, %Y"))]
    long = body.melt(id_vars=["app", "device"], value_vars=day_cols, var_name="day", value_name=value_name)
    long["date"] = pd.to_datetime(long["day"], format="%B %d, %Y").dt.date
    return long.drop(columns="day")


def parse_stayfree_report(path: Path, category_rules: dict[str, list[str]]) -> dict[str, pd.DataFrame]:
    """Parse a StayFree workbook into ``screen_usage`` rows and daily phone unlocks.

    Returns:
        ``{"screen_usage": ..., "s24_health": ...}`` (the latter carries ``phone_unlocks``).
    """
    workbook = pd.ExcelFile(path)
    frames: dict[str, pd.DataFrame] = {}

    if "Usage Time" in workbook.sheet_names:
        usage = _melt(workbook.parse("Usage Time", header=None), "duration")
        usage["minutes"] = usage["duration"].map(lambda v: parse_duration_to_minutes(v, "min"))
        if "Usage Count" in workbook.sheet_names:
            counts = _melt(workbook.parse("Usage Count", header=None), "launches")
            usage = usage.merge(counts, on=["app", "device", "date"], how="left")
        else:
            usage["launches"] = pd.NA
        usage = usage.loc[usage["minutes"].fillna(0) > 0]
        rows = pd.DataFrame({
            "date": usage["date"],
            "device": usage["device"].map(_device_name),
            "app": usage["app"].astype(str).str.strip(),
            "category": usage["app"].map(lambda a: categorize_app(str(a), category_rules)),
            "minutes": usage["minutes"].round(2),
            "launches": pd.to_numeric(usage["launches"], errors="coerce").round().astype("Int64"),
            "source": "stayfree",
        })
        frames["screen_usage"] = rows.groupby(["date", "device", "app"], as_index=False).agg(
            category=("category", "first"), minutes=("minutes", "sum"),
            launches=("launches", lambda s: s.sum(min_count=1)), source=("source", "first"),
        )

    if "Device Unlocks" in workbook.sheet_names:
        grid = workbook.parse("Device Unlocks", header=None)
        header, values = grid.iloc[0], grid.loc[grid[0].astype(str).str.strip().str.lower() == "device unlocks"]
        if not values.empty:
            unlocks = []
            for col in grid.columns[1:]:
                day = pd.to_datetime(str(header[col]), errors="coerce", format="%B %d, %Y")
                if pd.notna(day):
                    unlocks.append({"date": day.date(), "phone_unlocks": pd.to_numeric(values.iloc[0][col], errors="coerce")})
            frame = pd.DataFrame(unlocks).dropna()
            # Days before StayFree was installed are exported as 0 - drop that leading run.
            started = frame.loc[frame["phone_unlocks"] > 0, "date"]
            frame = frame.loc[frame["date"] >= started.min()] if not started.empty else frame.iloc[0:0]
            if not frame.empty:
                frame["phone_unlocks"] = frame["phone_unlocks"].astype(int)
                frames["s24_health"] = frame
    return frames
