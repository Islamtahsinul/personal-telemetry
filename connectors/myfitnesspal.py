"""MyFitnessPal daily nutrition connector.

Sources (``MFP_SOURCE``):

* ``export`` - the official MFP Premium export (*Settings -> Export data*). Drop
  the ``.zip`` or the ``Nutrition-Summary-*.csv`` into ``MFP_EXPORT_DIR``
  (default ``data/inbox/myfitnesspal``). Rows are per meal and summed per day.
* ``health_connect`` - MFP's Health Connect sync (Nutrition + Hydration records)
  read from ``ANDROID_EXPORT_DIR`` (Health Sync dumps).
* ``scraper`` - the community `myfitnesspal <https://pypi.org/project/myfitnesspal/>`_
  package, which reuses your browser's logged-in MFP cookies. It pins old
  dependencies, so install it in a separate environment if it conflicts.
* ``auto`` (default) - export files if present, then Health Connect nutrition
  files, then the scraper if installed.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import pandas as pd

from connectors.android_s24 import NUTRITION_FILE_HINTS
from connectors.base import BaseConnector, ConnectorError, Frames
from connectors.utils import (
    date_range_days,
    find_column,
    normalise_name,
    read_tabular,
    to_local_dates,
)

NUTRIENT_ALIASES: dict[str, list[str]] = {
    "calories_kcal": ["calories", "calories_kcal", "energy_kcal", "energy_in_kilocalories", "energy", "kcal"],
    "protein_g": ["protein_g", "protein", "protein_in_grams"],
    "carbs_g": ["carbohydrates_g", "carbohydrates", "carbs", "total_carbohydrate", "total_carbohydrate_in_grams"],
    "fat_g": ["fat_g", "fat", "total_fat", "totalfat", "total_fat_in_grams"],
    "fiber_g": ["fiber", "fiber_g", "dietary_fiber", "dietary_fiber_in_grams", "fibre"],
    "sugar_g": ["sugar", "sugar_g", "sugars", "sugar_in_grams"],
    "sodium_mg": ["sodium_mg", "sodium", "sodium_in_milligrams"],
    "water_ml": ["water_ml", "water", "hydration", "volume_in_milliliters", "volume_ml", "volume_in_liters", "volume"],
}
NUTRITION_COLUMNS = list(NUTRIENT_ALIASES)
WATER_UNIT_TO_ML: dict[str, float] = {"ml": 1.0, "cups": 236.588, "fl_oz": 29.5735}
MfpSource = Literal["export", "scraper", "health_connect"]


@dataclass(slots=True)
class MfpRaw:
    """Raw nutrition frames (per file or per scraped day) and their source."""

    source: MfpSource
    frames: list[pd.DataFrame] = field(default_factory=list)


def parse_nutrition_frame(frame: pd.DataFrame, tz: ZoneInfo, source_filter: str | None = None) -> pd.DataFrame:
    """Sum any per-meal/per-record nutrition table to daily totals.

    Args:
        frame: MFP export or Health Connect nutrition/hydration records.
        tz: Local timezone for timestamped records.
        source_filter: Optional regex on a data-origin column (e.g. ``"myfitnesspal"``).

    Returns:
        DataFrame with ``date`` and any of :data:`NUTRITION_COLUMNS` that were found.
    """
    if frame.empty:
        return pd.DataFrame(columns=["date"])
    date_col = find_column(frame, ["date", "start_time", "starttime", "time", "timestamp", "day"])
    if date_col is None:
        return pd.DataFrame(columns=["date"])
    origin_col = find_column(frame, ["data_origin", "dataorigin", "source", "package_name"])
    if source_filter and origin_col:
        mask = frame[origin_col].astype(str).str.contains(source_filter, case=False, regex=True, na=False)
        if mask.any():
            frame = frame.loc[mask]

    out = pd.DataFrame({"date": to_local_dates(frame[date_col], tz)})
    for target, aliases in NUTRIENT_ALIASES.items():
        col = find_column(frame, aliases)
        if col is None or col == date_col:
            continue
        values = pd.to_numeric(frame[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
        name = normalise_name(col)
        if target == "water_ml" and re.search(r"liter|litre|(^|_)l($|_)", name):
            values = values * 1000.0
        if target == "sodium_mg" and re.search(r"(^|_)g($|_)|grams", name) and "milli" not in name:
            values = values * 1000.0
        out[target] = values.to_numpy()
    out = out.dropna(subset=["date"])
    value_cols = [c for c in out.columns if c != "date"]
    if not value_cols:
        return pd.DataFrame(columns=["date"])
    return out.groupby("date", as_index=False)[value_cols].sum(min_count=1)


def read_mfp_export(path: Path) -> list[pd.DataFrame]:
    """Read nutrition summary CSV(s) from an MFP export ``.zip`` or a bare CSV."""
    if path.suffix.lower() == ".zip":
        frames: list[pd.DataFrame] = []
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if member.lower().endswith(".csv") and "nutrition" in member.lower():
                    with archive.open(member) as handle:
                        frames.append(pd.read_csv(io.TextIOWrapper(handle, encoding="utf-8-sig")))
        return frames
    return [read_tabular(path)]


class MyFitnessPalConnector(BaseConnector[MfpRaw]):
    """Daily calories and macros from MyFitnessPal."""

    name = "myfitnesspal"
    description = "MyFitnessPal calories, protein, carbs, fat, fiber and water (export / Health Connect / scraper)."
    tables = ("mfp_nutrition",)
    filter_to_range = False

    # ------------------------------------------------------------ source choice
    def _export_files(self) -> list[Path]:
        directory = self.settings.mfp_export_dir
        if directory is None or not directory.exists():
            return []
        return sorted(
            p for p in directory.rglob("*")
            if p.is_file() and (p.suffix.lower() == ".zip" or (p.suffix.lower() == ".csv" and "nutrition" in p.name.lower()))
        )

    def _health_connect_files(self) -> list[Path]:
        directory = self.settings.android_export_dir
        if directory is None or not directory.exists():
            return []
        return sorted(
            p for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in {".csv", ".json", ".jsonl"} and NUTRITION_FILE_HINTS.search(p.stem)
        )

    @staticmethod
    def _scraper_installed() -> bool:
        try:
            import myfitnesspal  # noqa: F401
        except ImportError:
            return False
        return True

    def _resolve_source(self) -> MfpSource | None:
        configured = self.settings.mfp_source
        if configured != "auto":
            return configured
        if self._export_files():
            return "export"
        if self._health_connect_files():
            return "health_connect"
        if self._scraper_installed():
            return "scraper"
        return None

    def missing_config(self) -> list[str]:
        """At least one source must be available."""
        source = self._resolve_source()
        if source is None:
            return [f"MFP export in {self.settings.mfp_export_dir}, Health Connect nutrition files, or the myfitnesspal package"]
        if source == "scraper" and not self._scraper_installed():
            return ["myfitnesspal package (pip install myfitnesspal)"]
        return []

    # ------------------------------------------------------------------- hooks
    def extract(self, start: date, end: date) -> MfpRaw:
        """Read export/Health Connect files, or scrape each day in ``[start, end]``."""
        source = self._resolve_source()
        if source == "export":
            frames = [f for path in self._export_files() for f in read_mfp_export(path)]
            return MfpRaw("export", frames)
        if source == "health_connect":
            return MfpRaw("health_connect", [read_tabular(p) for p in self._health_connect_files()])
        if source == "scraper":
            return MfpRaw("scraper", [self._scrape(start, end)])
        raise ConnectorError("No MyFitnessPal source available")

    def _scrape(self, start: date, end: date) -> pd.DataFrame:
        import myfitnesspal

        try:
            client = myfitnesspal.Client()
        except Exception as exc:  # noqa: BLE001 - library raises varied auth errors
            raise ConnectorError(f"MyFitnessPal login via browser cookies failed: {exc}") from exc
        water_factor = WATER_UNIT_TO_ML[self.settings.mfp_water_unit]
        rows: list[dict[str, Any]] = []
        for day in date_range_days(start, end):
            entry = client.get_date(day)
            totals = entry.totals
            if not totals and not entry.water:
                continue
            rows.append(
                {
                    "date": day,
                    "calories": totals.get("calories"),
                    "protein": totals.get("protein"),
                    "carbohydrates": totals.get("carbohydrates"),
                    "fat": totals.get("fat"),
                    "fiber": totals.get("fiber"),
                    "sugar": totals.get("sugar"),
                    "sodium": totals.get("sodium"),
                    "water_ml": float(entry.water or 0) * water_factor or None,
                }
            )
        return pd.DataFrame(rows)

    def transform(self, raw: MfpRaw) -> Frames:
        """Merge all source frames into one ``mfp_nutrition`` row per day."""
        tz = self.settings.tz
        source_filter = "myfitnesspal|mfp" if raw.source == "health_connect" else None
        daily_parts = [parse_nutrition_frame(f, tz, source_filter) for f in raw.frames]
        daily_parts = [d for d in daily_parts if len(d.columns) > 1 and not d.empty]
        if not daily_parts:
            return {"mfp_nutrition": self.empty_frame("mfp_nutrition")}
        combined = pd.concat(daily_parts, ignore_index=True)
        value_cols = [c for c in NUTRITION_COLUMNS if c in combined.columns]
        if raw.source == "export":
            # Overlapping export files repeat the same days; keep the latest file's totals.
            daily = combined.groupby("date", as_index=False)[value_cols].last()
        else:
            # Hydration and nutrition arrive in separate files: sum disjoint columns.
            daily = combined.groupby("date", as_index=False)[value_cols].sum(min_count=1)
        daily["source"] = raw.source
        return {"mfp_nutrition": daily}
