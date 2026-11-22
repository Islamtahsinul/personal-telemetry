"""Samsung S24 Ultra parser: Health Connect dumps and screen-usage exports.

Drop files into ``ANDROID_EXPORT_DIR`` (default ``data/inbox/android``) - or point
that setting at a folder synced by Health Sync / Tasker (e.g. a Google Drive or
Syncthing folder). Files are re-read only when their content hash changes, and
loads are idempotent upserts, so re-exporting the same history is safe.

Supported layouts (column names are matched case/punctuation-insensitively):

* **Long Health Connect records** - one row per record with a type column
  (``type`` / ``dataType`` / ``recordType``), a time column and a value column.
  JSON dumps such as ``{"steps": [...], "weight": [...]}`` are flattened the same way.
* **Per-metric files** (Health Sync style) - metric inferred from the filename,
  e.g. ``Steps.csv`` with ``Date, Time, Steps`` columns.
* **Wide daily files** - ``Date, Steps, Active calories, Weight, ...``.
* **Screen usage** (Digital Wellbeing via Tasker / ActionDash / StayFree) - rows
  with an app/package column and a duration column; filenames containing
  ``screen``, ``usage``, ``wellbeing``, ``tasker`` or ``actiondash`` are routed here.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import pandas as pd

from connectors.base import BaseConnector, Frames
from connectors.health_connect_export import is_health_connect_db, parse_health_connect_db
from connectors.samsung_health import is_samsung_csv, parse_samsung_export
from connectors.stayfree import is_stayfree_report, parse_stayfree_report
from connectors.utils import (
    DurationUnit,
    categorize_app,
    file_sha256,
    find_column,
    normalise_name,
    parse_duration_to_minutes,
    read_tabular,
    read_tabular_bytes,
    to_local_dates,
)
from core.config import Settings
from core.database import Database

TABULAR_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl"}

Aggregation = Literal["sum", "mean", "last"]


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """How a Health Connect metric maps to a column and aggregates per day."""

    column: str
    aggregation: Aggregation
    patterns: tuple[str, ...]


# Order matters: more specific patterns first ("active calories" before "calories").
HEALTH_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("steps", "sum", (r"step",)),
    MetricSpec("active_calories_kcal", "sum", (r"active_?calor", r"active_?energy", r"activecaloriesburned")),
    MetricSpec("total_calories_kcal", "sum", (r"total_?calor", r"totalcaloriesburned", r"calories_?burned")),
    MetricSpec("distance_m", "sum", (r"distance",)),
    MetricSpec("floors_climbed", "sum", (r"floor",)),
    MetricSpec("resting_heart_rate_bpm", "mean", (r"resting_?heart",)),
    MetricSpec("body_fat_pct", "last", (r"body_?fat",)),
    MetricSpec("skeletal_muscle_mass_kg", "last", (r"skeletal_?muscle", r"lean_?body", r"muscle_?mass")),
    MetricSpec("body_water_kg", "last", (r"body_?water",)),
    MetricSpec("bmr_kcal", "last", (r"basal_?metabolic", r"\bbmr\b")),
    MetricSpec("weight_kg", "last", (r"weight",)),
)
METRIC_BY_COLUMN: dict[str, MetricSpec] = {m.column: m for m in HEALTH_METRICS}

SCREEN_FILE_HINTS = re.compile(r"screen|usage|wellbeing|tasker|actiondash|stayfree", re.IGNORECASE)
NUTRITION_FILE_HINTS = re.compile(r"nutrition|food|meal|hydration|water", re.IGNORECASE)

TYPE_ALIASES = ["type", "data_type", "datatype", "record_type", "recordtype", "_type", "metric"]
TIME_ALIASES = ["start_time", "starttime", "start", "timestamp", "time", "datetime", "date_time", "date", "day"]
DATE_ONLY_ALIASES = ["date", "day"]
CLOCK_ALIASES = ["time"]
VALUE_ALIASES = [
    "value", "count", "in_kilocalories", "inkilocalories", "kcal", "energy", "calories", "in_kilograms",
    "inkilograms", "in_grams", "ingrams", "kg", "mass", "in_meters", "inmeters", "meters", "in_percent",
    "percentage", "percent", "beats_per_minute", "bpm", "floors", "amount",
]
SOURCE_ALIASES = ["data_origin", "dataorigin", "source", "origin", "package_name", "app"]
APP_ALIASES = ["app_name", "appname", "app", "application", "package_name", "package", "label"]
DURATION_ALIASES = [
    "usage_time", "screen_time", "foreground_time", "total_time", "duration", "usage", "time_used",
    "minutes", "seconds", "millis", "ms",
]
LAUNCH_ALIASES = ["launches", "launch_count", "opens", "times_opened", "unlocks", "count"]


def classify_metric(label: str) -> MetricSpec | None:
    """Return the :class:`MetricSpec` whose patterns match ``label`` (filename/type/column)."""
    text = normalise_name(label)
    if NUTRITION_FILE_HINTS.search(text):
        return None
    for spec in HEALTH_METRICS:
        if any(re.search(p, text) for p in spec.patterns):
            return spec
    return None


@dataclass(slots=True)
class AndroidRaw:
    """Parsed-but-not-aggregated Android data plus provenance for idempotency."""

    health_records: list[pd.DataFrame] = field(default_factory=list)
    screen_records: list[pd.DataFrame] = field(default_factory=list)
    samsung_files: dict[str, bytes] = field(default_factory=dict)
    health_connect_dbs: list[bytes] = field(default_factory=list)
    stayfree: list[dict[str, pd.DataFrame]] = field(default_factory=list)
    files: list[tuple[Path, str]] = field(default_factory=list)


# --------------------------------------------------------------------------- health
def parse_health_connect_frame(frame: pd.DataFrame, label: str, tz: ZoneInfo) -> pd.DataFrame:
    """Normalise any supported Health Connect layout to long records.

    Args:
        frame: Raw table read from a CSV/JSON file.
        label: File stem, used to infer the metric for per-metric files.
        tz: Local timezone for day bucketing.

    Returns:
        DataFrame with columns ``date``, ``metric``, ``value``, ``source_app``.
    """
    empty = pd.DataFrame(columns=["date", "metric", "value", "source_app"])
    if frame.empty:
        return empty

    dates = _resolve_dates(frame, tz)
    if dates is None:
        return empty
    source_col = find_column(frame, SOURCE_ALIASES)
    sources = frame[source_col].astype(str) if source_col else pd.Series("unknown", index=frame.index)

    type_col = find_column(frame, TYPE_ALIASES)
    value_col = _value_column(frame, exclude={type_col, source_col})
    if type_col and value_col:
        metrics = frame[type_col].map(lambda t: (spec.column if (spec := classify_metric(str(t))) else None))
        values = _scaled(frame[value_col], value_col)
        long = pd.DataFrame({"date": dates, "metric": metrics, "value": values, "source_app": sources})
        return long.dropna(subset=["date", "metric", "value"])

    # Wide layout: several columns are metrics themselves.
    metric_cols = {c: spec for c in frame.columns if (spec := classify_metric(str(c)))}
    file_spec = classify_metric(label)
    if len(metric_cols) > 1 or (metric_cols and file_spec is None):
        parts = [
            pd.DataFrame(
                {"date": dates, "metric": spec.column, "value": _scaled(frame[col], str(col)), "source_app": sources}
            )
            for col, spec in metric_cols.items()
        ]
        return pd.concat(parts, ignore_index=True).dropna(subset=["date", "value"])

    # Per-metric file: metric from the filename, value from the best numeric column.
    if file_spec is not None:
        col = next(iter(metric_cols)) if metric_cols else value_col
        if col is None:
            return empty
        long = pd.DataFrame(
            {"date": dates, "metric": file_spec.column, "value": _scaled(frame[col], str(col)), "source_app": sources}
        )
        return long.dropna(subset=["date", "value"])
    return empty


def aggregate_health_daily(records: pd.DataFrame, source_filter: str | None = None) -> pd.DataFrame:
    """Aggregate long Health Connect records to one row per day.

    Additive metrics (steps, calories, distance) are summed per source app and the
    largest per-source daily total is kept - this mirrors Health Connect's own
    de-duplication when the phone and watch both report the same activity.

    Args:
        records: Output of :func:`parse_health_connect_frame` (possibly concatenated).
        source_filter: Optional case-insensitive regex; only matching ``source_app`` rows are used.
    """
    if records.empty:
        return pd.DataFrame(columns=["date"])
    data = records.copy()
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["value"])
    if source_filter:
        data = data.loc[data["source_app"].str.contains(source_filter, case=False, na=False, regex=True)]
    if data.empty:
        return pd.DataFrame(columns=["date"])

    columns: list[pd.Series] = []
    for metric, group in data.groupby("metric"):
        spec = METRIC_BY_COLUMN.get(str(metric))
        if spec is None:
            continue
        if spec.aggregation == "sum":
            series = group.groupby(["date", "source_app"])["value"].sum().groupby("date").max()
        elif spec.aggregation == "mean":
            series = group.groupby("date")["value"].mean()
        else:
            series = group.groupby("date")["value"].last()
        columns.append(series.rename(spec.column))
    if not columns:
        return pd.DataFrame(columns=["date"])
    daily = pd.concat(columns, axis=1).reset_index().rename(columns={"index": "date"})
    if "steps" in daily:
        daily["steps"] = daily["steps"].round().astype("Int64")
    return daily


# --------------------------------------------------------------------------- screen
def parse_screen_usage_frame(
    frame: pd.DataFrame,
    tz: ZoneInfo,
    category_rules: dict[str, list[str]],
    source: str = "android_export",
) -> pd.DataFrame:
    """Normalise a Digital Wellbeing / Tasker usage export to ``screen_usage`` rows.

    Numeric durations take their unit from the column name (``ms``, ``sec``,
    ``min``, ``hour``). For an unlabelled numeric column the unit is inferred from
    magnitude: per-app daily totals above 1440 imply seconds, above 86 400 ms.
    """
    columns = ["date", "device", "app", "category", "minutes", "launches", "source"]
    app_col = find_column(frame, APP_ALIASES)
    duration_col = find_column(frame, DURATION_ALIASES)
    dates = _resolve_dates(frame, tz)
    if frame.empty or app_col is None or duration_col is None or dates is None:
        return pd.DataFrame(columns=columns)

    durations = frame[duration_col]
    unit = _duration_unit(str(duration_col), durations, dates, frame[app_col])
    minutes = durations.map(lambda v: parse_duration_to_minutes(v, unit))
    launch_col = find_column(frame, LAUNCH_ALIASES)
    launches = (
        pd.to_numeric(frame[launch_col], errors="coerce")
        if launch_col and launch_col != duration_col
        else pd.Series(pd.NA, index=frame.index)
    )
    rows = pd.DataFrame(
        {"date": dates, "app": frame[app_col].astype(str).str.strip(), "minutes": minutes, "launches": launches}
    ).dropna(subset=["date", "minutes"])
    rows = rows.loc[rows["app"].ne("") & rows["app"].str.lower().ne("nan")]
    daily = rows.groupby(["date", "app"], as_index=False).agg(
        minutes=("minutes", "sum"), launches=("launches", lambda s: s.sum(min_count=1))
    )
    daily["device"] = "s24_ultra"
    daily["category"] = daily["app"].map(lambda a: categorize_app(a, category_rules))
    daily["source"] = source
    daily["minutes"] = daily["minutes"].round(2)
    daily["launches"] = daily["launches"].astype("Int64")
    return daily[columns]


# --------------------------------------------------------------------------- connector
class AndroidS24Connector(BaseConnector[AndroidRaw]):
    """Ingest Health Connect and screen-usage exports from the S24 Ultra."""

    name = "android_s24"
    description = "Samsung S24 Ultra Health Connect dumps (Health Sync) and Digital Wellbeing/Tasker usage exports."
    # Samsung Health exports can also carry food logged via MyFitnessPal / Health Connect.
    tables = ("s24_health", "screen_usage", "mfp_nutrition")
    filter_to_range = False

    def __init__(self, settings: Settings | None = None, db: Database | None = None, *, force: bool = False) -> None:
        """Args: force: re-ingest files even if their content hash was seen before."""
        super().__init__(settings, db)
        self.force = force
        self._pending_files: list[tuple[Path, str]] = []

    def missing_config(self) -> list[str]:
        """The export directory must exist."""
        directory = self.settings.android_export_dir
        return [] if directory and directory.exists() else ["ANDROID_EXPORT_DIR (folder not found)"]

    def extract(self, start: date, end: date) -> AndroidRaw:
        """Read every new/changed CSV, JSON, zip or Health Connect export in the export directory."""
        assert self.settings.android_export_dir is not None
        raw = AndroidRaw()
        tz = self.settings.tz
        for path in sorted(self.settings.android_export_dir.rglob("*")):
            if path.suffix.lower() not in {*TABULAR_SUFFIXES, ".zip", ".db", ".xls", ".xlsx"} or not path.is_file():
                continue
            digest = file_sha256(path)
            if not self.force and self.db.is_file_ingested(digest):
                continue
            if path.suffix.lower() in {".xls", ".xlsx"}:
                if is_stayfree_report(path):
                    raw.stayfree.append(parse_stayfree_report(path, self.settings.phone_category_rules))
                    raw.files.append((path, digest))
                else:
                    self.log.warning("Skipping %s - not a StayFree report", path.name)
                continue
            if path.suffix.lower() == ".zip":
                try:
                    with zipfile.ZipFile(path) as archive:
                        members = [(m, archive.read(m)) for m in archive.namelist()
                                   if Path(m).suffix.lower() in TABULAR_SUFFIXES or is_health_connect_db(m)]
                except zipfile.BadZipFile as exc:
                    self.log.warning("Skipping unreadable zip %s: %s", path.name, exc)
                    continue
            else:
                members = [(path.name, path.read_bytes())]
            for name, content in members:
                if is_health_connect_db(name):
                    raw.health_connect_dbs.append(content)
                    continue
                if is_samsung_csv(name, content[:64]):
                    raw.samsung_files[name] = content
                    continue
                try:
                    frame = read_tabular_bytes(name, content)
                except Exception as exc:  # noqa: BLE001 - skip unreadable files, keep going
                    self.log.warning("Skipping unreadable file %s: %s", name, exc)
                    continue
                member = Path(name)
                if _is_screen_file(member, frame):
                    raw.screen_records.append(parse_screen_usage_frame(frame, tz, self.settings.phone_category_rules))
                else:
                    raw.health_records.append(parse_health_connect_frame(frame, member.stem, tz))
            raw.files.append((path, digest))
        self.log.info("Parsed %d new Android export files", len(raw.files))
        return raw

    def transform(self, raw: AndroidRaw) -> Frames:
        """Aggregate parsed records into ``s24_health`` and ``screen_usage``."""
        frames: Frames = {}
        if raw.health_records:
            daily = aggregate_health_daily(pd.concat(raw.health_records, ignore_index=True))
            if len(daily.columns) > 1:
                frames["s24_health"] = daily
        # Later sources fill gaps in earlier ones: generic files < Samsung download < Health Connect export.
        extra: list[Frames] = []
        if raw.samsung_files:
            extra.append(parse_samsung_export(raw.samsung_files, self.settings.tz))
        extra.extend(parse_health_connect_db(content) for content in raw.health_connect_dbs)
        for report in raw.stayfree:
            if "screen_usage" in report:
                raw.screen_records.append(report["screen_usage"])
            if "s24_health" in report:
                extra.append({"s24_health": report["s24_health"]})
        for parsed in extra:
            for table in ("s24_health", "mfp_nutrition"):
                if table not in parsed:
                    continue
                existing = frames.get(table)
                frames[table] = (
                    parsed[table] if existing is None
                    else parsed[table].set_index("date").combine_first(existing.set_index("date")).reset_index()
                )
        if raw.screen_records:
            screen = pd.concat(raw.screen_records, ignore_index=True)
            if not screen.empty:
                frames["screen_usage"] = screen.drop_duplicates(["date", "device", "app"], keep="last")
        self._pending_files = raw.files
        return frames

    def load(self, frames: Frames) -> dict[str, int]:
        """Upsert, then mark the source files as ingested."""
        written = super().load(frames)
        total = sum(written.values())
        for path, digest in self._pending_files:
            self.db.mark_file_ingested(digest, path, self.name, total)
        return written


# --------------------------------------------------------------------------- helpers
def load_health_connect_records(directory: Path, tz: ZoneInfo) -> pd.DataFrame:
    """Parse every Health Connect file under ``directory`` into long records (no hashing).

    Used by the Renpho and MyFitnessPal connectors as a fallback source.
    """
    parts: list[pd.DataFrame] = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in {".csv", ".tsv", ".json", ".jsonl"} or not path.is_file():
            continue
        try:
            frame = read_tabular(path)
        except Exception:  # noqa: BLE001
            continue
        if not _is_screen_file(path, frame):
            parts.append(parse_health_connect_frame(frame, path.stem, tz))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["date", "metric", "value", "source_app"])


def _is_screen_file(path: Path, frame: pd.DataFrame) -> bool:
    if SCREEN_FILE_HINTS.search(path.stem):
        return True
    return find_column(frame, APP_ALIASES) is not None and find_column(frame, DURATION_ALIASES) is not None and (
        find_column(frame, TYPE_ALIASES) is None
    )


def _resolve_dates(frame: pd.DataFrame, tz: ZoneInfo) -> pd.Series | None:
    """Find the timestamp (or separate date + time) columns and return local dates."""
    date_col = find_column(frame, DATE_ONLY_ALIASES)
    time_col = find_column(frame, CLOCK_ALIASES)
    if date_col and time_col and date_col != time_col and normalise_name(time_col) == "time":
        combined = frame[date_col].astype(str) + " " + frame[time_col].astype(str)
        return to_local_dates(combined, tz)
    col = find_column(frame, TIME_ALIASES)
    if col is None:
        return None
    return to_local_dates(frame[col], tz)


def _value_column(frame: pd.DataFrame, exclude: set[str | None]) -> str | None:
    col = find_column(frame, VALUE_ALIASES)
    if col and col not in exclude:
        return col
    time_cols = {find_column(frame, TIME_ALIASES), find_column(frame, CLOCK_ALIASES)}
    for candidate in frame.columns:
        if candidate in exclude or candidate in time_cols:
            continue
        if pd.to_numeric(frame[candidate], errors="coerce").notna().mean() > 0.8:
            return str(candidate)
    return None


def _scaled(values: pd.Series, column: str) -> pd.Series:
    """Convert unit-suffixed values (grams, kilometres, kJ) to schema units."""
    numeric = pd.to_numeric(values.astype(str).str.replace(",", "", regex=False), errors="coerce")
    name = normalise_name(column)
    if re.search(r"(^|_)in_?grams|(^|_)grams?$", name) and "kilo" not in name:
        return numeric / 1000.0
    if re.search(r"kilomet|(^|_)km($|_)", name):
        return numeric * 1000.0
    if re.search(r"kilojoule|(^|_)kj($|_)", name):
        return numeric / 4.184
    return numeric


def _duration_unit(column: str, values: pd.Series, dates: pd.Series, apps: pd.Series) -> DurationUnit:
    name = normalise_name(column)
    if re.search(r"(^|_)(ms|millis|milliseconds)($|_)", name):
        return "ms"
    if re.search(r"sec", name):
        return "s"
    if re.search(r"hour|(^|_)h($|_)", name):
        return "h"
    if re.search(r"min", name):
        return "min"
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() < 0.8:
        return "min"  # human strings like "1h 5m" carry their own units
    per_app_day = numeric.groupby([dates, apps]).sum()
    peak = float(per_app_day.max()) if not per_app_day.empty else 0.0
    if peak > 86_400:
        return "ms"
    if peak > 1_440:
        return "s"
    return "min"
