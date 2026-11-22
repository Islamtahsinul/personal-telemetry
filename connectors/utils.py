"""Shared parsing helpers for connectors (durations, flexible columns, dates, files)."""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_HMS = re.compile(r"^\s*(\d+):(\d{1,2})(?::(\d{1,2}(?:\.\d+)?))?\s*$")
_UNIT_TOKEN = re.compile(r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes|s|sec|secs|second|seconds|ms)\b")

DurationUnit = Literal["ms", "s", "min", "h"]
_UNIT_TO_MINUTES: dict[str, float] = {
    "ms": 1 / 60000,
    "s": 1 / 60,
    "sec": 1 / 60,
    "secs": 1 / 60,
    "second": 1 / 60,
    "seconds": 1 / 60,
    "m": 1.0,
    "min": 1.0,
    "mins": 1.0,
    "minute": 1.0,
    "minutes": 1.0,
    "h": 60.0,
    "hr": 60.0,
    "hrs": 60.0,
    "hour": 60.0,
    "hours": 60.0,
}


def normalise_name(name: Any) -> str:
    """Lower-case a column name and collapse non-alphanumerics to ``_``."""
    return _NON_ALNUM.sub("_", str(name).strip().lower()).strip("_")


def slugify(text: Any) -> str:
    """Return a safe identifier fragment for ``text`` (e.g. a category)."""
    slug = normalise_name(text)
    return slug or "unknown"


def find_column(frame: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    """Return the first column of ``frame`` matching any alias (case/punctuation-insensitive).

    Exact normalised matches win; otherwise an alias contained in a column name matches.
    """
    normalised = {normalise_name(c): c for c in frame.columns}
    alias_list = [normalise_name(a) for a in aliases]
    for alias in alias_list:
        if alias in normalised:
            return normalised[alias]
    for alias in alias_list:
        for norm, original in normalised.items():
            if alias and alias in norm:
                return original
    return None


def parse_duration_to_minutes(value: Any, numeric_unit: DurationUnit = "min") -> float | None:
    """Parse a duration into minutes.

    Accepts numbers (interpreted with ``numeric_unit``), ``HH:MM[:SS]`` strings and
    human strings such as ``"1h 23m"``, ``"45 min"``, ``"2 hrs 5 mins 10 s"``.

    Returns:
        Minutes as a float, or ``None`` when the value cannot be parsed.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) * _UNIT_TO_MINUTES[numeric_unit]
    text = str(value).strip().lower()
    if not text:
        return None
    try:
        return float(text) * _UNIT_TO_MINUTES[numeric_unit]
    except ValueError:
        pass
    if match := _HMS.match(text):
        hours, minutes, seconds = match.groups()
        return int(hours) * 60 + int(minutes) + (float(seconds) / 60 if seconds else 0.0)
    tokens = _UNIT_TOKEN.findall(text)
    if not tokens:
        return None
    return sum(float(num) * _UNIT_TO_MINUTES[unit] for num, unit in tokens)


def to_local_dates(values: pd.Series, tz: ZoneInfo, dayfirst: bool = False) -> pd.Series:
    """Convert timestamps/dates to local calendar ``date`` objects.

    Timezone-aware values are converted to ``tz``; naive values are assumed local.
    Numeric values are treated as epoch seconds (or milliseconds when > 1e12).
    """
    if pd.api.types.is_numeric_dtype(values):
        numeric = values.astype("float64")
        unit = "ms" if numeric.dropna().gt(1e12).any() else "s"
        parsed = pd.to_datetime(numeric, unit=unit, utc=True)
    else:
        parsed = pd.to_datetime(values, errors="coerce", dayfirst=dayfirst, utc=False, format="mixed")
        if parsed.dt.tz is None:
            # Strings carrying offsets parse to object dtype; normalise via UTC.
            if parsed.isna().all() or values.astype(str).str.contains(r"[+-]\d{2}:?\d{2}$|Z$", regex=True).any():
                parsed = pd.to_datetime(values, errors="coerce", dayfirst=dayfirst, utc=True, format="mixed")
            else:
                return parsed.dt.date
    return parsed.dt.tz_convert(tz).dt.date


def coerce_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Convert the given columns to floats, stripping thousands separators and units."""
    for col in columns:
        if col in frame.columns and not pd.api.types.is_numeric_dtype(frame[col]):
            cleaned = frame[col].astype(str).str.replace(",", "", regex=False).str.extract(r"(-?\d+(?:\.\d+)?)")[0]
            frame[col] = pd.to_numeric(cleaned, errors="coerce")
    return frame


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tabular(path: Path) -> pd.DataFrame:
    """Read a CSV / TSV / JSON / JSON-Lines file into a DataFrame.

    JSON may be a list of records, an object with a single list-valued key
    (e.g. ``{"records": [...]}``) or newline-delimited records.
    """
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        with path.open(encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        return pd.json_normalize(_records_from_json(payload))
    raise ValueError(f"Unsupported file type: {path.name}")


def read_tabular_bytes(name: str, content: bytes) -> pd.DataFrame:
    """Like :func:`read_tabular` for in-memory content (e.g. a zip member)."""
    suffix = Path(name).suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        return pd.read_csv(io.BytesIO(content), sep=None, engine="python", encoding="utf-8-sig")
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(io.BytesIO(content), lines=True)
    if suffix == ".json":
        return pd.json_normalize(_records_from_json(json.loads(content.decode("utf-8-sig"))))
    raise ValueError(f"Unsupported file type: {name}")


def _records_from_json(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        list_values = [v for v in payload.values() if isinstance(v, list)]
        if len(list_values) == 1:
            return list_values[0]
        if list_values:
            # Several record lists (e.g. per data type) -> tag each with its key.
            records: list[dict[str, Any]] = []
            for key, value in payload.items():
                if isinstance(value, list):
                    records.extend({**item, "_type": key} for item in value if isinstance(item, dict))
            return records
        return [payload]
    raise ValueError("Unsupported JSON structure")


def categorize_app(app: str, rules: dict[str, list[str]], default: str = "other") -> str:
    """Return the first category whose regex patterns match ``app`` (case-insensitive)."""
    text = str(app).lower()
    for category, patterns in rules.items():
        if any(re.search(pattern, text) for pattern in patterns):
            return category
    return default


def date_range_days(start: date, end: date) -> list[date]:
    """Inclusive list of dates from ``start`` to ``end``."""
    return [d.date() for d in pd.date_range(start, end, freq="D")]
