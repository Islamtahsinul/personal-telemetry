"""Generic CSV/JSON ingester for manual file drops.

Drop files into ``data/inbox/<table_name>/`` (e.g. ``data/inbox/subjective_state/``),
or anywhere under ``data/inbox/`` with a filename starting with the table name
(e.g. ``weather_daily_backfill.csv``). Columns are matched to the table schema by
normalised name plus a small alias list; unknown columns are ignored. Rows should
already be at the table's grain (one per primary key); duplicates keep the last row.

A MyFitnessPal *Nutrition Summary* CSV dropped anywhere in the inbox is detected
and summed to daily totals automatically.

Successfully ingested files are moved to ``data/processed/<table>/``. Content
hashes are recorded so the same file is never loaded twice.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from connectors.base import BaseConnector, ConnectorError, Frames
from connectors.myfitnesspal import parse_nutrition_frame
from connectors.utils import file_sha256, normalise_name, read_tabular
from core.config import Settings
from core.database import DOMAIN_TABLES, TABLES, Database

# Folders owned by dedicated connectors.
RESERVED_SUBDIRS = {"android", "myfitnesspal", "monzo"}
SUPPORTED_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl"}

COLUMN_ALIASES: dict[str, list[str]] = {
    "date": ["day", "date_local", "timestamp", "datetime"],
    "mood": ["mood_score", "mood_1_5"],
    "energy": ["energy_score", "energy_level"],
    "focus": ["focus_score", "concentration"],
    "notes": ["note", "comment", "comments", "journal"],
    "amount_gbp": ["amount", "spend", "total"],
    "minutes": ["duration_minutes", "usage_minutes", "screen_time_minutes"],
    "weight_kg": ["weight"],
    "body_fat_pct": ["body_fat", "bodyfat", "fat_pct"],
    "steps": ["step_count"],
}


@dataclass(slots=True)
class PendingFile:
    """A file waiting to be loaded into ``table``."""

    path: Path
    file_hash: str
    table: str
    frame: pd.DataFrame


@dataclass(slots=True)
class FileDropRaw:
    """Files discovered in the inbox."""

    files: list[PendingFile] = field(default_factory=list)


def map_columns(frame: pd.DataFrame, table: str) -> pd.DataFrame:
    """Rename ``frame`` columns to the schema of ``table`` and drop the rest.

    Raises:
        ConnectorError: If a primary-key column cannot be matched.
    """
    spec = TABLES[table]
    by_norm = {normalise_name(c): c for c in frame.columns}
    renames: dict[str, str] = {}
    for target in spec.columns:
        if target == "ingested_at":
            continue
        for candidate in [target, *COLUMN_ALIASES.get(target, [])]:
            source = by_norm.get(normalise_name(candidate))
            if source is not None and source not in renames:
                renames[source] = target
                break
    mapped = frame[list(renames)].rename(columns=renames)
    missing = [k for k in spec.primary_key if k not in mapped.columns]
    if missing:
        raise ConnectorError(f"{table}: file is missing key column(s) {missing}; found {list(frame.columns)}")
    return mapped


def detect_table(path: Path, inbox: Path, frame: pd.DataFrame) -> str | None:
    """Infer the target table from folder name, filename prefix or known formats."""
    relative = path.relative_to(inbox)
    if len(relative.parts) > 1 and relative.parts[0] in TABLES:
        return relative.parts[0]
    stem = normalise_name(path.stem)
    for table in sorted(DOMAIN_TABLES, key=len, reverse=True):
        if stem.startswith(table):
            return table
    columns = {normalise_name(c) for c in frame.columns}
    if {"date", "meal", "calories"} <= columns:
        return "mfp_nutrition"
    if {"date", "mood"} <= columns or {"date", "energy", "focus"} <= columns:
        return "subjective_state"
    return None


class FileParserConnector(BaseConnector[FileDropRaw]):
    """Ingest manual CSV/JSON drops from the inbox folder."""

    name = "file_drop"
    description = "Manual CSV/JSON file drops in data/inbox/<table>/."
    tables = tuple(DOMAIN_TABLES)
    filter_to_range = False

    def __init__(
        self, settings: Settings | None = None, db: Database | None = None, *, move_processed: bool = True
    ) -> None:
        """Args: move_processed: move files to ``data/processed`` after loading."""
        super().__init__(settings, db)
        self.move_processed = move_processed
        self._pending: list[PendingFile] = []

    def missing_config(self) -> list[str]:
        """The inbox folder is created automatically, so nothing is required."""
        return []

    def discover(self) -> list[Path]:
        """List candidate files under the inbox, excluding connector-owned folders."""
        inbox = self.settings.inbox_dir
        files: list[Path] = []
        for path in sorted(inbox.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            relative = path.relative_to(inbox)
            if len(relative.parts) > 1 and relative.parts[0] in RESERVED_SUBDIRS:
                continue
            files.append(path)
        return files

    def extract(self, start: date, end: date) -> FileDropRaw:
        """Read new files and assign each a target table."""
        raw = FileDropRaw()
        for path in self.discover():
            digest = file_sha256(path)
            if self.db.is_file_ingested(digest):
                self.log.info("Already ingested, skipping %s", path.name)
                continue
            frame = read_tabular(path)
            table = detect_table(path, self.settings.inbox_dir, frame)
            if table is None:
                self.log.warning("Cannot determine target table for %s - put it in inbox/<table>/", path.name)
                continue
            raw.files.append(PendingFile(path, digest, table, frame))
        return raw

    def transform(self, raw: FileDropRaw) -> Frames:
        """Map each file's columns to its table schema and concatenate per table."""
        per_table: dict[str, list[pd.DataFrame]] = {}
        for pending in raw.files:
            if pending.table == "mfp_nutrition" and "meal" in {normalise_name(c) for c in pending.frame.columns}:
                frame = parse_nutrition_frame(pending.frame, self.settings.tz)
                frame["source"] = "export"
            else:
                frame = map_columns(pending.frame, pending.table)
            per_table.setdefault(pending.table, []).append(frame)
        self._pending = raw.files
        return {table: pd.concat(parts, ignore_index=True) for table, parts in per_table.items()}

    def load(self, frames: Frames) -> dict[str, int]:
        """Upsert, record hashes and move processed files."""
        written = super().load(frames)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        for pending in self._pending:
            self.db.mark_file_ingested(pending.file_hash, pending.path, self.name, written.get(pending.table, 0))
            if self.move_processed:
                target_dir = self.settings.processed_dir / pending.table
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(pending.path), target_dir / f"{stamp}_{pending.path.name}")
        return written

    def ingest_file(self, path: Path, table: str) -> int:
        """Load a single file into ``table`` immediately (used by the UI uploader).

        Returns:
            Number of rows written.
        """
        if table not in self.tables:
            raise ConnectorError(f"Unknown table {table}")
        digest = file_sha256(path)
        raw = FileDropRaw([PendingFile(path, digest, table, read_tabular(path))])
        previous = self.move_processed
        self.move_processed = False
        try:
            return sum(self.load(self.transform(raw)).values())
        finally:
            self.move_processed = previous
