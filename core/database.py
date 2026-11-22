"""DuckDB connection manager and the local time-series schema.

Design notes
------------
* **Single source of truth.** Every table is declared once as a :class:`TableSpec`
  in :data:`TABLES`. DDL, upsert key handling and column validation are all derived
  from it.
* **Daily grain.** Domain tables are keyed by local calendar ``date`` (plus extra
  dimensions such as ``category`` or ``app`` where needed). Raw sub-daily data is
  aggregated by the connectors before loading.
* **Idempotent loads.** Writes are ``INSERT ... ON CONFLICT DO UPDATE`` upserts, so
  re-running a sync for an overlapping date range never duplicates rows. By default
  non-null incoming values overwrite and nulls preserve existing values, which lets
  several sources contribute different columns to the same row.
* **Short-lived connections.** DuckDB allows only one read-write process per file.
  Connections are opened per operation so that the CLI and the Streamlit UI can
  take turns on the same database without holding a long-lived lock.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from core.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TableSpec:
    """Declarative definition of a DuckDB table.

    Attributes:
        name: Table name.
        columns: Ordered mapping of column name to DuckDB type (may include constraints).
        primary_key: Columns forming the upsert conflict target.
        domain: Logical domain used by analytics (e.g. ``"whoop"``); ``"meta"`` for internal tables.
        description: Human-readable description.
        checks: Optional table-level CHECK constraint expressions.
    """

    name: str
    columns: dict[str, str]
    primary_key: tuple[str, ...]
    domain: str
    description: str = ""
    checks: tuple[str, ...] = field(default_factory=tuple)

    @property
    def value_columns(self) -> list[str]:
        """Columns that are not part of the primary key or bookkeeping."""
        return [c for c in self.columns if c not in self.primary_key and c != "ingested_at"]

    def ddl(self) -> str:
        """Return the ``CREATE TABLE IF NOT EXISTS`` statement for this table."""
        cols = [f"    {name} {dtype}" for name, dtype in self.columns.items()]
        cols.append(f"    PRIMARY KEY ({', '.join(self.primary_key)})")
        cols.extend(f"    CHECK ({expr})" for expr in self.checks)
        body = ",\n".join(cols)
        return f"CREATE TABLE IF NOT EXISTS {self.name} (\n{body}\n);"


def _with_ingested_at(columns: dict[str, str]) -> dict[str, str]:
    return {**columns, "ingested_at": "TIMESTAMP DEFAULT current_timestamp"}


_SCORE_1_TO_5 = "TINYINT CHECK ({col} IS NULL OR {col} BETWEEN 1 AND 5)"

TABLES: dict[str, TableSpec] = {
    spec.name: spec
    for spec in (
        # ------------------------------------------------------------- WHOOP
        TableSpec(
            name="whoop_recovery",
            domain="whoop",
            description="Daily WHOOP recovery, keyed by the local date the physiological cycle started.",
            primary_key=("date",),
            columns=_with_ingested_at(
                {
                    "date": "DATE NOT NULL",
                    "cycle_id": "BIGINT",
                    "recovery_score": "DOUBLE",
                    "hrv_rmssd_ms": "DOUBLE",
                    "resting_heart_rate_bpm": "DOUBLE",
                    "spo2_pct": "DOUBLE",
                    "skin_temp_c": "DOUBLE",
                    "user_calibrating": "BOOLEAN",
                }
            ),
        ),
        TableSpec(
            name="whoop_cycle",
            domain="whoop",
            description="Daily WHOOP physiological cycle (day strain).",
            primary_key=("date",),
            columns=_with_ingested_at(
                {
                    "date": "DATE NOT NULL",
                    "cycle_id": "BIGINT",
                    "strain": "DOUBLE",
                    "kilojoules": "DOUBLE",
                    "avg_heart_rate_bpm": "DOUBLE",
                    "max_heart_rate_bpm": "DOUBLE",
                }
            ),
        ),
        TableSpec(
            name="whoop_sleep",
            domain="whoop",
            description="Main (non-nap) WHOOP sleep, keyed by the local date of waking.",
            primary_key=("date",),
            columns=_with_ingested_at(
                {
                    "date": "DATE NOT NULL",
                    "sleep_id": "VARCHAR",
                    "sleep_performance_pct": "DOUBLE",
                    "sleep_efficiency_pct": "DOUBLE",
                    "sleep_consistency_pct": "DOUBLE",
                    "respiratory_rate": "DOUBLE",
                    "total_in_bed_hours": "DOUBLE",
                    "total_sleep_hours": "DOUBLE",
                    "light_sleep_hours": "DOUBLE",
                    "deep_sleep_hours": "DOUBLE",
                    "rem_sleep_hours": "DOUBLE",
                    "awake_hours": "DOUBLE",
                    "disturbance_count": "INTEGER",
                    "sleep_cycle_count": "INTEGER",
                    "sleep_start": "TIMESTAMPTZ",
                    "sleep_end": "TIMESTAMPTZ",
                }
            ),
        ),
        # --------------------------------------------------- Samsung S24 Ultra
        TableSpec(
            name="s24_health",
            domain="s24",
            description="Daily Health Connect activity and body composition from the Samsung S24 Ultra / Galaxy Watch.",
            primary_key=("date",),
            columns=_with_ingested_at(
                {
                    "date": "DATE NOT NULL",
                    "steps": "INTEGER",
                    "active_calories_kcal": "DOUBLE",
                    "total_calories_kcal": "DOUBLE",
                    "distance_m": "DOUBLE",
                    "floors_climbed": "DOUBLE",
                    "phone_unlocks": "INTEGER",
                    "resting_heart_rate_bpm": "DOUBLE",
                    "weight_kg": "DOUBLE",
                    "body_fat_pct": "DOUBLE",
                    "skeletal_muscle_mass_kg": "DOUBLE",
                    "body_water_kg": "DOUBLE",
                    "bmr_kcal": "DOUBLE",
                }
            ),
        ),
        # ------------------------------------------------------------- Renpho
        TableSpec(
            name="renpho_body",
            domain="renpho",
            description="Daily Renpho smart-scale body composition (last weigh-in of the day).",
            primary_key=("date",),
            columns=_with_ingested_at(
                {
                    "date": "DATE NOT NULL",
                    "weight_kg": "DOUBLE",
                    "bmi": "DOUBLE",
                    "body_fat_pct": "DOUBLE",
                    "muscle_mass_pct": "DOUBLE",
                    "muscle_mass_kg": "DOUBLE",
                    "visceral_fat": "DOUBLE",
                    "subcutaneous_fat_pct": "DOUBLE",
                    "body_water_pct": "DOUBLE",
                    "bone_mass_pct": "DOUBLE",
                    "protein_pct": "DOUBLE",
                    "bmr_kcal": "DOUBLE",
                    "lean_body_mass_kg": "DOUBLE",
                    "body_age": "DOUBLE",
                    "measurement_count": "INTEGER",
                    "source": "VARCHAR",
                }
            ),
        ),
        # -------------------------------------------------------- MyFitnessPal
        TableSpec(
            name="mfp_nutrition",
            domain="mfp",
            description="Daily nutrition totals from MyFitnessPal (export, scraper or Health Connect).",
            primary_key=("date",),
            columns=_with_ingested_at(
                {
                    "date": "DATE NOT NULL",
                    "calories_kcal": "DOUBLE",
                    "protein_g": "DOUBLE",
                    "carbs_g": "DOUBLE",
                    "fat_g": "DOUBLE",
                    "fiber_g": "DOUBLE",
                    "sugar_g": "DOUBLE",
                    "sodium_mg": "DOUBLE",
                    "water_ml": "DOUBLE",
                    "source": "VARCHAR",
                }
            ),
        ),
        # ------------------------------------------------------- Screen usage
        TableSpec(
            name="screen_usage",
            domain="screen",
            description="Daily per-app foreground time for each device (phone and laptop).",
            primary_key=("date", "device", "app"),
            columns=_with_ingested_at(
                {
                    "date": "DATE NOT NULL",
                    "device": "VARCHAR NOT NULL",
                    "app": "VARCHAR NOT NULL",
                    "category": "VARCHAR",
                    "minutes": "DOUBLE",
                    "launches": "INTEGER",
                    "source": "VARCHAR",
                }
            ),
        ),
        TableSpec(
            name="laptop_activity",
            domain="laptop",
            description="Daily laptop activity summary derived from ActivityWatch.",
            primary_key=("date",),
            columns=_with_ingested_at(
                {
                    "date": "DATE NOT NULL",
                    "active_minutes": "DOUBLE",
                    "afk_minutes": "DOUBLE",
                    "coding_minutes": "DOUBLE",
                    "work_minutes": "DOUBLE",
                    "first_active": "TIMESTAMPTZ",
                    "last_active": "TIMESTAMPTZ",
                    "top_app": "VARCHAR",
                }
            ),
        ),
        # --------------------------------------------------------------- Monzo
        TableSpec(
            name="monzo_spending",
            domain="spend",
            description="Daily Monzo outgoing spend per normalised category (GBP, positive = money out).",
            primary_key=("date", "category"),
            columns=_with_ingested_at(
                {
                    "date": "DATE NOT NULL",
                    "category": "VARCHAR NOT NULL",
                    "amount_gbp": "DOUBLE",
                    "transaction_count": "INTEGER",
                }
            ),
        ),
        # ------------------------------------------------------------- Weather
        TableSpec(
            name="weather_daily",
            domain="weather",
            description="Daily local weather from Open-Meteo.",
            primary_key=("date",),
            columns=_with_ingested_at(
                {
                    "date": "DATE NOT NULL",
                    "temp_max_c": "DOUBLE",
                    "temp_min_c": "DOUBLE",
                    "temp_mean_c": "DOUBLE",
                    "apparent_temp_mean_c": "DOUBLE",
                    "precipitation_mm": "DOUBLE",
                    "rain_mm": "DOUBLE",
                    "precipitation_hours": "DOUBLE",
                    "daylight_hours": "DOUBLE",
                    "sunshine_hours": "DOUBLE",
                    "pressure_msl_hpa": "DOUBLE",
                    "humidity_mean_pct": "DOUBLE",
                    "cloud_cover_mean_pct": "DOUBLE",
                    "wind_speed_max_kmh": "DOUBLE",
                    "latitude": "DOUBLE",
                    "longitude": "DOUBLE",
                }
            ),
        ),
        # ---------------------------------------------------- Subjective state
        TableSpec(
            name="subjective_state",
            domain="subjective",
            description="Self-reported daily mood, energy and focus on a 1-5 scale.",
            primary_key=("date",),
            columns=_with_ingested_at(
                {
                    "date": "DATE NOT NULL",
                    "mood": _SCORE_1_TO_5.format(col="mood"),
                    "energy": _SCORE_1_TO_5.format(col="energy"),
                    "focus": _SCORE_1_TO_5.format(col="focus"),
                    "notes": "VARCHAR",
                }
            ),
        ),
        # ---------------------------------------------------------------- Meta
        TableSpec(
            name="ingestion_runs",
            domain="meta",
            description="Audit log of connector runs.",
            primary_key=("run_id",),
            columns={
                "run_id": "VARCHAR NOT NULL",
                "connector": "VARCHAR NOT NULL",
                "started_at": "TIMESTAMPTZ NOT NULL",
                "finished_at": "TIMESTAMPTZ",
                "range_start": "DATE",
                "range_end": "DATE",
                "rows_loaded": "INTEGER",
                "status": "VARCHAR",
                "error": "VARCHAR",
            },
        ),
        TableSpec(
            name="ingested_files",
            domain="meta",
            description="Content hashes of files already ingested, to make file drops idempotent.",
            primary_key=("file_hash",),
            columns={
                "file_hash": "VARCHAR NOT NULL",
                "path": "VARCHAR",
                "connector": "VARCHAR",
                "rows_loaded": "INTEGER",
                "ingested_at": "TIMESTAMP DEFAULT current_timestamp",
            },
        ),
        TableSpec(
            name="oauth_tokens",
            domain="meta",
            description="OAuth2 tokens for API connectors (stored locally only).",
            primary_key=("provider",),
            columns={
                "provider": "VARCHAR NOT NULL",
                "access_token": "VARCHAR",
                "refresh_token": "VARCHAR",
                "expires_at": "TIMESTAMPTZ",
                "scope": "VARCHAR",
                "updated_at": "TIMESTAMPTZ",
            },
        ),
        TableSpec(
            name="app_settings",
            domain="meta",
            description="Settings saved from the Connections page (secrets live in the OS keychain).",
            primary_key=("key",),
            columns={"key": "VARCHAR NOT NULL", "value": "VARCHAR", "updated_at": "TIMESTAMPTZ"},
        ),
    )
}

DOMAIN_TABLES: list[str] = [name for name, spec in TABLES.items() if spec.domain != "meta"]


class Database:
    """Thin manager around a local DuckDB file.

    Example:
        >>> db = Database()
        >>> db.init_schema()
        >>> with db.connect() as con:
        ...     con.sql("SELECT count(*) FROM weather_daily").fetchone()
    """

    def __init__(self, path: Path | str | None = None, settings: Settings | None = None) -> None:
        """Create a manager for ``path`` (defaults to ``settings.duckdb_path``)."""
        self.settings = settings or get_settings()
        resolved = Path(path) if path is not None else self.settings.duckdb_path
        assert resolved is not None
        self.path: Path = resolved
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ connections
    @contextmanager
    def connect(self, read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
        """Yield a short-lived DuckDB connection that is always closed afterwards.

        Args:
            read_only: Open in read-only mode (only valid once the file exists).
        """
        con = duckdb.connect(str(self.path), read_only=read_only and self.path.exists())
        try:
            con.execute(f"SET TimeZone = '{self.settings.timezone}'")
            yield con
        finally:
            con.close()

    def init_schema(self) -> None:
        """Create all tables if they do not already exist."""
        with self.connect() as con:
            for spec in TABLES.values():
                con.execute(spec.ddl())
                # Additive migration: columns added to a TableSpec after the table was created.
                existing = {r[0] for r in con.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_name = ?", [spec.name]).fetchall()}
                for column, dtype in spec.columns.items():
                    if column not in existing:
                        plain_type = dtype.split()[0]
                        con.execute(f"ALTER TABLE {spec.name} ADD COLUMN {column} {plain_type}")
        logger.info("Initialised DuckDB schema at %s (%d tables)", self.path, len(TABLES))

    # ------------------------------------------------------------------ writes
    def upsert(
        self,
        table: str,
        frame: pd.DataFrame,
        *,
        overwrite_nulls: bool = False,
    ) -> int:
        """Insert or update rows of ``frame`` into ``table`` keyed on its primary key.

        Args:
            table: Target table name (must exist in :data:`TABLES`).
            frame: Data to write. Unknown columns are dropped with a warning.
            overwrite_nulls: When ``True``, incoming NULLs overwrite stored values.
                When ``False`` (default), stored values are kept where the incoming
                value is NULL, allowing partial updates from multiple sources.

        Returns:
            Number of rows written.

        Raises:
            KeyError: If ``table`` is unknown.
            ValueError: If primary-key columns are missing or contain nulls.
        """
        spec = TABLES[table]
        if frame is None or frame.empty:
            return 0

        unknown = [c for c in frame.columns if c not in spec.columns]
        if unknown:
            logger.warning("Dropping unknown columns for %s: %s", table, unknown)
        cols = [c for c in spec.columns if c in frame.columns and c != "ingested_at"]
        missing_keys = [k for k in spec.primary_key if k not in cols]
        if missing_keys:
            raise ValueError(f"{table}: missing primary-key columns {missing_keys}")

        staged = frame[cols].copy()
        if "date" in staged.columns:
            staged["date"] = _to_date_series(staged["date"])
        if staged[list(spec.primary_key)].isna().any(axis=None):
            raise ValueError(f"{table}: primary-key columns contain nulls")
        staged = staged.drop_duplicates(subset=list(spec.primary_key), keep="last")
        staged["ingested_at"] = datetime.now(UTC).replace(tzinfo=None)
        cols = [*cols, "ingested_at"]

        updates = [c for c in cols if c not in spec.primary_key]
        if overwrite_nulls:
            set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in updates)
        else:
            set_clause = ", ".join(f"{c} = COALESCE(EXCLUDED.{c}, {table}.{c})" for c in updates)
        col_list = ", ".join(cols)
        conflict = ", ".join(spec.primary_key)
        sql = (
            f"INSERT INTO {table} ({col_list}) SELECT {col_list} FROM _staging "
            f"ON CONFLICT ({conflict}) DO UPDATE SET {set_clause}"
        )

        with self.connect() as con:
            con.register("_staging", staged)
            try:
                con.execute(sql)
            finally:
                con.unregister("_staging")
        logger.debug("Upserted %d rows into %s", len(staged), table)
        return len(staged)

    def delete_range(self, table: str, start: date, end: date) -> None:
        """Delete rows of a date-keyed table within ``[start, end]`` (inclusive)."""
        spec = TABLES[table]
        if "date" not in spec.columns:
            raise ValueError(f"{table} is not date-keyed")
        with self.connect() as con:
            con.execute(f"DELETE FROM {table} WHERE date BETWEEN ? AND ?", [start, end])

    # ------------------------------------------------------------------- reads
    def query_df(self, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        """Run ``sql`` and return the result as a pandas DataFrame."""
        with self.connect(read_only=True) as con:
            return con.execute(sql, list(params or [])).df()

    def read_table(self, table: str, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        """Read a domain table, optionally filtered to an inclusive date range."""
        spec = TABLES[table]
        clauses: list[str] = []
        params: list[Any] = []
        if start is not None and "date" in spec.columns:
            clauses.append("date >= ?")
            params.append(start)
        if end is not None and "date" in spec.columns:
            clauses.append("date <= ?")
            params.append(end)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        order = " ORDER BY date" if "date" in spec.columns else ""
        return self.query_df(f"SELECT * FROM {table}{where}{order}", params)

    def table_coverage(self) -> pd.DataFrame:
        """Return row counts and date coverage for every domain table."""
        rows: list[dict[str, Any]] = []
        with self.connect(read_only=True) as con:
            for name in DOMAIN_TABLES:
                n, first, last = con.execute(f"SELECT count(*), min(date), max(date) FROM {name}").fetchone()
                rows.append(
                    {"table": name, "domain": TABLES[name].domain, "rows": n, "first_date": first, "last_date": last}
                )
        return pd.DataFrame(rows)

    # --------------------------------------------------------------- run audit
    def start_run(self, connector: str, start: date | None, end: date | None) -> str:
        """Record the start of a connector run and return its id."""
        run_id = uuid.uuid4().hex
        with self.connect() as con:
            con.execute(
                "INSERT INTO ingestion_runs (run_id, connector, started_at, range_start, range_end, status) "
                "VALUES (?, ?, ?, ?, ?, 'running')",
                [run_id, connector, datetime.now(UTC), start, end],
            )
        return run_id

    def finish_run(self, run_id: str, rows_loaded: int, status: str, error: str | None = None) -> None:
        """Mark a connector run as finished."""
        with self.connect() as con:
            con.execute(
                "UPDATE ingestion_runs SET finished_at = ?, rows_loaded = ?, status = ?, error = ? WHERE run_id = ?",
                [datetime.now(UTC), rows_loaded, status, error, run_id],
            )

    # ------------------------------------------------------------ file hashes
    def is_file_ingested(self, file_hash: str) -> bool:
        """Return ``True`` if a file with this content hash was already ingested."""
        with self.connect(read_only=True) as con:
            return con.execute("SELECT 1 FROM ingested_files WHERE file_hash = ?", [file_hash]).fetchone() is not None

    def mark_file_ingested(self, file_hash: str, path: Path, connector: str, rows: int) -> None:
        """Record that a file was ingested."""
        with self.connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO ingested_files (file_hash, path, connector, rows_loaded, ingested_at) "
                "VALUES (?, ?, ?, ?, current_timestamp)",
                [file_hash, str(path), connector, rows],
            )

    # ------------------------------------------------------------ oauth tokens
    def get_token(self, provider: str) -> dict[str, Any] | None:
        """Return the stored OAuth token row for ``provider`` or ``None``."""
        with self.connect(read_only=True) as con:
            row = con.execute(
                "SELECT access_token, refresh_token, expires_at, scope FROM oauth_tokens WHERE provider = ?",
                [provider],
            ).fetchone()
        if row is None:
            return None
        return {"access_token": row[0], "refresh_token": row[1], "expires_at": row[2], "scope": row[3]}

    def save_token(
        self,
        provider: str,
        access_token: str,
        refresh_token: str | None,
        expires_at: datetime,
        scope: str | None,
    ) -> None:
        """Persist an OAuth token for ``provider``."""
        with self.connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO oauth_tokens (provider, access_token, refresh_token, expires_at, scope, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [provider, access_token, refresh_token, expires_at, scope, datetime.now(UTC)],
            )

    def set_meta(self, key: str, value: str | None) -> None:
        """Store (or with ``None`` delete) an internal key/value pair, e.g. a pending OAuth state."""
        with self.connect() as con:
            con.execute("DELETE FROM app_settings WHERE key = ?", [f"_meta:{key}"])
            if value is not None:
                con.execute("INSERT INTO app_settings (key, value, updated_at) VALUES (?, ?, ?)",
                            [f"_meta:{key}", value, datetime.now(UTC)])

    def get_meta(self, key: str) -> tuple[str, datetime] | None:
        """Return ``(value, updated_at)`` for an internal key, or ``None``."""
        with self.connect(read_only=True) as con:
            row = con.execute("SELECT value, updated_at FROM app_settings WHERE key = ?", [f"_meta:{key}"]).fetchone()
        return (row[0], row[1]) if row else None

    def delete_token(self, provider: str) -> None:
        """Forget the stored OAuth token for ``provider``."""
        with self.connect() as con:
            con.execute("DELETE FROM oauth_tokens WHERE provider = ?", [provider])


def _to_date_series(values: pd.Series) -> pd.Series:
    """Coerce a column of date-like values to python ``date`` objects."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return values.dt.date
    return pd.to_datetime(values).dt.date
