"""Abstract connector contract: ``extract`` -> ``transform`` -> ``load``.

A connector pulls raw data from one source (API, local service or files), shapes it
into DataFrames whose columns match the :data:`core.database.TABLES` schema, and
upserts them. :meth:`BaseConnector.run` orchestrates the three hooks, records an
audit row in ``ingestion_runs`` and never raises for expected failures, so that one
broken source cannot stop a full sync.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import ClassVar, Generic, Literal, TypeVar

import pandas as pd

from core.config import Settings, get_settings
from core.database import TABLES, Database

logger = logging.getLogger(__name__)

RawT = TypeVar("RawT")
Frames = dict[str, pd.DataFrame]
RunStatus = Literal["success", "skipped", "failed"]


class ConnectorError(RuntimeError):
    """Raised by connectors for expected, user-actionable failures."""


class ConnectorNotConfiguredError(ConnectorError):
    """Raised when required settings for a connector are missing."""


@dataclass(slots=True)
class ConnectorResult:
    """Outcome of a single connector run."""

    connector: str
    status: RunStatus
    rows_by_table: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    duration_seconds: float = 0.0

    @property
    def rows_loaded(self) -> int:
        """Total rows written across all tables."""
        return sum(self.rows_by_table.values())


class BaseConnector(ABC, Generic[RawT]):
    """Base class for every data source.

    Subclasses must set :attr:`name` and :attr:`tables` and implement
    :meth:`extract` and :meth:`transform`. Override :meth:`missing_config` to declare
    required settings.

    Attributes:
        name: Registry key used by the CLI and UI.
        description: One-line human description.
        tables: Tables this connector writes to.
        filter_to_range: When ``True``, rows outside the requested range are dropped
            before loading (API connectors). File connectors set this to ``False``.
        overwrite_nulls: Passed to :meth:`Database.upsert`.
    """

    name: ClassVar[str]
    description: ClassVar[str] = ""
    tables: ClassVar[tuple[str, ...]]
    filter_to_range: ClassVar[bool] = True
    overwrite_nulls: ClassVar[bool] = False

    def __init__(self, settings: Settings | None = None, db: Database | None = None) -> None:
        """Bind the connector to settings and a database (defaults to the global ones)."""
        self.settings = settings or get_settings()
        self.db = db or Database(settings=self.settings)
        self.log = logging.getLogger(f"connectors.{self.name}")

    # ----------------------------------------------------------- configuration
    def missing_config(self) -> list[str]:
        """Return names of required settings that are not set (empty = ready)."""
        return []

    def is_configured(self) -> bool:
        """Return ``True`` when all required settings are present."""
        return not self.missing_config()

    # ------------------------------------------------------------------- hooks
    @abstractmethod
    def extract(self, start: date, end: date) -> RawT:
        """Fetch raw data covering ``[start, end]`` (inclusive) from the source."""

    @abstractmethod
    def transform(self, raw: RawT) -> Frames:
        """Convert raw data into ``{table_name: DataFrame}`` matching the schema."""

    def load(self, frames: Frames) -> dict[str, int]:
        """Upsert each frame into its table and return rows written per table."""
        written: dict[str, int] = {}
        for table, frame in frames.items():
            if table not in self.tables:
                raise ConnectorError(f"{self.name} is not allowed to write to {table}")
            written[table] = self.db.upsert(table, frame, overwrite_nulls=self.overwrite_nulls)
        return written

    # ----------------------------------------------------------- orchestration
    def default_range(self) -> tuple[date, date]:
        """Default sync window: the last ``default_lookback_days`` up to today."""
        today = datetime.now(self.settings.tz).date()
        return today - timedelta(days=self.settings.default_lookback_days - 1), today

    def run(self, start: date | None = None, end: date | None = None) -> ConnectorResult:
        """Run extract -> transform -> load for ``[start, end]`` and audit the run.

        Missing configuration yields a ``skipped`` result; any exception yields a
        ``failed`` result with the error message (and is logged with traceback).
        """
        default_start, default_end = self.default_range()
        start = start or default_start
        end = end or default_end
        if start > end:
            raise ValueError("start must be on or before end")

        if missing := self.missing_config():
            msg = f"missing settings: {', '.join(missing)}"
            self.log.info("Skipping %s (%s)", self.name, msg)
            return ConnectorResult(self.name, "skipped", error=msg)

        t0 = time.perf_counter()
        run_id = self.db.start_run(self.name, start, end)
        try:
            raw = self.extract(start, end)
            frames = self.transform(raw)
            if self.filter_to_range:
                frames = {t: _clip(f, start, end) for t, f in frames.items()}
            written = self.load(frames)
        except Exception as exc:  # noqa: BLE001 - one source must not break the sync
            self.log.exception("Connector %s failed", self.name)
            self.db.finish_run(run_id, 0, "failed", f"{type(exc).__name__}: {exc}")
            return ConnectorResult(
                self.name, "failed", error=f"{type(exc).__name__}: {exc}", duration_seconds=time.perf_counter() - t0
            )

        result = ConnectorResult(self.name, "success", written, duration_seconds=time.perf_counter() - t0)
        self.db.finish_run(run_id, result.rows_loaded, "success")
        self.log.info("%s loaded %d rows %s in %.1fs", self.name, result.rows_loaded, written, result.duration_seconds)
        return result

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def empty_frame(table: str) -> pd.DataFrame:
        """Return an empty DataFrame with the columns of ``table``."""
        return pd.DataFrame(columns=[c for c in TABLES[table].columns if c != "ingested_at"])


def _clip(frame: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    if frame.empty or "date" not in frame.columns:
        return frame
    dates = pd.to_datetime(frame["date"]).dt.date
    return frame.loc[(dates >= start) & (dates <= end)]
