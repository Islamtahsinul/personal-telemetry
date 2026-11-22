"""Data-source connectors and a registry for the CLI and UI."""

from __future__ import annotations

from datetime import date

from connectors.activitywatch import ActivityWatchConnector
from connectors.android_s24 import AndroidS24Connector
from connectors.base import BaseConnector, ConnectorError, ConnectorNotConfiguredError, ConnectorResult
from connectors.file_parser import FileParserConnector
from connectors.monzo_sheets import MonzoSheetsConnector
from connectors.myfitnesspal import MyFitnessPalConnector
from connectors.renpho import RenphoConnector
from connectors.weather import WeatherConnector
from connectors.whoop import WhoopConnector
from core.config import Settings
from core.database import Database

# Order matters: the Android parser and file drops run before connectors that
# read the same folders as a fallback.
CONNECTORS: dict[str, type[BaseConnector]] = {
    cls.name: cls
    for cls in (
        FileParserConnector,
        AndroidS24Connector,
        WhoopConnector,
        RenphoConnector,
        MyFitnessPalConnector,
        MonzoSheetsConnector,
        WeatherConnector,
        ActivityWatchConnector,
    )
}


def get_connector(name: str, settings: Settings | None = None, db: Database | None = None) -> BaseConnector:
    """Instantiate a registered connector by name.

    Raises:
        KeyError: If no connector is registered under ``name``.
    """
    try:
        cls = CONNECTORS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown connector {name!r}; choose from {sorted(CONNECTORS)}") from exc
    return cls(settings=settings, db=db)


def run_all(
    start: date | None = None,
    end: date | None = None,
    names: list[str] | None = None,
    settings: Settings | None = None,
    db: Database | None = None,
) -> list[ConnectorResult]:
    """Run the selected (default: all) connectors sequentially and return their results."""
    selected = names or list(CONNECTORS)
    return [get_connector(n, settings, db).run(start, end) for n in selected]


__all__ = [
    "CONNECTORS",
    "BaseConnector",
    "ConnectorError",
    "ConnectorNotConfiguredError",
    "ConnectorResult",
    "get_connector",
    "run_all",
    "ActivityWatchConnector",
    "AndroidS24Connector",
    "FileParserConnector",
    "MonzoSheetsConnector",
    "MyFitnessPalConnector",
    "RenphoConnector",
    "WeatherConnector",
    "WhoopConnector",
]
