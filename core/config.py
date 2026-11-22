"""Application configuration loaded from environment variables and a local ``.env`` file.

All settings are optional at import time so that the framework can start with only
the connectors you actually use configured. Each connector checks its own
requirements via ``BaseConnector.is_configured()``.

List / dict fields accept JSON in the environment, e.g.::

    AW_CODING_APP_PATTERNS='["code", "pycharm", "terminal"]'
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DEFAULT_MONZO_CATEGORY_MAP: dict[str, str] = {
    "groceries": "groceries",
    "eating out": "eating_out",
    "eating_out": "eating_out",
    "transport": "transport",
    "shopping": "shopping",
    "entertainment": "entertainment",
    "bills": "bills",
    "personal care": "personal_care",
    "personal_care": "personal_care",
    "holidays": "travel",
    "travel": "travel",
    "general": "general",
    "expenses": "general",
    "family": "family",
    "gifts": "gifts",
    "charity": "charity",
    "cash": "cash",
    "finances": "finances",
}

DEFAULT_EXCLUDED_MONZO_CATEGORIES: list[str] = ["transfers", "savings", "income"]


class Settings(BaseSettings):
    """Typed, validated application settings."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ general
    data_dir: Path = Field(default=PROJECT_ROOT / "data", description="Root for all local data.")
    duckdb_path: Path | None = Field(default=None, description="Defaults to <data_dir>/telemetry.duckdb.")
    timezone: str = Field(default="Europe/London", description="IANA timezone used to bucket days.")
    log_level: str = "INFO"
    default_lookback_days: int = Field(default=30, ge=1, le=3650)
    http_timeout_seconds: float = Field(default=30.0, gt=0)

    # -------------------------------------------------------------------- WHOOP
    whoop_client_id: str | None = None
    whoop_client_secret: SecretStr | None = None
    # The dashboard's own Connections page receives the OAuth callback - no extra port needed.
    whoop_redirect_uri: str = "http://localhost:8501/connections"
    whoop_scopes: list[str] = Field(
        default=[
            "offline",
            "read:recovery",
            "read:cycles",
            "read:sleep",
            "read:workout",
            "read:profile",
            "read:body_measurement",
        ]
    )

    # -------------------------------------------------------- Monzo via Sheets
    google_service_account_file: Path | None = None
    monzo_sheet_id: str | None = None
    monzo_import_folder: Path | None = Field(
        default=None, description="Folder holding Monzo CSV/.xlsx exports, e.g. a Google Drive for desktop folder."
    )
    monzo_worksheet_name: str | None = Field(
        default=None, description="Worksheet/tab name; first sheet is used when unset."
    )
    monzo_category_map: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_MONZO_CATEGORY_MAP))
    monzo_excluded_categories: list[str] = Field(
        default_factory=lambda: list(DEFAULT_EXCLUDED_MONZO_CATEGORIES)
    )

    # ------------------------------------------------------------------ Weather
    weather_latitude: float | None = Field(default=None, ge=-90, le=90)
    weather_longitude: float | None = Field(default=None, ge=-180, le=180)

    # ----------------------------------------------------------- Android / S24
    android_export_dir: Path | None = Field(default=None, description="Defaults to <data_dir>/inbox/android.")
    phone_category_rules: dict[str, list[str]] = Field(
        default={
            "social": [r"instagram", r"whatsapp", r"facebook", r"twitter", r"\bx\b", r"reddit", r"tiktok",
                       r"snapchat", r"telegram", r"messenger", r"threads", r"linkedin", r"discord"],
            "entertainment": [r"youtube", r"netflix", r"spotify", r"twitch", r"disney", r"primevideo",
                              r"iplayer", r"game", r"music"],
            "productivity": [r"gmail", r"outlook", r"calendar", r"docs", r"sheets", r"notion", r"keep",
                             r"drive", r"teams", r"slack", r"todoist", r"obsidian"],
            "browsing": [r"chrome", r"firefox", r"sbrowser", r"internet", r"brave", r"edge"],
            "health": [r"whoop", r"health", r"myfitnesspal", r"renpho", r"strava", r"fitness"],
            "finance": [r"monzo", r"bank", r"revolut", r"paypal"],
        }
    )

    # ------------------------------------------------------------------ Renpho
    renpho_email: str | None = None
    renpho_password: SecretStr | None = None
    renpho_area_code: str | None = Field(default=None, description="Renpho account region, e.g. GB or US.")
    renpho_source: Literal["auto", "api", "health_connect"] = Field(
        default="auto", description="'auto' tries the renpho-api client, then Health Connect exports."
    )

    # ------------------------------------------------------------ MyFitnessPal
    mfp_source: Literal["auto", "export", "scraper", "health_connect"] = Field(
        default="auto", description="'auto' = export files, then Health Connect, then scraper."
    )
    mfp_export_dir: Path | None = Field(default=None, description="Defaults to <data_dir>/inbox/myfitnesspal.")
    mfp_water_unit: Literal["ml", "cups", "fl_oz"] = Field(
        default="ml", description="Unit MyFitnessPal reports water in (account preference)."
    )

    # ------------------------------------------------------------ ActivityWatch
    activitywatch_url: str = "http://localhost:5600"
    aw_coding_app_patterns: list[str] = Field(
        default=[
            r"code",
            r"cursor",
            r"pycharm",
            r"intellij",
            r"idea",
            r"webstorm",
            r"sublime",
            r"vim",
            r"emacs",
            r"terminal",
            r"iterm",
            r"windowsterminal",
            r"wezterm",
            r"alacritty",
            r"kitty",
        ]
    )
    aw_work_app_patterns: list[str] = Field(
        default=[r"outlook", r"teams", r"slack", r"zoom", r"excel", r"word", r"powerpnt", r"notion", r"obsidian"]
    )
    aw_category_rules: dict[str, list[str]] = Field(
        default={
            "coding": [r"code", r"cursor", r"pycharm", r"intellij", r"vim", r"terminal", r"iterm"],
            "communication": [r"slack", r"teams", r"outlook", r"zoom", r"discord", r"mail"],
            "productivity": [r"excel", r"word", r"powerpnt", r"notion", r"obsidian", r"onenote"],
            "browsing": [r"chrome", r"firefox", r"edge", r"safari", r"brave", r"arc"],
            "entertainment": [r"spotify", r"vlc", r"netflix", r"steam", r"youtube"],
        }
    )

    # -------------------------------------------------------------------- LLM
    llm_provider: Literal["anthropic", "ollama", "none"] = "ollama"
    llm_timeout_seconds: float = Field(default=600.0, gt=0)
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    anthropic_max_tokens: int = Field(default=16000, ge=256)
    anthropic_enable_fallbacks: bool = Field(
        default=True, description="Server-side refusal fallback (Claude API only)."
    )
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_temperature: float = Field(default=0.3, ge=0, le=2)

    # ------------------------------------------------------------- validators
    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"Unknown IANA timezone: {value!r}") from exc
        return value

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in logging.getLevelNamesMapping():
            raise ValueError(f"Unknown log level: {value}")
        return level

    @field_validator("monzo_category_map")
    @classmethod
    def _normalise_category_keys(cls, value: dict[str, str]) -> dict[str, str]:
        return {k.strip().lower(): v.strip().lower() for k, v in value.items()}

    @model_validator(mode="after")
    def _derive_paths(self) -> Settings:
        self.data_dir = self.data_dir.expanduser().resolve()
        if self.duckdb_path is None:
            self.duckdb_path = self.data_dir / "telemetry.duckdb"
        if self.android_export_dir is None:
            self.android_export_dir = self.data_dir / "inbox" / "android"
        if self.mfp_export_dir is None:
            self.mfp_export_dir = self.data_dir / "inbox" / "myfitnesspal"
        return self

    # ---------------------------------------------------------------- helpers
    @property
    def tz(self) -> ZoneInfo:
        """The configured timezone as a ``ZoneInfo`` object."""
        return ZoneInfo(self.timezone)

    @property
    def inbox_dir(self) -> Path:
        """Folder watched by the generic file ingester."""
        return self.data_dir / "inbox"

    @property
    def processed_dir(self) -> Path:
        """Folder that ingested files are moved into."""
        return self.data_dir / "processed"

    @property
    def monzo_import_dir(self) -> Path:
        """Folder for Monzo exports: ``MONZO_IMPORT_FOLDER`` (e.g. a synced Drive folder) or ``inbox/monzo``."""
        return self.monzo_import_folder or self.inbox_dir / "monzo"

    def ensure_directories(self) -> None:
        """Create the local data folders if they do not already exist."""
        for path in (
            self.data_dir,
            self.inbox_dir,
            self.processed_dir,
            self.android_export_dir,
            self.mfp_export_dir,
            self.monzo_import_dir,
        ):
            assert path is not None
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings: ``.env``/environment, overlaid with values saved in the UI."""
    from core.settings_store import SettingsStore

    base = Settings()
    assert base.duckdb_path is not None
    overrides = SettingsStore(base.duckdb_path).load()
    settings = base
    if overrides:
        try:
            settings = Settings(**{**_explicit_env_paths(base), **overrides})
        except ValueError as exc:  # pydantic ValidationError subclasses ValueError
            logging.getLogger(__name__).warning("Ignoring invalid saved settings: %s", exc)
    settings.ensure_directories()
    return settings


def reload_settings() -> Settings:
    """Drop the cached settings (e.g. after saving in the UI) and load them again."""
    get_settings.cache_clear()
    return get_settings()


def _explicit_env_paths(base: Settings) -> dict[str, Any]:
    # Keep the database location stable: saved settings live inside it.
    return {"data_dir": base.data_dir, "duckdb_path": base.duckdb_path}


def configure_logging(settings: Settings | None = None) -> None:
    """Configure root logging once using the configured level."""
    level = (settings or get_settings()).log_level
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
