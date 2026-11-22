"""Core framework: configuration and local DuckDB storage."""

from core.config import Settings, get_settings
from core.database import TABLES, Database, TableSpec

__all__ = ["Settings", "get_settings", "Database", "TableSpec", "TABLES"]
