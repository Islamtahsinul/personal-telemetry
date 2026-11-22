"""Persistent settings saved from the UI (no ``.env`` editing required).

Plain settings live in the ``app_settings`` table of the local DuckDB file.
Secrets (passwords, client secrets, API keys) go to the operating system's
credential store via :mod:`keyring` - Windows Credential Manager, macOS Keychain
or the Linux Secret Service. If no keyring backend is available (e.g. a headless
Linux box) secrets fall back to the local database and are flagged as such.

Values saved here take precedence over environment variables and ``.env``.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

logger = logging.getLogger(__name__)

KEYRING_SERVICE = "personal-telemetry"
SECRET_FIELDS: frozenset[str] = frozenset({"whoop_client_secret", "renpho_password", "anthropic_api_key"})
_KEYRING_MARKER = "__keyring__"

DDL = """
CREATE TABLE IF NOT EXISTS app_settings (
    key VARCHAR PRIMARY KEY,
    value VARCHAR,
    updated_at TIMESTAMPTZ
)
"""


def _keyring() -> Any | None:
    """Return the keyring module if a usable backend exists, else ``None``."""
    try:
        import keyring
        from keyring.backends import fail

        if isinstance(keyring.get_keyring(), fail.Keyring):
            return None
        return keyring
    except Exception:  # noqa: BLE001 - any import/backend problem means "no keyring"
        return None


class SettingsStore:
    """Read and write UI-managed settings for one database file."""

    def __init__(self, db_path: Path, use_keyring: bool = True) -> None:
        """Args: db_path: DuckDB file; use_keyring: store secrets in the OS credential store."""
        self.db_path = db_path
        self._keyring = _keyring() if use_keyring else None

    @property
    def secrets_in_keychain(self) -> bool:
        """Whether secrets are stored in the OS credential store."""
        return self._keyring is not None

    # ------------------------------------------------------------------- read
    def load(self) -> dict[str, Any]:
        """Return all saved settings (secrets resolved), or ``{}`` if none."""
        if not self.db_path.exists():
            return {}
        try:
            con = duckdb.connect(str(self.db_path), read_only=True)
        except duckdb.Error as exc:
            logger.warning("Could not read saved settings: %s", exc)
            return {}
        try:
            exists = con.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = 'app_settings'"
            ).fetchone()[0]
            rows = (con.execute("SELECT key, value FROM app_settings WHERE NOT starts_with(key, '_meta:')").fetchall()
                    if exists else [])
        finally:
            con.close()

        values: dict[str, Any] = {}
        for key, raw in rows:
            if raw == _KEYRING_MARKER:
                secret = self._keyring.get_password(KEYRING_SERVICE, key) if self._keyring else None
                if secret is not None:
                    values[key] = secret
                continue
            try:
                values[key] = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                values[key] = raw
        return values

    # ------------------------------------------------------------------ write
    def save(self, values: dict[str, Any]) -> None:
        """Persist settings. ``None`` or empty-string values delete the key."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(self.db_path))
        try:
            con.execute(DDL)
            for key, value in values.items():
                if value is None or value == "":
                    self._delete(con, key)
                    continue
                if key in SECRET_FIELDS and self._keyring is not None:
                    self._keyring.set_password(KEYRING_SERVICE, key, str(value))
                    stored = _KEYRING_MARKER
                else:
                    stored = json.dumps(value, default=str)
                con.execute(
                    "INSERT OR REPLACE INTO app_settings (key, value, updated_at) VALUES (?, ?, ?)",
                    [key, stored, datetime.now(UTC)],
                )
        finally:
            con.close()

    def delete(self, *keys: str) -> None:
        """Remove saved settings (and their keychain entries)."""
        if not self.db_path.exists():
            return
        con = duckdb.connect(str(self.db_path))
        try:
            con.execute(DDL)
            for key in keys:
                self._delete(con, key)
        finally:
            con.close()

    def _delete(self, con: duckdb.DuckDBPyConnection, key: str) -> None:
        con.execute("DELETE FROM app_settings WHERE key = ?", [key])
        if key in SECRET_FIELDS and self._keyring is not None:
            try:
                self._keyring.delete_password(KEYRING_SERVICE, key)
            except Exception:  # noqa: BLE001 - nothing stored is fine
                pass
