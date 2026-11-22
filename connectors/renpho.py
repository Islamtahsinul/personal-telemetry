"""Renpho smart-scale body composition connector.

Two sources, selected by ``RENPHO_SOURCE``:

* ``api`` - the unofficial `renpho-api <https://pypi.org/project/renpho-api/>`_ client
  (``pip install renpho-api``) using ``RENPHO_EMAIL`` / ``RENPHO_PASSWORD``.
  Gives the full metric set including visceral fat, BMI and metabolic age.
* ``health_connect`` - Renpho's Health Connect sync, read from the same exports
  as :mod:`connectors.android_s24` (weight, body fat, lean mass, BMR only).
* ``auto`` (default) - API when credentials are set and the client is installed,
  otherwise Health Connect.

Multiple weigh-ins on one day collapse to the last one; ``measurement_count``
records how many there were.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import pandas as pd

from connectors.android_s24 import aggregate_health_daily, load_health_connect_records
from connectors.base import BaseConnector, ConnectorError, Frames
from connectors.utils import to_local_dates

RENPHO_FIELD_MAP: dict[str, str] = {
    "weight": "weight_kg",
    "bmi": "bmi",
    "bodyfat": "body_fat_pct",
    "muscle": "muscle_mass_pct",
    "visfat": "visceral_fat",
    "subfat": "subcutaneous_fat_pct",
    "water": "body_water_pct",
    "bone": "bone_mass_pct",
    "protein": "protein_pct",
    "bmr": "bmr_kcal",
    "sinew": "lean_body_mass_kg",
    "bodyage": "body_age",
}
HEALTH_CONNECT_MAP: dict[str, str] = {
    "weight_kg": "weight_kg",
    "body_fat_pct": "body_fat_pct",
    "skeletal_muscle_mass_kg": "lean_body_mass_kg",
    "bmr_kcal": "bmr_kcal",
}
RenphoSource = Literal["api", "health_connect"]


@dataclass(slots=True)
class RenphoRaw:
    """Raw measurements and where they came from."""

    source: RenphoSource
    measurements: list[dict[str, Any]] | None = None
    health_records: pd.DataFrame | None = None


class RenphoConnector(BaseConnector[RenphoRaw]):
    """Pull Renpho scale weigh-ins (renpho-api client, Health Connect fallback)."""

    name = "renpho"
    description = "Renpho scale: weight, body fat %, muscle mass, visceral fat, BMI."
    tables = ("renpho_body",)

    def _api_available(self) -> bool:
        if not (self.settings.renpho_email and self.settings.renpho_password):
            return False
        try:
            import renpho  # noqa: F401
        except ImportError:
            return False
        return True

    def _resolve_source(self) -> RenphoSource:
        if self.settings.renpho_source == "api":
            return "api"
        if self.settings.renpho_source == "health_connect":
            return "health_connect"
        return "api" if self._api_available() else "health_connect"

    def missing_config(self) -> list[str]:
        """API mode needs credentials + client; Health Connect mode needs the export folder."""
        source = self._resolve_source()
        if source == "api":
            missing = []
            if not self.settings.renpho_email:
                missing.append("RENPHO_EMAIL")
            if not self.settings.renpho_password:
                missing.append("RENPHO_PASSWORD")
            if not missing and not self._api_available():
                missing.append("renpho-api package (pip install renpho-api)")
            return missing
        directory = self.settings.android_export_dir
        return [] if directory and directory.exists() else ["RENPHO_EMAIL/RENPHO_PASSWORD or ANDROID_EXPORT_DIR"]

    def extract(self, start: date, end: date) -> RenphoRaw:
        """Fetch all measurements (the API has no server-side date filter)."""
        source = self._resolve_source()
        if source == "health_connect":
            assert self.settings.android_export_dir is not None
            return RenphoRaw(
                source, health_records=load_health_connect_records(self.settings.android_export_dir, self.settings.tz)
            )

        from renpho import RenphoAPIError

        assert self.settings.renpho_email and self.settings.renpho_password
        client, _ = renpho_login(
            self.settings.renpho_email,
            self.settings.renpho_password.get_secret_value(),
            area_code_candidates(self.settings.renpho_area_code, self.settings.timezone),
        )
        try:
            measurements = client.get_all_measurements()
        except RenphoAPIError as exc:
            raise ConnectorError(f"Renpho API error: {exc}") from exc
        self.log.debug("Fetched %d Renpho measurements", len(measurements))
        return RenphoRaw(source, measurements=measurements)

    def transform(self, raw: RenphoRaw) -> Frames:
        """Collapse weigh-ins to one ``renpho_body`` row per day."""
        if raw.source == "api":
            daily = renpho_measurements_to_daily(raw.measurements or [], self.settings.tz)
        else:
            daily = self._health_connect_daily(raw.health_records)
        return {"renpho_body": daily if not daily.empty else self.empty_frame("renpho_body")}

    def _health_connect_daily(self, records: pd.DataFrame | None) -> pd.DataFrame:
        if records is None or records.empty:
            return pd.DataFrame()
        body = records.loc[records["metric"].isin(HEALTH_CONNECT_MAP)]
        daily = aggregate_health_daily(body, source_filter="renpho")
        if len(daily.columns) <= 1:
            self.log.info("No Renpho-tagged Health Connect records; using all body-composition records")
            daily = aggregate_health_daily(body)
        if len(daily.columns) <= 1:
            return pd.DataFrame()
        counts = body.loc[body["metric"] == "weight_kg"].groupby("date").size().rename("measurement_count")
        daily = daily.rename(columns=HEALTH_CONNECT_MAP).merge(counts, on="date", how="left")
        daily["source"] = "health_connect"
        return daily


def area_code_candidates(preferred: str | None, timezone: str) -> list[str]:
    """Area codes to try at login, most likely first (``""`` is what the current app sends)."""
    regional = ["GB", "US"] if timezone.startswith("Europe/") else ["US", "GB"]
    ordered = [preferred.upper()] if preferred else []
    return list(dict.fromkeys([*ordered, "", *regional]))


# Renpho's server (Oct 2026) rejects logins from renpho-api 0.2 with -114/-104 unless the
# request carries current app headers and payload values. See
# https://github.com/danvaneijck/renpho-api/issues/12 - remove once fixed upstream.
RENPHO_APP_VERSION = "7.0.0"


def renpho_login(email: str, password: str, area_codes: list[str]) -> tuple[Any, str]:
    """Sign in, trying each area code; return ``(logged-in RenphoClient, area code used)``.

    Mirrors ``RenphoClient.login`` (renpho-api 0.2) with a configurable ``areaCode``.

    Raises:
        ConnectorError: With a readable message if every attempt fails.
    """
    try:
        import renpho.client as renpho_client
        from renpho import RenphoAPIError, RenphoClient
        from renpho.client import _check_response
        from renpho.constants import ENDPOINTS, PLATFORM
        from renpho.crypto import decrypt_response, encrypt_request
    except ImportError as exc:
        raise ConnectorError("The renpho-api package is not installed (pip install renpho-api)") from exc

    # Later authenticated calls read this module-level constant for their appVersion header.
    renpho_client.APP_VERSION = RENPHO_APP_VERSION
    last_error: Exception | None = None
    for area in area_codes:
        client = RenphoClient(email, password)
        client._session.headers.update(
            {"appVersion": RENPHO_APP_VERSION, "platform": PLATFORM, "User-Agent": "okhttp/4.9.3"}
        )
        payload = {
            "questionnaire": {},
            "login": {"password": password, "areaCode": area, "appRevision": RENPHO_APP_VERSION,
                      "cellphoneType": "MCP-Server", "systemType": "11", "email": email, "platform": PLATFORM},
            "bindingList": {"deviceTypes": ["2"]},
        }
        try:
            result = client._post(ENDPOINTS["login"], encrypt_request(payload), auth=False)
            _check_response(result, "Login")
            info = decrypt_response(result["data"]).get("login", {})
        except RenphoAPIError as exc:
            last_error = exc
            continue
        except Exception as exc:  # noqa: BLE001 - network errors from requests
            raise ConnectorError(f"Could not reach Renpho: {exc}") from exc
        if info.get("token"):
            client.token, client.user_id, client.user_info = info["token"], info.get("id"), info
            return client, area
    text = str(last_error)
    if "code=104" in text:
        raise ConnectorError("Renpho says the password is wrong - retype it (no spaces).") from last_error
    if "code=106" in text:
        raise ConnectorError(
            "Renpho has no account with that e-mail. Use the exact e-mail shown in the Renpho app "
            "(Apple sign-ups may use a @privaterelay.appleid.com address)."
        ) from last_error
    if "-114" in text:
        raise ConnectorError(
            "Renpho rejected the sign-in request format (its app API changed again). "
            "Check https://github.com/danvaneijck/renpho-api/issues for an update."
        ) from last_error
    raise ConnectorError(f"Renpho sign-in failed: {last_error}") from last_error


def verify_renpho_login(email: str, password: str, timezone: str = "Europe/London",
                        preferred_area: str | None = None) -> str:
    """Check Renpho credentials and return the area code that worked."""
    _, area = renpho_login(email, password, area_code_candidates(preferred_area, timezone))
    return area


def renpho_measurements_to_daily(measurements: list[dict[str, Any]], tz: Any) -> pd.DataFrame:
    """Convert ``RenphoClient.get_all_measurements()`` output to daily rows.

    Args:
        measurements: Measurement dicts (keys per the renpho-api docs, e.g. ``weight``, ``bodyfat``).
        tz: Local timezone used to bucket ``timeStamp`` (epoch seconds or ms) into days.
    """
    if not measurements:
        return pd.DataFrame()
    frame = pd.DataFrame(measurements)
    ts_col = "timeStamp" if "timeStamp" in frame else "time_stamp" if "time_stamp" in frame else None
    if ts_col is None:
        raise ConnectorError("Renpho measurements have no timeStamp field")
    frame["_ts"] = pd.to_numeric(frame[ts_col], errors="coerce")
    frame = frame.dropna(subset=["_ts"])
    frame["date"] = to_local_dates(frame["_ts"], tz)

    present = {src: dst for src, dst in RENPHO_FIELD_MAP.items() if src in frame}
    out = frame[["date", "_ts", *present]].rename(columns=present)
    for col in present.values():
        out[col] = pd.to_numeric(out[col], errors="coerce")
        # The API reports unmeasured metrics as 0.
        out.loc[out[col] <= 0, col] = pd.NA
    out = out.dropna(subset=["weight_kg"]) if "weight_kg" in out else out

    out = out.sort_values("_ts")
    counts = out.groupby("date").size().rename("measurement_count")
    daily = out.drop_duplicates("date", keep="last").drop(columns="_ts").merge(counts, on="date")
    if {"weight_kg", "muscle_mass_pct"} <= set(daily.columns):
        daily["muscle_mass_kg"] = (daily["weight_kg"] * daily["muscle_mass_pct"] / 100).round(2)
    daily["source"] = "renpho_api"
    return daily.reset_index(drop=True)
