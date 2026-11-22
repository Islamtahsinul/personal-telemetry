"""Open-Meteo daily weather connector (free, no API key).

Uses the Historical Weather API for days older than :data:`ARCHIVE_LAG_DAYS`
(reanalysis data arrives with a short delay) and the Forecast API, which also
serves the recent past, for the remainder. Daily means for pressure, humidity,
cloud cover and temperature are computed from hourly values so that both APIs
yield identical columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import httpx
import pandas as pd

from connectors.base import BaseConnector, ConnectorError, Frames

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_LAG_DAYS = 6

DAILY_VARS: dict[str, str] = {
    "temperature_2m_max": "temp_max_c",
    "temperature_2m_min": "temp_min_c",
    "precipitation_sum": "precipitation_mm",
    "rain_sum": "rain_mm",
    "precipitation_hours": "precipitation_hours",
    "daylight_duration": "daylight_hours",
    "sunshine_duration": "sunshine_hours",
    "wind_speed_10m_max": "wind_speed_max_kmh",
}
HOURLY_MEANS: dict[str, str] = {
    "temperature_2m": "temp_mean_c",
    "apparent_temperature": "apparent_temp_mean_c",
    "pressure_msl": "pressure_msl_hpa",
    "relative_humidity_2m": "humidity_mean_pct",
    "cloud_cover": "cloud_cover_mean_pct",
}
_SECONDS_TO_HOURS = ("daylight_hours", "sunshine_hours")


@dataclass(slots=True)
class WeatherRaw:
    """Raw Open-Meteo responses (one per API segment)."""

    payloads: list[dict[str, Any]]


class WeatherConnector(BaseConnector[WeatherRaw]):
    """Fetch daily local weather metrics for the configured coordinates."""

    name = "weather"
    description = "Open-Meteo daily weather (temperature, precipitation, daylight, pressure)."
    tables = ("weather_daily",)

    def missing_config(self) -> list[str]:
        """Latitude and longitude are required."""
        missing = []
        if self.settings.weather_latitude is None:
            missing.append("WEATHER_LATITUDE")
        if self.settings.weather_longitude is None:
            missing.append("WEATHER_LONGITUDE")
        return missing

    def extract(self, start: date, end: date) -> WeatherRaw:
        """Call the archive and/or forecast endpoints covering ``[start, end]``."""
        today = datetime.now(self.settings.tz).date()
        end = min(end, today)
        archive_cutoff = today - timedelta(days=ARCHIVE_LAG_DAYS)
        segments: list[tuple[str, date, date]] = []
        if start <= archive_cutoff:
            segments.append((ARCHIVE_URL, start, min(end, archive_cutoff)))
        if end > archive_cutoff:
            segments.append((FORECAST_URL, max(start, archive_cutoff + timedelta(days=1)), end))

        payloads: list[dict[str, Any]] = []
        with httpx.Client(timeout=self.settings.http_timeout_seconds) as client:
            for url, seg_start, seg_end in segments:
                params = {
                    "latitude": self.settings.weather_latitude,
                    "longitude": self.settings.weather_longitude,
                    "start_date": seg_start.isoformat(),
                    "end_date": seg_end.isoformat(),
                    "daily": ",".join(DAILY_VARS),
                    "hourly": ",".join(HOURLY_MEANS),
                    "timezone": self.settings.timezone,
                }
                response = client.get(url, params=params)
                if response.status_code >= 400:
                    reason = response.json().get("reason", response.text) if response.content else response.text
                    raise ConnectorError(f"Open-Meteo {response.status_code}: {reason}")
                payloads.append(response.json())
                self.log.debug("Fetched weather %s..%s from %s", seg_start, seg_end, url)
        return WeatherRaw(payloads)

    def transform(self, raw: WeatherRaw) -> Frames:
        """Merge daily and hourly-derived metrics into ``weather_daily`` rows."""
        frames: list[pd.DataFrame] = []
        for payload in raw.payloads:
            daily = pd.DataFrame(payload.get("daily", {}))
            if daily.empty:
                continue
            daily = daily.rename(columns={"time": "date", **DAILY_VARS})
            daily["date"] = pd.to_datetime(daily["date"]).dt.date
            for col in _SECONDS_TO_HOURS:
                if col in daily:
                    daily[col] = daily[col] / 3600.0

            hourly = pd.DataFrame(payload.get("hourly", {}))
            if not hourly.empty:
                hourly["date"] = pd.to_datetime(hourly.pop("time")).dt.date
                means = hourly.groupby("date").mean(numeric_only=True).rename(columns=HOURLY_MEANS).reset_index()
                daily = daily.merge(means, on="date", how="left")

            daily["latitude"] = payload.get("latitude", self.settings.weather_latitude)
            daily["longitude"] = payload.get("longitude", self.settings.weather_longitude)
            frames.append(daily)

        if not frames:
            return {"weather_daily": self.empty_frame("weather_daily")}
        result = pd.concat(frames, ignore_index=True).drop_duplicates("date", keep="last")
        # Rows for "today" may be partial forecasts; drop days with no observations.
        result = result.dropna(subset=["temp_max_c", "temp_min_c"], how="all")
        return {"weather_daily": result}


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


def search_locations(query: str, count: int = 6, timeout: float = 15.0) -> list[dict[str, Any]]:
    """Look up places by name via Open-Meteo geocoding (no API key).

    Returns:
        Dicts with ``label``, ``latitude``, ``longitude`` and ``timezone``.
    """
    if not query.strip():
        return []
    response = httpx.get(GEOCODING_URL, params={"name": query.strip(), "count": count, "format": "json"}, timeout=timeout)
    response.raise_for_status()
    places = []
    for item in response.json().get("results", []) or []:
        parts = [item.get("name"), item.get("admin1"), item.get("country")]
        places.append(
            {
                "label": ", ".join(dict.fromkeys(p for p in parts if p)),
                "latitude": float(item["latitude"]),
                "longitude": float(item["longitude"]),
                "timezone": item.get("timezone"),
            }
        )
    return places
