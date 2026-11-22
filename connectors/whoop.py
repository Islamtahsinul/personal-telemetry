"""WHOOP API v2 connector with a local OAuth2 authorization-code flow.

One-time setup:

1. Create an app at https://developer-dashboard.whoop.com with redirect URI
   ``http://localhost:8501/connections`` (or set ``WHOOP_REDIRECT_URI``).
2. Paste the client id / secret on the dashboard's Connections page (or set
   ``WHOOP_CLIENT_ID`` / ``WHOOP_CLIENT_SECRET``).
3. Click *Connect WHOOP* (or run ``python cli.py auth whoop``) - a browser opens,
   you approve, and the tokens are stored in the local DuckDB ``oauth_tokens``
   table. The ``offline`` scope yields a refresh token so later syncs are
   non-interactive.
"""

from __future__ import annotations

import re
import secrets
import threading
import time
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import pandas as pd

from connectors.base import BaseConnector, ConnectorError, Frames

AUTH_URL = "https://api.prod.whoop.com/oauth/oauth2/auth"
TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"
API_BASE = "https://api.prod.whoop.com/developer"
PROVIDER = "whoop"
PAGE_LIMIT = 25
STATE_KEY = "whoop_oauth_state"
STATE_TTL = timedelta(minutes=15)
_MS_PER_HOUR = 3_600_000


@dataclass(slots=True)
class WhoopRaw:
    """Raw paginated WHOOP records."""

    cycles: list[dict[str, Any]] = field(default_factory=list)
    recoveries: list[dict[str, Any]] = field(default_factory=list)
    sleeps: list[dict[str, Any]] = field(default_factory=list)


class WhoopConnector(BaseConnector[WhoopRaw]):
    """Pull recovery, strain (cycles) and sleep from WHOOP API v2."""

    name = "whoop"
    description = "WHOOP recovery, HRV, strain and sleep via OAuth2 (API v2)."
    tables = ("whoop_recovery", "whoop_cycle", "whoop_sleep")

    # ---------------------------------------------------------- configuration
    def missing_config(self) -> list[str]:
        """Client credentials and a stored token are required."""
        missing = []
        if not self.settings.whoop_client_id:
            missing.append("WHOOP_CLIENT_ID")
        if not self.settings.whoop_client_secret:
            missing.append("WHOOP_CLIENT_SECRET")
        if not missing and self.db.get_token(PROVIDER) is None:
            missing.append("WHOOP OAuth token (run: python cli.py auth whoop)")
        return missing

    # ------------------------------------------------------------------ OAuth
    def authorization_url(self, state: str) -> str:
        """Build the WHOOP consent URL."""
        query = {
            "response_type": "code",
            "client_id": self.settings.whoop_client_id,
            "redirect_uri": self.settings.whoop_redirect_uri,
            "scope": " ".join(self.settings.whoop_scopes),
            "state": state,
        }
        return f"{AUTH_URL}?{urlencode(query)}"

    # --- browser flow used by the dashboard: WHOOP redirects back to the Connections page
    def begin_authorization(self) -> str:
        """Create and remember a fresh ``state`` and return the consent URL to open.

        Raises:
            ConnectorError: If the WHOOP app details are not saved yet.
        """
        if not self.settings.whoop_client_id or not self.settings.whoop_client_secret:
            raise ConnectorError("Save your WHOOP Client ID and Client Secret first")
        state = secrets.token_urlsafe(16)
        self.db.set_meta(STATE_KEY, state)
        return self.authorization_url(state)

    def complete_authorization(self, code: str, state: str) -> None:
        """Validate the returned ``state`` and exchange ``code`` for tokens.

        Raises:
            ConnectorError: If the state is unknown, expired or does not match.
        """
        pending = self.db.get_meta(STATE_KEY)
        self.db.set_meta(STATE_KEY, None)  # single use
        if pending is None:
            raise ConnectorError("No WHOOP sign-in was in progress - click Connect WHOOP again")
        expected, created = pending
        created = created if created.tzinfo else created.replace(tzinfo=UTC)
        if not secrets.compare_digest(expected, state):
            raise ConnectorError("Sign-in check failed (state mismatch) - click Connect WHOOP again")
        if datetime.now(UTC) - created > STATE_TTL:
            raise ConnectorError("The sign-in link expired - click Connect WHOOP again")
        self._exchange({"grant_type": "authorization_code", "code": code,
                        "redirect_uri": self.settings.whoop_redirect_uri})
        self.log.info("WHOOP tokens stored")

    # --- terminal flow used by `cli.py auth whoop`: a one-shot local callback server
    def authorize(
        self,
        open_browser: bool = True,
        timeout_seconds: int = 300,
        on_url: Callable[[str], None] | None = None,
    ) -> None:
        """Run the interactive authorization-code flow and persist tokens.

        Starts a one-shot HTTP server on the redirect URI's host/port, opens the
        consent page, validates ``state`` and exchanges the code for tokens.

        Args:
            open_browser: Open the consent page in the default browser.
            timeout_seconds: How long to wait for the user to approve.
            on_url: Called with the consent URL (e.g. to show it as a link in the UI).

        Raises:
            ConnectorError: On timeout, state mismatch or a denied consent.
        """
        if not self.settings.whoop_client_id or not self.settings.whoop_client_secret:
            raise ConnectorError("Set WHOOP_CLIENT_ID and WHOOP_CLIENT_SECRET first")
        redirect = urlparse(self.settings.whoop_redirect_uri)
        state = secrets.token_urlsafe(16)
        received: dict[str, str] = {}
        done = threading.Event()

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib API
                params = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
                if urlparse(self.path).path != redirect.path:
                    self.send_response(404)
                    self.end_headers()
                    return
                received.update(params)
                ok = "code" in params and params.get("state") == state
                self.send_response(200 if ok else 400)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                message = "WHOOP connected - you can close this tab." if ok else "Authorization failed."
                self.wfile.write(f"<html><body><h3>{message}</h3></body></html>".encode())
                done.set()

            def log_message(self, *_: Any) -> None:
                return

        try:
            server = HTTPServer((redirect.hostname or "localhost", redirect.port or 80), _Handler)
        except OSError as exc:
            raise ConnectorError(
                f"Port {redirect.port} is unavailable (another program may be using it) - "
                f"sign in from the dashboard's Connections page instead ({exc})"
            ) from exc
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = self.authorization_url(state)
        self.log.info("Open this URL to authorise WHOOP:\n%s", url)
        if on_url is not None:
            on_url(url)
        else:
            print(f"\nAuthorise WHOOP in your browser:\n{url}\n")
        if open_browser:
            webbrowser.open(url)
        try:
            if not done.wait(timeout_seconds):
                raise ConnectorError("Timed out waiting for WHOOP authorization")
        finally:
            server.shutdown()
            server.server_close()

        if "error" in received:
            raise ConnectorError(f"WHOOP authorization denied: {received.get('error_description', received['error'])}")
        if received.get("state") != state:
            raise ConnectorError("OAuth state mismatch - possible CSRF, aborting")
        self._exchange({"grant_type": "authorization_code", "code": received["code"],
                        "redirect_uri": self.settings.whoop_redirect_uri})
        self.log.info("WHOOP tokens stored")

    def profile(self) -> dict[str, Any]:
        """Return the connected user's basic profile (also verifies the token works)."""
        response = httpx.get(
            f"{API_BASE}/v2/user/profile/basic",
            headers={"Authorization": f"Bearer {self._access_token()}"},
            timeout=self.settings.http_timeout_seconds,
        )
        if response.status_code == 401:
            raise ConnectorError("WHOOP rejected the stored token - reconnect")
        response.raise_for_status()
        return dict(response.json())

    def disconnect(self) -> None:
        """Revoke WHOOP access (best effort) and delete the stored token."""
        if self.db.get_token(PROVIDER) is not None:
            try:
                httpx.delete(
                    f"{API_BASE}/v2/user/access",
                    headers={"Authorization": f"Bearer {self._access_token()}"},
                    timeout=self.settings.http_timeout_seconds,
                )
            except (httpx.HTTPError, ConnectorError) as exc:
                self.log.warning("Could not revoke WHOOP access remotely: %s", exc)
        self.db.delete_token(PROVIDER)

    def _exchange(self, data: dict[str, str]) -> str:
        """POST to the token endpoint, persist the result and return the access token."""
        assert self.settings.whoop_client_secret is not None
        payload = {
            **data,
            "client_id": self.settings.whoop_client_id or "",
            "client_secret": self.settings.whoop_client_secret.get_secret_value(),
        }
        response = httpx.post(TOKEN_URL, data=payload, timeout=self.settings.http_timeout_seconds)
        if response.status_code >= 400:
            raise ConnectorError(f"WHOOP token endpoint {response.status_code}: {response.text}")
        token = response.json()
        expires_at = datetime.now(UTC) + timedelta(seconds=int(token.get("expires_in", 3600)))
        self.db.save_token(
            PROVIDER,
            token["access_token"],
            token.get("refresh_token") or data.get("refresh_token"),
            expires_at,
            token.get("scope"),
        )
        return str(token["access_token"])

    def _access_token(self) -> str:
        """Return a valid access token, refreshing it when it is about to expire."""
        stored = self.db.get_token(PROVIDER)
        if stored is None:
            raise ConnectorError("No WHOOP token - run: python cli.py auth whoop")
        expires_at: datetime | None = stored["expires_at"]
        if expires_at is not None and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if expires_at and expires_at - timedelta(minutes=2) > datetime.now(UTC):
            return str(stored["access_token"])
        if not stored["refresh_token"]:
            raise ConnectorError("WHOOP token expired and no refresh token - re-run: python cli.py auth whoop")
        self.log.info("Refreshing WHOOP access token")
        return self._exchange(
            {"grant_type": "refresh_token", "refresh_token": stored["refresh_token"], "scope": "offline"}
        )

    # ---------------------------------------------------------------- extract
    def _paginate(self, client: httpx.Client, path: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        params: dict[str, Any] = {"limit": PAGE_LIMIT, "start": _iso(start), "end": _iso(end)}
        while True:
            response = client.get(f"{API_BASE}{path}", params=params)
            if response.status_code == 429:
                wait = float(response.headers.get("Retry-After", "10"))
                self.log.warning("WHOOP rate limited; sleeping %.0fs", wait)
                time.sleep(wait)
                continue
            if response.status_code == 401:
                raise ConnectorError("WHOOP rejected the token (401) - re-run: python cli.py auth whoop")
            response.raise_for_status()
            body = response.json()
            records.extend(body.get("records", []))
            next_token = body.get("next_token")
            if not next_token:
                return records
            params["nextToken"] = next_token

    def extract(self, start: date, end: date) -> WhoopRaw:
        """Fetch cycles, recoveries and sleeps overlapping ``[start, end]``.

        The window is padded by one day on each side because cycles and sleeps
        straddle midnight; out-of-range days are clipped after transform.
        """
        tz = self.settings.tz
        window_start = datetime.combine(start - timedelta(days=1), datetime.min.time(), tz)
        window_end = datetime.combine(end + timedelta(days=1), datetime.max.time(), tz)
        headers = {"Authorization": f"Bearer {self._access_token()}"}
        with httpx.Client(headers=headers, timeout=self.settings.http_timeout_seconds) as client:
            return WhoopRaw(
                cycles=self._paginate(client, "/v2/cycle", window_start, window_end),
                recoveries=self._paginate(client, "/v2/recovery", window_start, window_end),
                sleeps=self._paginate(client, "/v2/activity/sleep", window_start, window_end),
            )

    # -------------------------------------------------------------- transform
    def transform(self, raw: WhoopRaw) -> Frames:
        """Flatten WHOOP records into daily rows."""
        cycle_rows: list[dict[str, Any]] = []
        cycle_dates: dict[int, date] = {}
        for cycle in raw.cycles:
            local = _local_date(cycle.get("start"), cycle.get("timezone_offset"))
            if local is None:
                continue
            cycle_dates[cycle["id"]] = local
            score = cycle.get("score") or {}
            if cycle.get("score_state") != "SCORED":
                continue
            cycle_rows.append(
                {
                    "date": local,
                    "cycle_id": cycle["id"],
                    "strain": score.get("strain"),
                    "kilojoules": score.get("kilojoule"),
                    "avg_heart_rate_bpm": score.get("average_heart_rate"),
                    "max_heart_rate_bpm": score.get("max_heart_rate"),
                }
            )

        recovery_rows: list[dict[str, Any]] = []
        for rec in raw.recoveries:
            if rec.get("score_state") != "SCORED":
                continue
            local = cycle_dates.get(rec.get("cycle_id")) or _local_date(rec.get("created_at"), None, self.settings.tz)
            score = rec.get("score") or {}
            recovery_rows.append(
                {
                    "date": local,
                    "cycle_id": rec.get("cycle_id"),
                    "recovery_score": score.get("recovery_score"),
                    "hrv_rmssd_ms": score.get("hrv_rmssd_milli"),
                    "resting_heart_rate_bpm": score.get("resting_heart_rate"),
                    "spo2_pct": score.get("spo2_percentage"),
                    "skin_temp_c": score.get("skin_temp_celsius"),
                    "user_calibrating": score.get("user_calibrating"),
                }
            )

        sleep_rows: list[dict[str, Any]] = []
        for sleep in raw.sleeps:
            if sleep.get("nap") or sleep.get("score_state") != "SCORED":
                continue
            local = _local_date(sleep.get("end"), sleep.get("timezone_offset"))
            score = sleep.get("score") or {}
            stages = score.get("stage_summary") or {}
            light = stages.get("total_light_sleep_time_milli") or 0
            deep = stages.get("total_slow_wave_sleep_time_milli") or 0
            rem = stages.get("total_rem_sleep_time_milli") or 0
            sleep_rows.append(
                {
                    "date": local,
                    "sleep_id": str(sleep.get("id")),
                    "sleep_performance_pct": score.get("sleep_performance_percentage"),
                    "sleep_efficiency_pct": score.get("sleep_efficiency_percentage"),
                    "sleep_consistency_pct": score.get("sleep_consistency_percentage"),
                    "respiratory_rate": score.get("respiratory_rate"),
                    "total_in_bed_hours": (stages.get("total_in_bed_time_milli") or 0) / _MS_PER_HOUR,
                    "total_sleep_hours": (light + deep + rem) / _MS_PER_HOUR,
                    "light_sleep_hours": light / _MS_PER_HOUR,
                    "deep_sleep_hours": deep / _MS_PER_HOUR,
                    "rem_sleep_hours": rem / _MS_PER_HOUR,
                    "awake_hours": (stages.get("total_awake_time_milli") or 0) / _MS_PER_HOUR,
                    "disturbance_count": stages.get("disturbance_count"),
                    "sleep_cycle_count": stages.get("sleep_cycle_count"),
                    "sleep_start": pd.to_datetime(sleep.get("start"), utc=True),
                    "sleep_end": pd.to_datetime(sleep.get("end"), utc=True),
                }
            )

        sleep_df = pd.DataFrame(sleep_rows)
        if not sleep_df.empty:
            # Keep the longest main sleep when several end on the same day.
            sleep_df = sleep_df.sort_values("total_sleep_hours").drop_duplicates("date", keep="last")

        return {
            "whoop_cycle": pd.DataFrame(cycle_rows) if cycle_rows else self.empty_frame("whoop_cycle"),
            "whoop_recovery": pd.DataFrame(recovery_rows) if recovery_rows else self.empty_frame("whoop_recovery"),
            "whoop_sleep": sleep_df if not sleep_df.empty else self.empty_frame("whoop_sleep"),
        }


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _local_date(timestamp: str | None, offset: str | None, fallback_tz: Any = None) -> date | None:
    """Convert a UTC ISO timestamp to a local date using a ``±HH:MM`` offset or a timezone."""
    if not timestamp:
        return None
    moment = pd.Timestamp(timestamp)
    if moment.tzinfo is None:
        moment = moment.tz_localize("UTC")
    if offset:
        # WHOOP sends "+01:00", "-0500" or "Z" (UTC).
        match = re.fullmatch(r"([+-])(\d{2}):?(\d{2})", offset.strip())
        if offset.strip().upper() == "Z":
            return moment.tz_convert("UTC").date()
        if match:
            sign = -1 if match.group(1) == "-" else 1
            delta = pd.Timedelta(hours=int(match.group(2)), minutes=int(match.group(3)))
            return (moment.tz_convert("UTC") + sign * delta).date()
    if fallback_tz is not None:
        return moment.tz_convert(fallback_tz).date()
    return moment.date()
