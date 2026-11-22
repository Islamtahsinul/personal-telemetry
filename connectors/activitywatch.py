"""ActivityWatch connector for laptop screen time and focused work duration.

Talks to the local ActivityWatch server (default ``http://localhost:5600``; no
auth, data never leaves the machine). Uses the server-side query engine to
intersect window events with *not-AFK* periods, so only time you were actually
at the keyboard counts.

Produces:

* ``screen_usage`` rows (``device = 'laptop'``) - minutes per application per day.
* ``laptop_activity`` - active/AFK minutes, coding and work minutes, first/last activity.

Coding time is the larger of (a) time in apps matching ``AW_CODING_APP_PATTERNS``
and (b) editor-watcher time (e.g. ``aw-watcher-vscode``) when such a bucket exists.
"""

from __future__ import annotations

import re
import socket
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import httpx
import pandas as pd

from connectors.base import BaseConnector, ConnectorError, Frames
from connectors.utils import categorize_app, date_range_days

PERIODS_PER_REQUEST = 31


@dataclass(slots=True)
class AWBuckets:
    """Bucket ids selected for querying."""

    window: str
    afk: str
    editor: str | None = None


@dataclass(slots=True)
class ActivityWatchRaw:
    """Query results, one entry per local day."""

    days: list[date] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)


class ActivityWatchConnector(BaseConnector[ActivityWatchRaw]):
    """Query a local ActivityWatch server for per-app and focus metrics."""

    name = "activitywatch"
    description = "Laptop app usage and active coding/work time from local ActivityWatch."
    tables = ("screen_usage", "laptop_activity")

    @property
    def _api(self) -> str:
        return self.settings.activitywatch_url.rstrip("/") + "/api/0"

    def missing_config(self) -> list[str]:
        """The ActivityWatch server must be reachable."""
        try:
            httpx.get(f"{self._api}/info", timeout=3).raise_for_status()
        except httpx.HTTPError:
            return [f"ActivityWatch server at {self.settings.activitywatch_url} (is aw-server running?)"]
        return []

    # ---------------------------------------------------------------- buckets
    def discover_buckets(self, client: httpx.Client) -> AWBuckets:
        """Select window/AFK/editor buckets, preferring this machine's hostname."""
        response = client.get(f"{self._api}/buckets/")
        response.raise_for_status()
        buckets: dict[str, dict[str, Any]] = response.json()
        hostname = socket.gethostname().lower()

        def pick(bucket_type: str) -> str | None:
            candidates = [b for b in buckets.values() if b.get("type") == bucket_type]
            if not candidates:
                return None
            local = [b for b in candidates if str(b.get("hostname", "")).lower() == hostname]
            pool = local or candidates
            pool.sort(key=lambda b: b.get("last_updated") or "", reverse=True)
            return str(pool[0]["id"])

        window, afk = pick("currentwindow"), pick("afkstatus")
        if not window or not afk:
            raise ConnectorError("ActivityWatch window/AFK buckets not found - are aw-watcher-window/afk running?")
        return AWBuckets(window=window, afk=afk, editor=pick("app.editor.activity"))

    @staticmethod
    def build_query(buckets: AWBuckets) -> list[str]:
        """Return the AW query-language statements for one time period."""
        lines = [
            f'afk = flood(query_bucket("{buckets.afk}"));',
            'not_afk = filter_keyvals(afk, "status", ["not-afk"]);',
            'is_afk = filter_keyvals(afk, "status", ["afk"]);',
            f'win = flood(query_bucket("{buckets.window}"));',
            "win = filter_period_intersect(win, not_afk);",
            'apps = sort_by_duration(merge_events_by_keys(win, ["app"]));',
        ]
        if buckets.editor:
            lines += [
                f'editor = flood(query_bucket("{buckets.editor}"));',
                "editor = filter_period_intersect(editor, not_afk);",
                'RETURN = {"apps": apps, "not_afk": not_afk, "afk_seconds": sum_durations(is_afk), '
                '"editor_seconds": sum_durations(editor)};',
            ]
        else:
            lines.append(
                'RETURN = {"apps": apps, "not_afk": not_afk, "afk_seconds": sum_durations(is_afk), "editor_seconds": 0};'
            )
        return lines

    # ------------------------------------------------------------------ hooks
    def extract(self, start: date, end: date) -> ActivityWatchRaw:
        """Run one query per local day (batched) over ``[start, end]``."""
        tz = self.settings.tz
        days = date_range_days(start, min(end, datetime.now(tz).date()))
        raw = ActivityWatchRaw()
        with httpx.Client(timeout=max(self.settings.http_timeout_seconds, 120)) as client:
            buckets = self.discover_buckets(client)
            query = self.build_query(buckets)
            for i in range(0, len(days), PERIODS_PER_REQUEST):
                chunk = days[i : i + PERIODS_PER_REQUEST]
                periods = [
                    f"{datetime.combine(d, datetime.min.time(), tz).isoformat()}/"
                    f"{datetime.combine(d + timedelta(days=1), datetime.min.time(), tz).isoformat()}"
                    for d in chunk
                ]
                response = client.post(f"{self._api}/query/", json={"timeperiods": periods, "query": query})
                if response.status_code >= 400:
                    raise ConnectorError(f"ActivityWatch query failed {response.status_code}: {response.text[:300]}")
                raw.days.extend(chunk)
                raw.results.extend(response.json())
        return raw

    def transform(self, raw: ActivityWatchRaw) -> Frames:
        """Convert per-day query results to screen usage and activity summaries."""
        coding = [re.compile(p, re.IGNORECASE) for p in self.settings.aw_coding_app_patterns]
        work = [re.compile(p, re.IGNORECASE) for p in self.settings.aw_work_app_patterns]
        usage_rows: list[dict[str, Any]] = []
        activity_rows: list[dict[str, Any]] = []

        for day, result in zip(raw.days, raw.results, strict=True):
            apps: list[dict[str, Any]] = result.get("apps", [])
            coding_seconds = work_seconds = 0.0
            for event in apps:
                app = str(event.get("data", {}).get("app") or "unknown")
                seconds = float(event.get("duration", 0.0))
                if seconds < 1:
                    continue
                usage_rows.append(
                    {
                        "date": day,
                        "device": "laptop",
                        "app": app,
                        "category": categorize_app(app, self.settings.aw_category_rules),
                        "minutes": round(seconds / 60, 2),
                        "source": "activitywatch",
                    }
                )
                if any(p.search(app) for p in coding):
                    coding_seconds += seconds
                elif any(p.search(app) for p in work):
                    work_seconds += seconds

            not_afk: list[dict[str, Any]] = result.get("not_afk", [])
            active_seconds = sum(float(e.get("duration", 0.0)) for e in not_afk)
            if active_seconds == 0 and not apps:
                continue
            first = last = None
            if not_afk:
                starts = pd.to_datetime([e["timestamp"] for e in not_afk], utc=True, format="ISO8601")
                ends = starts + pd.to_timedelta([float(e.get("duration", 0.0)) for e in not_afk], unit="s")
                first, last = starts.min(), ends.max()
            coding_seconds = max(coding_seconds, float(result.get("editor_seconds") or 0.0))
            activity_rows.append(
                {
                    "date": day,
                    "active_minutes": round(active_seconds / 60, 2),
                    "afk_minutes": round(float(result.get("afk_seconds") or 0.0) / 60, 2),
                    "coding_minutes": round(coding_seconds / 60, 2),
                    # Coding counts as work too.
                    "work_minutes": round((work_seconds + coding_seconds) / 60, 2),
                    "first_active": first,
                    "last_active": last,
                    "top_app": apps[0]["data"].get("app") if apps else None,
                }
            )

        return {
            "screen_usage": pd.DataFrame(usage_rows) if usage_rows else self.empty_frame("screen_usage"),
            "laptop_activity": pd.DataFrame(activity_rows) if activity_rows else self.empty_frame("laptop_activity"),
        }
