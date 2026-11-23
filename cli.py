"""Command-line entry point.

Examples::

    python cli.py init                      # create local folders + DuckDB schema
    python cli.py demo --days 180           # load synthetic data to explore the UI
    python cli.py auth whoop                # one-time WHOOP OAuth consent
    python cli.py status                    # which connectors are ready
    python cli.py sync                      # run every configured connector (last 30 days)
    python cli.py sync -s whoop -s weather --days 365
    python cli.py log --mood 4 --energy 3 --focus 5 --notes "slept well"
    python cli.py insights --provider ollama --days 30
    python cli.py export --days 90 --out matrix.csv
    python cli.py ui                        # launch the Streamlit dashboard
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from core.config import configure_logging, get_settings
from core.database import Database


def _date(value: str) -> date:
    return date.fromisoformat(value)


def _range(args: argparse.Namespace) -> tuple[date, date]:
    today = datetime.now(get_settings().tz).date()
    end = args.end or today
    start = args.start or end - timedelta(days=args.days - 1)
    return start, end


def cmd_init(_: argparse.Namespace) -> int:
    """Create directories and tables."""
    settings = get_settings()
    Database(settings=settings).init_schema()
    print(f"Database: {settings.duckdb_path}\nInbox:    {settings.inbox_dir}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Seed synthetic data."""
    from analytics.demo_data import seed_demo_data

    written = seed_demo_data(Database(), days=args.days, seed=args.seed)
    for table, rows in written.items():
        print(f"{table:<18} {rows:>6} rows")
    return 0


def cmd_auth(args: argparse.Namespace) -> int:
    """Interactive OAuth for API connectors."""
    from connectors.whoop import WhoopConnector

    if args.provider != "whoop":
        print(f"No OAuth flow for {args.provider}", file=sys.stderr)
        return 2
    WhoopConnector().authorize(open_browser=not args.no_browser)
    print("WHOOP authorised.")
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    """Print connector readiness and table coverage."""
    from connectors import CONNECTORS, get_connector

    for name in CONNECTORS:
        missing = get_connector(name).missing_config()
        print(f"{'ready  ' if not missing else 'missing'}  {name:<14} {', '.join(missing)}")
    print()
    print(Database().table_coverage().to_string(index=False))
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    """Run connectors."""
    from connectors import run_all

    start, end = _range(args)
    results = run_all(start, end, names=args.source or None)
    failed = 0
    for r in results:
        detail = f" ({r.error})" if r.error else ""
        print(f"{r.status:<8} {r.connector:<14} {r.rows_loaded:>6} rows  {r.duration_seconds:5.1f}s{detail}")
        failed += r.status == "failed"
    return 1 if failed else 0


def cmd_log(args: argparse.Namespace) -> int:
    """Record subjective state."""
    day = args.date or datetime.now(get_settings().tz).date()
    row = {"date": day, "mood": args.mood, "energy": args.energy, "focus": args.focus, "notes": args.notes}
    Database().upsert("subjective_state", pd.DataFrame([row]))
    print(f"Logged {day}: mood={args.mood} energy={args.energy} focus={args.focus}")
    return 0


def cmd_insights(args: argparse.Namespace) -> int:
    """Generate an AI executive summary."""
    from analytics import build_daily_matrix, build_insight_payload
    from llm import LLMProviderError, get_provider

    start, end = _range(args)
    matrix = build_daily_matrix(start, end)
    payload = build_insight_payload(matrix, start, end)
    try:
        report = get_provider(name=args.provider).synthesize_insights(payload)
    except LLMProviderError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if args.out:
        Path(args.out).write_text(report.markdown, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(report.markdown)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export the daily master matrix to CSV or Parquet."""
    from analytics import build_daily_matrix

    start, end = _range(args)
    matrix = build_daily_matrix(start, end)
    out = Path(args.out)
    if out.suffix.lower() == ".parquet":
        matrix.to_parquet(out)
    else:
        matrix.to_csv(out)
    print(f"Wrote {matrix.shape[0]} days x {matrix.shape[1]} features to {out}")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    """Launch Streamlit."""
    app = Path(__file__).resolve().parent / "ui" / "app.py"
    # Pass the bind address explicitly: .streamlit/config.toml is only read from the
    # current directory, and Streamlit otherwise listens on every network interface.
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app),
                            "--server.address", "localhost", "--server.port", str(args.port)],
                           cwd=str(app.parent.parent))


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(prog="telemetry", description="Local-first personal telemetry pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_range(p: argparse.ArgumentParser, default_days: int) -> None:
        p.add_argument("--days", type=int, default=default_days, help="Window length ending today/--end")
        p.add_argument("--start", type=_date, help="YYYY-MM-DD (overrides --days)")
        p.add_argument("--end", type=_date, help="YYYY-MM-DD (default today)")

    sub.add_parser("init", help="Create folders and database").set_defaults(func=cmd_init)

    p = sub.add_parser("demo", help="Load synthetic demo data")
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("auth", help="Run an OAuth flow")
    p.add_argument("provider", choices=["whoop"])
    p.add_argument("--no-browser", action="store_true", help="Print the URL instead of opening a browser")
    p.set_defaults(func=cmd_auth)

    sub.add_parser("status", help="Show connector readiness and data coverage").set_defaults(func=cmd_status)

    p = sub.add_parser("sync", help="Run connectors")
    p.add_argument("-s", "--source", action="append", help="Connector name (repeatable; default all)")
    with_range(p, get_settings().default_lookback_days)
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("log", help="Log subjective mood/energy/focus (1-5)")
    for name in ("mood", "energy", "focus"):
        p.add_argument(f"--{name}", type=int, choices=range(1, 6), required=True)
    p.add_argument("--notes")
    p.add_argument("--date", type=_date)
    p.set_defaults(func=cmd_log)

    p = sub.add_parser("insights", help="Generate an AI executive summary")
    p.add_argument("--provider", choices=["anthropic", "ollama"])
    p.add_argument("--out", help="Write Markdown to this file")
    with_range(p, 30)
    p.set_defaults(func=cmd_insights)

    p = sub.add_parser("export", help="Export the daily matrix")
    p.add_argument("--out", required=True, help=".csv or .parquet")
    with_range(p, 90)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("ui", help="Launch the dashboard")
    p.add_argument("--port", type=int, default=8501)
    p.set_defaults(func=cmd_ui)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    configure_logging()
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
