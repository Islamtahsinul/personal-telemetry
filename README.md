# Personal Telemetry

A local-first, open-source pipeline that pulls your health, behaviour, spending and
environment data into a single DuckDB file on your own machine, aligns it into one
daily matrix, finds cross-domain patterns, and writes an AI executive summary, either
with a fully local model (Ollama) or with Claude via your own API key.

Runs on macOS, Linux and Windows. It needs no cloud infrastructure and no account
beyond the data sources you choose to connect.

```
WHOOP ─┐  Renpho ─┐  MyFitnessPal ─┐  S24 Health Connect ─┐  Monzo (Sheets) ─┐  Open-Meteo ─┐  ActivityWatch ─┐  Files ─┐
       └──────────┴────────────────┴──────────────────────┴──────────────────┴──────────────┴─────────────────┴─────────┘
                                       connectors/*  extract → transform → load (idempotent upserts)
                                                            │
                                               core/database.py  (DuckDB, daily grain)
                                                            │
                                     analytics/pipeline.py  →  daily master matrix (1 row/day, ~80 features)
                                                            │
                     analytics/correlations.py  (MAs, FDR-corrected cross-domain + lagged correlations, anomalies)
                                                            │
                                     llm/*  (Ollama local │ Claude)  →  executive summary
                                                            │
                                                  ui/app.py  (Streamlit)
```

## Quick start

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate      Windows: .venv\Scripts\activate
pip install -r requirements.txt
python cli.py ui                # → http://localhost:8501
```

Open the **Connections** page and connect your sources. Nothing has to be edited by hand:

* **Location / weather:** type your town.
* **WHOOP:** a two-minute guided one-off (create your own free WHOOP developer app), then **Connect WHOOP**
  signs you in through the browser. Tokens refresh automatically after that.
* **Renpho:** sign in with your Renpho e-mail and password.
* **MyFitnessPal, Monzo, Samsung Health Connect / screen time:** upload the export files, or point the
  Samsung section at a synced folder.
* **ActivityWatch:** detected automatically when it's running.
* **AI summaries:** pick local Ollama or paste an Anthropic API key.

Saved settings live in the local database. Passwords, client secrets and API keys are stored in the OS
credential store (Windows Credential Manager, macOS Keychain or the Linux Secret Service). A `.env` file
([`.env.example`](.env.example)) still works for headless or server setups; values saved in the UI take precedence.

To explore with synthetic data first, run `python cli.py demo --days 180`. To keep it separate from real data,
set `DATA_DIR`, e.g. `DATA_DIR=./demo-data python cli.py demo`.

## Project layout

```
core/
  config.py           pydantic-settings loader for .env
  database.py         DuckDB schema (TableSpec registry), upserts, run audit, token store
  settings_store.py   Settings saved from the UI; secrets in the OS keychain
connectors/
  base.py             BaseConnector: extract() → transform() → load(), run() with audit + error isolation
  whoop.py            WHOOP API v2, OAuth2 authorization-code flow with local callback + refresh
  renpho.py           renpho-api client, Health Connect fallback
  myfitnesspal.py     MFP export (.zip/.csv), Health Connect nutrition, or scraper
  monzo_sheets.py     Monzo → Google Sheets export via gspread, daily spend per category
  weather.py          Open-Meteo archive + forecast APIs (no key)
  android_s24.py      Health Connect CSV/JSON dumps + Digital Wellbeing/Tasker usage exports
  activitywatch.py    Local ActivityWatch query API: per-app time, coding/work minutes
  file_parser.py      Generic CSV/JSON drops into data/inbox/<table>/
  utils.py            Duration parsing, fuzzy column matching, timezone-aware day bucketing
analytics/
  pipeline.py         Daily master matrix (alignment, forward-fill, zero-fill, derived features)
  correlations.py     Moving averages, correlation matrices, FDR, lags, hypotheses, anomalies, LLM payload
  demo_data.py        Synthetic data with known embedded effects
llm/
  base_provider.py    BaseLLMProvider.synthesize_insights(payload) + shared prompt
  anthropic_provider.py  Claude (adaptive thinking, effort, server-side refusal fallback)
  ollama_provider.py  Local Ollama /api/chat
ui/
  app.py              Streamlit entry point (navigation)
  dashboard.py        Dashboard page
  connections.py      Connections page (sign-ins, uploads, location, AI provider)
  charts.py, theme.py Plotly figure builders and colour-blind-safe palette
cli.py                init · demo · auth · status · sync · log · insights · export · ui
tests/                pytest suite (no network)
```

## Connecting sources

| Source | Setup | Command |
|---|---|---|
| **WHOOP** | Create an app at developer-dashboard.whoop.com with redirect `http://localhost:8501/connections`. Set `WHOOP_CLIENT_ID/SECRET`. | `python cli.py auth whoop` once, then `sync -s whoop` |
| **Renpho** | `pip install renpho-api`, set `RENPHO_EMAIL/PASSWORD`. Or enable Renpho → Health Connect and use the Android exports. | `sync -s renpho` |
| **MyFitnessPal** | Premium export (Settings → Export) → drop the `.zip` in `data/inbox/myfitnesspal/`. Or MFP → Health Connect. | `sync -s myfitnesspal` |
| **Samsung S24** | Health Sync (or similar) exporting Health Connect data as CSV/JSON to `data/inbox/android/`, or point `ANDROID_EXPORT_DIR` at a synced folder. Screen time: Tasker/ActionDash/StayFree CSV export to the same folder. | `sync -s android_s24` |
| **Monzo** | Upload a CSV from *Account → Export transactions*, or for automatic sync enable Monzo's Google Sheets export. Create a Google Cloud service account with the Sheets + Drive APIs, download its key, and share the sheet with its e-mail. Set `GOOGLE_SERVICE_ACCOUNT_FILE`, `MONZO_SHEET_ID`. | `sync -s monzo` |
| **Weather** | Set `WEATHER_LATITUDE/LONGITUDE`. | `sync -s weather --days 365` |
| **ActivityWatch** | Install from activitywatch.net and keep `aw-server` running. | `sync -s activitywatch` |
| **Mood/energy/focus** | Sidebar form in the UI, `python cli.py log --mood 4 --energy 3 --focus 5`, or CSV in `data/inbox/subjective_state/`. | — |

`python cli.py status` shows which connectors are ready and what is missing.

### Scheduling

Every load is an idempotent upsert, so you can schedule a daily sync safely:

* **macOS/Linux (cron):** `15 7 * * * cd /path/to/repo && .venv/bin/python cli.py sync --days 7`
* **Windows (Task Scheduler):** action `C:\path\to\repo\.venv\Scripts\python.exe`, arguments `cli.py sync --days 7`, start in `C:\path\to\repo`.

## Analytics notes

* **Alignment:** everything is bucketed to local calendar days (`TIMEZONE`). WHOOP sleep is keyed by
  wake-up date, so `whoop_sleep_*` on day D is the night before D. The *next-day* correlations pair
  behaviour on D with outcomes on D+1, which is how "screen time vs sleep quality" is measured.
* **Body composition** (Renpho/S24) is forward-filled for up to 7 days. Spend days without transactions
  count as £0 within the period your Monzo export covers.
* **Weekday adjustment:** correlations use day-of-week-adjusted values. Without that, every
  workday-driven metric (laptop time, commute spend) correlates with every other one.
* **Trivial pairs** (same domain, the same quantity from two devices, derived features against
  their inputs) are excluded from the cross-domain ranking.
* **Multiple testing:** with ~2,000 pairs, Benjamini–Hochberg q-values are reported next to r.
  Read the results as hypotheses to test, not as causal findings.
* **Anomalies:** robust z-scores (median/MAD) against a trailing, weekday-aware baseline.

## Privacy

* All data stays in `data/` (git-ignored). `python cli.py ui` binds the Streamlit server to `localhost` only.
  If you run `streamlit run ui/app.py` yourself, run it from the repo root (so `.streamlit/config.toml` applies)
  or add `--server.address localhost`; otherwise Streamlit listens on every network interface.
  The dashboard has no login, so never expose it to other machines or the internet.
* With `LLM_PROVIDER=ollama`, nothing leaves the machine.
* With `LLM_PROVIDER=anthropic`, only the aggregated payload (averages, correlations, anomalies) is
  sent. Notes, transactions and app titles are never sent, and the UI shows the exact payload before you generate.
* OAuth tokens are stored in the local DuckDB file. Protect `data/` as you would any personal health data.

## Development

```bash
pip install -r requirements.txt -r requirements-optional.txt
pytest
ruff check .
```

Adding a source means subclassing `BaseConnector`, declaring its table(s) as a `TableSpec`
in `core/database.py`, adding a SQL block to `DOMAIN_QUERIES` in `analytics/pipeline.py`,
and registering it in `connectors/__init__.py`.
