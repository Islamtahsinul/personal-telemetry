"""Connections page: connect every data source from the browser - no ``.env`` editing.

Settings are saved to the local database; passwords, client secrets and API keys
go to the operating system's credential store (see :mod:`core.settings_store`).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connectors import get_connector  # noqa: E402
from connectors.base import ConnectorError, ConnectorResult  # noqa: E402
from connectors.renpho import verify_renpho_login  # noqa: E402
from connectors.utils import file_sha256  # noqa: E402
from connectors.weather import search_locations  # noqa: E402
from connectors.whoop import WhoopConnector  # noqa: E402
from core.config import get_settings, reload_settings  # noqa: E402
from core.database import Database  # noqa: E402
from core.settings_store import SettingsStore  # noqa: E402
from llm.ollama_provider import OllamaProvider  # noqa: E402

settings = get_settings()
db = Database(settings=settings)
db.init_schema()
store = SettingsStore(settings.duckdb_path)
SECRET_HOME = "Windows Credential Manager" if sys.platform == "win32" else (
    "macOS Keychain" if sys.platform == "darwin" else "system keyring")
SECRET_NOTE = (f"Stored in {SECRET_HOME}." if store.secrets_in_keychain
               else "No system keychain available - stored in the local database file.")
WHOOP_DEV_URL = "https://developer-dashboard.whoop.com"


# =============================================================================== helpers
def save(values: dict[str, Any], message: str = "Saved") -> None:
    """Persist settings, reload them everywhere and refresh the page."""
    store.save(values)
    reload_settings()
    st.cache_data.clear()
    st.toast(message, icon="✅")
    st.rerun()


def badge(connected: bool, detail: str = "") -> None:
    """Connection status line."""
    if connected:
        st.markdown(f":green[● Connected]{' · ' + detail if detail else ''}")
    else:
        st.markdown(f":gray[○ Not connected]{' · ' + detail if detail else ''}")


def run_sync(name: str, days: int) -> ConnectorResult:
    """Run one connector over the last ``days`` days and report the outcome."""
    today = datetime.now(settings.tz).date()
    with st.spinner(f"Syncing {name}…"):
        result = get_connector(name, reload_settings(), db).run(today - timedelta(days=days - 1), today)
    st.cache_data.clear()
    if result.status == "success":
        st.success(f"Loaded {result.rows_loaded} rows "
                   + ", ".join(f"{t}: {n}" for t, n in result.rows_by_table.items()))
    elif result.status == "skipped":
        st.warning(f"Nothing to sync yet: {result.error}")
    else:
        st.error(f"Sync failed: {result.error}")
    return result


def save_uploads(files: list[Any], folder: Path) -> list[Path]:
    """Write uploaded files into ``folder`` (filenames sanitised)."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for upload in files:
        path = folder / Path(upload.name).name
        path.write_bytes(upload.getvalue())
        paths.append(path)
    return paths


def section(title: str, subtitle: str) -> Any:
    """Bordered card with a heading."""
    box = st.container(border=True)
    box.subheader(title)
    box.caption(subtitle)
    return box


# =============================================================================== page
st.title("Connections")
st.caption("Connect your sources here. Everything is stored on this computer; nothing is sent anywhere "
           "except to the service you are connecting to.")

whoop = WhoopConnector(settings, db)

# WHOOP sends the browser back here with ?code=...&state=... (or ?error=...) after sign-in.
callback = dict(st.query_params)
if "code" in callback or "error" in callback:
    st.query_params.clear()
    if "error" in callback:
        st.error(f"WHOOP sign-in was cancelled or refused: {callback.get('error_description', callback['error'])}")
    else:
        try:
            with st.spinner("Finishing WHOOP sign-in…"):
                whoop.complete_authorization(callback["code"], callback.get("state", ""))
            st.success("WHOOP connected ✓ Pulling your last 90 days now - you can close any other dashboard tab.")
            run_sync("whoop", 90)
        except ConnectorError as exc:
            st.error(str(exc))

whoop_token = db.get_token("whoop") is not None
summary = {
    "Location": settings.weather_latitude is not None,
    "WHOOP": whoop_token,
    "Renpho": bool(settings.renpho_email and settings.renpho_password),
    "Monzo": not get_connector("monzo", settings, db).missing_config(),
    "Samsung": any(p.is_file() for p in settings.android_export_dir.rglob("*")) if settings.android_export_dir else False,
}
cols = st.columns(len(summary))
for col, (name, ok) in zip(cols, summary.items(), strict=True):
    col.markdown(f"{'🟢' if ok else '⚪'} **{name}**")

# ------------------------------------------------------------------------------ location
with section("📍 Location & weather", "Free Open-Meteo weather for your area - just type your town."):
    if settings.weather_latitude is not None:
        badge(True, f"{settings.weather_latitude:.3f}, {settings.weather_longitude:.3f} · timezone {settings.timezone}")
    query = st.text_input("City or town", placeholder="e.g. Manchester", key="loc_query")
    if query:
        try:
            places = search_locations(query)
        except Exception as exc:  # noqa: BLE001 - show network errors inline
            places = []
            st.error(f"Location search failed: {exc}")
        if places:
            choice = st.selectbox("Pick the right place", range(len(places)), format_func=lambda i: places[i]["label"])
            place = places[choice]
            if st.button("Use this location", type="primary"):
                values: dict[str, Any] = {"weather_latitude": place["latitude"], "weather_longitude": place["longitude"]}
                if place["timezone"]:
                    values["timezone"] = place["timezone"]
                save(values, f"Location set to {place['label']}")
        elif query:
            st.caption("No matches - try a nearby larger town.")
    if settings.weather_latitude is not None and st.button("Load the last year of weather"):
        run_sync("weather", 365)

# ------------------------------------------------------------------------------ whoop
with section("💚 WHOOP", "Recovery, HRV, strain and sleep."):
    has_app = bool(settings.whoop_client_id and settings.whoop_client_secret)
    if whoop_token:
        try:
            person = whoop.profile()
            badge(True, f"{person.get('first_name', '')} {person.get('last_name', '')}".strip() or person.get("email", ""))
        except ConnectorError as exc:
            badge(False, f"{exc}")
            whoop_token = False
        except Exception as exc:  # noqa: BLE001 - network problems shouldn't hide the controls
            badge(True, f"couldn't reach WHOOP right now ({type(exc).__name__})")

    if not has_app:
        st.markdown(
            "WHOOP only lets registered apps read your data, so the first time you create your own free "
            "*developer app* - about two minutes, once. It is linked to your WHOOP account and only you use it."
        )
        st.markdown(f"**1.** Open the [WHOOP Developer Dashboard]({WHOOP_DEV_URL}) and sign in with your normal WHOOP login.")
        st.markdown("**2.** Create a team if asked (any name), then click **Create App**. Name it e.g. *Personal Telemetry*.")
        st.markdown("**3.** Under **Redirect URIs** add exactly:")
        st.code(settings.whoop_redirect_uri, language=None)
        st.markdown("**4.** Tick every **read** scope (recovery, cycles, sleep, workout, profile, body measurement), then save.")
        st.markdown("**5.** Copy the **Client ID** and **Client Secret** into the boxes below.")
    with st.form("whoop_app"):
        client_id = st.text_input("Client ID", value=settings.whoop_client_id or "")
        client_secret = st.text_input("Client Secret", type="password",
                                      placeholder="saved ✓ (leave blank to keep)" if settings.whoop_client_secret else "")
        if st.form_submit_button("Save WHOOP app details"):
            values = {"whoop_client_id": client_id.strip()}
            if client_secret.strip():
                values["whoop_client_secret"] = client_secret.strip()
            save(values, "WHOOP app details saved")
    if has_app:
        st.caption(SECRET_NOTE)
        st.caption(f"Your WHOOP app's redirect URI must be exactly `{settings.whoop_redirect_uri}`.")
        c1, c2, c3 = st.columns(3)
        if c1.button("Reconnect WHOOP" if whoop_token else "Connect WHOOP", type="primary", width="stretch"):
            try:
                st.session_state["whoop_auth_url"] = whoop.begin_authorization()
            except ConnectorError as exc:
                st.error(str(exc))
        if url := st.session_state.get("whoop_auth_url"):
            st.link_button("Sign in to WHOOP ↗", url, type="primary")
            st.caption("A WHOOP tab opens. After you approve, it comes back to this page and finishes automatically "
                       "- you can then close this tab.")
        if whoop_token and c2.button("Sync last 90 days", width="stretch"):
            run_sync("whoop", 90)
        if whoop_token and c3.button("Disconnect", width="stretch"):
            whoop.disconnect()
            st.toast("WHOOP disconnected")
            st.rerun()

# ------------------------------------------------------------------------------ renpho
with section("⚖️ Renpho", "Weight, body fat %, muscle mass, visceral fat, BMI."):
    connected = bool(settings.renpho_email and settings.renpho_password)
    badge(connected, settings.renpho_email or "")
    if not connected:
        with st.form("renpho"):
            st.markdown("Sign in with the same e-mail and password you use in the Renpho app.")
            email = st.text_input("E-mail")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in", type="primary"):
                try:
                    with st.spinner("Checking with Renpho…"):
                        area = verify_renpho_login(email.strip(), password, settings.timezone, settings.renpho_area_code)
                    save({"renpho_email": email.strip(), "renpho_password": password, "renpho_source": "api",
                          "renpho_area_code": area}, "Renpho connected")
                except ConnectorError as exc:
                    st.error(str(exc))
        st.caption("Signed up to Renpho with Google/Apple? Set a password in the Renpho app first, "
                   "or enable Renpho → Health Connect and use the Samsung section below.")
    else:
        st.caption(SECRET_NOTE)
        c1, c2 = st.columns(2)
        if c1.button("Sync all weigh-ins", type="primary", width="stretch"):
            run_sync("renpho", 3650)
        if c2.button("Disconnect Renpho", width="stretch"):
            save({"renpho_email": None, "renpho_password": None, "renpho_source": None, "renpho_area_code": None},
                 "Renpho disconnected")

# ------------------------------------------------------------------------------ monzo
with section("💳 Monzo", "Daily spending by category (groceries, eating out, transport…)."):
    monzo_files = get_connector("monzo", settings, db).csv_files()
    badge(bool(monzo_files) or bool(settings.monzo_sheet_id),
          f"{len(monzo_files)} export file(s)" if monzo_files else ("Google Sheets" if settings.monzo_sheet_id else ""))
    st.markdown("In the Monzo app: **Account → Export transactions → CSV**, pick a date range and upload it here. "
                "Overlapping exports are fine - duplicates are removed.")
    files = st.file_uploader("Monzo export (.xlsx from Google Drive, or .csv)", type=["xlsx", "csv"],
                             accept_multiple_files=True, key="monzo_upload")
    if files and st.button("Import Monzo file(s)", type="primary"):
        save_uploads(files, settings.monzo_import_dir)
        run_sync("monzo", 3650)
    with st.expander("Advanced: automatic sync via Monzo's Google Sheets export"):
        st.markdown("Requires Monzo Plus/Premium and a Google Cloud *service account* with the Sheets and Drive APIs "
                    "enabled. Share the Monzo spreadsheet with the service account's e-mail.")
        with st.form("monzo_sheets"):
            sheet_id = st.text_input("Spreadsheet ID (from the sheet URL)", value=settings.monzo_sheet_id or "")
            key_file = st.file_uploader("Service-account JSON key", type=["json"])
            if st.form_submit_button("Save Google Sheets settings"):
                values = {"monzo_sheet_id": sheet_id.strip() or None}
                if key_file is not None:
                    path = save_uploads([key_file], settings.data_dir / "secrets")[0]
                    values["google_service_account_file"] = str(path)
                save(values, "Monzo Sheets settings saved")

# ------------------------------------------------------------------------------ samsung
with section("📱 Samsung S24 · Health Connect", "Steps, calories, distance, resting HR, weight - and MyFitnessPal meals."):
    folder = settings.android_export_dir
    count = sum(1 for p in folder.rglob("*") if p.is_file()) if folder and folder.exists() else 0
    badge(count > 0, f"{count} file(s) in {folder}")
    st.markdown(
        "**Best: Health Connect scheduled export.** On the phone: Health Connect → Export and import → "
        "Scheduled export → Google Drive. It includes everything apps write to Health Connect - Samsung steps, "
        "WHOOP, Renpho and MyFitnessPal meals. Upload the *Health Connect.zip* here, or set the synced folder "
        "below to the Drive folder so new exports are picked up automatically.\n\n"
        "Also accepted: a **Samsung Health → Download personal data** zip."
    )
    files = st.file_uploader("Health Connect / Samsung Health export (.zip) or CSV", type=["zip", "db", "csv", "json", "tsv", "jsonl"],
                             accept_multiple_files=True, key="android_upload")
    if files and st.button("Import Samsung file(s)", type="primary"):
        saved = save_uploads(files, folder)
        seen = [p.name for p in saved if db.is_file_ingested(file_sha256(p))]
        if len(seen) == len(saved):
            st.info(f"Already imported: {', '.join(seen)} - that data is up to date. "
                    "Upload a newer export to add recent days.")
        else:
            if seen:
                st.caption(f"Skipping already-imported: {', '.join(seen)}")
            run_sync("android_s24", 3650)
    with st.expander("Hands-free: read from a synced folder"):
        st.markdown("Point this at the folder your export app writes to (e.g. a Google Drive for desktop folder) "
                    "and every sync picks up new files automatically.")
        with st.form("android_folder"):
            new_folder = st.text_input("Folder path", value=str(folder))
            if st.form_submit_button("Save folder"):
                if Path(new_folder).expanduser().is_dir():
                    save({"android_export_dir": str(Path(new_folder).expanduser())}, "Folder saved")
                else:
                    st.error("That folder doesn't exist on this computer.")

# ------------------------------------------------------------------------------ stayfree
with section("📊 Phone screen time · StayFree", "Per-app phone time, app launches and daily unlocks."):
    stayfree_dir = folder / "stayfree"
    reports = sorted(p for p in stayfree_dir.glob("*.xls*")) if stayfree_dir.exists() else []
    badge(bool(reports), f"{len(reports)} report(s) imported" if reports else "")
    st.markdown("In **StayFree → Settings → Export data**, export the *Total Usage* report (.xls) and upload it here. "
                "Each export contains the full history, so just upload the newest one.")
    files = st.file_uploader("StayFree report (.xls / .xlsx)", type=["xls", "xlsx"],
                             accept_multiple_files=True, key="stayfree_upload")
    if files and st.button("Import StayFree report", type="primary"):
        saved = save_uploads(files, stayfree_dir)
        if all(db.is_file_ingested(file_sha256(p)) for p in saved):
            st.info("That report was already imported - upload a newer export to add recent days.")
        else:
            run_sync("android_s24", 3650)

# ------------------------------------------------------------------------------ activitywatch
with section("💻 Laptop · ActivityWatch", "App usage and coding/work time on this computer."):
    aw_missing = get_connector("activitywatch", settings, db).missing_config()
    badge(not aw_missing, "found running on this computer" if not aw_missing else "")
    if aw_missing:
        st.markdown("Install the free, open-source [ActivityWatch](https://activitywatch.net/downloads/) and leave it "
                    "running - it's detected automatically, no login needed.")
        if st.button("Check again"):
            st.rerun()
    elif st.button("Sync last 30 days", type="primary"):
        run_sync("activitywatch", 30)

# ------------------------------------------------------------------------------ AI
with section("🤖 AI summaries", "Who writes the weekly executive summary."):
    choice = st.radio("Provider", ["ollama", "anthropic"], horizontal=True,
                      index=1 if settings.llm_provider == "anthropic" else 0,
                      format_func=lambda p: {"ollama": "Ollama (free, runs on this computer)",
                                             "anthropic": "Claude (your Anthropic API key)"}[p])
    if choice == "ollama":
        try:
            models = OllamaProvider(host=settings.ollama_host).list_models()
        except Exception:  # noqa: BLE001 - not installed / not running
            models = []
        if not models:
            st.markdown(f"No Ollama server found at `{settings.ollama_host}`. Install "
                        "[Ollama](https://ollama.com/download) and run `ollama pull llama3.1:8b` - or, if it runs on "
                        "another computer on your network, enter that computer's address below.")
            if st.button("Check again", key="ollama_check"):
                st.rerun()
        with st.form("ollama_host"):
            host = st.text_input("Ollama server address", value=settings.ollama_host,
                                 help="e.g. http://192.168.1.20:11434 for another PC on your home network. Ollama "
                                      "has no authentication: on that PC set OLLAMA_HOST to its LAN address only, "
                                      "allow port 11434 from your local network in its firewall, and never forward "
                                      "that port to the internet.")
            if st.form_submit_button("Save address"):
                save({"ollama_host": host.strip().rstrip("/")}, "Ollama address saved")
        if models:
            badge(True, f"{len(models)} local model(s)")
            index = models.index(settings.ollama_model) if settings.ollama_model in models else 0
            model = st.selectbox("Model", models, index=index)
            if st.button("Use Ollama", type="primary"):
                save({"llm_provider": "ollama", "ollama_model": model}, "AI provider set to Ollama")
    else:
        badge(bool(settings.anthropic_api_key), "API key saved" if settings.anthropic_api_key else "")
        st.markdown("Create a key at [console.anthropic.com](https://console.anthropic.com/settings/keys). "
                    "Only aggregated numbers are sent - never notes, transactions or app names.")
        with st.form("anthropic"):
            key = st.text_input("API key", type="password",
                                placeholder="saved ✓ (leave blank to keep)" if settings.anthropic_api_key else "sk-ant-…")
            if st.form_submit_button("Use Claude", type="primary"):
                values = {"llm_provider": "anthropic"}
                if key.strip():
                    values["anthropic_api_key"] = key.strip()
                save(values, "AI provider set to Claude")
        if settings.anthropic_api_key:
            st.caption(SECRET_NOTE)
