"""Transform-level tests for every connector (no network access required)."""

from __future__ import annotations

import json
import zipfile
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from connectors.activitywatch import ActivityWatchConnector, ActivityWatchRaw
from connectors.android_s24 import (
    AndroidS24Connector,
    aggregate_health_daily,
    parse_health_connect_frame,
    parse_screen_usage_frame,
)
from connectors.file_parser import FileParserConnector, detect_table, map_columns
from connectors.monzo_sheets import aggregate_monzo_transactions
from connectors.myfitnesspal import MyFitnessPalConnector, parse_nutrition_frame, read_mfp_export
from connectors.renpho import renpho_measurements_to_daily
from connectors.utils import categorize_app, find_column, parse_duration_to_minutes
from connectors.weather import WeatherConnector, WeatherRaw
from connectors.whoop import WhoopConnector, WhoopRaw, _local_date
from core.config import DEFAULT_EXCLUDED_MONZO_CATEGORIES, DEFAULT_MONZO_CATEGORY_MAP

LONDON = ZoneInfo("Europe/London")


# ------------------------------------------------------------------ utils
@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [(90, "s", 1.5), ("1h 30m", "min", 90), ("01:15:30", "min", 75.5), ("45 min", "min", 45),
     ("2 hrs 5 mins", "min", 125), (120000, "ms", 2), ("", "min", None), ("abc", "min", None)],
)
def test_parse_duration(value, unit, expected) -> None:
    assert parse_duration_to_minutes(value, unit) == expected


def test_find_column_prefers_exact() -> None:
    frame = pd.DataFrame(columns=["Saturated Fat", "Fat (g)"])
    assert find_column(frame, ["fat_g", "fat"]) == "Fat (g)"


def test_categorize_app() -> None:
    rules = {"social": [r"instagram"], "coding": [r"code"]}
    assert categorize_app("com.instagram.android", rules) == "social"
    assert categorize_app("Notepad", rules) == "other"


# ------------------------------------------------------------------ whoop
def test_whoop_local_date_uses_offset() -> None:
    # 23:30 UTC with +02:00 offset is the next local day.
    assert _local_date("2026-03-01T23:30:00.000Z", "+02:00") == date(2026, 3, 2)
    assert _local_date("2026-03-01T03:30:00.000Z", "-05:00") == date(2026, 2, 28)


def test_whoop_transform(settings, db) -> None:
    raw = WhoopRaw(
        cycles=[{"id": 10, "start": "2026-03-02T06:00:00.000Z", "timezone_offset": "+00:00", "score_state": "SCORED",
                 "score": {"strain": 12.5, "kilojoule": 9000, "average_heart_rate": 70, "max_heart_rate": 160}}],
        recoveries=[{"cycle_id": 10, "sleep_id": "abc", "score_state": "SCORED", "created_at": "2026-03-02T07:00:00Z",
                     "score": {"recovery_score": 66, "hrv_rmssd_milli": 55.1, "resting_heart_rate": 52,
                               "spo2_percentage": 96.5, "skin_temp_celsius": 33.7, "user_calibrating": False}}],
        sleeps=[
            {"id": "s1", "start": "2026-03-01T23:00:00Z", "end": "2026-03-02T06:30:00Z", "timezone_offset": "+00:00",
             "nap": False, "score_state": "SCORED",
             "score": {"stage_summary": {"total_in_bed_time_milli": 27_000_000, "total_awake_time_milli": 1_800_000,
                                         "total_light_sleep_time_milli": 12_600_000,
                                         "total_slow_wave_sleep_time_milli": 5_400_000,
                                         "total_rem_sleep_time_milli": 7_200_000, "disturbance_count": 8,
                                         "sleep_cycle_count": 4},
                       "sleep_performance_percentage": 91, "sleep_efficiency_percentage": 93,
                       "sleep_consistency_percentage": 80, "respiratory_rate": 15.2}},
            {"id": "nap", "start": "2026-03-02T13:00:00Z", "end": "2026-03-02T13:30:00Z", "timezone_offset": "+00:00",
             "nap": True, "score_state": "SCORED", "score": {}},
        ],
    )
    frames = WhoopConnector(settings, db).transform(raw)
    assert frames["whoop_recovery"].iloc[0]["date"] == date(2026, 3, 2)
    assert frames["whoop_cycle"].iloc[0]["strain"] == 12.5
    sleep = frames["whoop_sleep"]
    assert len(sleep) == 1 and sleep.iloc[0]["total_sleep_hours"] == pytest.approx(7.0)
    assert sum(WhoopConnector(settings, db).load(frames).values()) == 3


# ------------------------------------------------------------------ weather
def test_weather_transform(settings, db) -> None:
    payload = {
        "latitude": 51.5, "longitude": -0.12,
        "daily": {"time": ["2026-03-01", "2026-03-02"], "temperature_2m_max": [10, 12], "temperature_2m_min": [3, 4],
                  "precipitation_sum": [0, 4.2], "rain_sum": [0, 4.2], "precipitation_hours": [0, 3],
                  "daylight_duration": [39600, 40000], "sunshine_duration": [18000, 3600], "wind_speed_10m_max": [12, 20]},
        "hourly": {"time": ["2026-03-01T00:00", "2026-03-01T12:00", "2026-03-02T00:00"],
                   "temperature_2m": [4, 8, 6], "apparent_temperature": [2, 6, 4], "pressure_msl": [1010, 1012, 1000],
                   "relative_humidity_2m": [80, 70, 90], "cloud_cover": [10, 30, 100]},
    }
    frame = WeatherConnector(settings, db).transform(WeatherRaw([payload]))["weather_daily"]
    first = frame.iloc[0]
    assert first["daylight_hours"] == pytest.approx(11.0)
    assert first["pressure_msl_hpa"] == pytest.approx(1011)
    assert first["temp_mean_c"] == pytest.approx(6)


# ------------------------------------------------------------------ monzo
def test_monzo_aggregation() -> None:
    tx = pd.DataFrame({
        "Transaction ID": list("abcdef"),
        "Date": ["01/03/2026", "01/03/2026", "01/03/2026", "02/03/2026", "02/03/2026", "02/03/2026"],
        "Type": ["Card payment", "Card payment", "Pot transfer", "Card payment", "Faster payment", "Card payment"],
        "Category": ["Groceries", "Eating out", "Savings", "Groceries", "Income", "Groceries"],
        "Amount": ["-12.50", "-20.00", "-100.00", "-5.00", "1500.00", "2.50"],
        "Currency": ["GBP"] * 6,
    })
    daily = aggregate_monzo_transactions(tx, DEFAULT_MONZO_CATEGORY_MAP, DEFAULT_EXCLUDED_MONZO_CATEGORIES)
    lookup = {(r.date, r.category): r.amount_gbp for r in daily.itertuples()}
    assert lookup[(date(2026, 3, 1), "groceries")] == 12.5
    assert lookup[(date(2026, 3, 1), "eating_out")] == 20.0
    assert lookup[(date(2026, 3, 2), "groceries")] == 2.5  # refund netted
    assert not any(cat in {"savings", "income"} for _, cat in lookup)


# ------------------------------------------------------------------ android / health connect
def test_health_connect_long_format() -> None:
    frame = pd.DataFrame({
        "type": ["Steps", "Steps", "Steps", "Weight", "ActiveCaloriesBurned"],
        "start_time": ["2026-03-01T08:00:00Z", "2026-03-01T18:00:00Z", "2026-03-01T09:00:00Z",
                       "2026-03-01T07:00:00Z", "2026-03-01T10:00:00Z"],
        "value": [3000, 2000, 4500, 80.2, 300],
        "data_origin": ["com.sec.android.app.shealth", "com.sec.android.app.shealth", "com.samsung.watch",
                        "com.renpho", "com.sec.android.app.shealth"],
    })
    daily = aggregate_health_daily(parse_health_connect_frame(frame, "export", LONDON))
    row = daily.iloc[0]
    assert row["steps"] == 5000  # max of per-source totals (phone 5000 vs watch 4500), not 9500
    assert row["weight_kg"] == pytest.approx(80.2)
    assert row["active_calories_kcal"] == 300


def test_health_connect_per_metric_file_with_grams() -> None:
    frame = pd.DataFrame({"Date": ["2026-03-01", "2026-03-02"], "Time": ["07:00", "07:10"],
                          "Weight in grams": [80500, 80100]})
    daily = aggregate_health_daily(parse_health_connect_frame(frame, "Weight", LONDON))
    assert daily["weight_kg"].tolist() == pytest.approx([80.5, 80.1])


def test_health_connect_wide_file() -> None:
    frame = pd.DataFrame({"date": ["2026-03-01"], "Steps": [9000], "Active calories": [420], "Distance (km)": [6.5]})
    daily = aggregate_health_daily(parse_health_connect_frame(frame, "daily_summary", LONDON))
    assert daily.iloc[0]["distance_m"] == pytest.approx(6500)
    assert daily.iloc[0]["steps"] == 9000


def test_screen_usage_parsing_infers_seconds() -> None:
    frame = pd.DataFrame({"date": ["2026-03-01"] * 3, "app_name": ["Instagram", "Instagram", "Gmail"],
                          "usage": [1800, 1800, 600]})
    rows = parse_screen_usage_frame(frame, LONDON, {"social": [r"instagram"]})
    insta = rows.loc[rows["app"] == "Instagram"].iloc[0]
    assert insta["minutes"] == 60 and insta["category"] == "social" and insta["device"] == "s24_ultra"


def test_android_connector_end_to_end_is_idempotent(settings, db) -> None:
    folder = settings.android_export_dir
    pd.DataFrame({"Date": ["2026-03-01"], "Time": ["12:00"], "Steps": [7000]}).to_csv(folder / "Steps.csv", index=False)
    pd.DataFrame({"date": ["2026-03-01"], "app": ["YouTube"], "usage_minutes": [42]}).to_csv(
        folder / "screen_time.csv", index=False)
    first = AndroidS24Connector(settings, db).run(date(2026, 3, 1), date(2026, 3, 1))
    assert first.status == "success" and first.rows_by_table == {"s24_health": 1, "screen_usage": 1}
    second = AndroidS24Connector(settings, db).run(date(2026, 3, 1), date(2026, 3, 1))
    assert second.rows_loaded == 0  # unchanged files are skipped


# ------------------------------------------------------------------ renpho
def test_renpho_measurements_to_daily() -> None:
    measurements = [
        {"timeStamp": 1772352000, "weight": 81.0, "bodyfat": 20.0, "muscle": 40.0, "visfat": 9, "bmi": 25.0},
        {"timeStamp": 1772380800, "weight": 80.6, "bodyfat": 19.8, "muscle": 40.5, "visfat": 9, "bmi": 24.9,
         "bone": 0},
    ]
    daily = renpho_measurements_to_daily(measurements, LONDON)
    row = daily.iloc[0]
    assert len(daily) == 1 and row["measurement_count"] == 2
    assert row["weight_kg"] == 80.6  # last weigh-in of the day
    assert row["muscle_mass_kg"] == pytest.approx(32.64, abs=0.01)
    assert pd.isna(row["bone_mass_pct"])  # zero means "not measured"


# ------------------------------------------------------------------ myfitnesspal
def _mfp_csv() -> pd.DataFrame:
    return pd.DataFrame({
        "Date": ["2026-03-01", "2026-03-01", "2026-03-02"], "Meal": ["Breakfast", "Dinner", "Lunch"],
        "Calories": [500, 900, 700], "Fat (g)": [20, 30, 25], "Saturated Fat": [5, 9, 7],
        "Carbohydrates (g)": [60, 90, 70], "Protein (g)": [30, 60, 45], "Fiber": [8, 6, 5],
        "Sugar": [10, 12, 8], "Sodium (mg)": [400, 900, 600],
    })


def test_mfp_nutrition_summary_parsing() -> None:
    daily = parse_nutrition_frame(_mfp_csv(), LONDON)
    first = daily.loc[daily["date"] == date(2026, 3, 1)].iloc[0]
    assert first["calories_kcal"] == 1400 and first["fat_g"] == 50 and first["protein_g"] == 90


def test_mfp_connector_reads_export_zip(settings, db, tmp_path: Path) -> None:
    archive = settings.mfp_export_dir / "File-Export-2026.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Nutrition-Summary-2026-03-01-to-2026-03-02.csv", _mfp_csv().to_csv(index=False))
        zf.writestr("Measurement-Summary.csv", "Date,Weight\n2026-03-01,80\n")
    assert len(read_mfp_export(archive)) == 1
    result = MyFitnessPalConnector(settings, db).run(date(2026, 3, 1), date(2026, 3, 2))
    assert result.status == "success" and result.rows_loaded == 2


# ------------------------------------------------------------------ activitywatch
def test_activitywatch_transform(settings, db) -> None:
    raw = ActivityWatchRaw(
        days=[date(2026, 3, 2)],
        results=[{
            "apps": [{"duration": 3600, "data": {"app": "Code.exe"}}, {"duration": 1800, "data": {"app": "Slack.exe"}},
                     {"duration": 900, "data": {"app": "chrome.exe"}}],
            "not_afk": [{"timestamp": "2026-03-02T08:00:00+00:00", "duration": 7200},
                        {"timestamp": "2026-03-02T21:00:00+00:00", "duration": 600}],
            "afk_seconds": 3600, "editor_seconds": 4000,
        }],
    )
    frames = ActivityWatchConnector(settings, db).transform(raw)
    activity = frames["laptop_activity"].iloc[0]
    assert activity["coding_minutes"] == pytest.approx(4000 / 60, abs=0.01)  # editor bucket wins
    assert activity["active_minutes"] == 130
    assert activity["top_app"] == "Code.exe"
    assert set(frames["screen_usage"]["category"]) >= {"coding", "communication", "browsing"}


def test_activitywatch_query_mentions_buckets() -> None:
    from connectors.activitywatch import AWBuckets

    query = "\n".join(ActivityWatchConnector.build_query(AWBuckets("aw-watcher-window_host", "aw-watcher-afk_host")))
    assert "aw-watcher-window_host" in query and "RETURN" in query


# ------------------------------------------------------------------ file drops
def test_file_parser_maps_and_moves(settings, db) -> None:
    folder = settings.inbox_dir / "subjective_state"
    folder.mkdir(parents=True)
    path = folder / "journal.csv"
    pd.DataFrame({"Day": ["2026-03-01"], "Mood": [4], "Energy": [3], "Focus": [5], "Journal": ["ok"]}).to_csv(path, index=False)
    result = FileParserConnector(settings, db).run()
    assert result.rows_by_table == {"subjective_state": 1}
    assert not path.exists() and any((settings.processed_dir / "subjective_state").iterdir())
    assert db.read_table("subjective_state").iloc[0]["notes"] == "ok"


def test_file_parser_detects_mfp_and_json(settings) -> None:
    inbox = settings.inbox_dir
    assert detect_table(inbox / "x.csv", inbox, _mfp_csv()) == "mfp_nutrition"
    assert detect_table(inbox / "weather_daily_backfill.json", inbox, pd.DataFrame({"date": []})) == "weather_daily"
    (inbox / "subjective_state_export.json").write_text(json.dumps({"records": [{"date": "2026-03-01", "mood": 2}]}))
    mapped = map_columns(pd.DataFrame({"date": ["2026-03-01"], "mood": [2]}), "subjective_state")
    assert list(mapped.columns) == ["date", "mood"]


# ------------------------------------------------------------------ whoop browser sign-in
def _whoop_ready(settings):
    from pydantic import SecretStr

    settings.whoop_client_id = "cid"
    settings.whoop_client_secret = SecretStr("secret")
    return settings


def test_whoop_browser_flow_validates_state(settings, db, monkeypatch) -> None:
    connector = WhoopConnector(_whoop_ready(settings), db)
    exchanged: dict = {}
    monkeypatch.setattr(connector, "_exchange", lambda data: exchanged.update(data) or "token")

    url = connector.begin_authorization()
    assert "redirect_uri=http%3A%2F%2Flocalhost%3A8501%2Fconnections" in url
    state = url.split("state=")[1].split("&")[0]
    connector.complete_authorization("the-code", state)
    assert exchanged["code"] == "the-code"

    from connectors.base import ConnectorError

    with pytest.raises(ConnectorError, match="No WHOOP sign-in"):  # state is single-use
        connector.complete_authorization("the-code", state)
    connector.begin_authorization()
    with pytest.raises(ConnectorError, match="state mismatch"):
        connector.complete_authorization("the-code", "forged")


# ------------------------------------------------------------------ samsung health download
def _samsung_zip(path: Path) -> Path:
    steps = ("com.samsung.shealth.step_daily_trend,7010053,6\n"
             "binning_data,update_time,create_time,source_pkg_name,source_type,count,speed,distance,calorie,deviceuuid,pkg_name,datauuid,day_time\n"
             "x,2026-10-18 20:00:00.000,2026-10-18 07:00:00.000,com.sec.android.app.shealth,0,6000,1.3,4700,190,A,p,u1,2026-10-18 00:00:00.000,\n"
             "x,2026-10-18 20:00:00.000,2026-10-18 07:00:00.000,com.sec.android.app.shealth,-2,6691,1.3,5280.4,203,B,p,u2,2026-10-18 00:00:00.000,\n"
             "x,2026-10-19 20:00:00.000,2026-10-19 07:00:00.000,com.sec.android.app.shealth,0,5508,1.3,4368.6,168,A,p,u3,2026-10-19 00:00:00.000,\n")
    calories = ("com.samsung.shealth.calories_burned.details,7010053,4\n"
                "active_calories_goal,version,extra_data,exercise_calories,total_exercise_calories,"
                "com.samsung.shealth.calories_burned.tef_calorie,com.samsung.shealth.calories_burned.rest_calorie,"
                "com.samsung.shealth.calories_burned.active_calorie,com.samsung.shealth.calories_burned.day_time\n"
                ",,,,190.0,10.0,1565.3,205.1,2026-10-18 00:00:00.000,\n")
    archive = path / "Samsung health.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Samsung health/com.samsung.shealth.step_daily_trend.20261020183425.csv", steps)
        zf.writestr("Samsung health/com.samsung.shealth.calories_burned.details.20261020183425.csv", calories)
        zf.writestr("Samsung health/com.samsung.health.user_profile.20261020183425.csv", "com.samsung.health.user_profile,1,1\nkey,value\n")
    return archive


def test_samsung_health_zip_import(settings, db) -> None:
    _samsung_zip(settings.android_export_dir)
    result = AndroidS24Connector(settings, db).run(date(2026, 10, 18), date(2026, 10, 19))
    assert result.status == "success", result.error
    health = db.read_table("s24_health").assign(date=lambda f: pd.to_datetime(f["date"]).dt.date).set_index("date")
    assert health.loc[date(2026, 10, 18), "steps"] == 6691  # merged all-device row preferred
    assert health.loc[date(2026, 10, 19), "steps"] == 5508  # fallback to single device
    assert health.loc[date(2026, 10, 18), "active_calories_kcal"] == pytest.approx(205.1)
    assert health.loc[date(2026, 10, 18), "total_calories_kcal"] == pytest.approx(1780.4)


# ------------------------------------------------------------------ health connect scheduled export
def _health_connect_zip(folder: Path) -> Path:
    import sqlite3

    db_path = folder / "health_connect_export.db"
    con = sqlite3.connect(db_path)
    con.executescript("""
        CREATE TABLE application_info_table (row_id INTEGER, package_name TEXT);
        INSERT INTO application_info_table VALUES (3, 'com.myfitnesspal.android'), (8, 'com.whoop.android'), (9, 'phone');
        CREATE TABLE steps_record_table (local_date INTEGER, app_info_id INTEGER, count INTEGER, start_time INTEGER);
        INSERT INTO steps_record_table VALUES (20744, 9, 4000, 1), (20744, 9, 2691, 2), (20744, 8, 3012, 3);
        CREATE TABLE active_calories_burned_record_table (local_date INTEGER, app_info_id INTEGER, energy REAL);
        INSERT INTO active_calories_burned_record_table VALUES (20744, 9, 156688.0);
        CREATE TABLE weight_record_table (local_date INTEGER, time INTEGER, weight REAL);
        INSERT INTO weight_record_table VALUES (20744, 1, 73450.0), (20744, 2, 73300.0);
        CREATE TABLE nutrition_record_table (local_date INTEGER, app_info_id INTEGER, energy REAL, protein REAL,
            total_carbohydrate REAL, total_fat REAL, dietary_fiber REAL, sugar REAL, sodium REAL);
        INSERT INTO nutrition_record_table VALUES (20744, 3, 650000, 40, 70, 20, 8, 10, 0.9),
                                                  (20744, 3, 850000, 55, 90, 30, 6, 12, 1.1);
        CREATE TABLE hydration_record_table (local_date INTEGER, app_info_id INTEGER, volume REAL);
        INSERT INTO hydration_record_table VALUES (20744, 3, 1.5);
    """)
    con.close()
    archive = folder / "Health Connect.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(db_path, "health_connect_export.db")
    db_path.unlink()
    return archive


def test_health_connect_export_import(settings, db) -> None:
    _health_connect_zip(settings.android_export_dir)
    result = AndroidS24Connector(settings, db).run()
    assert result.status == "success", result.error
    day = date(2026, 10, 18)  # local_date 20744
    health = db.read_table("s24_health").assign(date=lambda f: pd.to_datetime(f["date"]).dt.date).set_index("date")
    assert health.loc[day, "steps"] == 6691  # phone total beats WHOOP; apps not summed
    assert health.loc[day, "active_calories_kcal"] == pytest.approx(156.688)
    assert health.loc[day, "weight_kg"] == pytest.approx(73.3)  # last reading of the day
    food = db.read_table("mfp_nutrition").assign(date=lambda f: pd.to_datetime(f["date"]).dt.date).set_index("date")
    assert food.loc[day, "calories_kcal"] == pytest.approx(1500)
    assert food.loc[day, "protein_g"] == 95 and food.loc[day, "sodium_mg"] == pytest.approx(2000)
    assert food.loc[day, "water_ml"] == pytest.approx(1500)


def test_whoop_local_date_accepts_z_and_compact_offsets() -> None:
    assert _local_date("2026-03-01T23:30:00.000Z", "Z") == date(2026, 3, 1)
    assert _local_date("2026-03-01T23:30:00.000Z", "+0100") == date(2026, 3, 2)


# ------------------------------------------------------------------ stayfree
def test_stayfree_report_import(settings, db) -> None:
    usage = pd.DataFrame([
        [None, "Device", "October 9, 2026", "October 10, 2026", "October 11, 2026", "Total Usage"],
        ["Instagram", "samsung SM-S928B", "0s", "1h 2m 30s", "45m", "1h 47m 30s"],
        ["Chrome", "samsung SM-S928B", "0s", "10m", "0s", "10m"],
    ])
    counts = usage.copy()
    counts.iloc[1, 2:5] = [0, 12, 9]
    counts.iloc[2, 2:5] = [0, 3, 0]
    unlocks = pd.DataFrame([
        [None, "October 9, 2026", "October 10, 2026", "October 11, 2026", "Total Usage"],
        ["Device Unlocks", 0, 80, 95, 175],
        ["Created by StayFree.", None, None, None, None],
    ])
    folder = settings.android_export_dir / "stayfree"
    folder.mkdir()
    path = folder / "StayFree Export - Total Usage - 10_11_26.xlsx"
    with pd.ExcelWriter(path) as writer:
        usage.to_excel(writer, sheet_name="Usage Time", header=False, index=False)
        counts.to_excel(writer, sheet_name="Usage Count", header=False, index=False)
        unlocks.to_excel(writer, sheet_name="Device Unlocks", header=False, index=False)

    db.init_schema()  # exercises the additive migration path too
    result = AndroidS24Connector(settings, db).run()
    assert result.status == "success", result.error
    screen = db.read_table("screen_usage")
    insta = screen.loc[screen["app"] == "Instagram"].sort_values("date")
    assert insta["minutes"].tolist() == pytest.approx([62.5, 45.0])  # zero day dropped
    assert insta["launches"].tolist() == [12, 9] and set(screen["device"]) == {"s24_ultra"}
    health = db.read_table("s24_health")
    assert health["phone_unlocks"].tolist() == [80, 95]  # pre-install zero day dropped
