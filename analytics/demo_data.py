"""Synthetic demo data with realistic, deliberately embedded cross-domain effects.

Lets you explore the dashboard (and validate the analytics) before connecting any
real source: ``python cli.py demo --days 180``. Embedded relationships include:

* phone screen time on day D lowers WHOOP sleep performance on D+1,
* sleep drives next-morning recovery, HRV and subjective energy,
* calorie surplus drives weight change; protein intake slowly lowers body fat,
* rain raises eating-out spend; sunshine raises steps; daylight lifts mood.

Writes into whatever database it is given - point ``DATA_DIR`` at a scratch
folder if you want to keep demo data separate from real data.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from core.database import Database


def generate_demo_frames(end: date, days: int = 180, seed: int = 7) -> dict[str, pd.DataFrame]:
    """Return ``{table: DataFrame}`` of synthetic data for ``days`` days ending at ``end``."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end - timedelta(days=days - 1), end, freq="D")
    n = len(dates)
    doy = dates.dayofyear.to_numpy()
    weekend = np.asarray(dates.dayofweek >= 5)

    # Weather (London-ish seasonality).
    season = np.cos((doy - 200) / 365 * 2 * np.pi)
    temp_mean = 11 + 7 * season + rng.normal(0, 2.5, n)
    rain = np.where(rng.random(n) < 0.45, rng.gamma(1.4, 3.0, n), 0.0)
    daylight = 12.2 + 4.3 * season
    sunshine = np.clip(daylight * rng.beta(2, 3, n) - rain * 0.3, 0, None)
    weather = pd.DataFrame(
        {
            "date": dates.date,
            "temp_mean_c": temp_mean,
            "temp_max_c": temp_mean + rng.uniform(3, 6, n),
            "temp_min_c": temp_mean - rng.uniform(3, 6, n),
            "apparent_temp_mean_c": temp_mean - 1.5,
            "precipitation_mm": rain,
            "rain_mm": rain,
            "precipitation_hours": np.minimum(rain * 1.2, 24),
            "daylight_hours": daylight,
            "sunshine_hours": sunshine,
            "pressure_msl_hpa": 1013 + rng.normal(0, 8, n) - rain * 0.8,
            "humidity_mean_pct": np.clip(72 + rain * 1.5 + rng.normal(0, 6, n), 30, 100),
            "cloud_cover_mean_pct": np.clip(100 - sunshine / daylight * 100 + rng.normal(0, 8, n), 0, 100),
            "wind_speed_max_kmh": rng.gamma(4, 5, n),
        }
    )

    # Screen time and laptop.
    phone_minutes = np.clip(rng.normal(190, 55, n) + weekend * 40 + rain * 3, 30, None)
    social_share = rng.uniform(0.3, 0.5, n)
    laptop_active = np.where(weekend, rng.normal(90, 40, n), rng.normal(430, 60, n)).clip(0)
    coding = laptop_active * rng.uniform(0.35, 0.6, n)
    last_active_hour = np.where(weekend, rng.normal(20, 2, n), rng.normal(19.5, 1.5, n)) + phone_minutes / 400

    # Sleep depends on the previous day's phone use and late laptop use.
    prev_phone = np.roll(phone_minutes, 1)
    prev_phone[0] = phone_minutes.mean()
    prev_late = np.roll(last_active_hour, 1)
    prev_late[0] = last_active_hour.mean()
    sleep_perf = np.clip(88 - 0.07 * (prev_phone - 190) - 2.5 * (prev_late - 20) + rng.normal(0, 6, n), 35, 100)
    sleep_hours = np.clip(7.3 * sleep_perf / 85 + rng.normal(0, 0.4, n), 4, 10)
    recovery = np.clip(25 + 0.55 * sleep_perf + rng.normal(0, 10, n), 1, 99)
    hrv = np.clip(35 + 0.35 * recovery + rng.normal(0, 6, n), 15, 140)
    strain = np.clip(rng.normal(11, 3, n) + weekend * 2.5, 2, 21)
    kilojoules = 7600 + strain * 380 + rng.normal(0, 600, n)

    # Nutrition, body composition.
    calories = np.clip(rng.normal(2350, 320, n) + strain * 25 + weekend * 250, 1200, None)
    protein = np.clip(rng.normal(140, 28, n), 50, None)
    weight = np.empty(n)
    body_fat = np.empty(n)
    weight[0], body_fat[0] = 82.0, 19.0
    expenditure = kilojoules / 4.184
    for i in range(1, n):
        weight[i] = weight[i - 1] + (calories[i - 1] - expenditure[i - 1]) / 7700 + rng.normal(0, 0.05)
        body_fat[i] = body_fat[i - 1] + (calories[i - 1] - expenditure[i - 1]) / 25000 - (protein[i - 1] - 140) / 9000
    weighed = rng.random(n) < 0.7
    noise_w = rng.normal(0, 0.35, n)
    renpho = pd.DataFrame(
        {
            "date": dates.date,
            "weight_kg": weight + noise_w,
            "body_fat_pct": body_fat + rng.normal(0, 0.4, n),
            "muscle_mass_pct": 100 - body_fat - 38 + rng.normal(0, 0.3, n),
            "visceral_fat": 8 + (body_fat - 19) * 0.5,
            "body_water_pct": 57 - (body_fat - 19) * 0.6,
            "bmr_kcal": 1750 + (weight - 82) * 14,
        }
    )
    renpho["bmi"] = renpho["weight_kg"] / 1.80**2
    renpho["muscle_mass_kg"] = renpho["weight_kg"] * renpho["muscle_mass_pct"] / 100
    renpho["measurement_count"] = 1
    renpho["source"] = "demo"
    renpho = renpho.loc[weighed]

    mfp = pd.DataFrame(
        {
            "date": dates.date,
            "calories_kcal": calories,
            "protein_g": protein,
            "carbs_g": (calories - protein * 4) * 0.55 / 4,
            "fat_g": (calories - protein * 4) * 0.45 / 9,
            "fiber_g": rng.normal(28, 7, n).clip(5),
            "sugar_g": rng.normal(60, 18, n).clip(5),
            "sodium_mg": rng.normal(2400, 500, n).clip(500),
            "water_ml": rng.normal(2200, 450, n).clip(500),
            "source": "demo",
        }
    )

    steps = np.clip(7500 + 650 * sunshine - rain * 120 + rng.normal(0, 2200, n) + strain * 150, 800, None)
    s24 = pd.DataFrame(
        {
            "date": dates.date,
            "steps": steps.round().astype(int),
            "active_calories_kcal": steps * 0.045 + rng.normal(0, 60, n),
            "total_calories_kcal": expenditure,
            "distance_m": steps * 0.78,
            "resting_heart_rate_bpm": 62 - 0.12 * (recovery - 60) + rng.normal(0, 2, n),
        }
    )

    # Screen usage rows.
    usage_rows = []
    phone_apps = {"Instagram": "social", "WhatsApp": "social", "YouTube": "entertainment",
                  "Chrome": "browsing", "Gmail": "productivity", "Spotify": "entertainment"}
    for i, d in enumerate(dates.date):
        social = phone_minutes[i] * social_share[i]
        rest = phone_minutes[i] - social
        split = {"Instagram": social * 0.6, "WhatsApp": social * 0.4, "YouTube": rest * 0.4,
                 "Chrome": rest * 0.3, "Gmail": rest * 0.1, "Spotify": rest * 0.2}
        for app, minutes in split.items():
            usage_rows.append({"date": d, "device": "s24_ultra", "app": app, "category": phone_apps[app],
                               "minutes": round(minutes, 1), "source": "demo"})
        usage_rows += [
            {"date": d, "device": "laptop", "app": "Code.exe", "category": "coding", "minutes": round(coding[i], 1),
             "source": "demo"},
            {"date": d, "device": "laptop", "app": "chrome.exe", "category": "browsing",
             "minutes": round((laptop_active[i] - coding[i]) * 0.6, 1), "source": "demo"},
            {"date": d, "device": "laptop", "app": "Slack.exe", "category": "communication",
             "minutes": round((laptop_active[i] - coding[i]) * 0.4, 1), "source": "demo"},
        ]
    tz_midnight = [datetime.combine(d, datetime.min.time(), UTC) for d in dates.date]
    laptop = pd.DataFrame(
        {
            "date": dates.date,
            "active_minutes": laptop_active,
            "afk_minutes": rng.normal(120, 40, n).clip(0),
            "coding_minutes": coding,
            "work_minutes": np.where(weekend, coding * 0.5, laptop_active * 0.9),
            "first_active": [m + timedelta(hours=8.5) for m in tz_midnight],
            "last_active": [m + timedelta(hours=float(h)) for m, h in zip(tz_midnight, last_active_hour, strict=True)],
            "top_app": "Code.exe",
        }
    )

    # Spending.
    spend_rows = []
    for i, d in enumerate(dates.date):
        if rng.random() < 0.35:
            spend_rows.append({"date": d, "category": "groceries", "amount_gbp": round(rng.gamma(3, 12), 2),
                               "transaction_count": 1})
        if rng.random() < 0.25 + min(rain[i], 10) * 0.05:
            spend_rows.append({"date": d, "category": "eating_out",
                               "amount_gbp": round(rng.gamma(2.5, 9) + rain[i] * 1.5, 2), "transaction_count": 1})
        if not weekend[i]:
            spend_rows.append({"date": d, "category": "transport", "amount_gbp": round(5.6 + rng.normal(0, 1), 2),
                               "transaction_count": 2})
        if rng.random() < 0.12:
            spend_rows.append({"date": d, "category": "shopping", "amount_gbp": round(rng.gamma(2, 25), 2),
                               "transaction_count": 1})

    subjective = pd.DataFrame(
        {
            "date": dates.date,
            "mood": np.clip(np.round(2.2 + 0.12 * daylight + 0.01 * (sleep_perf - 80) + rng.normal(0, 0.6, n)), 1, 5),
            "energy": np.clip(np.round(1 + recovery / 30 + rng.normal(0, 0.6, n)), 1, 5),
            "focus": np.clip(np.round(1.2 + hrv / 35 + rng.normal(0, 0.7, n)), 1, 5),
        }
    ).astype({"mood": int, "energy": int, "focus": int})
    subjective = subjective.loc[rng.random(n) < 0.85]

    return {
        "weather_daily": weather,
        "whoop_sleep": pd.DataFrame(
            {"date": dates.date, "sleep_performance_pct": sleep_perf, "total_sleep_hours": sleep_hours,
             "deep_sleep_hours": sleep_hours * 0.2, "rem_sleep_hours": sleep_hours * 0.24,
             "light_sleep_hours": sleep_hours * 0.56, "sleep_efficiency_pct": np.clip(sleep_perf + 5, 0, 100),
             "respiratory_rate": rng.normal(15, 0.6, n), "disturbance_count": rng.poisson(9, n)}
        ),
        "whoop_recovery": pd.DataFrame(
            {"date": dates.date, "recovery_score": recovery, "hrv_rmssd_ms": hrv,
             "resting_heart_rate_bpm": 62 - 0.12 * (recovery - 60), "spo2_pct": rng.normal(96.5, 0.8, n),
             "skin_temp_c": rng.normal(33.8, 0.3, n)}
        ),
        "whoop_cycle": pd.DataFrame(
            {"date": dates.date, "strain": strain, "kilojoules": kilojoules, "avg_heart_rate_bpm": 68 + strain}
        ),
        "renpho_body": renpho,
        "mfp_nutrition": mfp,
        "s24_health": s24,
        "screen_usage": pd.DataFrame(usage_rows),
        "laptop_activity": laptop,
        "monzo_spending": pd.DataFrame(spend_rows),
        "subjective_state": subjective,
    }


def seed_demo_data(db: Database, end: date | None = None, days: int = 180, seed: int = 7) -> dict[str, int]:
    """Write synthetic data into ``db`` and return rows written per table."""
    db.init_schema()
    frames = generate_demo_frames(end or date.today(), days, seed)
    return {table: db.upsert(table, frame, overwrite_nulls=True) for table, frame in frames.items()}
