"""Analytics tests on synthetic data with known embedded effects."""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from analytics.correlations import (
    benjamini_hochberg,
    build_insight_payload,
    correlation_matrix,
    detect_anomalies,
    evaluate_hypotheses,
    lagged_correlations,
    ranked_correlations,
    rolling_averages,
)
from analytics.demo_data import seed_demo_data
from analytics.pipeline import DailyMatrixBuilder, feature_families
from core.database import Database

END = date(2026, 6, 30)
DAYS = 150


@pytest.fixture()
def matrix(db: Database) -> pd.DataFrame:
    seed_demo_data(db, end=END, days=DAYS, seed=11)
    return DailyMatrixBuilder(db).build(END - timedelta(days=DAYS - 1), END)


def test_matrix_shape_and_domains(matrix: pd.DataFrame) -> None:
    assert len(matrix) == DAYS
    assert isinstance(matrix.index, pd.DatetimeIndex)
    prefixes = {c.split("_")[0] for c in matrix.columns}
    assert {"whoop", "renpho", "mfp", "s24", "screen", "laptop", "spend", "weather", "subjective"} <= prefixes


def test_body_composition_forward_filled(matrix: pd.DataFrame) -> None:
    # Demo weigh-ins happen ~70% of days; forward-fill closes the gaps.
    assert matrix["renpho_weight_kg"].iloc[1:].notna().mean() > 0.97


def test_spend_zero_filled_within_coverage(matrix: pd.DataFrame) -> None:
    assert matrix["spend_total_gbp"].notna().all()


def test_rolling_averages(matrix: pd.DataFrame) -> None:
    ma = rolling_averages(matrix[["s24_steps"]], windows=(7,))
    expected = matrix["s24_steps"].iloc[:7].mean()
    assert ma["s24_steps_ma7"].iloc[6] == pytest.approx(expected)


def test_correlation_matrix_symmetric(matrix: pd.DataFrame) -> None:
    corr = correlation_matrix(matrix)
    assert np.allclose(corr.values, corr.values.T, equal_nan=True)
    assert "calendar_is_weekend" not in corr.columns


def test_cross_domain_ranking_excludes_trivial_pairs(matrix: pd.DataFrame) -> None:
    ranked = ranked_correlations(matrix)
    for row in ranked.itertuples():
        assert not feature_families(row.feature_x) & feature_families(row.feature_y)
    assert ((ranked["q_value"] >= ranked["p_value"] - 1e-12) | ranked["q_value"].isna()).all()


def test_recovers_embedded_effects(matrix: pd.DataFrame) -> None:
    same_day = ranked_correlations(matrix).set_index(["feature_x", "feature_y"])["r"]
    assert same_day.get(("s24_steps", "weather_sunshine_hours"), same_day.get(("weather_sunshine_hours", "s24_steps"))) > 0.4

    lagged = lagged_correlations(matrix, lags=(1,)).set_index(["feature_x", "feature_y"])["r"]
    assert lagged[("laptop_last_active_hour", "whoop_sleep_performance_pct")] < -0.3
    assert lagged[("screen_phone_minutes", "whoop_sleep_performance_pct")] < -0.15


def test_hypotheses(matrix: pd.DataFrame) -> None:
    hyp = evaluate_hypotheses(matrix).set_index("name")
    assert hyp.loc["Rain vs eating-out spend", "r"] > 0.15
    assert hyp.loc["Late laptop use vs next-night sleep performance", "r"] < -0.3


def test_benjamini_hochberg_matches_reference() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.005, np.nan])
    q = benjamini_hochberg(p)
    assert q[3] == pytest.approx(0.02)
    assert q[0] == pytest.approx(0.02)
    assert q[1] == pytest.approx(0.04)
    assert np.isnan(q[4])


def test_detect_injected_anomaly(matrix: pd.DataFrame) -> None:
    spiked = matrix.copy()
    target = spiked.index[-5]
    spiked.loc[target, "whoop_hrv_ms"] = spiked["whoop_hrv_ms"].median() * 3
    anomalies = detect_anomalies(spiked)
    hit = anomalies.loc[(anomalies["feature"] == "whoop_hrv_ms") & (anomalies["date"] == target)]
    assert len(hit) == 1 and hit.iloc[0]["direction"] == "high"


def test_insight_payload_is_json_serialisable(matrix: pd.DataFrame) -> None:
    payload = build_insight_payload(matrix, END - timedelta(days=DAYS - 1), END)
    text = json.dumps(payload)
    assert payload["period"]["days"] == DAYS
    assert payload["top_cross_domain_correlations"]
    assert "NaN" not in text
