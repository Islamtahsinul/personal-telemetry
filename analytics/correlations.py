"""Statistical analysis over the daily master matrix.

* Moving averages (7/30 day).
* Correlation matrices and ranked cross-domain pairs with p-values and
  Benjamini-Hochberg false-discovery-rate correction - with ~60 features there
  are ~1,800 pairs, so uncorrected p < 0.05 would yield ~90 false positives.
* Lagged correlations (behaviour on day D vs outcome on D+k).
* Curated hypotheses (weight vs calories, screen time vs sleep, weather vs spend...).
* Robust rolling anomaly detection (median/MAD z-scores against a trailing window).
* An LLM-ready, JSON-serialisable insight payload.

Correlation is not causation, and n-of-1 daily data is autocorrelated; p-values
here are screening heuristics, not confirmatory statistics.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from analytics.pipeline import DOMAIN_LABELS, column_domain, feature_families, feature_label

Method = Literal["pearson", "spearman", "kendall"]
EXCLUDED_FROM_CORRELATION: frozenset[str] = frozenset({"calendar_day_of_week"})
MAD_SCALE = 1.4826  # MAD -> standard deviation for normal data


# --------------------------------------------------------------------------- smoothing
def rolling_averages(
    matrix: pd.DataFrame,
    windows: Sequence[int] = (7, 30),
    columns: Iterable[str] | None = None,
    min_fraction: float = 0.5,
) -> pd.DataFrame:
    """Return trailing moving averages as new ``<col>_ma<window>`` columns.

    Args:
        matrix: Daily matrix (DatetimeIndex).
        windows: Window lengths in days.
        columns: Columns to smooth (default: all numeric).
        min_fraction: Minimum fraction of non-null days required inside a window.
    """
    cols = list(columns) if columns is not None else list(matrix.select_dtypes("number").columns)
    out = {}
    for window in windows:
        rolled = matrix[cols].rolling(window, min_periods=max(1, math.ceil(window * min_fraction))).mean()
        out.update({f"{c}_ma{window}": rolled[c] for c in cols})
    return pd.DataFrame(out, index=matrix.index)


def with_rolling_averages(matrix: pd.DataFrame, windows: Sequence[int] = (7, 30)) -> pd.DataFrame:
    """Return ``matrix`` with moving-average columns appended."""
    return matrix.join(rolling_averages(matrix, windows))


# --------------------------------------------------------------------------- correlation
def usable_columns(matrix: pd.DataFrame, min_periods: int = 14, include_calendar: bool = False) -> list[str]:
    """Numeric, non-constant columns with at least ``min_periods`` observations.

    Calendar features are excluded by default: weekday/weekend effects are real but
    obvious and would otherwise dominate every ranking.
    """
    numeric = matrix.select_dtypes("number")
    keep = []
    for col in numeric.columns:
        if col in EXCLUDED_FROM_CORRELATION or (not include_calendar and column_domain(col) == "calendar"):
            continue
        series = numeric[col].dropna()
        if len(series) >= min_periods and series.nunique() > 1:
            keep.append(col)
    return keep


def remove_weekday_effect(matrix: pd.DataFrame) -> pd.DataFrame:
    """Subtract each column's day-of-week mean (partial out the weekly rhythm).

    Without this, anything that differs between workdays and weekends (laptop time,
    commuting spend, alarm-driven sleep) correlates with everything else that does.
    """
    numeric = matrix.select_dtypes("number")
    weekday_means = numeric.groupby(numeric.index.dayofweek).transform("mean")
    return numeric - weekday_means


def correlation_matrix(
    matrix: pd.DataFrame, method: Method = "spearman", min_periods: int = 14, adjust_weekday: bool = True
) -> pd.DataFrame:
    """Pairwise correlation matrix over usable columns (pairwise-complete observations)."""
    cols = usable_columns(matrix, min_periods)
    data = remove_weekday_effect(matrix[cols]) if adjust_weekday else matrix[cols]
    return data.corr(method=method, min_periods=min_periods)


def _pairwise_counts(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    a = left.notna().astype("int64")
    b = right.notna().astype("int64")
    return pd.DataFrame(a.T.to_numpy() @ b.to_numpy(), index=left.columns, columns=right.columns)


def _cross_correlation(left: pd.DataFrame, right: pd.DataFrame, method: Method, min_periods: int) -> pd.DataFrame:
    """Correlation of every column in ``left`` with every column in ``right``."""
    joined = pd.concat([left.add_prefix("x::"), right.add_prefix("y::")], axis=1)
    full = joined.corr(method=method, min_periods=min_periods)
    block = full.loc[[f"x::{c}" for c in left.columns], [f"y::{c}" for c in right.columns]]
    block.index = left.columns
    block.columns = right.columns
    return block


def correlation_p_values(r: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Two-sided p-values for correlation coefficients via the t-distribution.

    Exact for Pearson; a standard large-sample approximation for Spearman.
    """
    r = np.clip(np.asarray(r, dtype="float64"), -0.999999, 0.999999)
    n = np.asarray(n, dtype="float64")
    dof = np.maximum(n - 2, 1)
    t = r * np.sqrt(dof / (1 - r**2))
    return 2 * stats.t.sf(np.abs(t), dof)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted q-values (NaNs preserved)."""
    p = np.asarray(p_values, dtype="float64")
    q = np.full_like(p, np.nan)
    mask = ~np.isnan(p)
    m = int(mask.sum())
    if m == 0:
        return q
    order = np.argsort(p[mask])
    ranked = p[mask][order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(ranked, 1.0)
    q[mask] = adjusted
    return q


def ranked_correlations(
    matrix: pd.DataFrame,
    method: Method = "spearman",
    min_periods: int = 14,
    lag: int = 0,
    cross_domain_only: bool = True,
    columns: Sequence[str] | None = None,
    adjust_weekday: bool = True,
) -> pd.DataFrame:
    """Rank feature pairs by absolute correlation, with p- and FDR q-values.

    Args:
        matrix: Daily matrix.
        method: Correlation method.
        min_periods: Minimum overlapping days for a pair to be scored.
        lag: Correlate ``x`` on day D with ``y`` on day D+``lag``. With ``lag > 0``
            pairs are directional (x leads y); with 0 each unordered pair appears once.
        cross_domain_only: Only keep pairs whose measurement families are disjoint.
        columns: Restrict to these columns (default: all usable).
        adjust_weekday: Correlate day-of-week-adjusted values (see :func:`remove_weekday_effect`).

    Returns:
        DataFrame with ``feature_x, feature_y, domain_x, domain_y, lag, r, n, p_value,
        q_value, label`` sorted by ``|r|`` descending.
    """
    cols = list(columns) if columns is not None else usable_columns(matrix, min_periods)
    if len(cols) < 2:
        return pd.DataFrame(
            columns=["feature_x", "feature_y", "domain_x", "domain_y", "lag", "r", "n", "p_value", "q_value", "label"]
        )
    left = remove_weekday_effect(matrix[cols]) if adjust_weekday else matrix[cols]
    right = left.shift(-lag) if lag else left
    r = _cross_correlation(left, right, method, min_periods) if lag else left.corr(method=method, min_periods=min_periods)
    n = _pairwise_counts(left, right)

    families = {c: feature_families(c) for c in cols}
    rows: list[dict[str, Any]] = []
    for i, x in enumerate(cols):
        for j, y in enumerate(cols):
            if x == y or (lag == 0 and j <= i):
                continue
            if cross_domain_only and families[x] & families[y]:
                continue
            value = r.at[x, y]
            if pd.isna(value):
                continue
            rows.append({"feature_x": x, "feature_y": y, "r": float(value), "n": int(n.at[x, y])})
    result = pd.DataFrame(rows)
    if result.empty:
        return ranked_correlations(pd.DataFrame(), columns=[])  # empty, correctly-shaped frame
    result["domain_x"] = result["feature_x"].map(column_domain)
    result["domain_y"] = result["feature_y"].map(column_domain)
    result["lag"] = lag
    result["p_value"] = correlation_p_values(result["r"].to_numpy(), result["n"].to_numpy())
    result["q_value"] = benjamini_hochberg(result["p_value"].to_numpy())
    arrow = f" → (+{lag}d) " if lag else " ↔ "
    result["label"] = result["feature_x"].map(feature_label) + arrow + result["feature_y"].map(feature_label)
    result = result.reindex(result["r"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return result[["feature_x", "feature_y", "domain_x", "domain_y", "lag", "r", "n", "p_value", "q_value", "label"]]


def lagged_correlations(
    matrix: pd.DataFrame, lags: Sequence[int] = (1, 2, 3), **kwargs: Any
) -> pd.DataFrame:
    """Concatenate :func:`ranked_correlations` across several positive lags."""
    frames = [ranked_correlations(matrix, lag=lag, **kwargs) for lag in lags]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return ranked_correlations(pd.DataFrame(), columns=[])
    combined = pd.concat(frames, ignore_index=True)
    return combined.reindex(combined["r"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# --------------------------------------------------------------------------- hypotheses
@dataclass(frozen=True, slots=True)
class Hypothesis:
    """A named, pre-registered relationship to monitor.

    Attributes:
        name: Display name.
        x: Driver feature (candidates in preference order; first present is used).
        y: Outcome feature (candidates in preference order).
        lag: Days by which ``y`` follows ``x``.
        smooth: Trailing-mean window applied to both series first (1 = raw daily).
    """

    name: str
    x: tuple[str, ...]
    y: tuple[str, ...]
    lag: int = 0
    smooth: int = 1


HYPOTHESES: tuple[Hypothesis, ...] = (
    Hypothesis("Calories (7d) vs weekly weight change", ("mfp_calories_kcal",), ("derived_weight_change_7d_kg",), 0, 7),
    Hypothesis("Energy balance (7d) vs weekly weight change", ("derived_energy_balance_kcal",),
               ("derived_weight_change_7d_kg",), 0, 7),
    Hypothesis("Protein (7d) vs body fat %", ("mfp_protein_g",), ("renpho_body_fat_pct", "s24_body_fat_pct"), 0, 7),
    Hypothesis("Protein per kg (7d) vs muscle mass", ("derived_protein_g_per_kg",),
               ("renpho_muscle_mass_kg", "s24_skeletal_muscle_kg"), 0, 7),
    Hypothesis("Calories vs day strain", ("whoop_strain",), ("mfp_calories_kcal",), 0, 1),
    Hypothesis("Protein vs next-day recovery", ("mfp_protein_g",), ("whoop_recovery_score",), 1, 1),
    Hypothesis("Phone screen time vs next-night sleep performance", ("screen_phone_minutes",),
               ("whoop_sleep_performance_pct",), 1, 1),
    Hypothesis("Total screen time vs next-night sleep hours", ("screen_total_minutes",), ("whoop_sleep_hours",), 1, 1),
    Hypothesis("Late laptop use vs next-night sleep performance", ("laptop_last_active_hour",),
               ("whoop_sleep_performance_pct",), 1, 1),
    Hypothesis("Social-media time vs mood", ("screen_social_minutes",), ("subjective_mood",), 0, 1),
    Hypothesis("Temperature vs total spend", ("weather_temp_mean_c",), ("spend_total_gbp",), 0, 1),
    Hypothesis("Rain vs eating-out spend", ("weather_precipitation_mm",), ("spend_eating_out_gbp",), 0, 1),
    Hypothesis("Daylight vs mood", ("weather_daylight_hours",), ("subjective_mood",), 0, 7),
    Hypothesis("Sunshine vs steps", ("weather_sunshine_hours",), ("s24_steps",), 0, 1),
    Hypothesis("Recovery vs subjective energy", ("whoop_recovery_score",), ("subjective_energy",), 0, 1),
    Hypothesis("HRV vs focus", ("whoop_hrv_ms",), ("subjective_focus",), 0, 1),
    Hypothesis("Coding time vs focus", ("laptop_coding_minutes",), ("subjective_focus",), 0, 1),
)


def evaluate_hypotheses(
    matrix: pd.DataFrame,
    hypotheses: Sequence[Hypothesis] = HYPOTHESES,
    method: Method = "spearman",
    min_periods: int = 10,
) -> pd.DataFrame:
    """Score each hypothesis whose features are present in ``matrix``.

    Returns:
        DataFrame with ``name, x, y, lag, smooth, r, n, p_value, strength``.
    """
    rows: list[dict[str, Any]] = []
    for hyp in hypotheses:
        x = next((c for c in hyp.x if c in matrix and matrix[c].notna().any()), None)
        y = next((c for c in hyp.y if c in matrix and matrix[c].notna().any()), None)
        if x is None or y is None:
            continue
        xs, ys = matrix[x], matrix[y]
        if hyp.smooth > 1:
            min_p = math.ceil(hyp.smooth / 2)
            xs = xs.rolling(hyp.smooth, min_periods=min_p).mean()
            ys = ys.rolling(hyp.smooth, min_periods=min_p).mean()
        if hyp.lag:
            ys = ys.shift(-hyp.lag)
        pair = pd.concat([xs, ys], axis=1).dropna()
        n = len(pair)
        if n < min_periods or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
            rows.append({"name": hyp.name, "x": x, "y": y, "lag": hyp.lag, "smooth": hyp.smooth,
                         "r": np.nan, "n": n, "p_value": np.nan, "strength": "insufficient data"})
            continue
        r = float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))
        p = float(correlation_p_values(np.array([r]), np.array([n]))[0])
        rows.append({"name": hyp.name, "x": x, "y": y, "lag": hyp.lag, "smooth": hyp.smooth,
                     "r": r, "n": n, "p_value": p, "strength": describe_strength(r)})
    return pd.DataFrame(rows)


def describe_strength(r: float) -> str:
    """Qualitative label for a correlation coefficient."""
    if np.isnan(r):
        return "n/a"
    magnitude = abs(r)
    label = "negligible" if magnitude < 0.1 else "weak" if magnitude < 0.3 else "moderate" if magnitude < 0.5 else "strong"
    if label == "negligible":
        return label
    return f"{label} {'positive' if r > 0 else 'negative'}"


# --------------------------------------------------------------------------- anomalies
def _robust_z(series: pd.Series, window: int, min_periods: int) -> tuple[pd.Series, pd.Series]:
    """Trailing robust z-score of ``series`` against its previous ``window`` observations."""
    observed = series.dropna()
    history = observed.shift(1).rolling(window, min_periods=min_periods)
    median = history.median()
    mad = (observed.shift(1) - median).abs().rolling(window, min_periods=min_periods).median() * MAD_SCALE
    spread = mad.where(mad > 0, history.std())
    z = (observed - median) / spread
    return z.reindex(series.index), median.reindex(series.index)


def detect_anomalies(
    matrix: pd.DataFrame,
    window: int = 30,
    threshold: float = 3.5,
    min_periods: int = 10,
    columns: Iterable[str] | None = None,
    weekday_aware: bool = True,
    max_zero_share: float = 0.3,
) -> pd.DataFrame:
    """Flag days whose value deviates strongly from the trailing baseline.

    Uses a robust z-score ``(x - median) / (1.4826 * MAD)`` over the previous
    ``window`` observed days (the current day is excluded from its own baseline).
    When MAD is zero the trailing standard deviation is used instead.

    Args:
        matrix: Daily matrix.
        window: Trailing baseline length in observed days (scaled per day type).
        threshold: Minimum ``|z|`` to flag.
        min_periods: Minimum baseline observations.
        columns: Columns to scan (default: all usable, non-calendar).
        weekday_aware: Compare weekdays with weekdays and weekends with weekends, so
            routine weekly rhythms (no commute on Saturday) are not flagged.
        max_zero_share: Skip zero-inflated series (e.g. per-category spend) where
            more than this share of days is exactly zero; their totals are still scanned.

    Returns:
        Long DataFrame ``date, feature, domain, value, baseline, z_score, direction, label``
        sorted by date descending then ``|z|``.
    """
    cols = list(columns) if columns is not None else usable_columns(matrix, min_periods)
    is_weekend = pd.Series(matrix.index.dayofweek >= 5, index=matrix.index)
    records: list[pd.DataFrame] = []
    for col in cols:
        series = matrix[col]
        observed = series.dropna()
        if observed.empty or (observed == 0).mean() > max_zero_share:
            continue
        if weekday_aware:
            parts = []
            for weekend_flag, share in ((False, 5 / 7), (True, 2 / 7)):
                group = series[is_weekend == weekend_flag]
                group_window = max(min_periods, round(window * share))
                group_min = max(4, round(min_periods * share))
                parts.append(_robust_z(group, group_window, group_min))
            z = pd.concat([p[0] for p in parts]).reindex(series.index)
            median = pd.concat([p[1] for p in parts]).reindex(series.index)
        else:
            z, median = _robust_z(series, window, min_periods)
        flagged = z.abs() >= threshold
        if flagged.any():
            records.append(
                pd.DataFrame(
                    {
                        "date": series.index[flagged],
                        "feature": col,
                        "domain": column_domain(col),
                        "value": series[flagged].to_numpy(),
                        "baseline": median[flagged].to_numpy(),
                        "z_score": z[flagged].to_numpy(),
                    }
                )
            )
    if not records:
        return pd.DataFrame(columns=["date", "feature", "domain", "value", "baseline", "z_score", "direction", "label"])
    result = pd.concat(records, ignore_index=True)
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=["z_score"])
    result["direction"] = np.where(result["z_score"] > 0, "high", "low")
    result["label"] = result["feature"].map(feature_label)
    result["_abs"] = result["z_score"].abs()
    return result.sort_values(["date", "_abs"], ascending=[False, False]).drop(columns="_abs").reset_index(drop=True)


# --------------------------------------------------------------------------- trends
def trend_summary(matrix: pd.DataFrame, recent_days: int = 7, baseline_days: int = 30) -> pd.DataFrame:
    """Compare the most recent ``recent_days`` mean with the preceding baseline.

    Returns:
        DataFrame ``feature, recent_mean, baseline_mean, change, change_pct, effect_size``
        sorted by ``|effect_size|`` (change in baseline standard deviations).
    """
    if len(matrix) < recent_days + 3:
        return pd.DataFrame(columns=["feature", "recent_mean", "baseline_mean", "change", "change_pct", "effect_size"])
    recent = matrix.iloc[-recent_days:]
    baseline = matrix.iloc[-(recent_days + baseline_days) : -recent_days]
    rows = []
    for col in usable_columns(matrix, min_periods=5):
        if column_domain(col) == "calendar":
            continue
        r_mean, b_mean, b_std = recent[col].mean(), baseline[col].mean(), baseline[col].std()
        if pd.isna(r_mean) or pd.isna(b_mean):
            continue
        change = r_mean - b_mean
        rows.append(
            {
                "feature": col,
                "recent_mean": r_mean,
                "baseline_mean": b_mean,
                "change": change,
                "change_pct": (change / abs(b_mean) * 100) if b_mean else np.nan,
                "effect_size": (change / b_std) if b_std and not pd.isna(b_std) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.reindex(out["effect_size"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# --------------------------------------------------------------------------- payload
def build_insight_payload(
    matrix: pd.DataFrame,
    start: date,
    end: date,
    max_correlations: int = 15,
    max_anomalies: int = 15,
    anomaly_lookback_days: int = 14,
    q_threshold: float = 0.2,
) -> dict[str, Any]:
    """Assemble a compact, JSON-serialisable summary for LLM synthesis.

    Only aggregates are included - never raw notes, app titles or transactions.
    """
    coverage = {
        DOMAIN_LABELS.get(domain, domain): int(matrix[[c for c in matrix if column_domain(c) == domain]].notna().any(axis=1).sum())
        for domain in sorted({column_domain(c) for c in matrix.columns})
        if domain != "calendar"
    }
    averages = {
        feature_label(c): _round(matrix[c].mean())
        for c in usable_columns(matrix, min_periods=3)
        if column_domain(c) != "calendar"
    }

    same_day = ranked_correlations(matrix)
    lagged = lagged_correlations(matrix, lags=(1,))
    significant = same_day.loc[same_day["q_value"] <= q_threshold] if not same_day.empty else same_day
    top = significant if len(significant) >= 5 else same_day
    trends = trend_summary(matrix)
    anomalies = detect_anomalies(matrix)
    if not anomalies.empty:
        cutoff = pd.Timestamp(end) - pd.Timedelta(days=anomaly_lookback_days)
        anomalies = anomalies.loc[pd.to_datetime(anomalies["date"]) >= cutoff]
    hypotheses = evaluate_hypotheses(matrix)

    return {
        "period": {"start": start.isoformat(), "end": end.isoformat(), "days": len(matrix)},
        "days_with_data_by_domain": coverage,
        "period_averages": averages,
        "trends_last_7d_vs_prior_30d": [
            {
                "metric": feature_label(row.feature),
                "recent": _round(row.recent_mean),
                "baseline": _round(row.baseline_mean),
                "change_pct": _round(row.change_pct, 1),
                "effect_size_sd": _round(row.effect_size, 2),
            }
            for row in trends.head(12).itertuples()
        ],
        "top_cross_domain_correlations": _records(top.head(max_correlations)),
        "top_next_day_correlations": _records(lagged.head(max_correlations // 2 + 1)),
        "hypotheses": [
            {"name": row.name, "r": _round(row.r, 2), "n": int(row.n), "p_value": _round(row.p_value, 4),
             "strength": row.strength, "lag_days": int(row.lag), "smoothing_days": int(row.smooth)}
            for row in hypotheses.itertuples()
        ],
        "recent_anomalies": [
            {"date": pd.Timestamp(row.date).date().isoformat(), "metric": row.label, "value": _round(row.value),
             "baseline_median": _round(row.baseline), "z_score": _round(row.z_score, 1), "direction": row.direction}
            for row in anomalies.head(max_anomalies).itertuples()
        ],
        "notes": [
            "Correlations are Spearman rank correlations on day-of-week-adjusted daily values; q_value is Benjamini-Hochberg FDR.",
            "WHOOP sleep on date D is the night before D; next-day correlations pair day D with D+1.",
            "Observational n-of-1 data: correlations are hypotheses, not causal findings.",
        ],
    }


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {"pair": row.label, "r": _round(row.r, 2), "n": int(row.n), "q_value": _round(row.q_value, 4)}
        for row in frame.itertuples()
    ]


def _round(value: Any, digits: int = 2) -> float | None:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None
