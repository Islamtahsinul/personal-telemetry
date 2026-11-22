"""Analytics: daily alignment pipeline and statistical analysis."""

from analytics.correlations import (
    HYPOTHESES,
    build_insight_payload,
    correlation_matrix,
    detect_anomalies,
    evaluate_hypotheses,
    lagged_correlations,
    ranked_correlations,
    rolling_averages,
    trend_summary,
    with_rolling_averages,
)
from analytics.pipeline import DailyMatrixBuilder, PipelineOptions, build_daily_matrix, feature_label

__all__ = [
    "HYPOTHESES",
    "DailyMatrixBuilder",
    "PipelineOptions",
    "build_daily_matrix",
    "build_insight_payload",
    "correlation_matrix",
    "detect_anomalies",
    "evaluate_hypotheses",
    "feature_label",
    "lagged_correlations",
    "ranked_correlations",
    "rolling_averages",
    "trend_summary",
    "with_rolling_averages",
]
