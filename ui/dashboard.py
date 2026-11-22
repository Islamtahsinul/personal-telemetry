"""Dashboard page of the Personal Telemetry app.

Run from the project root::

    streamlit run ui/app.py

Everything is read from the local DuckDB file; the only network calls are the
connector syncs you trigger and, if you choose the Anthropic provider, the
AI summary request.
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.correlations import (  # noqa: E402
    build_insight_payload,
    correlation_matrix,
    detect_anomalies,
    evaluate_hypotheses,
    lagged_correlations,
    ranked_correlations,
    remove_weekday_effect,
    trend_summary,
)
from analytics.pipeline import DOMAIN_LABELS, DailyMatrixBuilder, column_domain, feature_label  # noqa: E402
from connectors import CONNECTORS, get_connector  # noqa: E402
from connectors.file_parser import FileParserConnector  # noqa: E402
from core.config import configure_logging, get_settings  # noqa: E402
from core.database import DOMAIN_TABLES, Database  # noqa: E402
from llm import LLMProviderError, get_provider  # noqa: E402
from ui import theme  # noqa: E402
from ui.charts import (  # noqa: E402
    correlation_heatmap,
    indexed_overlay,
    scatter_with_fit,
    spend_stacked,
    trend_panels,
)

theme.register_template()
settings = get_settings()
configure_logging(settings)
db = Database(settings=settings)
db.init_schema()
INSIGHTS_DIR = settings.data_dir / "insights"
CHART_CONFIG: dict[str, Any] = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

DEFAULT_TREND_METRICS = [
    "whoop_recovery_score", "whoop_sleep_performance_pct", "renpho_weight_kg", "mfp_calories_kcal",
]


# =============================================================================== data access
@st.cache_data(ttl=300, show_spinner="Building daily matrix…")
def load_matrix(start: date, end: date, db_mtime: float) -> pd.DataFrame:
    """Build the daily matrix (cached; ``db_mtime`` invalidates after writes)."""
    return DailyMatrixBuilder(db).build(start, end)


@st.cache_data(ttl=300)
def load_table(table: str, start: date, end: date, db_mtime: float) -> pd.DataFrame:
    """Read a raw domain table (cached)."""
    return db.read_table(table, start, end)


def db_mtime() -> float:
    """Modification time of the DuckDB file, used as a cache key."""
    return db.path.stat().st_mtime if db.path.exists() else 0.0


@st.cache_data(ttl=60, show_spinner=False)
def connector_status() -> dict[str, list[str]]:
    """Missing settings per connector (cached: some checks probe local services)."""
    return {name: get_connector(name, settings, db).missing_config() for name in CONNECTORS}


def latest_insight() -> dict[str, Any] | None:
    """Most recently saved AI summary, if any."""
    if not INSIGHTS_DIR.exists():
        return None
    files = sorted(INSIGHTS_DIR.glob("*.json"))
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


def save_insight(report: dict[str, Any], start: date, end: date) -> None:
    """Persist an AI summary as JSON under ``data/insights``."""
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    payload = {**report, "period_start": start.isoformat(), "period_end": end.isoformat()}
    (INSIGHTS_DIR / f"{stamp}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


# =============================================================================== chart helpers
def options_for(matrix: pd.DataFrame, domains: set[str] | None = None) -> list[str]:
    """Feature columns available for selection, optionally restricted to domains."""
    return [
        c for c in matrix.columns
        if matrix[c].notna().any() and column_domain(c) != "calendar"
        and (domains is None or column_domain(c) in domains)
    ]


def kpi(container: Any, label: str, matrix: pd.DataFrame, column: str, fmt: str = "{:.0f}",
        window: int = 7, invert: bool = False) -> None:
    """Stat tile: trailing ``window``-day mean with delta vs the preceding window."""
    if column not in matrix or matrix[column].dropna().empty:
        container.metric(label, "—")
        return
    recent = matrix[column].iloc[-window:].mean()
    prior = matrix[column].iloc[-2 * window : -window].mean()
    delta = None if pd.isna(prior) or pd.isna(recent) else recent - prior
    container.metric(
        label, "—" if pd.isna(recent) else fmt.format(recent),
        None if delta is None else fmt.format(delta), delta_color="inverse" if invert else "normal",
        help=f"Mean of the last {window} days; delta vs the {window} days before.",
    )


# =============================================================================== sidebar
with st.sidebar:
    st.header("Personal Telemetry")
    today = datetime.now(settings.tz).date()
    preset = st.selectbox("Period", ["Last 30 days", "Last 90 days", "Last 180 days", "Last 365 days", "Custom"], index=1)
    if preset == "Custom":
        picked = st.date_input("Date range", (today - timedelta(days=89), today), max_value=today)
        start, end = (picked if isinstance(picked, tuple) and len(picked) == 2 else (today - timedelta(days=89), today))
    else:
        days = int(preset.split()[1])
        start, end = today - timedelta(days=days - 1), today
    smooth = st.select_slider("Smoothing", options=[1, 3, 7, 14, 30], value=7, format_func=lambda d: f"{d}-day avg")

    st.divider()
    st.subheader("Sync sources")
    status = connector_status()
    ready = [n for n, missing in status.items() if not missing]
    chosen = st.multiselect("Connectors", list(CONNECTORS), default=ready,
                            format_func=lambda n: n + ("" if not status[n] else " (not configured)"))
    sync_days = st.number_input("Days to sync", min_value=1, max_value=3650, value=settings.default_lookback_days)
    if st.button("Sync now", type="primary", width="stretch"):
        with st.status("Syncing…", expanded=True) as box:
            for name in chosen:
                result = get_connector(name, settings, db).run(today - timedelta(days=int(sync_days) - 1), today)
                icon = {"success": "✅", "skipped": "⏭️", "failed": "❌"}[result.status]
                st.write(f"{icon} **{name}** — {result.rows_loaded} rows" + (f" · {result.error}" if result.error else ""))
            box.update(label="Sync complete", state="complete")
        st.cache_data.clear()
    st.page_link("connections.py", label="Manage connections", icon="🔌")

    st.divider()
    st.subheader("Log today")
    with st.form("subjective", clear_on_submit=False):
        log_date = st.date_input("Date", today, max_value=today)
        mood = st.slider("Mood", 1, 5, 3)
        energy = st.slider("Energy", 1, 5, 3)
        focus = st.slider("Focus", 1, 5, 3)
        notes = st.text_input("Notes (stays local)")
        if st.form_submit_button("Save", width="stretch"):
            db.upsert("subjective_state", pd.DataFrame(
                [{"date": log_date, "mood": mood, "energy": energy, "focus": focus, "notes": notes or None}]
            ), overwrite_nulls=True)
            st.cache_data.clear()
            st.success(f"Saved for {log_date:%a %d %b}")

    with st.expander("Upload a file"):
        target = st.selectbox("Target table", DOMAIN_TABLES)
        upload = st.file_uploader("CSV or JSON", type=["csv", "json", "jsonl", "tsv"])
        if upload is not None and st.button("Ingest file"):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / upload.name
                path.write_bytes(upload.getvalue())
                try:
                    rows = FileParserConnector(settings, db).ingest_file(path, target)
                    st.success(f"Loaded {rows} rows into {target}")
                    st.cache_data.clear()
                except Exception as exc:  # noqa: BLE001 - surface any parse error to the user
                    st.error(f"Could not ingest: {exc}")

# =============================================================================== data
matrix = load_matrix(start, end, db_mtime())
features = options_for(matrix)

st.title("Personal Telemetry")
st.caption(f"{start:%d %b %Y} → {end:%d %b %Y} · {len(features)} metrics across "
           f"{len({column_domain(c) for c in features})} domains · stored locally in `{db.path.name}`")

if not features:
    st.info(
        "No data yet. Connect your sources on the **Connections** page, "
        "or load synthetic data with `python cli.py demo` to explore the dashboard."
    )
    st.page_link("connections.py", label="Open Connections", icon="🔌")
    st.stop()

tiles = st.columns(6)
kpi(tiles[0], "Recovery %", matrix, "whoop_recovery_score")
kpi(tiles[1], "Sleep performance %", matrix, "whoop_sleep_performance_pct")
kpi(tiles[2], "Weight kg", matrix, "renpho_weight_kg" if "renpho_weight_kg" in matrix else "s24_weight_kg", "{:.1f}", invert=True)
kpi(tiles[3], "Calories kcal", matrix, "mfp_calories_kcal")
kpi(tiles[4], "Phone screen min", matrix, "screen_phone_minutes", invert=True)
kpi(tiles[5], "Spend £/day", matrix, "spend_total_gbp", "{:.2f}", invert=True)

tabs = st.tabs(["Overview", "Trends", "Body & nutrition", "Weather", "Spending", "Correlations", "Anomalies",
                "AI summary", "Data"])

# ------------------------------------------------------------------------------ overview
with tabs[0]:
    left, right = st.columns([3, 2])
    with left:
        insight = latest_insight()
        st.subheader("Latest executive summary")
        if insight:
            st.caption(f"{insight['provider']} · {insight['model']} · generated {insight['generated_at'][:16]} "
                       f"for {insight.get('period_start')} → {insight.get('period_end')}")
            st.markdown(insight["markdown"])
        else:
            st.info("No AI summary yet — generate one in the **AI summary** tab.")
    with right:
        st.subheader("Biggest shifts this week")
        trends = trend_summary(matrix)
        if trends.empty:
            st.caption("Not enough data for trend comparison.")
        else:
            top = trends.head(8).copy()
            top["metric"] = top["feature"].map(feature_label)
            st.dataframe(
                top[["metric", "recent_mean", "baseline_mean", "change_pct", "effect_size"]],
                hide_index=True, width="stretch",
                column_config={
                    "recent_mean": st.column_config.NumberColumn("last 7d", format="%.1f"),
                    "baseline_mean": st.column_config.NumberColumn("prior 30d", format="%.1f"),
                    "change_pct": st.column_config.NumberColumn("change %", format="%+.0f%%"),
                    "effect_size": st.column_config.NumberColumn("effect (SD)", format="%+.2f"),
                },
            )
        st.subheader("Strongest next-day links")
        lagged = lagged_correlations(matrix, lags=(1,))
        if not lagged.empty:
            st.dataframe(lagged.head(6)[["label", "r", "n"]], hide_index=True, width="stretch",
                         column_config={"r": st.column_config.NumberColumn(format="%+.2f")})

# ------------------------------------------------------------------------------ trends
with tabs[1]:
    defaults = [m for m in DEFAULT_TREND_METRICS if m in features] or features[:4]
    picked = st.multiselect("Metrics (any domain)", features, default=defaults, format_func=feature_label,
                            max_selections=8)
    mode = st.radio("View", ["Small multiples", "Indexed overlay (z-score)"], horizontal=True)
    if picked:
        fig = trend_panels(matrix, picked, smooth) if mode == "Small multiples" else indexed_overlay(matrix, picked, smooth)
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
        with st.expander("Table view"):
            st.dataframe(matrix[picked].rename(columns=feature_label), width="stretch")

# ------------------------------------------------------------------------------ body & nutrition
with tabs[2]:
    body_metrics = [m for m in ["renpho_weight_kg", "renpho_body_fat_pct", "renpho_muscle_mass_kg",
                                "mfp_calories_kcal", "mfp_protein_g", "whoop_strain", "whoop_recovery_score"]
                    if m in features]
    if not body_metrics:
        st.info("Connect Renpho, MyFitnessPal or WHOOP on the Connections page to see body composition vs nutrition and strain.")
    else:
        st.markdown("**Weight & macros vs strain & recovery** — shared timeline, "
                    f"dots are daily values, lines are {smooth}-day averages.")
        st.plotly_chart(trend_panels(matrix, body_metrics, smooth), width="stretch", config=CHART_CONFIG)

    macro_cols = {"protein": "mfp_protein_g", "carbs": "mfp_carbs_g", "fat": "mfp_fat_g"}
    if all(c in matrix for c in macro_cols.values()):
        kcal = pd.DataFrame({
            "protein": matrix["mfp_protein_g"] * 4, "carbs": matrix["mfp_carbs_g"] * 4, "fat": matrix["mfp_fat_g"] * 9,
        }).dropna(how="all")
        weekly = kcal.resample("W-MON", label="left", closed="left").mean()
        fig = go.Figure([
            go.Bar(x=weekly.index, y=weekly[m], name=m.capitalize(), marker_color=theme.MACRO_COLORS[m],
                   hovertemplate=f"{m}: %{{y:.0f}} kcal<extra></extra>")
            for m in ["protein", "carbs", "fat"]
        ])
        fig.update_layout(barmode="stack", height=320, title="Average daily calories by macro, per week",
                          yaxis_title="kcal / day")
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

    c1, c2 = st.columns(2)
    if {"mfp_calories_kcal", "derived_weight_change_7d_kg"} <= set(matrix.columns):
        c1.plotly_chart(scatter_with_fit(matrix["mfp_calories_kcal"].rolling(7, min_periods=4).mean(),
                                         matrix["derived_weight_change_7d_kg"],
                                         "calories, 7-day avg (kcal)", "weight change over 7 days (kg)"),
                        width="stretch", config=CHART_CONFIG)
    fat_col = next((c for c in ["renpho_body_fat_pct", "s24_body_fat_pct"] if c in matrix), None)
    if fat_col and "mfp_protein_g" in matrix:
        c2.plotly_chart(scatter_with_fit(matrix["mfp_protein_g"].rolling(7, min_periods=4).mean(), matrix[fat_col],
                                         "protein, 7-day avg (g)", "body fat %", theme.SERIES[1]),
                        width="stretch", config=CHART_CONFIG)

# ------------------------------------------------------------------------------ weather
with tabs[3]:
    weather_cols = options_for(matrix, {"weather"})
    if not weather_cols:
        st.info("Set your location on the Connections page to add weather.")
    else:
        other = [c for c in features if column_domain(c) != "weather"]
        c1, c2 = st.columns(2)
        metric = c1.selectbox("Metric", other, format_func=feature_label,
                              index=other.index("spend_total_gbp") if "spend_total_gbp" in other else 0)
        driver = c2.selectbox("Weather variable", weather_cols, format_func=feature_label)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.45, 0.3, 0.25],
                            subplot_titles=[feature_label(metric), feature_label(driver), "Precipitation (mm)"])
        series = matrix[metric]
        fig.add_trace(go.Scatter(x=series.index, y=series, mode="markers", marker={"size": 5, "color": theme.RAW_POINT},
                                 showlegend=False, hovertemplate="%{y:.1f}<extra>daily</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=series.index, y=series.rolling(smooth, min_periods=max(1, smooth // 2)).mean(),
                                 mode="lines", line={"color": theme.SERIES[0], "width": 2}, showlegend=False,
                                 hovertemplate=f"%{{y:.1f}}<extra>{smooth}d avg</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=matrix.index, y=matrix[driver], mode="lines",
                                 line={"color": theme.SERIES[1], "width": 2}, showlegend=False,
                                 hovertemplate="%{y:.1f}<extra></extra>"), row=2, col=1)
        if "weather_precipitation_mm" in matrix:
            fig.add_trace(go.Bar(x=matrix.index, y=matrix["weather_precipitation_mm"], marker_color=theme.SERIES[2],
                                 showlegend=False, hovertemplate="%{y:.1f} mm<extra></extra>"), row=3, col=1)
        for annotation in fig.layout.annotations:
            annotation.update(x=0, xanchor="left", font={"size": 12, "color": theme.TEXT_SECONDARY})
        fig.update_layout(height=620)
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

        c1, c2 = st.columns(2)
        c1.plotly_chart(scatter_with_fit(matrix[driver], matrix[metric], feature_label(driver), feature_label(metric)),
                        width="stretch", config=CHART_CONFIG)
        if "weather_precipitation_mm" in matrix:
            wet = matrix["weather_precipitation_mm"] >= 1.0
            groups = pd.DataFrame({"day": np.where(wet, "Wet (≥1 mm)", "Dry"), "value": matrix[metric]}).dropna()
            summary = groups.groupby("day")["value"].agg(["mean", "count"]).reset_index()
            fig = go.Figure(go.Bar(x=summary["day"], y=summary["mean"], marker_color=[theme.SERIES[2], theme.SERIES[0]][: len(summary)],
                                   text=[f"{m:.1f} (n={n})" for m, n in zip(summary["mean"], summary["count"], strict=True)],
                                   textposition="outside", hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
            fig.update_layout(height=360, title=f"Mean {feature_label(metric)} on wet vs dry days", showlegend=False)
            c2.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

# ------------------------------------------------------------------------------ spending
with tabs[4]:
    spend = load_table("monzo_spending", start, end, db_mtime())
    if spend.empty:
        st.info("Upload a Monzo export on the Connections page to see spending.")
    else:
        spend["date"] = pd.to_datetime(spend["date"])
        spend["category"] = spend["category"].where(spend["category"].isin(theme.SPEND_CATEGORY_ORDER), "other")
        order = [c for c in [*theme.SPEND_CATEGORY_ORDER, "other"] if c in set(spend["category"])]
        grain = st.radio("Grain", ["Weekly", "Daily", "Monthly"], horizontal=True)
        rule = {"Daily": "D", "Weekly": "W-MON", "Monthly": "MS"}[grain]
        pivot = (spend.pivot_table(index="date", columns="category", values="amount_gbp", aggfunc="sum")
                 .resample(rule, label="left", closed="left").sum())
        fig = spend_stacked(pivot, order, f"{grain} spend by category")
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

        c1, c2 = st.columns(2)
        totals = spend.groupby("category")["amount_gbp"].sum().reindex(order).dropna().sort_values()
        fig = go.Figure(go.Bar(x=totals.values, y=[c.replace("_", " ").capitalize() for c in totals.index],
                               orientation="h", marker_color=[theme.spend_color(c) for c in totals.index],
                               text=[f"£{v:,.0f}" for v in totals.values], textposition="outside",
                               hovertemplate="%{y}: £%{x:,.2f}<extra></extra>"))
        fig.update_layout(height=340, title="Total by category", showlegend=False, hovermode="closest",
                          xaxis={"showgrid": True})
        c1.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

        if "spend_total_gbp" in matrix:
            by_dow = matrix["spend_total_gbp"].groupby(matrix.index.dayofweek).mean()
            fig = go.Figure(go.Bar(x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], y=by_dow.reindex(range(7)).values,
                                   marker_color=theme.SERIES[0], hovertemplate="%{x}: £%{y:.2f}<extra></extra>"))
            fig.update_layout(height=340, title="Average spend by weekday", yaxis_title="£ / day")
            c2.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

# ------------------------------------------------------------------------------ correlations
with tabs[5]:
    st.caption("Spearman rank correlations on day-of-week-adjusted values. q = Benjamini–Hochberg false-discovery "
               "rate. Correlation ≠ causation — treat these as hypotheses to test.")
    available_domains = sorted({column_domain(c) for c in features})
    chosen_domains = st.multiselect("Domains in heatmap", available_domains,
                                    default=[d for d in available_domains if d not in {"derived"}],
                                    format_func=lambda d: DOMAIN_LABELS.get(d, d))
    min_n = st.slider("Minimum overlapping days", 7, 90, 14)
    heat_cols = [c for c in features if column_domain(c) in set(chosen_domains)]
    corr = correlation_matrix(matrix[heat_cols], min_periods=min_n) if heat_cols else pd.DataFrame()
    if corr.shape[0] >= 2:
        fig = correlation_heatmap(corr)
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

    c1, c2 = st.columns([2, 1])
    lag = c2.selectbox("Lag (y follows x by)", [0, 1, 2, 3, 7], format_func=lambda d: "same day" if d == 0 else f"{d} day(s)")
    cross_only = c2.checkbox("Cross-domain pairs only", value=True)
    ranked = ranked_correlations(matrix, min_periods=min_n, lag=lag, cross_domain_only=cross_only)
    with c1:
        st.markdown("**Top relationships**")
        st.dataframe(ranked.head(40)[["label", "r", "n", "q_value"]], hide_index=True, width="stretch",
                     column_config={"r": st.column_config.NumberColumn(format="%+.2f"),
                                    "q_value": st.column_config.NumberColumn("q", format="%.4f")}, height=380)
    if not ranked.empty:
        pair = st.selectbox("Inspect pair", ranked.head(40).index, format_func=lambda i: ranked.at[i, "label"])
        x_col, y_col = ranked.at[pair, "feature_x"], ranked.at[pair, "feature_y"]
        adjusted = remove_weekday_effect(matrix[[x_col, y_col]])
        st.plotly_chart(scatter_with_fit(adjusted[x_col], adjusted[y_col].shift(-lag),
                                         f"{feature_label(x_col)} (weekday-adjusted)",
                                         f"{feature_label(y_col)} (weekday-adjusted, +{lag}d)"),
                        width="stretch", config=CHART_CONFIG)

    st.markdown("**Tracked hypotheses**")
    hyp = evaluate_hypotheses(matrix)
    if hyp.empty:
        st.caption("No hypotheses have both features available yet.")
    else:
        st.dataframe(hyp[["name", "r", "n", "p_value", "strength", "lag", "smooth"]], hide_index=True,
                     width="stretch",
                     column_config={"r": st.column_config.NumberColumn(format="%+.2f"),
                                    "p_value": st.column_config.NumberColumn("p", format="%.4f"),
                                    "lag": st.column_config.NumberColumn("lag (d)"),
                                    "smooth": st.column_config.NumberColumn("smoothing (d)")})

# ------------------------------------------------------------------------------ anomalies
with tabs[6]:
    c1, c2 = st.columns(2)
    threshold = c1.slider("Robust z-score threshold", 2.5, 6.0, 3.5, 0.5)
    window = c2.slider("Baseline window (days)", 14, 90, 30)
    anomalies = detect_anomalies(matrix, window=window, threshold=threshold)
    if anomalies.empty:
        st.success("No anomalies at this threshold.")
    else:
        anomalies["domain_label"] = anomalies["domain"].map(lambda d: DOMAIN_LABELS.get(d, d))
        domains_present = list(dict.fromkeys(anomalies["domain_label"]))[: len(theme.SERIES)]
        fig = px.scatter(
            anomalies.loc[anomalies["domain_label"].isin(domains_present)], x="date", y="z_score",
            color="domain_label", hover_data={"label": True, "value": ":.1f", "baseline": ":.1f", "z_score": ":.1f"},
            color_discrete_sequence=list(theme.SERIES), category_orders={"domain_label": domains_present},
        )
        fig.update_traces(marker={"size": 9, "line": {"color": theme.SURFACE, "width": 2}})
        for level in (threshold, -threshold):
            fig.add_hline(y=level, line={"color": theme.BASELINE, "width": 1, "dash": "dot"})
        fig.update_layout(height=380, legend_title_text="", hovermode="closest", yaxis_title="robust z-score")
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
        st.dataframe(anomalies[["date", "label", "value", "baseline", "z_score", "direction"]], hide_index=True,
                     width="stretch",
                     column_config={"date": st.column_config.DateColumn(format="ddd D MMM YYYY"),
                                    "value": st.column_config.NumberColumn(format="%.1f"),
                                    "baseline": st.column_config.NumberColumn("baseline median", format="%.1f"),
                                    "z_score": st.column_config.NumberColumn("z", format="%+.1f")})

# ------------------------------------------------------------------------------ AI summary
with tabs[7]:
    provider_name = st.radio("Provider", ["ollama", "anthropic"], horizontal=True,
                             index=0 if settings.llm_provider != "anthropic" else 1,
                             captions=["local & private", "Claude API (sends aggregates only)"])
    payload = build_insight_payload(matrix, start, end)
    with st.expander("Payload sent to the model"):
        st.json(payload, expanded=False)
    if st.button("Generate executive summary", type="primary"):
        try:
            provider = get_provider(settings, provider_name)
            if not provider.is_available():
                st.warning(f"{provider_name} does not look ready (missing key, server down or model not pulled) — trying anyway.")
            with st.spinner(f"Asking {provider_name} ({provider.model})… this can take a minute"):
                report = provider.synthesize_insights(payload)
            save_insight(report.to_dict(), start, end)
            if report.truncated:
                st.warning("The response hit the output limit and may be cut off.")
            st.markdown(report.markdown)
            st.caption(f"{report.provider} · {report.model} · {report.usage}")
        except LLMProviderError as exc:
            st.error(str(exc))
    elif (insight := latest_insight()) is not None:
        st.caption(f"Last summary · {insight['provider']} · {insight['generated_at'][:16]}")
        st.markdown(insight["markdown"])

# ------------------------------------------------------------------------------ data
with tabs[8]:
    st.markdown("**Coverage**")
    st.dataframe(db.table_coverage(), hide_index=True, width="stretch")
    st.markdown("**Daily master matrix**")
    st.dataframe(matrix, width="stretch", height=360)
    st.download_button("Download matrix as CSV", matrix.to_csv().encode("utf-8"),
                       file_name=f"telemetry_{start}_{end}.csv", mime="text/csv")
    st.markdown("**Recent sync runs**")
    st.dataframe(db.query_df("SELECT connector, started_at, status, rows_loaded, error FROM ingestion_runs "
                             "ORDER BY started_at DESC LIMIT 50"), hide_index=True, width="stretch")
