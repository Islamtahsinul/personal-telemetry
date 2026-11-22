"""Reusable Plotly figure builders for the dashboard (no Streamlit dependency)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analytics.pipeline import feature_label
from ui import theme


def trend_panels(matrix: pd.DataFrame, metrics: list[str], smooth: int, title: str | None = None) -> go.Figure:
    """Small multiples with a shared x-axis: raw daily points + trailing average line."""
    fig = make_subplots(
        rows=len(metrics), cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=[feature_label(m) for m in metrics],
    )
    for i, metric in enumerate(metrics, start=1):
        series = matrix[metric]
        fig.add_trace(
            go.Scatter(
                x=series.index, y=series, mode="markers", name="daily", showlegend=False,
                marker={"size": 5, "color": theme.RAW_POINT},
                hovertemplate="%{y:.1f}<extra>daily</extra>",
            ),
            row=i, col=1,
        )
        smoothed = series.rolling(smooth, min_periods=max(1, smooth // 2)).mean()
        fig.add_trace(
            go.Scatter(
                x=smoothed.index, y=smoothed, mode="lines", name=f"{smooth}d avg", showlegend=False,
                line={"color": theme.SERIES[0], "width": 2},
                hovertemplate=f"%{{y:.1f}}<extra>{smooth}d avg</extra>",
            ),
            row=i, col=1,
        )
    for annotation in fig.layout.annotations:
        annotation.update(x=0, xanchor="left", font={"size": 12, "color": theme.TEXT_SECONDARY})
    fig.update_layout(height=max(220, 190 * len(metrics)), title=title, hovermode="x unified")
    return fig


def indexed_overlay(matrix: pd.DataFrame, metrics: list[str], smooth: int) -> go.Figure:
    """All metrics on one axis as z-scores of their trailing average (common scale)."""
    fig = go.Figure()
    for i, metric in enumerate(metrics[: len(theme.SERIES)]):
        smoothed = matrix[metric].rolling(smooth, min_periods=max(1, smooth // 2)).mean()
        z = (smoothed - smoothed.mean()) / smoothed.std()
        fig.add_trace(
            go.Scatter(
                x=z.index, y=z, mode="lines", name=feature_label(metric),
                line={"color": theme.SERIES[i], "width": 2},
                customdata=smoothed, hovertemplate="%{customdata:.1f} (z %{y:.2f})",
            )
        )
    fig.add_hline(y=0, line={"color": theme.BASELINE, "width": 1})
    fig.update_layout(height=420, yaxis_title="standard deviations from period mean")
    return fig


def scatter_with_fit(
    x: pd.Series, y: pd.Series, x_label: str, y_label: str, color: str = theme.SERIES[0]
) -> go.Figure:
    """Scatter plot with an OLS trend line."""
    pair = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    fig = go.Figure(
        go.Scatter(
            x=pair["x"], y=pair["y"], mode="markers", marker={"size": 8, "color": color, "opacity": 0.75,
                                                              "line": {"color": theme.SURFACE, "width": 2}},
            text=[d.strftime("%a %d %b") for d in pair.index], hovertemplate="%{text}<br>x %{x:.1f}<br>y %{y:.1f}<extra></extra>",
            showlegend=False,
        )
    )
    if len(pair) >= 3 and pair["x"].nunique() > 1:
        slope, intercept = np.polyfit(pair["x"], pair["y"], 1)
        xs = np.linspace(pair["x"].min(), pair["x"].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", hoverinfo="skip", showlegend=False,
                                 line={"color": theme.TEXT_SECONDARY, "width": 2, "dash": "dot"}))
    fig.update_layout(height=360, xaxis_title=x_label, yaxis_title=y_label, hovermode="closest",
                      xaxis={"showgrid": True})
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    """Diverging heatmap of a correlation matrix (blue negative, red positive, gray zero)."""
    labels = [feature_label(c) for c in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels, zmin=-1, zmax=1, zmid=0, colorscale=theme.DIVERGING,
        xgap=2, ygap=2, colorbar={"title": "r", "thickness": 12},
        hovertemplate="%{y}<br>%{x}<br>r = %{z:.2f}<extra></extra>",
    ))
    # Show every label (Plotly thins categorical ticks by default).
    ticks = {"tickmode": "array", "tickvals": labels, "showgrid": False}
    fig.update_layout(height=max(520, 24 * len(labels) + 200), xaxis={**ticks, "tickangle": -50},
                      yaxis={**ticks, "autorange": "reversed"}, hovermode="closest")
    return fig


def spend_stacked(pivot: pd.DataFrame, order: list[str], title: str) -> go.Figure:
    """Stacked bars of spend per period with fixed per-category colours."""
    fig = go.Figure([
        go.Bar(x=pivot.index, y=pivot[c], name=c.replace("_", " ").capitalize(), marker_color=theme.spend_color(c),
               hovertemplate=f"{c.replace('_', ' ')}: £%{{y:.2f}}<extra></extra>")
        for c in order if c in pivot
    ])
    fig.update_layout(barmode="stack", height=380, yaxis_title="£", title=title)
    return fig
