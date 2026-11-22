"""Chart palette and Plotly styling for the dashboard.

Colours follow a validated colour-vision-deficiency-safe categorical order; assign
slots in this fixed order and keep a category on the same slot everywhere (colour
follows the entity, never its rank). Swap the hex values here to re-brand.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

SERIES: tuple[str, ...] = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#6250d6",  # 7 violet
    "#e34948",  # 8 red
)
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
RAW_POINT = "#c3c2b7"

# Diverging blue <-> red with a neutral gray midpoint (correlation heatmaps).
DIVERGING: list[list[float | str]] = [
    [0.0, "#104281"],
    [0.2, "#2a78d6"],
    [0.4, "#9ec5f4"],
    [0.5, "#f0efec"],
    [0.6, "#f2a7a6"],
    [0.8, "#e34948"],
    [1.0, "#9e2424"],
]

# Fixed category -> slot assignments (stable across filters).
SPEND_CATEGORY_ORDER: tuple[str, ...] = (
    "groceries", "eating_out", "transport", "shopping", "bills", "entertainment", "travel",
)
MACRO_COLORS: dict[str, str] = {"protein": SERIES[0], "carbs": SERIES[1], "fat": SERIES[2]}
OTHER_COLOR = TEXT_MUTED


def spend_color(category: str) -> str:
    """Stable colour for a spending category; unlisted categories share the 'other' gray."""
    if category in SPEND_CATEGORY_ORDER:
        return SERIES[SPEND_CATEGORY_ORDER.index(category)]
    return OTHER_COLOR


def register_template() -> None:
    """Register and activate the ``telemetry`` Plotly template."""
    axis = {
        "showgrid": True,
        "gridcolor": GRIDLINE,
        "gridwidth": 1,
        "zeroline": False,
        "automargin": True,
        "linecolor": BASELINE,
        "tickfont": {"color": TEXT_MUTED, "size": 11},
        "title": {"font": {"color": TEXT_SECONDARY, "size": 12}},
    }
    template = go.layout.Template(
        layout=go.Layout(
            font={"family": 'system-ui, -apple-system, "Segoe UI", sans-serif', "color": TEXT_PRIMARY, "size": 12},
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            colorway=list(SERIES),
            xaxis={**axis, "showgrid": False, "showline": True},
            yaxis=axis,
            title={"x": 0, "xanchor": "left", "font": {"size": 14, "color": TEXT_PRIMARY}},
            # Legend top-right so it never collides with the left-aligned title; keep series order.
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 1, "xanchor": "right",
                    "traceorder": "normal", "font": {"color": TEXT_SECONDARY}},
            hovermode="x unified",
            hoverlabel={"bgcolor": "white", "font": {"color": TEXT_PRIMARY}},
            margin={"l": 48, "r": 16, "t": 40, "b": 32},
            bargap=0.15,
        ),
        data={
            "scatter": [go.Scatter(line={"width": 2}, marker={"size": 8})],
            "bar": [go.Bar(marker={"line": {"color": SURFACE, "width": 2}})],
        },
    )
    pio.templates["telemetry"] = template
    pio.templates.default = "telemetry"
