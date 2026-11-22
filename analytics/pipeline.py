"""Daily master-matrix pipeline.

Aligns every source onto one continuous local-date index and emits a wide
DataFrame with one row per day and one column per feature. Column names are
prefixed with their domain (``whoop_``, ``renpho_``, ``mfp_``, ``spend_`` ...), which
the correlation engine uses to find *cross-domain* relationships.

Alignment conventions:

* WHOOP sleep is keyed by the **wake-up date**, so ``whoop_sleep_*`` on day D
  describes the night *before* D. Evening behaviour on D (screen time, food,
  spend) therefore relates to sleep on D+1 - see ``lagged_correlations``.
* Body composition is measured sporadically; it is forward-filled for up to
  ``body_ffill_days`` days (configurable, set 0 to disable).
* Spending days without transactions are filled with 0 inside the date span the
  Monzo export covers (a missing day there means "spent nothing").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from connectors.utils import slugify
from core.database import Database

logger = logging.getLogger(__name__)

DOMAIN_LABELS: dict[str, str] = {
    "whoop": "WHOOP",
    "s24": "Samsung S24",
    "renpho": "Renpho",
    "mfp": "MyFitnessPal",
    "screen": "Screen time",
    "laptop": "Laptop",
    "spend": "Spending",
    "weather": "Weather",
    "subjective": "Subjective",
    "derived": "Derived",
    "calendar": "Calendar",
}

BODY_COMPOSITION_COLUMNS: tuple[str, ...] = (
    "renpho_weight_kg",
    "renpho_bmi",
    "renpho_body_fat_pct",
    "renpho_muscle_mass_kg",
    "renpho_muscle_mass_pct",
    "renpho_visceral_fat",
    "renpho_body_water_pct",
    "renpho_bmr_kcal",
    "s24_weight_kg",
    "s24_body_fat_pct",
    "s24_skeletal_muscle_kg",
)

# Domain -> SQL producing one row per date with already-prefixed feature columns.
DOMAIN_QUERIES: dict[str, str] = {
    "whoop": """
        SELECT coalesce(r.date, c.date, s.date) AS date,
               r.recovery_score            AS whoop_recovery_score,
               r.hrv_rmssd_ms              AS whoop_hrv_ms,
               r.resting_heart_rate_bpm    AS whoop_rhr_bpm,
               r.spo2_pct                  AS whoop_spo2_pct,
               r.skin_temp_c               AS whoop_skin_temp_c,
               c.strain                    AS whoop_strain,
               c.kilojoules                AS whoop_kilojoules,
               c.avg_heart_rate_bpm        AS whoop_avg_hr_bpm,
               s.sleep_performance_pct     AS whoop_sleep_performance_pct,
               s.sleep_efficiency_pct      AS whoop_sleep_efficiency_pct,
               s.sleep_consistency_pct     AS whoop_sleep_consistency_pct,
               s.total_sleep_hours         AS whoop_sleep_hours,
               s.deep_sleep_hours          AS whoop_deep_sleep_hours,
               s.rem_sleep_hours           AS whoop_rem_sleep_hours,
               s.respiratory_rate          AS whoop_respiratory_rate,
               s.disturbance_count         AS whoop_sleep_disturbances
        FROM whoop_recovery r
        FULL OUTER JOIN whoop_cycle c USING (date)
        FULL OUTER JOIN whoop_sleep s ON s.date = coalesce(r.date, c.date)
    """,
    "s24": """
        SELECT date,
               steps                    AS s24_steps,
               active_calories_kcal     AS s24_active_kcal,
               total_calories_kcal      AS s24_total_kcal,
               distance_m / 1000.0      AS s24_distance_km,
               resting_heart_rate_bpm   AS s24_resting_hr_bpm,
               weight_kg                AS s24_weight_kg,
               body_fat_pct             AS s24_body_fat_pct,
               skeletal_muscle_mass_kg  AS s24_skeletal_muscle_kg,
               phone_unlocks            AS screen_phone_unlocks
        FROM s24_health
    """,
    "renpho": """
        SELECT date,
               weight_kg         AS renpho_weight_kg,
               bmi               AS renpho_bmi,
               body_fat_pct      AS renpho_body_fat_pct,
               muscle_mass_kg    AS renpho_muscle_mass_kg,
               muscle_mass_pct   AS renpho_muscle_mass_pct,
               visceral_fat      AS renpho_visceral_fat,
               body_water_pct    AS renpho_body_water_pct,
               bmr_kcal          AS renpho_bmr_kcal
        FROM renpho_body
    """,
    "mfp": """
        SELECT date,
               calories_kcal AS mfp_calories_kcal,
               protein_g     AS mfp_protein_g,
               carbs_g       AS mfp_carbs_g,
               fat_g         AS mfp_fat_g,
               fiber_g       AS mfp_fiber_g,
               sugar_g       AS mfp_sugar_g,
               sodium_mg     AS mfp_sodium_mg,
               water_ml      AS mfp_water_ml
        FROM mfp_nutrition
    """,
    "laptop": """
        SELECT date,
               active_minutes  AS laptop_active_minutes,
               afk_minutes     AS laptop_afk_minutes,
               coding_minutes  AS laptop_coding_minutes,
               work_minutes    AS laptop_work_minutes,
               -- Hours after local midnight of the date (late nights exceed 24).
               CASE WHEN last_active IS NULL THEN NULL
                    ELSE date_diff('second', CAST(date AS TIMESTAMPTZ), last_active) / 3600.0
               END AS laptop_last_active_hour
        FROM laptop_activity
    """,
    "weather": """
        SELECT date,
               temp_mean_c          AS weather_temp_mean_c,
               temp_max_c           AS weather_temp_max_c,
               temp_min_c           AS weather_temp_min_c,
               precipitation_mm     AS weather_precipitation_mm,
               daylight_hours       AS weather_daylight_hours,
               sunshine_hours       AS weather_sunshine_hours,
               pressure_msl_hpa     AS weather_pressure_hpa,
               humidity_mean_pct    AS weather_humidity_pct,
               cloud_cover_mean_pct AS weather_cloud_cover_pct,
               wind_speed_max_kmh   AS weather_wind_max_kmh
        FROM weather_daily
    """,
    "subjective": """
        SELECT date, mood AS subjective_mood, energy AS subjective_energy, focus AS subjective_focus
        FROM subjective_state
    """,
}


# Derived features inherit the domains they are computed from, so that e.g.
# protein-per-kg is never reported as a "cross-domain" finding against protein.
DERIVED_SOURCES: dict[str, set[str]] = {
    "derived_protein_g_per_kg": {"mfp", "body"},
    "derived_protein_pct_kcal": {"mfp"},
    "derived_energy_balance_kcal": {"mfp", "whoop", "s24", "energy_expenditure"},
    "derived_weight_change_7d_kg": {"body"},
    "derived_strain_recovery_gap": {"whoop", "energy_expenditure"},
}


# Different devices measuring the same physiological quantity.
EQUIVALENT_FAMILIES: dict[str, frozenset[str]] = {
    "energy_expenditure": frozenset(
        {"whoop_kilojoules", "whoop_strain", "whoop_avg_hr_bpm", "s24_total_kcal", "s24_active_kcal"}
    ),
    # WHOOP recovery is computed from resting HR and HRV.
    "resting_hr": frozenset({"whoop_rhr_bpm", "s24_resting_hr_bpm", "whoop_recovery_score", "whoop_hrv_ms"}),
}


def column_domain(column: str) -> str:
    """Return the domain prefix of a feature column (``"whoop_hrv_ms"`` -> ``"whoop"``)."""
    return column.split("_", 1)[0]


def feature_families(column: str) -> set[str]:
    """Return the measurement families a feature belongs to.

    Two features are *cross-domain* only if their families are disjoint. Beyond the
    domain prefix, this groups body-composition columns from Renpho and the S24
    ("body"), and laptop screen time with ActivityWatch activity ("laptop_usage").
    """
    base = column.removesuffix("_ma7").removesuffix("_ma30")
    families = {column_domain(base)}
    if base in BODY_COMPOSITION_COLUMNS or families == {"renpho"}:
        families.add("body")
    # Every screen_* column except phone-only ones includes ActivityWatch time.
    if families == {"laptop"} or (base.startswith("screen_") and not base.startswith("screen_phone")):
        families.add("laptop_usage")
    families |= DERIVED_SOURCES.get(base, set())
    families |= {name for name, members in EQUIVALENT_FAMILIES.items() if base in members}
    return families


def feature_label(column: str) -> str:
    """Human-readable label for a feature column, e.g. ``"WHOOP · hrv ms"``."""
    domain, _, rest = column.partition("_")
    rest = rest.replace("_ma7", " (7d avg)").replace("_ma30", " (30d avg)")
    return f"{DOMAIN_LABELS.get(domain, domain)} · {rest.replace('_', ' ')}"


@dataclass(slots=True)
class PipelineOptions:
    """Tunable alignment behaviour."""

    body_ffill_days: int = 7
    fill_spend_zero: bool = True
    add_derived: bool = True
    add_calendar: bool = True
    drop_empty_columns: bool = True
    extra_domain_queries: dict[str, str] = field(default_factory=dict)


class DailyMatrixBuilder:
    """Build the unified daily master matrix from the DuckDB store."""

    def __init__(self, db: Database | None = None, options: PipelineOptions | None = None) -> None:
        """Args: db: database to read; options: alignment options."""
        self.db = db or Database()
        self.options = options or PipelineOptions()

    # ---------------------------------------------------------------- public
    def build(self, start: date, end: date) -> pd.DataFrame:
        """Return a DataFrame indexed by every date in ``[start, end]``.

        Args:
            start: First local date (inclusive).
            end: Last local date (inclusive).
        """
        index = pd.date_range(start, end, freq="D", name="date")
        parts: list[pd.DataFrame] = []
        for sql in {**DOMAIN_QUERIES, **self.options.extra_domain_queries}.values():
            frame = self._query(sql, start, end)
            if not frame.empty:
                parts.append(frame)
        parts.append(self._screen_features(start, end))
        parts.append(self._spend_features(start, end))

        matrix = pd.DataFrame(index=index)
        for part in parts:
            if part.empty:
                continue
            part = part.groupby(level=0).last()
            matrix = matrix.join(part, how="left")
        matrix = matrix.apply(pd.to_numeric, errors="coerce").astype("float64")

        self._forward_fill_body(matrix)
        if self.options.add_derived:
            self._add_derived(matrix)
        if self.options.add_calendar:
            matrix["calendar_day_of_week"] = matrix.index.dayofweek.astype("float64")
            matrix["calendar_is_weekend"] = (matrix.index.dayofweek >= 5).astype("float64")
        if self.options.drop_empty_columns:
            matrix = matrix.dropna(axis=1, how="all")
        logger.info("Built daily matrix %s x %s for %s..%s", *matrix.shape, start, end)
        return matrix

    # --------------------------------------------------------------- helpers
    def _query(self, sql: str, start: date, end: date) -> pd.DataFrame:
        wrapped = f"SELECT * FROM ({sql}) WHERE date BETWEEN ? AND ?"
        frame = self.db.query_df(wrapped, [start, end])
        if frame.empty:
            return frame
        frame["date"] = pd.to_datetime(frame["date"])
        return frame.set_index("date")

    def _screen_features(self, start: date, end: date) -> pd.DataFrame:
        usage = self.db.query_df(
            "SELECT date, device, coalesce(category, 'other') AS category, minutes, launches "
            "FROM screen_usage WHERE date BETWEEN ? AND ?",
            [start, end],
        )
        if usage.empty:
            return pd.DataFrame()
        usage["date"] = pd.to_datetime(usage["date"])
        usage["device_kind"] = np.where(usage["device"].eq("laptop"), "laptop", "phone")
        by_device = usage.pivot_table(index="date", columns="device_kind", values="minutes", aggfunc="sum")
        out = by_device.add_prefix("screen_").add_suffix("_minutes")
        out["screen_total_minutes"] = by_device.sum(axis=1, min_count=1)
        by_category = usage.pivot_table(index="date", columns="category", values="minutes", aggfunc="sum")
        by_category.columns = [f"screen_{slugify(c)}_minutes" for c in by_category.columns]
        out = out.join(by_category.drop(columns=[c for c in by_category if c in out], errors="ignore"))
        phone = usage.loc[usage["device_kind"].eq("phone")]
        if phone["launches"].notna().any():
            out["screen_phone_launches"] = phone.groupby("date")["launches"].sum(min_count=1)
        return out

    def _spend_features(self, start: date, end: date) -> pd.DataFrame:
        spend = self.db.query_df(
            "SELECT date, category, amount_gbp, transaction_count FROM monzo_spending WHERE date BETWEEN ? AND ?",
            [start, end],
        )
        if spend.empty:
            return pd.DataFrame()
        spend["date"] = pd.to_datetime(spend["date"])
        wide = spend.pivot_table(index="date", columns="category", values="amount_gbp", aggfunc="sum")
        wide.columns = [f"spend_{slugify(c)}_gbp" for c in wide.columns]
        wide["spend_total_gbp"] = wide.sum(axis=1, min_count=1)
        wide["spend_transactions"] = spend.groupby("date")["transaction_count"].sum()

        if self.options.fill_spend_zero:
            bounds = self.db.query_df("SELECT min(date) AS lo, max(date) AS hi FROM monzo_spending")
            lo, hi = pd.Timestamp(bounds.at[0, "lo"]), pd.Timestamp(bounds.at[0, "hi"])
            covered = pd.date_range(max(lo, pd.Timestamp(start)), min(hi, pd.Timestamp(end)), freq="D")
            wide = wide.reindex(wide.index.union(covered))
            wide.loc[covered] = wide.loc[covered].fillna(0.0)
        return wide

    def _forward_fill_body(self, matrix: pd.DataFrame) -> None:
        limit = self.options.body_ffill_days
        if limit <= 0:
            return
        cols = [c for c in BODY_COMPOSITION_COLUMNS if c in matrix.columns]
        if cols:
            matrix[cols] = matrix[cols].ffill(limit=limit)

    @staticmethod
    def _add_derived(matrix: pd.DataFrame) -> None:
        """Add cross-source features that no single source provides."""
        weight = _first_available(matrix, ["renpho_weight_kg", "s24_weight_kg"])
        if "mfp_protein_g" in matrix and weight is not None:
            matrix["derived_protein_g_per_kg"] = matrix["mfp_protein_g"] / weight
        if {"mfp_protein_g", "mfp_calories_kcal"} <= set(matrix.columns):
            calories = matrix["mfp_calories_kcal"].where(matrix["mfp_calories_kcal"] > 0)
            matrix["derived_protein_pct_kcal"] = 400.0 * matrix["mfp_protein_g"] / calories
        expenditure = None
        if "whoop_kilojoules" in matrix:
            expenditure = matrix["whoop_kilojoules"] / 4.184
        elif "s24_total_kcal" in matrix:
            expenditure = matrix["s24_total_kcal"]
        if expenditure is not None and "mfp_calories_kcal" in matrix:
            matrix["derived_energy_balance_kcal"] = matrix["mfp_calories_kcal"] - expenditure
        if weight is not None:
            matrix["derived_weight_change_7d_kg"] = weight - weight.shift(7)
        if {"whoop_strain", "whoop_recovery_score"} <= set(matrix.columns):
            # Positive = strain above what recovery supports (WHOOP strain 0-21 vs recovery 0-100).
            matrix["derived_strain_recovery_gap"] = matrix["whoop_strain"] / 21 * 100 - matrix["whoop_recovery_score"]


def _first_available(matrix: pd.DataFrame, columns: list[str]) -> pd.Series | None:
    present = [matrix[c] for c in columns if c in matrix]
    if not present:
        return None
    result = present[0].copy()
    for series in present[1:]:
        result = result.fillna(series)
    return result


def build_daily_matrix(start: date, end: date, db: Database | None = None, **options: object) -> pd.DataFrame:
    """Convenience wrapper around :class:`DailyMatrixBuilder`."""
    return DailyMatrixBuilder(db, PipelineOptions(**options)).build(start, end)  # type: ignore[arg-type]
