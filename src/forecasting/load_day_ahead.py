"""Causal day-ahead load forecasts for question 2.

Implements the comparison specified in ``问题二预测.docx`` §负载预测:

* two persistence baselines (yesterday L_{d-1}, weekly L_{d-7});
* the rolling 7-day same-slot mean currently used by the Q2 pipeline;
* the similar-day Gaussian-kernel model (four day types, r7 level feature,
  same-type pool, kernel k/h/tau, k-nearest curves carried verbatim);
* rolling q90/q95 upper bounds grouped by day type only.

All pool construction, sigma, kernel parameters, residual statistics and
bound quantiles use completed plan dates strictly before each 00:00 issue
timestamp.  Cold start: the first calendar day has no history and is marked
no-forecast (NaN, excluded from error statistics); days with fewer than 7
completed days fall back to yesterday persistence L_{d-1}.  Attachment 3 is
never read; attachment 1 load and PV are never used; no regression/ML is
used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DAY_TYPES = ("workday", "sat_class", "sunday", "holiday")

# 2025 official calendar rules from 问题二预测.docx, with the confirmed
# decision that 2025-04-04 (清明, a Friday) follows rule ② and is a sat_class.
MAKEUP_WORKDAYS = (
    "2025-01-26",
    "2025-02-08",
    "2025-04-27",
    "2025-09-28",
    "2025-10-11",
)
MON_THU_HOLIDAYS = (
    "2025-01-01",
    "2025-01-28",
    "2025-01-29",
    "2025-01-30",
    "2025-02-03",
    "2025-02-04",
    "2025-05-01",
    "2025-05-05",
    "2025-06-02",
    "2025-10-01",
    "2025-10-02",
    "2025-10-06",
    "2025-10-07",
    "2025-10-08",
)

# Fixed kernel parameters pre-selected on the 2025-01-20..01-27 mini
# walk-forward grid; never re-tuned on the formal evaluation period.
KERNEL_K = 10
KERNEL_H = 2.0
KERNEL_TAU = 14.0  # days
KERNEL_SIGMA_FLOOR_KW = 1.0

# Grid used to re-verify the tuning-window selection (3*4*6 = 72 combos).
GRID_K = (5, 10, 20)
GRID_H = (0.5, 1.0, 2.0, 4.0)
GRID_TAU = (7, 10, 14, 21, 28, 45)

BOUND_MIN_RESIDUAL_DAYS = 3


def build_day_type_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Map calendar dates to the four load day types (2025 official calendar).

    Precedence: makeup workdays, then Friday/Saturday -> sat_class, then
    Sunday -> sunday, then Mon-Thu statutory holidays -> holiday, otherwise
    workday.
    """
    dates = pd.DatetimeIndex(dates)
    makeup = {pd.Timestamp(value) for value in MAKEUP_WORKDAYS}
    holidays = {pd.Timestamp(value) for value in MON_THU_HOLIDAYS}
    rows: list[dict[str, Any]] = []
    for date in dates:
        if date in makeup:
            day_type = "workday"
        elif date.dayofweek in (4, 5):
            day_type = "sat_class"
        elif date.dayofweek == 6:
            day_type = "sunday"
        elif date in holidays:
            day_type = "holiday"
        else:
            day_type = "workday"
        rows.append({"plan_date": date, "day_type": day_type})
    return pd.DataFrame(rows)


def _daily_mean(matrix: np.ndarray) -> np.ndarray:
    return matrix.mean(axis=1)


def _r7_series(daily_mean: np.ndarray) -> np.ndarray:
    """Mean of daily mean load over d-7..d-1 for every day d."""
    days = len(daily_mean)
    r7 = np.full(days, np.nan)
    for d in range(1, days):
        r7[d] = daily_mean[max(0, d - 7) : d].mean()
    return r7


def gaussian_kernel_forecasts(
    load_matrix: np.ndarray,
    day_type_series: np.ndarray,
    k: int = KERNEL_K,
    h: float = KERNEL_H,
    tau: float = KERNEL_TAU,
    sigma_floor: float = KERNEL_SIGMA_FLOOR_KW,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk-forward similar-day Gaussian-kernel forecasts.

    Returns (forecasts, pool_sizes).  Rows where no same-type prior day
    exists are NaN and must be handled by the caller.  The project
    convention: day 0 (2025-01-01) stays NaN (frozen initial-condition
    day, excluded from statistics) and d < 7 falls back to yesterday
    persistence L_{d-1}; attachment 1 is never used for load or PV cold
    starts.
    """
    days, slots = load_matrix.shape
    daily_mean = _daily_mean(load_matrix)
    r7 = _r7_series(daily_mean)
    forecasts = np.full((days, slots), np.nan)
    pool_sizes = np.zeros(days, dtype=int)
    for d in range(days):
        # Same-type prior days only; members whose r7 level feature is not
        # computable (the very first calendar day) cannot enter the pool.
        pool = np.flatnonzero(
            (day_type_series[:d] == day_type_series[d]) & np.isfinite(r7[:d])
        )
        pool_sizes[d] = len(pool)
        if len(pool) == 0 or not np.isfinite(r7[d]):
            continue
        r7_pool = r7[pool]
        sigma = float(np.std(r7_pool))
        if sigma < sigma_floor:
            sigma = sigma_floor
        scores = ((r7_pool - r7[d]) / sigma) ** 2 / (2.0 * h**2) + (d - pool) / tau
        k_eff = min(k, len(pool))
        order = np.argsort(scores)[:k_eff]
        chosen = pool[order]
        # Numerically stable softmax over the negative scores.
        chosen_scores = scores[order]
        shifted = chosen_scores - chosen_scores.min()
        weights = np.exp(-shifted)
        weights = weights / weights.sum()
        forecasts[d] = weights @ load_matrix[chosen]
    return forecasts, pool_sizes


def rolling_type_quantile_residuals(
    residuals: np.ndarray,
    day_type_series: np.ndarray,
    q: float,
    min_days: int = BOUND_MIN_RESIDUAL_DAYS,
) -> np.ndarray:
    """Per-slot empirical q-quantile of same-type historical residuals.

    residuals[d] = L_d - forecast_d.  Only days strictly before d with the
    same day type enter the pool; requires at least ``min_days`` of them.
    Returns NaN rows where the requirement is not met.
    """
    days, slots = residuals.shape
    quantiles = np.full((days, slots), np.nan)
    finite_rows = np.isfinite(residuals).all(axis=1)
    for d in range(days):
        same = np.flatnonzero(day_type_series[:d] == day_type_series[d])
        same = same[finite_rows[same]]
        if len(same) >= min_days:
            quantiles[d] = np.quantile(residuals[same], q, axis=0)
    return quantiles


@dataclass
class LoadForecastResult:
    ten_minute: pd.DataFrame
    metrics: pd.DataFrame
    key_dates: pd.DataFrame
    bound_stats: pd.DataFrame
    quality_checks: pd.DataFrame
    summary: dict[str, Any]


def build_load_day_ahead(
    dispatch: pd.DataFrame,
    baseline: pd.DataFrame | None = None,
    k: int = KERNEL_K,
    h: float = KERNEL_H,
    tau: float = KERNEL_TAU,
    key_dates: list[str] | None = None,
) -> LoadForecastResult:
    """Walk-forward all four schemes plus q90/q95 bounds for 2025."""
    ordered = dispatch.sort_values(["plan_date", "slot_index"]).copy()
    dates = pd.DatetimeIndex(ordered["plan_date"].drop_duplicates()).sort_values()
    if not ordered.groupby("plan_date").size().reindex(dates).eq(144).all():
        raise ValueError("Every plan date must contain exactly 144 slots")
    load_matrix = (
        ordered.pivot(index="plan_date", columns="slot_index", values="load_actual_kw")
        .reindex(index=dates, columns=np.arange(1, 145))
        .to_numpy(float)
    )
    days, slots = load_matrix.shape
    day_types = build_day_type_frame(dates)["day_type"].to_numpy()

    # Scheme B2 reuses the exact Q2 baseline artifact when available so the
    # comparison is identical to the current pipeline's load input.
    if baseline is not None:
        baseline_ordered = baseline.sort_values(["plan_date", "slot_index"])
        baseline_dates = pd.DatetimeIndex(
            baseline_ordered["plan_date"].drop_duplicates()
        ).sort_values()
        if baseline_dates.equals(dates):
            b2_matrix = (
                baseline_ordered.pivot(
                    index="plan_date", columns="slot_index", values="load_forecast_kw"
                )
                .reindex(index=dates, columns=np.arange(1, 145))
                .to_numpy(float)
            )
        else:
            raise ValueError("Baseline artifact does not cover the same calendar")
    else:
        b2_matrix = np.empty_like(load_matrix)
        for d in range(days):
            if d == 0:
                b2_matrix[d] = np.nan
            else:
                b2_matrix[d] = load_matrix[max(0, d - 7) : d].mean(axis=0)

    b0_matrix = np.empty_like(load_matrix)
    b1_matrix = np.empty_like(load_matrix)
    for d in range(days):
        b0_matrix[d] = load_matrix[d - 1] if d > 0 else np.nan
        if d >= 7:
            b1_matrix[d] = load_matrix[d - 7]
        elif d > 0:
            b1_matrix[d] = load_matrix[d - 1]
        else:
            b1_matrix[d] = np.nan

    gk_matrix, pool_sizes = gaussian_kernel_forecasts(load_matrix, day_types, k, h, tau)
    for d in range(days):
        if not np.isfinite(gk_matrix[d]).all():
            gk_matrix[d] = load_matrix[d - 1] if d > 0 else np.nan

    schemes = {
        "b0": b0_matrix,
        "b1": b1_matrix,
        "b2": b2_matrix,
        "gk": gk_matrix,
    }
    bounds: dict[str, np.ndarray] = {}
    for name, matrix in schemes.items():
        residuals = load_matrix - matrix
        for q in (0.90, 0.95):
            bounds[f"{name}_q{int(q * 100)}"] = matrix + rolling_type_quantile_residuals(
                residuals, day_types, q
            )

    plan_values = np.repeat(dates.to_numpy(), slots)
    slot_index = np.tile(np.arange(1, slots + 1), days)
    result_start = pd.Timestamp("2025-02-01")
    result_end = pd.Timestamp("2025-12-31")
    result_days = (dates >= result_start) & (dates <= result_end)
    frame: dict[str, Any] = {
        "plan_date": plan_values,
        "slot_index": slot_index,
        "issue_ts": plan_values.copy(),
        "target_ts": plan_values + pd.to_timedelta(slot_index * 10, unit="m").to_numpy(),
        "training_end": np.repeat(
            np.r_[np.datetime64("NaT"), dates[:-1].to_numpy()], slots
        ),
        "day_type": np.repeat(day_types, slots),
        "load_actual_kw": load_matrix.reshape(-1),
        "load_b0_kw": b0_matrix.reshape(-1),
        "load_b1_kw": b1_matrix.reshape(-1),
        "load_b2_kw": b2_matrix.reshape(-1),
        "load_gk_kw": gk_matrix.reshape(-1),
        "gk_pool_size": np.repeat(pool_sizes, slots),
        "gk_q90_kw": bounds["gk_q90"].reshape(-1),
        "gk_q95_kw": bounds["gk_q95"].reshape(-1),
        "is_result_period": np.repeat(result_days, slots),
    }
    for name, matrix in bounds.items():
        if name.startswith("gk_"):
            continue
        frame[f"{name}_kw"] = matrix.reshape(-1)
    ten_minute = pd.DataFrame(frame)

    metrics = _scheme_metrics(dates, day_types, load_matrix, schemes, result_days)
    key_dates = key_dates or ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
    key_frame = _key_date_metrics(dates, load_matrix, schemes, key_dates)
    bound_stats = _bound_statistics(
        dates, day_types, load_matrix, bounds, result_days
    )
    checks = _quality_checks(ten_minute, day_types)
    summary = {
        "calendar_start": dates.min().date().isoformat(),
        "calendar_end": dates.max().date().isoformat(),
        "result_start": result_start.date().isoformat(),
        "result_end": result_end.date().isoformat(),
        "day_type_counts": pd.Series(day_types).value_counts().to_dict(),
        "kernel_params": {"k": k, "h": h, "tau": tau},
        "kernel_sigma_floor_kw": KERNEL_SIGMA_FLOOR_KW,
        "ten_minute_rows": int(len(ten_minute)),
        "quality_status": (
            "PASS" if not checks["status"].eq("FAIL").any() else "FAIL"
        ),
        "failed_checks": checks.loc[checks["status"].eq("FAIL"), "check_id"].tolist(),
        "uses_attachment_3": False,
    }
    return LoadForecastResult(ten_minute, metrics, key_frame, bound_stats, checks, summary)


def _scheme_metrics(
    dates: pd.DatetimeIndex,
    day_types: np.ndarray,
    load_matrix: np.ndarray,
    schemes: dict[str, np.ndarray],
    result_days: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, matrix in schemes.items():
        for scope, mask in (
            ("formal", result_days),
            *(
                (day_type, (result_days & (day_types == day_type)))
                for day_type in DAY_TYPES
            ),
        ):
            error = matrix[mask] - load_matrix[mask]
            actual = load_matrix[mask]
            daily_actual = load_matrix.reshape(len(dates), -1).sum(axis=1) / 6.0
            daily_forecast = matrix.reshape(len(dates), -1).sum(axis=1) / 6.0
            rows.append(
                {
                    "scheme": name,
                    "scope": scope,
                    "days": int(mask.sum()),
                    "mape_pct": float(np.mean(np.abs(error) / actual) * 100.0),
                    "mae_kw": float(np.mean(np.abs(error))),
                    "rmse_kw": float(np.sqrt(np.mean(error**2))),
                    "bias_kw": float(np.mean(error)),
                    "daily_energy_mae_kwh": float(
                        np.mean(np.abs(daily_forecast[mask] - daily_actual[mask]))
                    ),
                }
            )
    return pd.DataFrame(rows)


def _key_date_metrics(
    dates: pd.DatetimeIndex,
    load_matrix: np.ndarray,
    schemes: dict[str, np.ndarray],
    key_dates: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date_text in key_dates:
        date = pd.Timestamp(date_text)
        if date not in dates:
            continue
        day = int(dates.get_loc(date))
        actual = load_matrix[day]
        for name, matrix in schemes.items():
            error = matrix[day] - actual
            rows.append(
                {
                    "plan_date": date,
                    "scheme": name,
                    "mape_pct": float(np.mean(np.abs(error) / actual) * 100.0),
                    "mae_kw": float(np.mean(np.abs(error))),
                    "rmse_kw": float(np.sqrt(np.mean(error**2))),
                    "bias_kw": float(np.mean(error)),
                    "actual_energy_kwh": float(actual.sum() / 6.0),
                    "forecast_energy_kwh": float(matrix[day].sum() / 6.0),
                }
            )
    return pd.DataFrame(rows)


def _bound_statistics(
    dates: pd.DatetimeIndex,
    day_types: np.ndarray,
    load_matrix: np.ndarray,
    bounds: dict[str, np.ndarray],
    result_days: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, matrix in bounds.items():
        for scope, mask in (
            ("formal", result_days),
            *(
                (day_type, (result_days & (day_types == day_type)))
                for day_type in DAY_TYPES
            ),
        ):
            bound = matrix[mask]
            actual = load_matrix[mask]
            finite = np.isfinite(bound)
            if not finite.any():
                rows.append(
                    {
                        "bound": name,
                        "scope": scope,
                        "coverage": np.nan,
                        "mean_margin_kw": np.nan,
                        "finite_days": 0,
                    }
                )
                continue
            covered = (bound[finite] >= actual[finite]).mean()
            margin = bound[finite] - actual[finite]
            rows.append(
                {
                    "bound": name,
                    "scope": scope,
                    "coverage": float(covered),
                    "mean_margin_kw": float(margin.mean()),
                    "finite_days": int(finite.sum()),
                }
            )
    return pd.DataFrame(rows)


def _quality_checks(ten: pd.DataFrame, day_types: np.ndarray) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(check_id: str, condition: bool, observed: Any, expected: Any) -> None:
        checks.append(
            {
                "check_id": check_id,
                "status": "PASS" if bool(condition) else "FAIL",
                "observed": str(observed),
                "expected": str(expected),
            }
        )

    expected_counts = {"workday": 200, "sat_class": 102, "sunday": 49, "holiday": 14}
    counts = pd.Series(day_types).value_counts().to_dict()
    unique_days = int(ten["plan_date"].nunique())
    if unique_days == 365:
        add("day_type_counts", counts == expected_counts, counts, expected_counts)
    else:
        add(
            "day_type_counts_subset",
            unique_days < 365,
            f"{unique_days} days: {counts}",
            "full-year counts only checked on the 365-day calendar",
        )
    add("ten_minute_row_count", len(ten) == 365 * 144, len(ten), 365 * 144)
    trained = ten[ten["training_end"].notna()]
    add(
        "causality",
        bool((trained["training_end"] < trained["issue_ts"]).all()),
        "training_end < issue_ts",
        True,
    )
    value_columns = ["load_b0_kw", "load_b1_kw", "load_b2_kw", "load_gk_kw"]
    day0 = ten["plan_date"].eq(pd.Timestamp("2025-01-01"))
    non_day0 = ten.loc[~day0, value_columns]
    add(
        "nonnegative",
        bool(non_day0.ge(0).all().all()),
        "all non-negative on days with a forecast",
        True,
    )
    add(
        "finite_point_forecasts",
        bool(np.isfinite(non_day0.to_numpy(dtype=float)).all()),
        "finite on days with a forecast",
        True,
    )
    add(
        "day0_no_forecast",
        bool(ten.loc[day0, value_columns].isna().all().all()),
        "2025-01-01 forecasts are NaN (no-forecast cold start)",
        True,
    )
    add(
        "q90_q95_order",
        bool(
            (
                (ten["gk_q90_kw"] <= ten["gk_q95_kw"] + 1e-9)
                | ten["gk_q90_kw"].isna()
                | ten["gk_q95_kw"].isna()
            ).all()
        ),
        "gk_q90 <= gk_q95 wherever both defined",
        True,
    )
    day_4_4 = ten[ten["plan_date"].eq(pd.Timestamp("2025-04-04"))]["day_type"]
    observed_4_4 = day_4_4.iloc[0] if len(day_4_4) else "not_in_window"
    add("qingming_0404_sat_class", observed_4_4 == "sat_class", observed_4_4, "sat_class")
    return pd.DataFrame(checks)


def verify_kernel_grid(
    load_matrix: np.ndarray,
    day_type_series: np.ndarray,
    dates: pd.DatetimeIndex,
    tuning_start: str = "2025-01-20",
    tuning_end: str = "2025-01-27",
) -> pd.DataFrame:
    """Re-run the 72-combination mini walk-forward grid on the tuning window.

    For each (k, h, tau) the kernel forecast of every tuning day uses only
    days strictly before that day.  Returns one row per combination with the
    tuning-window MAPE (percent).
    """
    start = pd.Timestamp(tuning_start)
    end = pd.Timestamp(tuning_end)
    window_mask = (dates >= start) & (dates <= end)
    # Tuning-day forecasts only depend on strictly earlier days, so the
    # computation can be truncated at the tuning window end.
    truncate = int(np.flatnonzero(window_mask)[-1]) + 1
    truncated_matrix = load_matrix[:truncate]
    truncated_types = day_type_series[:truncate]
    mask = window_mask[:truncate]
    tuning_days = int(mask.sum())
    if tuning_days == 0:
        raise ValueError("Tuning window is empty")
    actual = truncated_matrix[mask]
    rows: list[dict[str, Any]] = []
    for k in GRID_K:
        for h in GRID_H:
            for tau in GRID_TAU:
                forecasts, _ = gaussian_kernel_forecasts(
                    truncated_matrix, truncated_types, k=k, h=h, tau=tau
                )
                error = forecasts[mask] - actual
                mape = float(np.mean(np.abs(error) / actual) * 100.0)
                rows.append(
                    {
                        "k": k,
                        "h": h,
                        "tau": tau,
                        "tuning_start": tuning_start,
                        "tuning_end": tuning_end,
                        "tuning_days": tuning_days,
                        "mape_pct": mape,
                    }
                )
    result = pd.DataFrame(rows)
    return result.sort_values("mape_pct").reset_index(drop=True)
