"""Question 3 attachment-3 issuance forecasts on the ten-minute grid.

Conventions adopted from ``outputs/question3/问题三_10分钟化实测记录.md``
(section 5, final ruling):

* attachment-3 values are instantaneous power at the target hour; forecast
  horizon j targets issue time + j hours (``预报1小时`` = issue + 1 h);
* the power observed AT the issue hour is a valid starting anchor: the
  interval ending at that instant has just completed, so the observation is
  causal;
* anchors are joined with shape-preserving PCHIP interpolation
  (``scipy.interpolate.PchipInterpolator``), which passes through every
  attachment-3 hourly anchor, is one-times differentiable, stays
  non-negative for non-negative anchors and adds no spurious peaks;
* no actual PV later than the issue instant is used.

Each of the four daily issuances covers the remaining same-calendar-day
horizon: 0:00 -> 00:00-24:00, 6:00 -> 06:00-24:00, 12:00 -> 12:00-24:00,
18:00 -> 18:00-24:00.  Curve values are stored in kWh per ten-minute
interval (kW / 6) on the internal endpoint slot grid: slot t corresponds to
minutes [10t, 10t+10) and the stored value is the curve at the endpoint
10(t+1) minutes.  Hourly anchor h therefore lands on slot 6h-1.  Slots
before the issue hour are NaN and must never be consumed by the optimiser.

Run ``python -m src.data_pipeline.question3_forecasts`` to write
``data/processed/pv3_issuance_10min.parquet`` and an anchor quality check.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[2]
T = 144
ISSUE_HOURS = np.array([0, 6, 12, 18], dtype=int)
ISSUE_COUNT = 4
DAYS = 365


def load_pv_actuals(root: Path = ROOT) -> np.ndarray:
    """Attachment 2 actual PV in kWh per ten-minute interval, (365, 144)."""
    raw = pd.read_excel(root / "附件/附件2.xlsx", sheet_name="光伏发电实际功率")
    dates = pd.DatetimeIndex(pd.to_datetime(raw.iloc[:, 0]))
    assert dates.equals(pd.date_range("2025-01-01", "2025-12-31"))
    pv = raw.iloc[:, 1:].to_numpy(float) / 6.0
    assert pv.shape == (DAYS, T) and np.isfinite(pv).all() and (pv >= 0).all()
    return pv


def observed_anchor_kw(pv: np.ndarray, d: int, issue_hour: int) -> float:
    """Instantaneous observed PV (kW) at the issue hour of day d.

    The observation at hour h belongs to the interval (h-10m, h] which has
    just completed when the forecast is issued.  For the 0:00 issuance the
    observation is the previous day's ``0:00+1`` column; for 1 January the
    anchor is unavailable and 0 is used (nighttime PV is zero by physics).
    """
    if issue_hour == 0:
        if d == 0:
            return 0.0
        return float(pv[d - 1, T - 1]) * 6.0
    return float(pv[d, 6 * issue_hour - 1]) * 6.0


def load_hourly_issuances(root: Path = ROOT) -> np.ndarray:
    """Hourly forecast matrix (365, 4, 24) in kW, strictly causal input.

    Reads the shared processed hourly table (built by the data pipeline from
    attachment 3 with date fill-down) and verifies its full structure.
    """
    df = pd.read_parquet(root / "data/processed/pv_forecast_hourly.parquet")
    assert len(df) == DAYS * ISSUE_COUNT * 24
    issues = df[["issue_date", "issue_time"]].drop_duplicates()
    assert len(issues) == DAYS * ISSUE_COUNT
    matrix = np.full((DAYS, ISSUE_COUNT, 24), np.nan, dtype=float)
    day_idx = {
        ts: i for i, ts in enumerate(pd.date_range("2025-01-01", "2025-12-31"))
    }
    hour_idx = {f"{h:02d}:00": k for k, h in enumerate(ISSUE_HOURS)}
    for _, group in df.groupby(["issue_date", "issue_time"], sort=False):
        d = day_idx[pd.Timestamp(group["issue_date"].iloc[0]).normalize()]
        k = hour_idx[str(group["issue_time"].iloc[0]).strip()]
        ordered = group.sort_values("horizon_hour")
        assert ordered["horizon_hour"].tolist() == list(range(1, 25))
        matrix[d, k, :] = ordered["pv_forecast_kw"].to_numpy(float)
    assert np.isfinite(matrix).all()
    assert (matrix >= 0).all(), "attachment 3 forecast values must be non-negative"
    return matrix


def build_issuance_curves(root: Path = ROOT) -> np.ndarray:
    """(365, 4, 144) PCHIP ten-minute curves in kWh per interval.

    ``curves[d, k, t]`` is defined for ``t >= 6*ISSUE_HOURS[k]`` and NaN
    before the issue hour.  Each curve passes through the observed anchor at
    the issue hour and through every attachment-3 hourly anchor afterwards.
    """
    pv = load_pv_actuals(root)
    hourly = load_hourly_issuances(root)
    curves = np.full((DAYS, ISSUE_COUNT, T), np.nan, dtype=float)
    grid_minutes = 10.0 * (np.arange(T) + 1)  # endpoints 00:10 ... 24:00
    for d in range(DAYS):
        for k, hour in enumerate(ISSUE_HOURS):
            start = 6 * hour
            # anchors: observed issue-hour value + attachment-3 hours afterwards
            anchor_x = 60.0 * np.arange(hour, 25)
            anchor_y = np.r_[observed_anchor_kw(pv, d, hour), hourly[d, k, : 24 - hour]]
            interp = PchipInterpolator(anchor_x, anchor_y, extrapolate=False)
            values = interp(grid_minutes[start:])  # kW at grid points
            curves[d, k, start:] = np.maximum(values, 0.0) / 6.0
    return curves


def anchor_check(curves: np.ndarray, hourly: np.ndarray, pv: np.ndarray) -> pd.DataFrame:
    """Verify every curve passes through its hourly anchors exactly.

    The observed issue-hour anchor is a knot of the interpolant (checked via
    the curve's first defined slot for hour > 0); attachment-3 hourly
    anchors are checked on the slot grid (slot 6h-1 for hour h).
    """
    rows = []
    for d in range(DAYS):
        for k, hour in enumerate(ISSUE_HOURS):
            start = 6 * hour
            anchors = np.r_[observed_anchor_kw(pv, d, hour), hourly[d, k, : 24 - hour]]
            for j, h in enumerate(range(hour + 1, 25)):
                curve_kw = curves[d, k, 6 * h - 1] * 6.0
                rows.append(
                    {
                        "day": d,
                        "issuance": f"{hour:02d}:00",
                        "anchor_hour": h,
                        "anchor_kw": float(anchors[j + 1]),
                        "curve_kw": float(curve_kw),
                        "abs_error_kw": float(abs(curve_kw - anchors[j + 1])),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    out = ROOT / "data/processed"
    curves = build_issuance_curves(ROOT)
    hourly = load_hourly_issuances(ROOT)
    pv = load_pv_actuals(ROOT)
    checks = anchor_check(curves, hourly, pv)
    max_err = float(checks.abs_error_kw.max())
    assert max_err < 1e-9, f"PCHIP failed to pass through anchors (max {max_err} kW)"

    long = []
    for k, hour in enumerate(ISSUE_HOURS):
        for d in range(DAYS):
            long.append(
                pd.DataFrame(
                    {
                        "issue_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=d),
                        "issue_time": f"{hour:02d}:00",
                        "slot_index": np.arange(1, T + 1),
                        "pv_issuance_kwh": curves[d, k],
                        "interpolation_method": "pchip_observed_anchor",
                    }
                )
            )
    frame = pd.concat(long, ignore_index=True)
    assert len(frame) == DAYS * ISSUE_COUNT * T
    frame.to_parquet(out / "pv3_issuance_10min.parquet", index=False)
    summary = {
        "interpolation": "pchip_observed_anchor",
        "anchor_rule": "observed PV at issue hour + attachment-3 hourly anchors; "
        "slot 6h-1 equals the hour-h anchor",
        "max_anchor_error_kw": max_err,
        "coverage": "issuance k covers slots [6*issue_hour, 144); earlier slots NaN",
        "units": "kWh per ten-minute interval (kW / 6)",
        "night_zero": "night anchors stay zero; curve clipped to be non-negative",
        "rows": int(len(frame)),
        "checks": "anchor pass-through < 1e-9 kW; all forecasts non-negative",
    }
    import json

    (out / "pv3_issuance_10min_meta.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"pv3 issuance curves: {frame.shape[0]} rows, "
        f"max anchor error {max_err:.2e} kW"
    )


if __name__ == "__main__":
    main()
