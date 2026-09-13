# -*- coding: utf-8 -*-
"""Question 4 fluctuating-price data layer (attachment 4).

Confirmed rule set (问题四口径):

* R0 alignment: attachment 4 time labels are interval STARTS, the same
  convention as attachment 1.  Physical interval t of day d uses column t-1
  of row d for t = 2..144 (labels 00:10 .. 23:50); the day's 00:00-00:10
  interval uses the PREVIOUS row's last column (label 0:00+1).  The single
  missing value (2025-01-01 00:00-00:10, no 2024-12-31 row) is filled with
  the 1 January 00:10 label price (warm-up ledger only).
* R1 information set: at 0:00 the day's prices are UNKNOWN and revealed
  interval by interval.  Decisions use the weekly-persistence forecast
  p_hat[d] = p_actual[d-7]; warm-up 1/2-1/7 uses the mean of available
  history, 1/1 is the zero-plan cold start (forecast inert, filled with the
  attachment-1 fixed curve as a placeholder).  Settlement always uses the
  realized attachment-4 price.
* R4 price scenarios: the same 21 historical days that supply the load/PV
  residuals also supply price residuals eps_p[i] = p_actual[i] - p_hat[i]
  (out-of-sample: p_hat[i] was built from history before day i).  Scenario
  price paths are p_hat[d] + eps_p[i] clipped at zero, paired same-day with
  the net-load scenarios (window first = max(7, d - residual_days)).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline.build_timeline import validate_source_headers

ROOT = Path(__file__).resolve().parents[2]
T = 144


def load_price_actual(root: Path = ROOT) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Realized prices, physically aligned, shape (365, 144), yuan/kWh.

    Row d column j carries the price of the interval STARTING at label j
    (j = 0 .. 143 -> labels 00:10 .. 0:00+1).  Physical interval 00:00-00:10
    of day d therefore uses row d-1's last column; 2025-01-01's first
    interval is filled with its own 00:10 label price per the confirmed R0.
    """
    raw = pd.read_excel(root / "附件/附件4.xlsx", sheet_name=0)
    validate_source_headers(raw.columns[1:])
    dates = pd.DatetimeIndex(pd.to_datetime(raw.iloc[:, 0])).normalize()
    assert dates.equals(pd.date_range("2025-01-01", "2025-12-31"))
    p = raw.iloc[:, 1:145].to_numpy(float)
    assert np.isfinite(p).all() and (p >= 0).all()
    out = np.empty((len(dates), T), dtype=float)
    out[:, 1:] = p[:, :143]
    out[1:, 0] = p[:-1, 143]
    out[0, 0] = p[0, 0]  # R0 confirmed fill for the 1 January first interval
    return dates, out


def load_fixed_price(root: Path = ROOT) -> np.ndarray:
    """Attachment-1 fixed price in the internal interval-start order (144,)."""
    raw = pd.read_excel(root / "附件/附件1.xlsx", sheet_name=0)
    src = raw.iloc[:, 1].to_numpy(float)
    assert src.shape == (T,)
    return np.r_[src[-1], src[:-1]]


def build_price_forecast(actual: np.ndarray, representative: np.ndarray) -> np.ndarray:
    """Weekly-persistence price forecast p_hat[d] = actual[d-7], (365, 144).

    d = 0 (1 January) is the zero-plan cold start: its forecast never enters
    any decision and is filled with the attachment-1 fixed curve as an inert
    placeholder.  d = 1..6 use the mean of the available history rows
    (confirmed warm-up rule); d >= 7 uses the full weekly window.
    """
    actual = np.asarray(actual, float)
    if actual.shape != (365, T):
        raise ValueError(f"expected (365, 144) actual prices, got {actual.shape}")
    out = np.full_like(actual, np.nan)
    out[0] = representative
    for d in range(1, 7):
        out[d] = actual[:d].mean(axis=0)
    for d in range(7, len(actual)):
        out[d] = actual[d - 7]
    assert np.isfinite(out).all()
    return out


def price_residual(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Out-of-sample residual eps_p[d] = actual[d] - p_hat[d], (365, 144)."""
    eps = actual - forecast
    return eps


def price_scenario_paths(d: int, forecast: np.ndarray, residual: np.ndarray,
                         residual_days: int = 21, h0: int = 0) -> np.ndarray | None:
    """Scenario price paths for day d over slots [h0, 144), (M, 144-h0).

    Uses exactly the window of the net-load scenario matrices
    (first = max(7, d - residual_days), M = d - first) so each price path
    stays same-day paired with the matching load/PV residual day.
    """
    first = max(7, d - residual_days)
    count = d - first
    if count <= 0:
        return None
    rows = np.maximum(0.0, forecast[d, h0:] + residual[first:d, h0:])
    return rows


def determinize_scenario_first_column(paths: np.ndarray, realized: float) -> np.ndarray:
    """Fix every scenario price path's first column to the realized price.

    The delivery interval starting at the update instant (0:00/6:00/12:00/
    18:00) has an interval-START price that is already realized when the
    decision is made (review correction 2), so that column carries no
    scenario uncertainty.  Returns a copy; the caller keeps the original
    unchanged.
    """
    out = np.asarray(paths, dtype=float).copy()
    out[:, 0] = float(realized)
    return out


def load_question4_prices(root: Path = ROOT):
    """Bundle used by the question 4 entry points."""
    dates, actual = load_price_actual(root)
    representative = load_fixed_price(root)
    forecast = build_price_forecast(actual, representative)
    residual = price_residual(actual, forecast)
    return {
        "dates": dates,
        "price_actual": actual,
        "price_forecast": forecast,
        "price_residual": residual,
    }
