"""Forecast normalization, interpolation, and causal day-ahead baselines."""

from __future__ import annotations

from datetime import time
from typing import Any

import numpy as np
import pandas as pd


def _time_delta(value: Any) -> pd.Timedelta:
    if isinstance(value, time):
        return pd.Timedelta(hours=value.hour, minutes=value.minute, seconds=value.second)
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) < 2:
        raise ValueError(f"Unsupported issue time: {value!r}")
    return pd.Timedelta(hours=int(parts[0]), minutes=int(parts[1]))


def build_pv_forecast_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    date_values = frame.iloc[:, 0].replace(r"^\s*$", np.nan, regex=True).ffill()
    issue_dates = pd.to_datetime(date_values, errors="raise").dt.normalize()
    issue_deltas = frame.iloc[:, 1].map(_time_delta)
    issue_ts = issue_dates + issue_deltas
    forecasts = frame.iloc[:, 2:26].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if forecasts.shape[1] != 24:
        raise ValueError("Attachment 3 must provide 24 hourly horizons per issue")

    rows = len(frame)
    horizons = np.tile(np.arange(1, 25, dtype=int), rows)
    repeated_issue = pd.DatetimeIndex(issue_ts).repeat(24)
    hourly = pd.DataFrame(
        {
            "issue_date": repeated_issue.normalize(),
            "issue_time": repeated_issue.strftime("%H:%M"),
            "issue_ts": repeated_issue,
            "horizon_hour": horizons,
            "target_ts": repeated_issue + pd.to_timedelta(horizons, unit="h"),
            "pv_forecast_kw": forecasts.reshape(-1),
        }
    )
    return hourly


def build_pv_forecast_10min(
    hourly: pd.DataFrame, dispatch: pd.DataFrame
) -> pd.DataFrame:
    # Forecast target timestamps denote power-observation times.  Under the
    # endpoint convention, they must be matched to the delivery interval end.
    actual_lookup = dispatch.set_index("observation_ts")["pv_actual_kw"]
    blocks: list[pd.DataFrame] = []
    anchor_minutes = np.arange(0, 1441, 60, dtype=float)
    target_minutes = np.arange(10, 1441, 10, dtype=float)

    for issue_ts, group in hourly.groupby("issue_ts", sort=True):
        ordered = group.sort_values("horizon_hour")
        if ordered["horizon_hour"].tolist() != list(range(1, 25)):
            raise ValueError(f"Issue {issue_ts} does not contain horizons 1-24")
        anchor_actual = float(actual_lookup.get(issue_ts, 0.0))
        anchor_values = np.r_[anchor_actual, ordered["pv_forecast_kw"].to_numpy(float)]
        interpolated = np.maximum(np.interp(target_minutes, anchor_minutes, anchor_values), 0.0)
        targets = pd.Timestamp(issue_ts) + pd.to_timedelta(target_minutes, unit="m")
        blocks.append(
            pd.DataFrame(
                {
                    "issue_date": pd.Timestamp(issue_ts).normalize(),
                    "issue_time": pd.Timestamp(issue_ts).strftime("%H:%M"),
                    "issue_ts": pd.Timestamp(issue_ts),
                    "horizon_10min": np.arange(1, 145, dtype=int),
                    "lead_minutes": target_minutes.astype(int),
                    "target_ts": targets,
                    "pv_forecast_kw": interpolated,
                    "interpolation_method": "linear_observed_anchor",
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)


def build_day_ahead_baseline(
    dispatch: pd.DataFrame, representative: pd.DataFrame
) -> pd.DataFrame:
    """Build 00:00 forecasts of calendar-day endpoint observations.

    Each forecast row contains the endpoints 00:10, ..., 00:00+1.  These are
    observation targets and are not yet the official result-template slots.
    ``training_start`` and ``training_end`` identify source calendar dates.
    """
    ordered_dates = pd.DatetimeIndex(dispatch["plan_date"].drop_duplicates()).sort_values()
    load_matrix = (
        dispatch.pivot(index="plan_date", columns="slot_index", values="load_actual_kw")
        .reindex(ordered_dates)
        .to_numpy(float)
    )
    pv_matrix = (
        dispatch.pivot(index="plan_date", columns="slot_index", values="pv_actual_kw")
        .reindex(ordered_dates)
        .to_numpy(float)
    )
    representative_load = pd.to_numeric(representative.iloc[:, 2], errors="raise").to_numpy(float)
    representative_pv = pd.to_numeric(representative.iloc[:, 3], errors="raise").to_numpy(float)

    blocks: list[pd.DataFrame] = []
    for idx, plan_date in enumerate(ordered_dates):
        if idx == 0:
            load_forecast = representative_load
            pv_forecast = representative_pv
            history_days = 0
            training_start = pd.NaT
            training_end = pd.NaT
            source = "attachment_1_cold_start"
        else:
            start = max(0, idx - 7)
            load_forecast = load_matrix[start:idx].mean(axis=0)
            pv_forecast = pv_matrix[start:idx].mean(axis=0)
            history_days = idx - start
            training_start = ordered_dates[start]
            training_end = ordered_dates[idx - 1]
            source = "expanding_same_slot_mean" if idx < 7 else "rolling_7d_same_slot_mean"

        issue_ts = pd.Timestamp(plan_date)
        slot_index = np.arange(1, 145, dtype=int)
        blocks.append(
            pd.DataFrame(
                {
                    "plan_date": pd.Timestamp(plan_date),
                    "slot_index": slot_index,
                    "issue_ts": issue_ts,
                    "target_ts": issue_ts + pd.to_timedelta(slot_index * 10, unit="m"),
                    "training_start": training_start,
                    "training_end": training_end,
                    "history_days": history_days,
                    "load_forecast_kw": load_forecast,
                    "pv_forecast_kw": pv_forecast,
                    "forecast_source": source,
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)
