"""Build the unified 10-minute actual-operation table."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .build_timeline import build_timeline, parse_time_label


def _numeric_matrix(frame: pd.DataFrame) -> np.ndarray:
    matrix = frame.iloc[:, 1:145].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if matrix.shape[1] != 144:
        raise ValueError(f"Expected 144 value columns, received {matrix.shape[1]}")
    return matrix


def _display_time_label(value: Any) -> str:
    minute, next_day = parse_time_label(value)
    if next_day:
        return "0:00+1"
    return f"{minute // 60:02d}:{minute % 60:02d}"


def build_dispatch_10min(raw: dict[str, Any]) -> pd.DataFrame:
    load = raw["attachment_2_load"]
    pv = raw["attachment_2_pv"]
    price_variable = raw["attachment_4"]
    representative = raw["attachment_1"]

    dates = pd.to_datetime(load.iloc[:, 0]).dt.normalize()
    if not dates.equals(pd.to_datetime(pv.iloc[:, 0]).dt.normalize()):
        raise ValueError("Load and PV dates are not aligned")
    if not dates.equals(pd.to_datetime(price_variable.iloc[:, 0]).dt.normalize()):
        raise ValueError("Actuals and variable-price dates are not aligned")

    timeline = build_timeline(dates, load.columns[1:145])
    load_matrix = _numeric_matrix(load)
    pv_matrix = _numeric_matrix(pv)
    variable_matrix = _numeric_matrix(price_variable)
    load_values = load_matrix.reshape(-1)
    pv_values = pv_matrix.reshape(-1)

    # Load and PV columns are endpoint observations.  Tariffs are attached to
    # the interval start, so each calendar interval uses the source tariff whose
    # label equals that interval start.  A source flag records whether that
    # tariff is present in the supplied observations.
    variable_aligned = np.full_like(variable_matrix, np.nan, dtype=float)
    variable_aligned[:, 1:] = variable_matrix[:, :-1]
    variable_aligned[1:, 0] = variable_matrix[:-1, -1]
    variable_values = variable_aligned.reshape(-1)

    fixed_price_source = pd.to_numeric(
        representative.iloc[:, 1], errors="raise"
    ).to_numpy(float)
    if len(fixed_price_source) != 144:
        raise ValueError("Attachment 1 must contain 144 representative slots")
    fixed_price_calendar = np.r_[fixed_price_source[-1], fixed_price_source[:-1]]

    source_labels = np.asarray(
        [_display_time_label(value) for value in load.columns[1:145]], dtype=object
    )
    endpoint_labels = np.tile(source_labels, len(dates))
    fixed_price_labels = np.tile(
        np.concatenate((source_labels[-1:], source_labels[:-1])), len(dates)
    )

    dispatch = timeline.copy()
    dispatch["load_pv_source_time_label"] = endpoint_labels
    dispatch["load_pv_source_ts"] = dispatch["observation_ts"]
    dispatch["price_source_ts"] = dispatch["interval_start"]
    dispatch["price_fixed_source_time_label"] = fixed_price_labels
    dispatch["load_actual_kw"] = load_values
    dispatch["pv_actual_kw"] = pv_values
    dispatch["net_load_actual_kw"] = load_values - pv_values
    dispatch["load_actual_kwh"] = load_values / 6.0
    dispatch["pv_actual_kwh"] = pv_values / 6.0
    dispatch["net_load_actual_kwh"] = (load_values - pv_values) / 6.0
    dispatch["price_fixed_yuan_per_kwh"] = np.tile(
        fixed_price_calendar, len(dates)
    )
    dispatch["price_variable_yuan_per_kwh"] = variable_values
    dispatch["price_variable_available"] = np.isfinite(variable_values)
    columns = [
        "plan_date", "slot_index", "interval_start", "interval_end", "observation_ts",
        "calendar_date", "is_cross_day", "template_plan_date", "template_slot_index",
        "load_pv_source_time_label", "load_pv_source_ts", "price_source_ts",
        "price_fixed_source_time_label", "load_actual_kw", "pv_actual_kw", "net_load_actual_kw",
        "load_actual_kwh", "pv_actual_kwh", "net_load_actual_kwh",
        "price_fixed_yuan_per_kwh", "price_variable_yuan_per_kwh",
        "price_variable_available", "is_result_period",
    ]
    return dispatch.loc[:, columns]
