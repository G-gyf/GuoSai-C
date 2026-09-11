"""Build the unified 10-minute actual-operation table."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .build_timeline import build_timeline


def _numeric_matrix(frame: pd.DataFrame) -> np.ndarray:
    matrix = frame.iloc[:, 1:145].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if matrix.shape[1] != 144:
        raise ValueError(f"Expected 144 value columns, received {matrix.shape[1]}")
    return matrix


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
    load_values = _numeric_matrix(load).reshape(-1)
    pv_values = _numeric_matrix(pv).reshape(-1)
    variable_values = _numeric_matrix(price_variable).reshape(-1)

    fixed_price = pd.to_numeric(representative.iloc[:, 1], errors="raise").to_numpy(float)
    if len(fixed_price) != 144:
        raise ValueError("Attachment 1 must contain 144 representative slots")

    dispatch = timeline.copy()
    dispatch["load_actual_kw"] = load_values
    dispatch["pv_actual_kw"] = pv_values
    dispatch["net_load_actual_kw"] = load_values - pv_values
    dispatch["load_actual_kwh"] = load_values / 6.0
    dispatch["pv_actual_kwh"] = pv_values / 6.0
    dispatch["net_load_actual_kwh"] = (load_values - pv_values) / 6.0
    dispatch["price_fixed_yuan_per_kwh"] = np.tile(fixed_price, len(dates))
    dispatch["price_variable_yuan_per_kwh"] = variable_values
    columns = [
        "plan_date", "slot_index", "interval_start", "interval_end", "calendar_date",
        "is_cross_day", "load_actual_kw", "pv_actual_kw", "net_load_actual_kw",
        "load_actual_kwh", "pv_actual_kwh", "net_load_actual_kwh",
        "price_fixed_yuan_per_kwh", "price_variable_yuan_per_kwh", "is_result_period",
    ]
    return dispatch.loc[:, columns]
