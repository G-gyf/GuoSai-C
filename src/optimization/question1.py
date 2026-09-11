"""Question 1 deterministic day-ahead dispatch using continuous LP only.

The model follows the definitions in ``C:\\Users\\lenovo\\Desktop\\1(1).docx``:
grid purchase, charge, discharge and curtailment are AC-side energy quantities,
while the storage state accounts for charge and discharge efficiency separately.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src.data_pipeline.build_timeline import parse_time_label, validate_source_headers


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SLOTS_PER_DAY = 144


@dataclass(frozen=True)
class Q1Parameters:
    interval_hours: float = 1.0 / 6.0
    initial_soc_kwh: float = 6000.0
    terminal_soc_kwh: float = 6000.0
    min_soc_kwh: float = 1200.0
    max_soc_kwh: float = 10800.0
    max_charge_power_kw: float = 5000.0
    max_discharge_power_kw: float = 5000.0
    max_battery_ramp_power_kw: float = 2000.0
    smoothing_cost_slack_rate: float = 0.0005
    charge_efficiency: float = 0.9
    discharge_efficiency: float = 0.9
    feasibility_tolerance: float = 1e-6
    zero_tolerance: float = 1e-9

    @property
    def max_charge_kwh(self) -> float:
        return self.max_charge_power_kw * self.interval_hours

    @property
    def max_discharge_kwh(self) -> float:
        return self.max_discharge_power_kw * self.interval_hours

    @property
    def max_battery_ramp_kwh(self) -> float:
        return self.max_battery_ramp_power_kw * self.interval_hours


@dataclass
class Q1Solution:
    schedule: pd.DataFrame
    block_summary: pd.DataFrame
    selected_intervals: pd.DataFrame
    summary: dict[str, Any]
    validation: dict[str, Any]


def _display_time_label(value: Any) -> str:
    minute, next_day = parse_time_label(value)
    if next_day:
        return "0:00+1"
    return f"{minute // 60:02d}:{minute % 60:02d}"


def load_question1_inputs(path: Path | str) -> pd.DataFrame:
    """Read and validate the 144-row representative day in attachment 1."""
    source = Path(path)
    frame = pd.read_excel(source, sheet_name=0)
    if frame.shape != (SLOTS_PER_DAY, 4):
        raise ValueError(
            f"Attachment 1 must contain 144 data rows and 4 columns; got {frame.shape}"
        )

    validate_source_headers(frame.iloc[:, 0].tolist())
    numeric = frame.iloc[:, 1:4].apply(pd.to_numeric, errors="raise")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Attachment 1 contains missing or non-finite numeric values")
    if (numeric < 0).any().any():
        raise ValueError("Price, load and photovoltaic power must be non-negative")

    return pd.DataFrame(
        {
            "slot_index": np.arange(1, SLOTS_PER_DAY + 1, dtype=int),
            "time_label": [_display_time_label(value) for value in frame.iloc[:, 0]],
            "clock_minute": [parse_time_label(value)[0] for value in frame.iloc[:, 0]],
            "is_next_day": [parse_time_label(value)[1] for value in frame.iloc[:, 0]],
            "price_yuan_per_kwh": numeric.iloc[:, 0].to_numpy(dtype=float),
            "load_kw": numeric.iloc[:, 1].to_numpy(dtype=float),
            "pv_kw": numeric.iloc[:, 2].to_numpy(dtype=float),
        }
    )


def _build_equalities(
    load_kwh: np.ndarray,
    pv_kwh: np.ndarray,
    parameters: Q1Parameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Build balance, storage-transition and terminal-state equalities."""
    n = len(load_kwh)
    variable_count = 5 * n
    grid, charge, discharge, curtailment, soc = 0, n, 2 * n, 3 * n, 4 * n
    matrix = np.zeros((2 * n + 1, variable_count), dtype=float)
    rhs = np.zeros(2 * n + 1, dtype=float)

    for t in range(n):
        matrix[t, grid + t] = 1.0
        matrix[t, charge + t] = -1.0
        matrix[t, discharge + t] = 1.0
        matrix[t, curtailment + t] = -1.0
        rhs[t] = load_kwh[t] - pv_kwh[t]

    for t in range(n):
        row = n + t
        matrix[row, soc + t] = 1.0
        matrix[row, charge + t] = -parameters.charge_efficiency
        matrix[row, discharge + t] = 1.0 / parameters.discharge_efficiency
        if t == 0:
            rhs[row] = parameters.initial_soc_kwh
        else:
            matrix[row, soc + t - 1] = -1.0

    matrix[-1, soc + n - 1] = 1.0
    rhs[-1] = parameters.terminal_soc_kwh
    return matrix, rhs


def _variable_bounds(n: int, parameters: Q1Parameters) -> list[tuple[float, float | None]]:
    return (
        [(0.0, None)] * n
        + [(0.0, parameters.max_charge_kwh)] * n
        + [(0.0, parameters.max_discharge_kwh)] * n
        + [(0.0, None)] * n
        + [(parameters.min_soc_kwh, parameters.max_soc_kwh)] * n
    )


def _build_battery_ramp_inequalities(
    n: int,
    parameters: Q1Parameters,
    variable_count: int,
    *,
    include_variation_epigraphs: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the cyclic battery ramp constraint and its variation epigraph."""
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    charge, discharge = n, 2 * n
    battery_variation = 5 * n

    for t in range(n):
        previous = (t - 1) % n
        battery_delta = np.zeros(variable_count, dtype=float)
        battery_delta[charge + t] = 1.0
        battery_delta[discharge + t] = -1.0
        battery_delta[charge + previous] = -1.0
        battery_delta[discharge + previous] = 1.0

        rows.extend([battery_delta, -battery_delta])
        rhs.extend(
            [
                parameters.max_battery_ramp_kwh,
                parameters.max_battery_ramp_kwh,
            ]
        )

        if include_variation_epigraphs:
            positive_battery = battery_delta.copy()
            negative_battery = -battery_delta.copy()
            positive_battery[battery_variation + t] = -1.0
            negative_battery[battery_variation + t] = -1.0
            rows.extend([positive_battery, negative_battery])
            rhs.extend([0.0, 0.0])

    return np.asarray(rows), np.asarray(rhs)


def _solve_three_stage_lp(inputs: pd.DataFrame, parameters: Q1Parameters) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve cost, smoothness and throughput objectives in strict lexicographic order."""
    n = len(inputs)
    price = inputs["price_yuan_per_kwh"].to_numpy(dtype=float)
    load_kwh = inputs["load_kw"].to_numpy(dtype=float) * parameters.interval_hours
    pv_kwh = inputs["pv_kw"].to_numpy(dtype=float) * parameters.interval_hours
    equalities, equality_rhs = _build_equalities(load_kwh, pv_kwh, parameters)
    bounds = _variable_bounds(n, parameters)

    cost_objective = np.zeros(5 * n, dtype=float)
    cost_objective[:n] = price

    unconstrained_reference = linprog(
        cost_objective,
        A_eq=equalities,
        b_eq=equality_rhs,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    if not unconstrained_reference.success:
        raise RuntimeError(
            "Question 1 unconstrained reference LP failed: "
            f"{unconstrained_reference.message}"
        )

    ramp_matrix, ramp_rhs = _build_battery_ramp_inequalities(
        n,
        parameters,
        5 * n,
        include_variation_epigraphs=False,
    )
    first = linprog(
        cost_objective,
        A_ub=ramp_matrix,
        b_ub=ramp_rhs,
        A_eq=equalities,
        b_eq=equality_rhs,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    if not first.success:
        raise RuntimeError(f"Question 1 primary LP failed: {first.message}")

    balance_shadow_price = np.asarray(first.eqlin.marginals[:n], dtype=float)
    storage_water_value = -np.asarray(first.eqlin.marginals[n : 2 * n], dtype=float)
    soc_lower_shadow_value = np.maximum(
        np.asarray(first.lower.marginals[4 * n : 5 * n], dtype=float), 0.0
    )
    soc_upper_shadow_value = np.maximum(
        -np.asarray(first.upper.marginals[4 * n : 5 * n], dtype=float), 0.0
    )
    ramp_marginals = np.asarray(first.ineqlin.marginals, dtype=float)
    ramp_residuals = np.asarray(first.ineqlin.residual, dtype=float)
    battery_ramp_shadow_value = -(
        ramp_marginals[0::2] + ramp_marginals[1::2]
    )
    battery_ramp_binding = (
        np.minimum(ramp_residuals[0::2], ramp_residuals[1::2]) <= 1e-7
    )

    cost_tolerance = max(1e-7, 1e-10 * abs(float(first.fun)))
    smoothing_cost_cap = (
        float(first.fun) * (1.0 + parameters.smoothing_cost_slack_rate)
        + cost_tolerance
    )
    extended_count = 6 * n
    extended_equalities = np.pad(equalities, ((0, 0), (0, n)))
    extended_bounds = bounds + [(0.0, None)] * n
    extended_cost_objective = np.zeros(extended_count, dtype=float)
    extended_cost_objective[: 5 * n] = cost_objective
    ramp_variation_matrix, ramp_variation_rhs = _build_battery_ramp_inequalities(
        n,
        parameters,
        extended_count,
        include_variation_epigraphs=True,
    )
    secondary_matrix = np.vstack(
        [ramp_variation_matrix, extended_cost_objective.reshape(1, -1)]
    )
    secondary_rhs = np.r_[ramp_variation_rhs, smoothing_cost_cap]

    variation_objective = np.zeros(extended_count, dtype=float)
    variation_objective[5 * n : 6 * n] = 1.0
    second = linprog(
        variation_objective,
        A_ub=secondary_matrix,
        b_ub=secondary_rhs,
        A_eq=extended_equalities,
        b_eq=equality_rhs,
        bounds=extended_bounds,
        method="highs",
        options={"presolve": True},
    )
    if not second.success:
        raise RuntimeError(f"Question 1 smoothness LP failed: {second.message}")

    variation_tolerance = max(1e-6, 1e-9 * abs(float(second.fun)))
    variation_cap_row = variation_objective.reshape(1, -1)
    tertiary_matrix = np.vstack([secondary_matrix, variation_cap_row])
    tertiary_rhs = np.r_[secondary_rhs, float(second.fun) + variation_tolerance]

    throughput_objective = np.zeros(extended_count, dtype=float)
    throughput_objective[n : 3 * n] = 1.0
    throughput_objective[3 * n : 4 * n] = 1e-9
    third = linprog(
        throughput_objective,
        A_ub=tertiary_matrix,
        b_ub=tertiary_rhs,
        A_eq=extended_equalities,
        b_eq=equality_rhs,
        bounds=extended_bounds,
        method="highs",
        options={"presolve": True},
    )
    if not third.success:
        raise RuntimeError(f"Question 1 throughput LP failed: {third.message}")

    metadata = {
        "solver": "scipy.optimize.linprog(method='highs')",
        "model_class": "continuous_linear_programming",
        "optimization_stages": 3,
        "unconstrained_reference_cost_yuan": float(unconstrained_reference.fun),
        "primary_status": int(first.status),
        "primary_message": first.message,
        "primary_optimal_cost_yuan": float(first.fun),
        "secondary_status": int(second.status),
        "secondary_message": second.message,
        "cost_tolerance_yuan": float(cost_tolerance),
        "smoothing_cost_cap_yuan": float(smoothing_cost_cap),
        "secondary_cost_yuan": float(extended_cost_objective @ second.x),
        "secondary_total_variation_kwh": float(second.fun),
        "variation_tolerance_kwh": float(variation_tolerance),
        "tertiary_status": int(third.status),
        "tertiary_message": third.message,
        "tertiary_cost_yuan": float(extended_cost_objective @ third.x),
        "tertiary_total_variation_kwh": float(variation_objective @ third.x),
        "tertiary_throughput_objective": float(throughput_objective @ third.x),
        "_primary_balance_shadow_price": balance_shadow_price,
        "_primary_storage_water_value": storage_water_value,
        "_primary_soc_lower_shadow_value": soc_lower_shadow_value,
        "_primary_soc_upper_shadow_value": soc_upper_shadow_value,
        "_primary_battery_ramp_shadow_value": battery_ramp_shadow_value,
        "_primary_battery_ramp_binding": battery_ramp_binding,
    }
    return third.x[: 5 * n], metadata


def _block_name(clock_minute: int) -> str:
    start_hour = (clock_minute // 240) * 4
    return f"{start_hour}:00-{start_hour + 4}:00"


def _build_outputs(
    inputs: pd.DataFrame,
    vector: np.ndarray,
    solver_metadata: dict[str, Any],
    parameters: Q1Parameters,
) -> Q1Solution:
    solver_metadata = dict(solver_metadata)
    balance_shadow_price = np.asarray(
        solver_metadata.pop("_primary_balance_shadow_price"), dtype=float
    )
    storage_water_value = np.asarray(
        solver_metadata.pop("_primary_storage_water_value"), dtype=float
    )
    soc_lower_shadow_value = np.asarray(
        solver_metadata.pop("_primary_soc_lower_shadow_value"), dtype=float
    )
    soc_upper_shadow_value = np.asarray(
        solver_metadata.pop("_primary_soc_upper_shadow_value"), dtype=float
    )
    battery_ramp_shadow_value = np.asarray(
        solver_metadata.pop("_primary_battery_ramp_shadow_value"), dtype=float
    )
    battery_ramp_binding = np.asarray(
        solver_metadata.pop("_primary_battery_ramp_binding"), dtype=bool
    )
    n = len(inputs)
    grid = vector[:n].copy()
    charge = vector[n : 2 * n].copy()
    discharge = vector[2 * n : 3 * n].copy()
    curtailment = vector[3 * n : 4 * n].copy()
    soc_end = vector[4 * n : 5 * n].copy()
    for values in (grid, charge, discharge, curtailment, soc_end):
        values[np.abs(values) < parameters.zero_tolerance] = 0.0

    schedule = inputs.copy()
    schedule["load_kwh"] = schedule["load_kw"] * parameters.interval_hours
    schedule["pv_kwh"] = schedule["pv_kw"] * parameters.interval_hours
    schedule["grid_purchase_kwh"] = grid
    schedule["charge_kwh"] = charge
    schedule["discharge_kwh"] = discharge
    schedule["curtailment_kwh"] = curtailment
    schedule["soc_start_kwh"] = np.r_[parameters.initial_soc_kwh, soc_end[:-1]]
    schedule["soc_end_kwh"] = soc_end
    schedule["interval_cost_yuan"] = schedule["price_yuan_per_kwh"] * grid
    schedule["primary_balance_shadow_price_yuan_per_kwh"] = balance_shadow_price
    schedule["primary_storage_water_value_yuan_per_kwh"] = storage_water_value
    schedule["primary_soc_lower_shadow_value_yuan_per_kwh"] = (
        soc_lower_shadow_value
    )
    schedule["primary_soc_upper_shadow_value_yuan_per_kwh"] = (
        soc_upper_shadow_value
    )
    schedule["primary_battery_ramp_shadow_value_yuan_per_kwh"] = (
        battery_ramp_shadow_value
    )
    schedule["primary_battery_ramp_binding"] = battery_ramp_binding
    schedule["block"] = schedule["clock_minute"].map(_block_name)

    block_order = [f"{hour}:00-{hour + 4}:00" for hour in range(0, 24, 4)]
    block_summary = (
        schedule.groupby("block", sort=False)[["charge_kwh", "discharge_kwh"]]
        .sum()
        .reindex(block_order)
        .reset_index()
        .rename(columns={"block": "time_block"})
    )
    if block_summary[["charge_kwh", "discharge_kwh"]].isna().any().any():
        raise RuntimeError("Four-hour charge/discharge aggregation is incomplete")

    selected_times = ["10:00", "12:00", "14:00", "16:00", "18:00", "20:00"]
    selected = schedule[schedule["time_label"].isin(selected_times)].copy()
    selected = selected.set_index("time_label").reindex(selected_times).reset_index()
    if selected["grid_purchase_kwh"].isna().any():
        raise RuntimeError("One or more paper-table time points are missing")

    baseline_purchase = np.maximum(schedule["load_kwh"] - schedule["pv_kwh"], 0.0)
    baseline_cost = float(np.dot(schedule["price_yuan_per_kwh"], baseline_purchase))
    total_cost = float(schedule["interval_cost_yuan"].sum())
    summary: dict[str, Any] = {
        **solver_metadata,
        "parameters": asdict(parameters),
        "total_purchase_kwh": float(grid.sum()),
        "total_cost_yuan": total_cost,
        "total_charge_kwh": float(charge.sum()),
        "total_discharge_kwh": float(discharge.sum()),
        "total_curtailment_kwh": float(curtailment.sum()),
        "min_soc_kwh": float(soc_end.min()),
        "max_soc_kwh": float(soc_end.max()),
        "baseline_purchase_kwh": float(baseline_purchase.sum()),
        "baseline_cost_yuan": baseline_cost,
        "cost_saving_yuan": baseline_cost - total_cost,
        "cost_saving_rate": (baseline_cost - total_cost) / baseline_cost,
        "engineering_cost_increase_yuan": (
            total_cost - solver_metadata["unconstrained_reference_cost_yuan"]
        ),
        "engineering_cost_increase_rate": (
            total_cost / solver_metadata["unconstrained_reference_cost_yuan"] - 1.0
        ),
    }
    validation = validate_solution(schedule, summary, parameters)
    return Q1Solution(schedule, block_summary, selected, summary, validation)


def validate_solution(
    schedule: pd.DataFrame,
    summary: dict[str, Any],
    parameters: Q1Parameters,
) -> dict[str, Any]:
    tol = parameters.feasibility_tolerance
    balance = (
        schedule["grid_purchase_kwh"]
        + schedule["pv_kwh"]
        + schedule["discharge_kwh"]
        - schedule["load_kwh"]
        - schedule["charge_kwh"]
        - schedule["curtailment_kwh"]
    )
    soc_transition = (
        schedule["soc_end_kwh"]
        - schedule["soc_start_kwh"]
        - parameters.charge_efficiency * schedule["charge_kwh"]
        + schedule["discharge_kwh"] / parameters.discharge_efficiency
    )
    simultaneous = (schedule["charge_kwh"] > tol) & (schedule["discharge_kwh"] > tol)
    grid_energy = schedule["grid_purchase_kwh"].to_numpy(dtype=float)
    battery_net_energy = (
        schedule["charge_kwh"] - schedule["discharge_kwh"]
    ).to_numpy(dtype=float)
    grid_ramp_kw = np.abs(
        np.diff(np.r_[grid_energy[-1], grid_energy])
    ) / parameters.interval_hours
    battery_ramp_kw = np.abs(
        np.diff(np.r_[battery_net_energy[-1], battery_net_energy])
    ) / parameters.interval_hours
    final_battery_total_variation_kwh = float(
        np.abs(
            np.diff(np.r_[battery_net_energy[-1], battery_net_energy])
        ).sum()
    )
    operating_mode = np.where(
        schedule["charge_kwh"].to_numpy(dtype=float) > tol,
        1,
        np.where(schedule["discharge_kwh"].to_numpy(dtype=float) > tol, -1, 0),
    )
    checks = {
        "status": "PASS",
        "slot_count": int(len(schedule)),
        "max_balance_residual_kwh": float(balance.abs().max()),
        "max_soc_transition_residual_kwh": float(soc_transition.abs().max()),
        "min_soc_kwh": float(schedule["soc_end_kwh"].min()),
        "max_soc_kwh": float(schedule["soc_end_kwh"].max()),
        "terminal_soc_kwh": float(schedule["soc_end_kwh"].iloc[-1]),
        "max_charge_kwh": float(schedule["charge_kwh"].max()),
        "max_discharge_kwh": float(schedule["discharge_kwh"].max()),
        "simultaneous_charge_discharge_slots": int(simultaneous.sum()),
        "minimum_grid_purchase_kwh": float(schedule["grid_purchase_kwh"].min()),
        "minimum_curtailment_kwh": float(schedule["curtailment_kwh"].min()),
        "observed_max_grid_ramp_power_kw": float(grid_ramp_kw.max()),
        "max_battery_ramp_power_kw": float(battery_ramp_kw.max()),
        "final_battery_total_variation_kwh": final_battery_total_variation_kwh,
        "operating_mode_changes_including_wrap": int(
            (operating_mode != np.roll(operating_mode, 1)).sum()
        ),
        "cost_recalculation_error_yuan": float(
            abs(
                summary["total_cost_yuan"]
                - np.dot(schedule["price_yuan_per_kwh"], schedule["grid_purchase_kwh"])
            )
        ),
        "smoothing_cost_cap_margin_yuan": float(
            summary["smoothing_cost_cap_yuan"] - summary["total_cost_yuan"]
        ),
        "variation_preservation_gap_kwh": float(
            final_battery_total_variation_kwh
            - summary["secondary_total_variation_kwh"]
        ),
    }
    failures = []
    if len(schedule) != SLOTS_PER_DAY:
        failures.append("slot_count")
    if checks["max_balance_residual_kwh"] > tol:
        failures.append("power_balance")
    if checks["max_soc_transition_residual_kwh"] > tol:
        failures.append("soc_transition")
    if checks["min_soc_kwh"] < parameters.min_soc_kwh - tol:
        failures.append("soc_lower_bound")
    if checks["max_soc_kwh"] > parameters.max_soc_kwh + tol:
        failures.append("soc_upper_bound")
    if abs(checks["terminal_soc_kwh"] - parameters.terminal_soc_kwh) > tol:
        failures.append("terminal_soc")
    if checks["max_charge_kwh"] > parameters.max_charge_kwh + tol:
        failures.append("charge_limit")
    if checks["max_discharge_kwh"] > parameters.max_discharge_kwh + tol:
        failures.append("discharge_limit")
    if checks["simultaneous_charge_discharge_slots"]:
        failures.append("simultaneous_charge_discharge")
    if checks["minimum_grid_purchase_kwh"] < -tol:
        failures.append("grid_purchase_nonnegative")
    if checks["minimum_curtailment_kwh"] < -tol:
        failures.append("curtailment_nonnegative")
    if checks["max_battery_ramp_power_kw"] > parameters.max_battery_ramp_power_kw + tol:
        failures.append("battery_ramp_limit")
    if checks["cost_recalculation_error_yuan"] > tol:
        failures.append("cost_recalculation")
    if checks["smoothing_cost_cap_margin_yuan"] < -tol:
        failures.append("smoothing_cost_cap")
    if checks["variation_preservation_gap_kwh"] > summary["variation_tolerance_kwh"] + tol:
        failures.append("lexicographic_variation")
    if failures:
        checks["status"] = "FAIL"
        checks["failed_checks"] = failures
        raise RuntimeError(f"Question 1 validation failed: {', '.join(failures)}")
    checks["failed_checks"] = []
    return checks


def solve_question1(
    inputs: pd.DataFrame,
    parameters: Q1Parameters | None = None,
) -> Q1Solution:
    parameters = parameters or Q1Parameters()
    vector, metadata = _solve_three_stage_lp(inputs, parameters)
    return _build_outputs(inputs, vector, metadata, parameters)


def run_battery_ramp_sensitivity(
    inputs: pd.DataFrame,
    ramp_power_values_kw: list[float] | tuple[float, ...] | np.ndarray,
    base_parameters: Q1Parameters | None = None,
) -> pd.DataFrame:
    """Resolve Question 1 over alternative battery ramp-power limits."""
    base = base_parameters or Q1Parameters()
    rows: list[dict[str, Any]] = []
    for ramp_power_kw in ramp_power_values_kw:
        value = float(ramp_power_kw)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Battery ramp-power limits must be positive finite values")
        parameters = replace(base, max_battery_ramp_power_kw=value)
        try:
            solution = solve_question1(inputs, parameters)
        except RuntimeError as error:
            rows.append(
                {
                    "battery_ramp_power_kw_per_10min": value,
                    "battery_ramp_rate_kw_per_min": value / 10.0,
                    "battery_ramp_energy_kwh": value * parameters.interval_hours,
                    "status": "INFEASIBLE",
                    "message": str(error),
                }
            )
            continue

        summary = solution.summary
        validation = solution.validation
        rows.append(
            {
                "battery_ramp_power_kw_per_10min": value,
                "battery_ramp_rate_kw_per_min": value / 10.0,
                "battery_ramp_energy_kwh": value * parameters.interval_hours,
                "status": validation["status"],
                "primary_cost_yuan": summary["primary_optimal_cost_yuan"],
                "primary_cost_increase_rate": (
                    summary["primary_optimal_cost_yuan"]
                    / summary["unconstrained_reference_cost_yuan"]
                    - 1.0
                ),
                "final_cost_yuan": summary["total_cost_yuan"],
                "final_cost_increase_rate": summary["engineering_cost_increase_rate"],
                "cost_saving_rate": summary["cost_saving_rate"],
                "observed_max_battery_ramp_power_kw": validation[
                    "max_battery_ramp_power_kw"
                ],
                "battery_total_variation_kwh": validation[
                    "final_battery_total_variation_kwh"
                ],
                "charge_discharge_throughput_kwh": summary["total_charge_kwh"]
                + summary["total_discharge_kwh"],
                "operating_mode_changes": validation[
                    "operating_mode_changes_including_wrap"
                ],
                "min_soc_kwh": validation["min_soc_kwh"],
                "max_soc_kwh": validation["max_soc_kwh"],
                "simultaneous_charge_discharge_slots": validation[
                    "simultaneous_charge_discharge_slots"
                ],
                "message": "",
            }
        )
    return pd.DataFrame(rows)


def _write_battery_ramp_sensitivity(
    inputs: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write the battery-ramp sensitivity table and its publication figure."""
    values = [500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7500, 10000]
    sensitivity = run_battery_ramp_sensitivity(inputs, values)
    csv_path = output_dir / "question1_battery_ramp_sensitivity.csv"
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / "储能爬坡率敏感性分析.png"
    sensitivity.to_csv(csv_path, index=False, encoding="utf-8-sig")

    feasible = sensitivity[sensitivity["status"].eq("PASS")].copy()
    x = feasible["battery_ramp_power_kw_per_10min"].to_numpy(dtype=float)
    primary_cost = 100.0 * feasible["primary_cost_increase_rate"].to_numpy(dtype=float)
    final_cost = 100.0 * feasible["final_cost_increase_rate"].to_numpy(dtype=float)
    variation = feasible["battery_total_variation_kwh"].to_numpy(dtype=float) / 1000.0
    switches = feasible["operating_mode_changes"].to_numpy(dtype=float)
    style = {
        "font.family": "Noto Sans SC",
        "font.size": 8.2,
        "axes.titlesize": 9.3,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.2,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,
        "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85))
        academic_blue = "#6F9FC1"
        academic_green = "#72AD98"
        warm_gray = "#D4A35F"
        cool_gray = "#9275A8"
        reference_gray = "#C4C8CD"
        marker_style = {"markersize": 3.0, "markeredgewidth": 0.7}

        axes[0].plot(
            x,
            primary_cost,
            marker="o",
            markerfacecolor="white",
            color=academic_blue,
            linewidth=1.15,
            label="第一阶段成本增幅",
            **marker_style,
        )
        axes[0].plot(
            x,
            final_cost,
            marker="s",
            markerfacecolor="white",
            color=cool_gray,
            linewidth=1.05,
            label="最终成本增幅",
            **marker_style,
        )
        axes[0].axvline(
            2000.0, color=reference_gray, linestyle="--", linewidth=0.9, zorder=0
        )
        axes[0].set_title("(a) 经济性代价", loc="left", fontweight="bold")
        axes[0].set_xlabel("储能净功率变化上限（kW/10 min）")
        axes[0].set_ylabel("成本增幅（%）")
        axes[0].legend(loc="upper right")

        axes[1].plot(
            x,
            variation,
            marker="o",
            markerfacecolor="white",
            color=academic_green,
            linewidth=1.15,
            label="净动作总变差",
            **marker_style,
        )
        switch_axis = axes[1].twinx()
        switch_axis.plot(
            x,
            switches,
            marker="s",
            markerfacecolor="white",
            color=warm_gray,
            linewidth=1.05,
            label="模式切换次数",
            **marker_style,
        )
        axes[1].axvline(
            2000.0, color=reference_gray, linestyle="--", linewidth=0.9, zorder=0
        )
        axes[1].set_title("(b) 运行平滑性", loc="left", fontweight="bold")
        axes[1].set_xlabel("储能净功率变化上限（kW/10 min）")
        axes[1].set_ylabel("净动作总变差（MWh）")
        switch_axis.set_ylabel("模式切换次数")
        lines = axes[1].get_lines()[:1] + switch_axis.get_lines()[:1]
        axes[1].legend(lines, [line.get_label() for line in lines], loc="lower right")

        display_ticks = [500, 1000, 2000, 5000, 10000]
        for axis in axes:
            axis.set_xscale("log")
            axis.set_xticks(display_ticks, [str(value) for value in display_ticks])
            axis.grid(axis="y", color="#E2E5E9", linewidth=0.5)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_color("#8F969E")
            axis.spines["bottom"].set_color("#8F969E")
            axis.tick_params(direction="out", length=2.5, width=0.6, colors="#40464D")
        switch_axis.spines["top"].set_visible(False)
        switch_axis.spines["right"].set_color("#8F969E")
        switch_axis.tick_params(direction="out", length=2.5, width=0.6, colors="#40464D")
        axes[0].annotate(
            "基准",
            xy=(2000.0, np.interp(2000.0, x, final_cost)),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=6.8,
            color="#777D84",
        )
        fig.text(
            0.5,
            0.012,
            "注：横轴采用对数尺度；浅灰虚线为基准情景 2000 kW/10 min。成本增幅均相对无爬坡理论最优成本计算。",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color="#59616A",
        )
        fig.subplots_adjust(left=0.09, right=0.91, bottom=0.27, top=0.90, wspace=0.34)
        fig.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
    return csv_path, png_path


def _write_figures_cn(solution: Q1Solution, output_dir: Path) -> list[Path]:
    """Export the Chinese publication figure set and the dual mechanism figure."""
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    schedule = solution.schedule
    parameters = solution.summary["parameters"]
    interval_hours = float(parameters["interval_hours"])
    time_hours = (
        schedule["clock_minute"].to_numpy(dtype=float) / 60.0
        + 24.0 * schedule["is_next_day"].to_numpy(dtype=float)
    )
    x_limits = (0.0, float(time_hours[-1] + interval_hours))
    time_ticks = np.arange(0.0, 24.1, 4.0)
    time_labels = [f"{int(hour):02d}:00" for hour in time_ticks]

    price = schedule["price_yuan_per_kwh"].to_numpy(dtype=float)
    load_power = schedule["load_kw"].to_numpy(dtype=float)
    pv_power = schedule["pv_kw"].to_numpy(dtype=float)
    net_load_power = load_power - pv_power
    grid_power = (
        schedule["grid_purchase_kwh"].to_numpy(dtype=float) / interval_hours
    )
    baseline_grid_power = np.maximum(net_load_power, 0.0)
    charge_power = schedule["charge_kwh"].to_numpy(dtype=float) / interval_hours
    discharge_power = schedule["discharge_kwh"].to_numpy(dtype=float) / interval_hours
    net_battery_power = charge_power - discharge_power
    curtailment_power = (
        schedule["curtailment_kwh"].to_numpy(dtype=float) / interval_hours
    )
    pv_surplus_power = np.maximum(pv_power - load_power, 0.0)
    absorbed_surplus_power = np.maximum(pv_surplus_power - curtailment_power, 0.0)
    soc = schedule["soc_end_kwh"].to_numpy(dtype=float)

    colors = {
        "ink": "#222222",
        "muted": "#6B7280",
        "grid": "#D9DEE7",
        "blue": "#0072B2",
        "sky": "#56B4E9",
        "orange": "#E69F00",
        "green": "#009E73",
        "vermillion": "#D55E00",
        "purple": "#7A5195",
        "yellow": "#F0E442",
    }
    style = {
        "font.family": "Noto Sans SC",
        "font.size": 8.5,
        "axes.titlesize": 9.8,
        "axes.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "axes.unicode_minus": False,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2,
        "legend.frameon": False,
        "lines.linewidth": 1.45,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }

    def format_time_axis(axis: Any, *, show_xlabels: bool) -> None:
        axis.set_xlim(*x_limits)
        axis.set_xticks(time_ticks, time_labels if show_xlabels else [])
        axis.grid(axis="y", color=colors["grid"], linewidth=0.55, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=3.0, width=0.7)

    def add_boolean_spans(axis: Any, mask: np.ndarray, color: str, alpha: float) -> None:
        padded = np.r_[False, mask, False].astype(int)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        for start, end in zip(starts, ends):
            left = time_hours[start]
            right = time_hours[end - 1] + interval_hours
            axis.axvspan(left, right, color=color, alpha=alpha, linewidth=0)

    def save_figure(figure: Any, stem: str) -> Path:
        png_path = figure_dir / f"{stem}.png"
        figure.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.04)
        plt.close(figure)
        return png_path

    output_paths: list[Path] = []
    with plt.rc_context(style):
        figure1, axes1 = plt.subplots(
            4,
            1,
            figsize=(7.2, 8.35),
            sharex=True,
            gridspec_kw={
                "height_ratios": [1.02, 0.18, 1.34, 1.26],
                "hspace": 0.16,
            },
        )
        figure1.suptitle(
            "图1 价格信号与需求侧响应",
            x=0.5,
            y=0.985,
            fontsize=12.5,
            fontweight="bold",
        )

        valley_threshold = 0.436
        peak_threshold = 0.984
        axes1[0].step(
            time_hours,
            price,
            where="post",
            color=colors["purple"],
            label="分时电价",
        )
        axes1[0].axhline(
            float(price.mean()),
            color=colors["muted"],
            linestyle="--",
            linewidth=1.0,
            label=f"日均电价 {price.mean():.3f} 元/kWh",
        )
        axes1[0].axhline(
            valley_threshold,
            color=colors["green"],
            linestyle=(0, (3, 2)),
            linewidth=0.95,
            label="低价阈值 0.436",
        )
        axes1[0].axhline(
            peak_threshold,
            color=colors["vermillion"],
            linestyle=(0, (3, 2)),
            linewidth=0.95,
            label="高价阈值 0.984",
        )
        axes1[0].set_title("(a) 电价信号与实际响应窗口", loc="left", fontweight="bold")
        axes1[0].set_ylabel("电价\n（元/kWh）")
        axes1[0].legend(loc="upper left", ncol=2)
        format_time_axis(axes1[0], show_xlabels=False)

        action_width = interval_hours * 0.92
        charge_active = charge_power > 1e-7
        discharge_active = discharge_power > 1e-7
        axes1[1].bar(
            time_hours[charge_active],
            np.full(int(charge_active.sum()), 0.34),
            bottom=0.56,
            width=action_width,
            color=colors["green"],
            linewidth=0,
        )
        axes1[1].bar(
            time_hours[discharge_active],
            np.full(int(discharge_active.sum()), 0.34),
            bottom=0.10,
            width=action_width,
            color=colors["orange"],
            linewidth=0,
        )
        axes1[1].set_ylim(0.0, 1.0)
        axes1[1].set_yticks([0.73, 0.27], ["充电", "放电"])
        axes1[1].set_ylabel(
            "实际窗口", fontsize=7.2, rotation=0, ha="right", va="center", labelpad=8
        )
        axes1[1].grid(False)
        for side in ("top", "right", "bottom"):
            axes1[1].spines[side].set_visible(False)
        axes1[1].spines["left"].set_color(colors["grid"])
        axes1[1].tick_params(axis="y", length=0, labelsize=6.8)
        axes1[1].tick_params(axis="x", length=0, labelbottom=False)

        axes1[2].plot(time_hours, load_power, color=colors["ink"], label="负荷")
        axes1[2].plot(time_hours, pv_power, color=colors["orange"], label="光伏")
        axes1[2].plot(
            time_hours,
            net_load_power,
            color=colors["blue"],
            linestyle="--",
            linewidth=1.2,
            label="净负荷",
        )
        absorbed_top = load_power + absorbed_surplus_power
        axes1[2].fill_between(
            time_hours,
            load_power,
            absorbed_top,
            where=pv_surplus_power > 1e-9,
            color=colors["green"],
            alpha=0.17,
            step="post",
            label="被储能吸收的光伏盈余",
        )
        axes1[2].axhline(0.0, color=colors["muted"], linewidth=0.7)
        axes1[2].set_title("(b) 负荷、光伏与净负荷", loc="left", fontweight="bold")
        axes1[2].set_ylabel("功率（kW）")
        axes1[2].legend(loc="upper right", ncol=2)
        axes1[2].text(
            0.99,
            0.06,
            f"光伏盈余 {pv_surplus_power.sum()*interval_hours/1000:.3f} MWh，全部由储能吸收；弃光为 0",
            transform=axes1[2].transAxes,
            ha="right",
            va="bottom",
            fontsize=7.4,
            color=colors["muted"],
        )
        format_time_axis(axes1[2], show_xlabels=False)

        purchase_delta = grid_power - baseline_grid_power
        increase_mask = purchase_delta >= 0.0
        axes1[3].bar(
            time_hours,
            np.where(increase_mask, purchase_delta, 0.0),
            width=action_width,
            color=colors["sky"],
            alpha=0.72,
            linewidth=0,
            label="增购（最优−基准 > 0）",
        )
        axes1[3].bar(
            time_hours,
            np.where(~increase_mask, purchase_delta, 0.0),
            width=action_width,
            color=colors["orange"],
            alpha=0.66,
            linewidth=0,
            label="减购（最优−基准 < 0）",
        )
        axes1[3].axhline(0.0, color=colors["ink"], linewidth=0.7)
        baseline_axis = axes1[3].twinx()
        baseline_axis.step(
            time_hours,
            baseline_grid_power,
            where="post",
            color="#AEB4BB",
            linestyle="--",
            linewidth=0.85,
            alpha=0.65,
            label="无储能基准购电（右轴）",
        )
        axes1[3].set_title("(c) 储能引起的购电量变化", loc="left", fontweight="bold")
        axes1[3].set_ylabel("Δ购电功率（kW）")
        baseline_axis.set_ylabel("基准购电功率（kW）", color="#8E949B")
        axes1[3].set_xlabel("时刻")
        delta_handles, delta_labels = axes1[3].get_legend_handles_labels()
        base_handles, base_labels = baseline_axis.get_legend_handles_labels()
        axes1[3].legend(
            delta_handles + base_handles,
            delta_labels + base_labels,
            loc="upper center",
            ncol=3,
        )
        format_time_axis(axes1[3], show_xlabels=True)
        baseline_axis.spines["top"].set_visible(False)
        baseline_axis.spines["right"].set_color("#C8CDD2")
        baseline_axis.tick_params(
            direction="out", length=2.5, width=0.6, colors="#8E949B"
        )

        figure1.text(
            0.5,
            0.012,
            "注：无储能基准为逐时段购电量 max(负荷电量−光伏电量, 0)，不配置储能且不售电。基准情景：储能净功率变化上限 2000 kW/10 min，ε=0.05%，充放电效率均为 90%，SOC 为 1200–10800 kWh，0:00 与 24:00 储电量均为 6000 kWh。",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=colors["muted"],
        )
        figure1.subplots_adjust(left=0.105, right=0.90, bottom=0.105, top=0.945)
        output_paths.append(save_figure(figure1, "图1_价格信号与需求侧响应"))

        figure2, axes2 = plt.subplots(
            3,
            1,
            figsize=(7.2, 6.9),
            sharex=True,
            gridspec_kw={"height_ratios": [1.02, 1.0, 1.0], "hspace": 0.12},
        )
        figure2.suptitle(
            "图2 储能策略与经济后果",
            x=0.5,
            y=0.985,
            fontsize=12.5,
            fontweight="bold",
        )

        axes2[0].bar(
            time_hours,
            charge_power,
            width=interval_hours * 0.82,
            color=colors["green"],
            label="充电功率",
        )
        axes2[0].bar(
            time_hours,
            -discharge_power,
            width=interval_hours * 0.82,
            color=colors["vermillion"],
            label="放电功率",
        )
        price_axis = axes2[0].twinx()
        price_axis.step(
            time_hours,
            price,
            where="post",
            color=colors["purple"],
            linewidth=1.15,
            alpha=0.85,
            label="分时电价",
        )
        axes2[0].axhline(0.0, color=colors["ink"], linewidth=0.7)
        axes2[0].set_title("(a) 储能充放电功率与电价", loc="left", fontweight="bold")
        axes2[0].set_ylabel("储能功率（kW）\n充电为正")
        price_axis.set_ylabel("电价（元/kWh）")
        handles1, labels1 = axes2[0].get_legend_handles_labels()
        handles2, labels2 = price_axis.get_legend_handles_labels()
        axes2[0].legend(handles1 + handles2, labels1 + labels2, loc="upper center", ncol=3)
        format_time_axis(axes2[0], show_xlabels=False)
        price_axis.spines["top"].set_visible(False)
        price_axis.tick_params(direction="out", length=3.0, width=0.7)

        min_soc = float(parameters["min_soc_kwh"])
        max_soc = float(parameters["max_soc_kwh"])
        axes2[1].axhspan(min_soc, max_soc, color=colors["green"], alpha=0.06)
        axes2[1].step(time_hours, soc, where="post", color=colors["green"], label="储电量")
        axes2[1].axhline(min_soc, color=colors["muted"], linestyle="--", linewidth=0.9)
        axes2[1].axhline(max_soc, color=colors["muted"], linestyle="--", linewidth=0.9)
        upper_hits = np.isclose(soc, max_soc, atol=1e-5)
        lower_hits = np.isclose(soc, min_soc, atol=1e-5)
        axes2[1].scatter(
            time_hours[upper_hits],
            soc[upper_hits],
            marker="v",
            s=22,
            color=colors["vermillion"],
            label="触及上限",
            zorder=4,
        )
        axes2[1].scatter(
            time_hours[lower_hits],
            soc[lower_hits],
            marker="^",
            s=22,
            color=colors["blue"],
            label="触及下限",
            zorder=4,
        )
        axes2[1].set_title("(b) 储电量轨迹与运行区间", loc="left", fontweight="bold")
        axes2[1].set_ylabel("储电量（kWh）")
        axes2[1].legend(loc="upper center", ncol=3)
        format_time_axis(axes2[1], show_xlabels=False)

        baseline_interval_cost = price * baseline_grid_power * interval_hours
        optimized_interval_cost = price * grid_power * interval_hours
        baseline_cumulative = np.cumsum(baseline_interval_cost)
        optimized_cumulative = np.cumsum(optimized_interval_cost)
        axes2[2].plot(
            time_hours,
            baseline_cumulative / 1000.0,
            color=colors["muted"],
            linestyle="--",
            label="无储能基准",
        )
        axes2[2].plot(
            time_hours,
            optimized_cumulative / 1000.0,
            color=colors["blue"],
            label="三阶段最优方案",
        )
        axes2[2].fill_between(
            time_hours,
            optimized_cumulative / 1000.0,
            baseline_cumulative / 1000.0,
            where=baseline_cumulative >= optimized_cumulative,
            color=colors["blue"],
            alpha=0.12,
        )
        gross_saving = float(np.maximum(baseline_interval_cost - optimized_interval_cost, 0.0).sum())
        extra_cost = float(np.maximum(optimized_interval_cost - baseline_interval_cost, 0.0).sum())
        net_saving = gross_saving - extra_cost
        axes2[2].text(
            0.50,
            0.16,
            f"高价少购节省 {gross_saving/1000:.2f} 千元  −  低价多购成本 {extra_cost/1000:.2f} 千元  =  净节省 {net_saving/1000:.2f} 千元",
            transform=axes2[2].transAxes,
            ha="center",
            va="center",
            fontsize=7.5,
            color=colors["ink"],
        )
        axes2[2].set_title("(c) 累计购电成本与节省归因", loc="left", fontweight="bold")
        axes2[2].set_ylabel("累计成本（千元）")
        axes2[2].set_xlabel("时刻")
        axes2[2].legend(loc="upper left")
        format_time_axis(axes2[2], show_xlabels=True)

        figure2.text(
            0.5,
            0.012,
            "注：累计成本按各时段电价×购电量计算；节省归因为高价时段少购电形成的费用减少，扣除低价时段为储能充电而增加的购电费用。基准情景参数同图1。",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=colors["muted"],
        )
        figure2.subplots_adjust(left=0.105, right=0.90, bottom=0.105, top=0.945)
        output_paths.append(save_figure(figure2, "图2_储能策略与经济后果"))

        figure3, axes3 = plt.subplots(
            2,
            1,
            figsize=(7.2, 4.95),
            sharex=True,
            gridspec_kw={"height_ratios": [1.35, 0.82], "hspace": 0.16},
        )
        figure3.suptitle(
            "图3 调度机制的对偶解释",
            x=0.5,
            y=0.99,
            fontsize=12.5,
            fontweight="bold",
        )
        balance_shadow = schedule[
            "primary_balance_shadow_price_yuan_per_kwh"
        ].to_numpy(dtype=float)
        water_value = schedule[
            "primary_storage_water_value_yuan_per_kwh"
        ].to_numpy(dtype=float)
        soc_lower_shadow = schedule[
            "primary_soc_lower_shadow_value_yuan_per_kwh"
        ].to_numpy(dtype=float)
        soc_upper_shadow = schedule[
            "primary_soc_upper_shadow_value_yuan_per_kwh"
        ].to_numpy(dtype=float)
        ramp_shadow = schedule[
            "primary_battery_ramp_shadow_value_yuan_per_kwh"
        ].to_numpy(dtype=float)
        charge_efficiency = float(parameters["charge_efficiency"])
        discharge_efficiency = float(parameters["discharge_efficiency"])

        axes3[0].step(
            time_hours,
            price,
            where="post",
            color=colors["purple"],
            linewidth=1.45,
            label=r"$\pi$：外网电价",
        )
        axes3[0].step(
            time_hours,
            balance_shadow,
            where="post",
            color=colors["blue"],
            linewidth=1.30,
            label=r"$\lambda_{\mathrm{balance}}$：功率平衡影子价格",
        )
        axes3[0].step(
            time_hours,
            water_value,
            where="post",
            color=colors["green"],
            linewidth=1.30,
            label=r"$\lambda_{\mathrm{water}}$：储能水价值",
        )
        axes3[0].step(
            time_hours,
            price / charge_efficiency,
            where="post",
            color="#AFB5BC",
            linestyle="-",
            linewidth=0.85,
            label=r"$\pi/\eta$：充电边界",
        )
        axes3[0].step(
            time_hours,
            discharge_efficiency * price,
            where="post",
            color="#D0D4D8",
            linestyle="-",
            linewidth=0.85,
            label=r"$\eta\cdot\pi$：放电边界",
        )
        axes3[0].set_title(
            "(a) 电价、功率平衡对偶与储能水价值",
            loc="left",
            fontweight="bold",
        )
        axes3[0].set_ylabel("边际价值（元/kWh）")
        axes3[0].legend(loc="upper left", ncol=2)
        format_time_axis(axes3[0], show_xlabels=False)

        def shadow_shades(values: np.ndarray, base_color: str) -> np.ndarray:
            maximum = max(float(values.max()), 1e-12)
            intensity = np.clip(values / maximum, 0.0, 1.0)
            base = np.asarray(matplotlib.colors.to_rgb(base_color), dtype=float)
            return np.asarray(
                [1.0 - (0.18 + 0.82 * level) * (1.0 - base) for level in intensity]
            )

        shadow_rows = [
            (ramp_shadow, 1.0, colors["purple"]),
            (soc_lower_shadow, 2.0, colors["blue"]),
            (soc_upper_shadow, 3.0, colors["vermillion"]),
        ]
        for shadow_values, row, base_color in shadow_rows:
            active = shadow_values > 1e-10
            if not active.any():
                continue
            axes3[1].bar(
                time_hours[active],
                np.full(int(active.sum()), 0.58),
                bottom=row - 0.29,
                width=interval_hours * 0.94,
                color=shadow_shades(shadow_values[active], base_color),
                linewidth=0,
            )
        axes3[1].set_yticks(
            [1, 2, 3],
            [
                f"储能爬坡  max={ramp_shadow.max():.3f}",
                f"SOC 下限  max={soc_lower_shadow.max():.3f}",
                f"SOC 上限  max={soc_upper_shadow.max():.3f}",
            ],
        )
        axes3[1].set_ylim(0.55, 3.45)
        axes3[1].set_title(
            "(b) 约束触界时段及影子价格强度",
            loc="left",
            fontweight="bold",
        )
        axes3[1].set_xlabel("时刻")
        axes3[1].text(
            0.995,
            0.04,
            "同一约束内：浅色 → 深色表示影子价格由低到高",
            transform=axes3[1].transAxes,
            ha="right",
            va="bottom",
            fontsize=7.0,
            color=colors["muted"],
        )
        format_time_axis(axes3[1], show_xlabels=True)

        figure3.text(
            0.5,
            0.012,
            "注：对偶量及约束影子价格均来自第一阶段成本最小化 LP；η=0.90。π/η 与 η·π 分别给出理想充电和放电边界；(b) 各行独立归一化色阶，仅比较同类约束内部强度。基准情景参数同图1。",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=colors["muted"],
        )
        figure3.subplots_adjust(left=0.19, right=0.985, bottom=0.15, top=0.86)
        output_paths.append(save_figure(figure3, "图3_调度机制的对偶解释"))

    return output_paths


def _write_word_combined_figure(solution: Q1Solution, output_dir: Path) -> Path:
    """Export a compact two-column Figure 1 + Figure 2 image for Word."""
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    schedule = solution.schedule
    parameters = solution.summary["parameters"]
    interval_hours = float(parameters["interval_hours"])
    time_hours = (
        schedule["clock_minute"].to_numpy(dtype=float) / 60.0
        + 24.0 * schedule["is_next_day"].to_numpy(dtype=float)
    )
    x_limits = (0.0, float(time_hours[-1] + interval_hours))
    time_ticks = np.arange(0.0, 24.1, 6.0)
    time_labels = [f"{int(hour):02d}:00" for hour in time_ticks]

    price = schedule["price_yuan_per_kwh"].to_numpy(dtype=float)
    load_power = schedule["load_kw"].to_numpy(dtype=float)
    pv_power = schedule["pv_kw"].to_numpy(dtype=float)
    net_load_power = load_power - pv_power
    grid_power = schedule["grid_purchase_kwh"].to_numpy(dtype=float) / interval_hours
    baseline_grid_power = np.maximum(net_load_power, 0.0)
    charge_power = schedule["charge_kwh"].to_numpy(dtype=float) / interval_hours
    discharge_power = schedule["discharge_kwh"].to_numpy(dtype=float) / interval_hours
    soc = schedule["soc_end_kwh"].to_numpy(dtype=float)
    pv_surplus_power = np.maximum(pv_power - load_power, 0.0)
    curtailment_power = (
        schedule["curtailment_kwh"].to_numpy(dtype=float) / interval_hours
    )
    absorbed_surplus_power = np.maximum(pv_surplus_power - curtailment_power, 0.0)

    colors = {
        "ink": "#292929",
        "muted": "#747B84",
        "grid": "#E0E5EA",
        "blue": "#5A91B8",
        "sky": "#86C5E3",
        "orange": "#E2AD52",
        "green": "#64A98E",
        "vermillion": "#D47756",
        "purple": "#8B6AA3",
    }
    style = {
        "font.family": "Noto Sans SC",
        "font.size": 7.2,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.3,
        "axes.linewidth": 0.7,
        "axes.unicode_minus": False,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.2,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }

    def format_axis(axis: Any, *, show_xlabels: bool) -> None:
        axis.set_xlim(*x_limits)
        axis.set_xticks(time_ticks, time_labels if show_xlabels else [])
        axis.grid(axis="y", color=colors["grid"], linewidth=0.45)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=2.2, width=0.55)

    with plt.rc_context(style):
        figure = plt.figure(figsize=(9.4, 8.4))
        outer = figure.add_gridspec(
            1,
            2,
            left=0.095,
            right=0.905,
            bottom=0.10,
            top=0.91,
            wspace=0.34,
        )
        left_grid = outer[0].subgridspec(
            4, 1, height_ratios=[0.88, 0.18, 1.10, 1.02], hspace=0.20
        )
        right_grid = outer[1].subgridspec(
            3, 1, height_ratios=[1.02, 1.0, 1.0], hspace=0.12
        )
        left_axes = [figure.add_subplot(left_grid[index]) for index in range(4)]
        right_axes = [figure.add_subplot(right_grid[index]) for index in range(3)]

        figure.text(
            0.285,
            0.965,
            "价格信号与需求侧响应",
            ha="center",
            va="top",
            fontsize=12.0,
            fontweight="bold",
        )
        figure.text(
            0.715,
            0.965,
            "储能策略与经济后果",
            ha="center",
            va="top",
            fontsize=12.0,
            fontweight="bold",
        )

        left_axes[0].step(
            time_hours, price, where="post", color=colors["purple"], label="分时电价"
        )
        left_axes[0].axhline(
            float(price.mean()),
            color=colors["muted"],
            linestyle="--",
            linewidth=0.8,
            label=f"均价 {price.mean():.3f}",
        )
        left_axes[0].axhline(
            0.436,
            color=colors["green"],
            linestyle=(0, (3, 2)),
            linewidth=0.8,
            label="低价阈值 0.436",
        )
        left_axes[0].axhline(
            0.984,
            color=colors["vermillion"],
            linestyle=(0, (3, 2)),
            linewidth=0.8,
            label="高价阈值 0.984",
        )
        left_axes[0].set_title("(a) 电价信号与实际响应窗口", loc="left", fontweight="bold")
        left_axes[0].set_ylabel("电价\n(元/kWh)")
        left_axes[0].legend(loc="upper left", ncol=2)
        format_axis(left_axes[0], show_xlabels=False)

        action_width = interval_hours * 0.92
        charge_active = charge_power > 1e-7
        discharge_active = discharge_power > 1e-7
        left_axes[1].bar(
            time_hours[charge_active],
            np.full(int(charge_active.sum()), 0.34),
            bottom=0.56,
            width=action_width,
            color=colors["green"],
            linewidth=0,
        )
        left_axes[1].bar(
            time_hours[discharge_active],
            np.full(int(discharge_active.sum()), 0.34),
            bottom=0.10,
            width=action_width,
            color=colors["orange"],
            linewidth=0,
        )
        left_axes[1].set_xlim(*x_limits)
        left_axes[1].set_ylim(0.0, 1.0)
        left_axes[1].set_yticks([0.73, 0.27], ["充电", "放电"])
        left_axes[1].set_ylabel(
            "实际窗口", fontsize=6.6, rotation=0, ha="right", va="center", labelpad=7
        )
        left_axes[1].grid(False)
        for side in ("top", "right", "bottom"):
            left_axes[1].spines[side].set_visible(False)
        left_axes[1].spines["left"].set_color(colors["grid"])
        left_axes[1].tick_params(axis="y", length=0, labelsize=6.2)
        left_axes[1].tick_params(axis="x", length=0, labelbottom=False)

        left_axes[2].plot(time_hours, load_power, color=colors["ink"], label="负荷")
        left_axes[2].plot(time_hours, pv_power, color=colors["orange"], label="光伏")
        left_axes[2].plot(
            time_hours,
            net_load_power,
            color=colors["blue"],
            linestyle="--",
            linewidth=1.0,
            label="净负荷",
        )
        absorbed_top = load_power + absorbed_surplus_power
        left_axes[2].fill_between(
            time_hours,
            load_power,
            absorbed_top,
            where=pv_surplus_power > 1e-9,
            color=colors["green"],
            alpha=0.20,
            step="post",
            label="光伏盈余吸收",
        )
        left_axes[2].axhline(0.0, color=colors["muted"], linewidth=0.6)
        left_axes[2].set_title("(b) 负荷、光伏与净负荷", loc="left", fontweight="bold")
        left_axes[2].set_ylabel("功率(kW)")
        left_axes[2].legend(loc="upper right", ncol=2)
        left_axes[2].text(
            0.985,
            0.05,
            f"光伏盈余 {pv_surplus_power.sum()*interval_hours/1000:.3f} MWh；弃光 0",
            transform=left_axes[2].transAxes,
            ha="right",
            va="bottom",
            fontsize=6.3,
            color=colors["muted"],
        )
        format_axis(left_axes[2], show_xlabels=False)

        purchase_delta = grid_power - baseline_grid_power
        increase_mask = purchase_delta >= 0.0
        left_axes[3].bar(
            time_hours,
            np.where(increase_mask, purchase_delta, 0.0),
            width=action_width,
            color=colors["sky"],
            alpha=0.80,
            linewidth=0,
            label="增购",
        )
        left_axes[3].bar(
            time_hours,
            np.where(~increase_mask, purchase_delta, 0.0),
            width=action_width,
            color=colors["orange"],
            alpha=0.76,
            linewidth=0,
            label="减购",
        )
        left_axes[3].axhline(0.0, color=colors["ink"], linewidth=0.65)
        baseline_axis = left_axes[3].twinx()
        baseline_axis.step(
            time_hours,
            baseline_grid_power,
            where="post",
            color="#B8C0C8",
            linestyle="--",
            linewidth=0.75,
            alpha=0.70,
            label="无储能基准(右轴)",
        )
        left_axes[3].set_title("(c) 储能引起的购电量变化", loc="left", fontweight="bold")
        left_axes[3].set_ylabel("Δ购电(kW)")
        baseline_axis.set_ylabel("基准购电(kW)", color="#9299A1")
        delta_handles, delta_labels = left_axes[3].get_legend_handles_labels()
        base_handles, base_labels = baseline_axis.get_legend_handles_labels()
        left_axes[3].legend(
            delta_handles + base_handles,
            delta_labels + base_labels,
            loc="upper center",
            ncol=3,
        )
        format_axis(left_axes[3], show_xlabels=True)
        left_axes[3].set_xlabel("时刻")
        baseline_axis.spines["top"].set_visible(False)
        baseline_axis.spines["right"].set_color("#C8CDD2")
        baseline_axis.tick_params(length=2.0, width=0.5, colors="#9299A1")

        right_axes[0].bar(
            time_hours,
            charge_power,
            width=interval_hours * 0.82,
            color=colors["green"],
            label="充电",
        )
        right_axes[0].bar(
            time_hours,
            -discharge_power,
            width=interval_hours * 0.82,
            color=colors["vermillion"],
            label="放电",
        )
        price_axis = right_axes[0].twinx()
        price_axis.step(
            time_hours,
            price,
            where="post",
            color=colors["purple"],
            linewidth=0.95,
            label="电价",
        )
        right_axes[0].axhline(0.0, color=colors["ink"], linewidth=0.65)
        right_axes[0].set_title("(a) 充放电功率与电价", loc="left", fontweight="bold")
        right_axes[0].set_ylabel("储能功率(kW)\n充电为正")
        price_axis.set_ylabel("电价(元/kWh)")
        handles1, labels1 = right_axes[0].get_legend_handles_labels()
        handles2, labels2 = price_axis.get_legend_handles_labels()
        right_axes[0].legend(handles1 + handles2, labels1 + labels2, loc="upper center", ncol=3)
        format_axis(right_axes[0], show_xlabels=False)
        price_axis.spines["top"].set_visible(False)
        price_axis.tick_params(length=2.0, width=0.5)

        min_soc = float(parameters["min_soc_kwh"])
        max_soc = float(parameters["max_soc_kwh"])
        right_axes[1].axhspan(min_soc, max_soc, color=colors["green"], alpha=0.07)
        right_axes[1].step(time_hours, soc, where="post", color=colors["green"], label="储电量")
        right_axes[1].axhline(min_soc, color=colors["muted"], linestyle="--", linewidth=0.7)
        right_axes[1].axhline(max_soc, color=colors["muted"], linestyle="--", linewidth=0.7)
        upper_hits = np.isclose(soc, max_soc, atol=1e-5)
        lower_hits = np.isclose(soc, min_soc, atol=1e-5)
        right_axes[1].scatter(
            time_hours[upper_hits], soc[upper_hits], marker="v", s=12, color=colors["vermillion"], label="上限"
        )
        right_axes[1].scatter(
            time_hours[lower_hits], soc[lower_hits], marker="^", s=12, color=colors["blue"], label="下限"
        )
        right_axes[1].set_title("(b) 储电量轨迹与运行区间", loc="left", fontweight="bold")
        right_axes[1].set_ylabel("储电量(kWh)")
        right_axes[1].legend(loc="upper center", ncol=3)
        format_axis(right_axes[1], show_xlabels=False)

        baseline_interval_cost = price * baseline_grid_power * interval_hours
        optimized_interval_cost = price * grid_power * interval_hours
        baseline_cumulative = np.cumsum(baseline_interval_cost)
        optimized_cumulative = np.cumsum(optimized_interval_cost)
        right_axes[2].plot(
            time_hours,
            baseline_cumulative / 1000.0,
            color=colors["muted"],
            linestyle="--",
            linewidth=1.0,
            label="无储能基准",
        )
        right_axes[2].plot(
            time_hours,
            optimized_cumulative / 1000.0,
            color=colors["blue"],
            linewidth=1.2,
            label="三阶段最优",
        )
        right_axes[2].fill_between(
            time_hours,
            optimized_cumulative / 1000.0,
            baseline_cumulative / 1000.0,
            where=baseline_cumulative >= optimized_cumulative,
            color=colors["sky"],
            alpha=0.18,
        )
        gross_saving = float(np.maximum(baseline_interval_cost - optimized_interval_cost, 0.0).sum())
        extra_cost = float(np.maximum(optimized_interval_cost - baseline_interval_cost, 0.0).sum())
        net_saving = gross_saving - extra_cost
        right_axes[2].text(
            0.50,
            0.13,
            f"高价少购 {gross_saving/1000:.2f} − 低价多购 {extra_cost/1000:.2f} = 净节省 {net_saving/1000:.2f} 千元",
            transform=right_axes[2].transAxes,
            ha="center",
            va="center",
            fontsize=6.2,
            color=colors["ink"],
        )
        right_axes[2].set_title("(c) 累计购电成本与节省归因", loc="left", fontweight="bold")
        right_axes[2].set_ylabel("累计成本(千元)")
        right_axes[2].legend(loc="upper left")
        format_axis(right_axes[2], show_xlabels=True)
        right_axes[2].set_xlabel("时刻")

        figure.text(
            0.5,
            0.025,
            "注：无储能基准为逐时段购电量 max(负荷电量−光伏电量, 0)，不配置储能且不售电；基准情景为储能净功率变化上限 2000 kW/10 min、ε=0.05%、充放电效率均为 90%。",
            ha="center",
            va="bottom",
            fontsize=6.7,
            color=colors["muted"],
        )
        output_path = figure_dir / "问题一_图1图2双栏全图.png"
        figure.savefig(output_path, dpi=400, bbox_inches="tight", pad_inches=0.05)
        plt.close(figure)
    return output_path


def _write_csv_and_json(solution: Q1Solution, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = output_dir / "question1_schedule.csv"
    summary_path = output_dir / "question1_summary.json"
    payload_path = output_dir / "question1_workbook_payload.json"
    solution.schedule.to_csv(schedule_path, index=False, encoding="utf-8-sig", float_format="%.10f")
    summary_payload = {
        "summary": solution.summary,
        "validation": solution.validation,
        "four_hour_blocks": solution.block_summary.to_dict(orient="records"),
        "paper_table_intervals": solution.selected_intervals[
            ["time_label", "grid_purchase_kwh"]
        ].to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    workbook_payload = {
        "purchase_kwh": solution.schedule["grid_purchase_kwh"].tolist(),
        "blocks": solution.block_summary.to_dict(orient="records"),
        "soc_0_kwh": solution.summary["parameters"]["initial_soc_kwh"],
        "soc_24_kwh": solution.summary["parameters"]["terminal_soc_kwh"],
    }
    payload_path.write_text(json.dumps(workbook_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return schedule_path, summary_path, payload_path


def _write_figures(solution: Q1Solution, output_dir: Path) -> list[Path]:
    """Export publication-ready diagnostic figures in raster and vector formats."""
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    schedule = solution.schedule
    parameters = solution.summary["parameters"]
    interval_hours = float(parameters["interval_hours"])
    time_hours = (
        schedule["clock_minute"].to_numpy(dtype=float) / 60.0
        + 24.0 * schedule["is_next_day"].to_numpy(dtype=float)
    )
    time_ticks = np.arange(0.0, 24.1, 4.0)
    time_labels = [f"{int(hour):02d}:00" for hour in time_ticks]
    x_limits = (0.0, float(time_hours[-1] + interval_hours))

    load = schedule["load_kwh"].to_numpy(dtype=float)
    pv = schedule["pv_kwh"].to_numpy(dtype=float)
    grid = schedule["grid_purchase_kwh"].to_numpy(dtype=float)
    charge = schedule["charge_kwh"].to_numpy(dtype=float)
    discharge = schedule["discharge_kwh"].to_numpy(dtype=float)
    price = schedule["price_yuan_per_kwh"].to_numpy(dtype=float)
    baseline_grid = np.maximum(load - pv, 0.0)

    colors = {
        "ink": "#222222",
        "muted": "#6B7280",
        "grid": "#D9DEE7",
        "blue": "#0072B2",
        "orange": "#E69F00",
        "green": "#009E73",
        "vermillion": "#D55E00",
        "purple": "#7A5195",
    }
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "lines.linewidth": 1.45,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }

    def format_axis(axis: Any, *, show_xlabels: bool) -> None:
        axis.set_xlim(*x_limits)
        axis.set_xticks(time_ticks, time_labels if show_xlabels else [])
        axis.grid(axis="y", color=colors["grid"], linewidth=0.55, alpha=0.75)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=3.0, width=0.7)

    def save_figure(figure: Any, stem: str) -> list[Path]:
        png_path = figure_dir / f"{stem}.png"
        pdf_path = figure_dir / f"{stem}.pdf"
        figure.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.04)
        figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
        plt.close(figure)
        return [png_path, pdf_path]

    output_paths: list[Path] = []
    with plt.rc_context(style):
        dispatch_fig, dispatch_axes = plt.subplots(
            3,
            1,
            figsize=(7.2, 6.6),
            sharex=True,
            gridspec_kw={"height_ratios": [1.08, 1.0, 0.92], "hspace": 0.12},
        )

        ax = dispatch_axes[0]
        ax.fill_between(
            time_hours,
            0.0,
            pv,
            color=colors["orange"],
            alpha=0.18,
            step="post",
        )
        ax.plot(time_hours, load, color=colors["ink"], label="Demand")
        ax.plot(time_hours, pv, color=colors["orange"], label="PV generation")
        ax.fill_between(
            time_hours,
            load,
            pv,
            where=pv > load,
            color=colors["green"],
            alpha=0.13,
            interpolate=True,
            label="PV surplus",
        )
        ax.set_ylabel("Energy (kWh/10 min)")
        ax.set_title("(a) Demand and photovoltaic generation", loc="left", fontweight="semibold")
        ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
        format_axis(ax, show_xlabels=False)

        ax = dispatch_axes[1]
        ax.plot(
            time_hours,
            baseline_grid,
            color=colors["muted"],
            linestyle=(0, (4, 2)),
            label="Without storage",
        )
        ax.plot(time_hours, grid, color=colors["blue"], linewidth=1.65, label="LP dispatch")
        ax.fill_between(
            time_hours,
            baseline_grid,
            grid,
            where=grid >= baseline_grid,
            color=colors["blue"],
            alpha=0.12,
            interpolate=True,
            label="Additional grid purchase",
        )
        ax.fill_between(
            time_hours,
            baseline_grid,
            grid,
            where=grid < baseline_grid,
            color=colors["vermillion"],
            alpha=0.13,
            interpolate=True,
            label="Avoided grid purchase",
        )
        ax.set_ylabel("Grid energy (kWh/10 min)")
        ax.set_title("(b) Grid procurement with and without storage", loc="left", fontweight="semibold")
        ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.01))
        format_axis(ax, show_xlabels=False)

        ax = dispatch_axes[2]
        bar_width = interval_hours * 0.88
        ax.bar(
            time_hours,
            charge,
            width=bar_width,
            color=colors["green"],
            alpha=0.82,
            label="Charge",
        )
        ax.bar(
            time_hours,
            -discharge,
            width=bar_width,
            color=colors["vermillion"],
            alpha=0.82,
            label="Discharge",
        )
        ax.axhline(0.0, color=colors["ink"], linewidth=0.7)
        ax.set_ylabel("Battery action\n(kWh/10 min)")
        ax.set_xlabel("Time of day")
        ax.set_title("(c) Optimal battery action", loc="left", fontweight="semibold")
        ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
        format_axis(ax, show_xlabels=True)

        dispatch_fig.align_ylabels(dispatch_axes)
        dispatch_fig.subplots_adjust(left=0.105, right=0.985, bottom=0.075, top=0.975)
        output_paths.extend(save_figure(dispatch_fig, "question1_dispatch"))

        soc_fig, soc_axes = plt.subplots(
            3,
            1,
            figsize=(7.2, 6.6),
            sharex=True,
            gridspec_kw={"height_ratios": [0.82, 1.2, 1.0], "hspace": 0.12},
        )

        ax = soc_axes[0]
        ax.step(time_hours, price, where="post", color=colors["purple"], linewidth=1.45)
        mean_price = float(price.mean())
        ax.axhline(
            mean_price,
            color=colors["muted"],
            linewidth=0.8,
            linestyle=(0, (3, 2)),
            label=f"Daily mean = {mean_price:.3f}",
        )
        ax.set_ylabel("Price\n(CNY/kWh)")
        ax.set_title("(a) Time-varying electricity price", loc="left", fontweight="semibold")
        ax.legend(loc="upper left")
        format_axis(ax, show_xlabels=False)

        min_soc = float(parameters["min_soc_kwh"])
        max_soc = float(parameters["max_soc_kwh"])
        initial_soc = float(parameters["initial_soc_kwh"])
        soc_time = np.r_[time_hours[0], time_hours + interval_hours]
        soc_state = np.r_[initial_soc, schedule["soc_end_kwh"].to_numpy(dtype=float)]

        ax = soc_axes[1]
        ax.axhspan(min_soc, max_soc, color=colors["green"], alpha=0.055)
        ax.step(
            soc_time,
            soc_state,
            where="post",
            color=colors["green"],
            linewidth=1.7,
            label="Stored energy",
        )
        ax.axhline(max_soc, color=colors["muted"], linewidth=0.8, linestyle=(0, (4, 2)))
        ax.axhline(min_soc, color=colors["muted"], linewidth=0.8, linestyle=(0, (4, 2)))
        ax.text(
            x_limits[1] - 0.25,
            max_soc - 180,
            r"$E_{max}=10{,}800$ kWh",
            ha="right",
            va="top",
            color=colors["muted"],
        )
        ax.text(
            x_limits[1] - 0.25,
            min_soc + 180,
            r"$E_{min}=1{,}200$ kWh",
            ha="right",
            va="bottom",
            color=colors["muted"],
        )
        ax.set_ylim(600.0, 11400.0)
        ax.set_ylabel("Stored energy (kWh)")
        ax.set_title("(b) Storage trajectory and feasible operating band", loc="left", fontweight="semibold")
        format_axis(ax, show_xlabels=False)

        baseline_cumulative_cost = np.cumsum(price * baseline_grid)
        optimized_cumulative_cost = np.cumsum(price * grid)
        final_saving = float(baseline_cumulative_cost[-1] - optimized_cumulative_cost[-1])
        saving_rate = float(solution.summary["cost_saving_rate"])

        ax = soc_axes[2]
        ax.plot(
            time_hours,
            baseline_cumulative_cost / 1000.0,
            color=colors["muted"],
            linestyle=(0, (4, 2)),
            label="Without storage",
        )
        ax.plot(
            time_hours,
            optimized_cumulative_cost / 1000.0,
            color=colors["blue"],
            linewidth=1.65,
            label="LP dispatch",
        )
        ax.fill_between(
            time_hours,
            optimized_cumulative_cost / 1000.0,
            baseline_cumulative_cost / 1000.0,
            color=colors["blue"],
            alpha=0.10,
        )
        ax.annotate(
            f"Daily saving: CNY {final_saving / 1000.0:.2f}k ({saving_rate:.1%})",
            xy=(time_hours[-1], optimized_cumulative_cost[-1] / 1000.0),
            xytext=(12.5, optimized_cumulative_cost[-1] / 1000.0 - 6.5),
            arrowprops={"arrowstyle": "-", "color": colors["muted"], "linewidth": 0.75},
            color=colors["ink"],
            ha="left",
            va="top",
        )
        ax.set_ylabel("Cumulative cost\n(thousand CNY)")
        ax.set_xlabel("Time of day")
        ax.set_title("(c) Economic consequence of storage dispatch", loc="left", fontweight="semibold")
        ax.legend(loc="upper left", ncol=2)
        format_axis(ax, show_xlabels=True)

        soc_fig.align_ylabels(soc_axes)
        soc_fig.subplots_adjust(left=0.105, right=0.985, bottom=0.075, top=0.975)
        output_paths.extend(save_figure(soc_fig, "question1_soc_price"))

    return output_paths


def _write_figures_v2(solution: Q1Solution, output_dir: Path) -> list[Path]:
    """Export an alternative publication figure set focused on mechanism and evidence."""
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    schedule = solution.schedule
    parameters = solution.summary["parameters"]
    interval_hours = float(parameters["interval_hours"])
    time_hours = (
        schedule["clock_minute"].to_numpy(dtype=float) / 60.0
        + 24.0 * schedule["is_next_day"].to_numpy(dtype=float)
    )
    time_ticks = np.arange(0.0, 24.1, 4.0)
    time_labels = [f"{int(hour):02d}:00" for hour in time_ticks]
    x_limits = (0.0, float(time_hours[-1] + interval_hours))

    price = schedule["price_yuan_per_kwh"].to_numpy(dtype=float)
    charge = schedule["charge_kwh"].to_numpy(dtype=float)
    discharge = schedule["discharge_kwh"].to_numpy(dtype=float)
    net_action = charge - discharge
    load = schedule["load_kwh"].to_numpy(dtype=float)
    pv = schedule["pv_kwh"].to_numpy(dtype=float)
    grid = schedule["grid_purchase_kwh"].to_numpy(dtype=float)
    baseline_grid = np.maximum(load - pv, 0.0)
    tolerance = float(parameters["feasibility_tolerance"])

    colors = {
        "ink": "#222222",
        "muted": "#667085",
        "grid": "#D9DEE7",
        "blue": "#0072B2",
        "orange": "#E69F00",
        "green": "#009E73",
        "vermillion": "#D55E00",
        "purple": "#7A5195",
    }
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "lines.linewidth": 1.45,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }

    def format_axis(axis: Any, *, show_xlabels: bool = True) -> None:
        axis.grid(axis="y", color=colors["grid"], linewidth=0.55, alpha=0.75)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=3.0, width=0.7)
        if show_xlabels:
            axis.set_xticks(time_ticks, time_labels)
        else:
            axis.set_xticks(time_ticks, [])

    def save_figure(figure: Any, stem: str) -> list[Path]:
        png_path = figure_dir / f"{stem}.png"
        pdf_path = figure_dir / f"{stem}.pdf"
        figure.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.04)
        figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
        plt.close(figure)
        return [png_path, pdf_path]

    output_paths: list[Path] = []
    with plt.rc_context(style):
        response_fig = plt.figure(figsize=(7.2, 4.9))
        response_grid = response_fig.add_gridspec(
            2,
            2,
            width_ratios=[1.9, 1.0],
            height_ratios=[1.0, 1.0],
            wspace=0.32,
            hspace=0.22,
        )
        price_axis = response_fig.add_subplot(response_grid[0, 0])
        action_axis = response_fig.add_subplot(response_grid[1, 0], sharex=price_axis)
        scatter_axis = response_fig.add_subplot(response_grid[:, 1])

        is_charging = net_action > tolerance
        is_discharging = net_action < -tolerance
        price_axis.fill_between(
            time_hours,
            0.0,
            price,
            where=is_charging,
            step="post",
            color=colors["green"],
            alpha=0.12,
            label="Charging interval",
        )
        price_axis.fill_between(
            time_hours,
            0.0,
            price,
            where=is_discharging,
            step="post",
            color=colors["vermillion"],
            alpha=0.11,
            label="Discharging interval",
        )
        price_axis.step(time_hours, price, where="post", color=colors["purple"], linewidth=1.5)
        minimum_price_index = int(np.argmin(price))
        maximum_price_index = int(np.argmax(price))
        price_axis.scatter(
            [time_hours[minimum_price_index], time_hours[maximum_price_index]],
            [price[minimum_price_index], price[maximum_price_index]],
            color=[colors["green"], colors["vermillion"]],
            s=22,
            zorder=4,
        )
        price_axis.annotate(
            f"Minimum {price[minimum_price_index]:.3f}",
            xy=(time_hours[minimum_price_index], price[minimum_price_index]),
            xytext=(time_hours[minimum_price_index] + 0.7, price[minimum_price_index] + 0.13),
            arrowprops={"arrowstyle": "-", "linewidth": 0.7, "color": colors["muted"]},
            color=colors["ink"],
        )
        price_axis.annotate(
            f"Maximum {price[maximum_price_index]:.3f}",
            xy=(time_hours[maximum_price_index], price[maximum_price_index]),
            xytext=(time_hours[maximum_price_index] - 4.7, price[maximum_price_index] - 0.16),
            arrowprops={"arrowstyle": "-", "linewidth": 0.7, "color": colors["muted"]},
            color=colors["ink"],
        )
        price_axis.set_xlim(*x_limits)
        price_axis.set_ylim(0.3, 1.48)
        price_axis.set_ylabel("Price (CNY/kWh)")
        price_axis.set_title("(a) Price signal and active intervals", loc="left", fontweight="semibold")
        price_axis.legend(loc="upper left", ncol=2)
        format_axis(price_axis, show_xlabels=False)

        bar_width = interval_hours * 0.88
        action_axis.bar(
            time_hours[is_charging],
            net_action[is_charging],
            width=bar_width,
            color=colors["green"],
            alpha=0.84,
            label="Charge (+)",
        )
        action_axis.bar(
            time_hours[is_discharging],
            net_action[is_discharging],
            width=bar_width,
            color=colors["vermillion"],
            alpha=0.84,
            label="Discharge (-)",
        )
        action_axis.axhline(0.0, color=colors["ink"], linewidth=0.7)
        action_axis.set_xlim(*x_limits)
        action_axis.set_ylabel("Net battery action\n(kWh/10 min)")
        action_axis.set_xlabel("Time of day")
        action_axis.set_title("(b) Optimal response", loc="left", fontweight="semibold")
        action_axis.legend(loc="upper left", ncol=2)
        format_axis(action_axis)

        idle = ~(is_charging | is_discharging)
        scatter_axis.scatter(
            price[idle],
            net_action[idle],
            s=13,
            facecolors="none",
            edgecolors=colors["muted"],
            linewidths=0.7,
            alpha=0.7,
            label="Idle",
        )
        scatter_axis.scatter(
            price[is_charging],
            net_action[is_charging],
            s=22,
            marker="^",
            color=colors["green"],
            alpha=0.78,
            label="Charge",
        )
        scatter_axis.scatter(
            price[is_discharging],
            net_action[is_discharging],
            s=22,
            marker="v",
            color=colors["vermillion"],
            alpha=0.78,
            label="Discharge",
        )
        active_rank_correlation = float(
            pd.Series(price[~idle]).rank().corr(pd.Series(net_action[~idle]).rank())
        )
        scatter_axis.axhline(0.0, color=colors["ink"], linewidth=0.7)
        scatter_axis.text(
            0.05,
            0.05,
            rf"Active intervals: Spearman $\rho={active_rank_correlation:.2f}$",
            transform=scatter_axis.transAxes,
            color=colors["ink"],
            ha="left",
            va="bottom",
        )
        scatter_axis.set_xlabel("Price (CNY/kWh)")
        scatter_axis.set_ylabel("Net battery action (kWh/10 min)")
        scatter_axis.set_title("(c) Price-response evidence", loc="left", fontweight="semibold")
        scatter_axis.legend(loc="upper right")
        scatter_axis.grid(color=colors["grid"], linewidth=0.55, alpha=0.75)
        scatter_axis.spines["top"].set_visible(False)
        scatter_axis.spines["right"].set_visible(False)
        scatter_axis.tick_params(direction="out", length=3.0, width=0.7)

        response_fig.subplots_adjust(left=0.09, right=0.985, bottom=0.11, top=0.96)
        output_paths.extend(save_figure(response_fig, "question1_price_response_v2"))

        block_order = [f"{hour}:00-{hour + 4}:00" for hour in range(0, 24, 4)]
        block_labels = [f"{hour:02d}-{hour + 4:02d}" for hour in range(0, 24, 4)]
        block_metrics = (
            schedule.assign(
                baseline_cost_yuan=price * baseline_grid,
                optimized_cost_yuan=price * grid,
            )
            .groupby("block", sort=False)
            .agg(
                average_price=("price_yuan_per_kwh", "mean"),
                charge_kwh=("charge_kwh", "sum"),
                discharge_kwh=("discharge_kwh", "sum"),
                baseline_cost_yuan=("baseline_cost_yuan", "sum"),
                optimized_cost_yuan=("optimized_cost_yuan", "sum"),
            )
            .reindex(block_order)
        )
        block_x = np.arange(len(block_metrics), dtype=float)
        block_saving = (
            block_metrics["baseline_cost_yuan"].to_numpy(dtype=float)
            - block_metrics["optimized_cost_yuan"].to_numpy(dtype=float)
        ) / 1000.0
        cumulative_saving = np.cumsum(block_saving)

        economics_fig, economics_axes = plt.subplots(
            3,
            1,
            figsize=(7.2, 6.2),
            sharex=True,
            gridspec_kw={"height_ratios": [0.8, 1.1, 1.05], "hspace": 0.14},
        )

        ax = economics_axes[0]
        ax.plot(
            block_x,
            block_metrics["average_price"],
            color=colors["purple"],
            marker="o",
            markersize=4.2,
        )
        ax.set_ylabel("Average price\n(CNY/kWh)")
        ax.set_title("(a) Four-hour average price", loc="left", fontweight="semibold")
        ax.grid(axis="y", color=colors["grid"], linewidth=0.55, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3.0, width=0.7)

        ax = economics_axes[1]
        ax.bar(
            block_x - 0.18,
            block_metrics["charge_kwh"] / 1000.0,
            width=0.36,
            color=colors["green"],
            alpha=0.84,
            label="Charge",
        )
        ax.bar(
            block_x + 0.18,
            -block_metrics["discharge_kwh"] / 1000.0,
            width=0.36,
            color=colors["vermillion"],
            alpha=0.84,
            label="Discharge",
        )
        ax.axhline(0.0, color=colors["ink"], linewidth=0.7)
        ax.set_ylabel("Battery throughput\n(thousand kWh)")
        ax.set_title("(b) Energy shifted between price periods", loc="left", fontweight="semibold")
        ax.legend(loc="upper left", ncol=2)
        ax.grid(axis="y", color=colors["grid"], linewidth=0.55, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3.0, width=0.7)

        ax = economics_axes[2]
        positive_saving = block_saving >= 0.0
        ax.bar(
            block_x[positive_saving],
            block_saving[positive_saving],
            width=0.62,
            color=colors["blue"],
            alpha=0.76,
            label="Positive block saving",
        )
        ax.bar(
            block_x[~positive_saving],
            block_saving[~positive_saving],
            width=0.62,
            color=colors["orange"],
            alpha=0.76,
            label="Block cost increase",
        )
        ax.plot(
            block_x,
            cumulative_saving,
            color=colors["ink"],
            marker="o",
            markersize=4.2,
            linewidth=1.45,
            label="Cumulative saving",
        )
        ax.axhline(0.0, color=colors["ink"], linewidth=0.7)
        ax.annotate(
            f"Final saving = CNY {cumulative_saving[-1]:.2f}k",
            xy=(block_x[-1], cumulative_saving[-1]),
            xytext=(block_x[-1] - 1.75, cumulative_saving[-1] - 3.6),
            arrowprops={"arrowstyle": "-", "linewidth": 0.7, "color": colors["muted"]},
            ha="left",
            va="top",
            color=colors["ink"],
        )
        ax.set_ylabel("Cost saving\n(thousand CNY)")
        ax.set_xlabel("Time block")
        ax.set_title("(c) Block contribution and cumulative economic value", loc="left", fontweight="semibold")
        ax.set_xticks(block_x, block_labels)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            [handles[2], handles[0], handles[1]],
            [labels[2], labels[0], labels[1]],
            loc="upper left",
            ncol=3,
        )
        ax.grid(axis="y", color=colors["grid"], linewidth=0.55, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3.0, width=0.7)

        economics_fig.align_ylabels(economics_axes)
        economics_fig.subplots_adjust(left=0.11, right=0.985, bottom=0.09, top=0.97)
        output_paths.extend(save_figure(economics_fig, "question1_block_economics_v2"))

    return output_paths


def _find_artifact_node_modules() -> Path:
    override = os.environ.get("CODEX_ARTIFACT_NODE_MODULES")
    candidates = [
        Path(override) if override else None,
        Path.home()
        / ".cache"
        / "codex-runtimes"
        / "codex-primary-runtime"
        / "dependencies"
        / "node"
        / "node_modules",
    ]
    for candidate in candidates:
        if candidate and (candidate / "@oai" / "artifact-tool").exists():
            return candidate
    raise FileNotFoundError(
        "@oai/artifact-tool was not found. Set CODEX_ARTIFACT_NODE_MODULES to its node_modules directory."
    )


def _run_workbook_writer(
    template_path: Path,
    payload_path: Path,
    output_path: Path,
    preview_dir: Path,
) -> None:
    writer_source = Path(__file__).with_name("write_question1_xlsx.mjs")
    node_modules = _find_artifact_node_modules()
    temp_dir = Path(tempfile.gettempdir()) / "codex_q1_artifact_runtime"
    temp_dir.mkdir(parents=True, exist_ok=True)
    writer_copy = temp_dir / writer_source.name
    shutil.copy2(writer_source, writer_copy)
    link = temp_dir / "node_modules"
    if not link.exists():
        if os.name == "nt":
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(node_modules)],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        else:
            link.symlink_to(node_modules, target_is_directory=True)
    command = [
        "node",
        str(writer_copy),
        "--template",
        str(template_path.resolve()),
        "--data",
        str(payload_path.resolve()),
        "--output",
        str(output_path.resolve()),
        "--preview-dir",
        str(preview_dir.resolve()),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=temp_dir,
        )
    except subprocess.CalledProcessError as exc:
        windows_shutdown_code = 3221226505
        if exc.returncode == windows_shutdown_code and output_path.exists():
            workbook_check = _validate_written_workbook(output_path)
            print(
                json.dumps(
                    {
                        "artifact_tool_notice": (
                            "The Windows Node process returned its shutdown code after export; "
                            "the saved workbook passed an independent read-back check."
                        ),
                        **workbook_check,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        details = "\n".join(part for part in [exc.stdout, exc.stderr] if part)
        raise RuntimeError(f"Artifact Tool workbook writer failed:\n{details}") from exc
    if completed.stdout.strip():
        print(completed.stdout.strip())


def _validate_written_workbook(path: Path) -> dict[str, Any]:
    """Independently read back required workbook values after export."""
    with pd.ExcelFile(path) as workbook:
        expected_sheets = ["计划购电量", "充放电量"]
        if workbook.sheet_names != expected_sheets:
            raise RuntimeError(f"Unexpected exported workbook sheets: {workbook.sheet_names}")
        purchase = pd.read_excel(workbook, sheet_name="计划购电量")
        storage = pd.read_excel(workbook, sheet_name="充放电量")
    if purchase.shape != (144, 2):
        raise RuntimeError(f"Exported purchase sheet has unexpected shape {purchase.shape}")
    purchase_values = pd.to_numeric(purchase.iloc[:, 1], errors="coerce")
    if purchase_values.notna().sum() != 144:
        raise RuntimeError("Exported workbook does not contain 144 numeric purchase values")
    charge_values = pd.to_numeric(storage.iloc[:6, 1], errors="coerce")
    discharge_values = pd.to_numeric(storage.iloc[:6, 2], errors="coerce")
    soc_values = pd.to_numeric(storage.iloc[:2, 4], errors="coerce")
    if charge_values.notna().sum() != 6 or discharge_values.notna().sum() != 6:
        raise RuntimeError("Exported workbook has incomplete four-hour storage summaries")
    if soc_values.notna().sum() != 2:
        raise RuntimeError("Exported workbook has incomplete 0:00/24:00 SOC values")
    return {
        "workbook": str(path),
        "sheets": expected_sheets,
        "purchase_value_count": 144,
        "purchase_total_kwh": float(purchase_values.sum()),
        "soc_0_kwh": float(soc_values.iloc[0]),
        "soc_24_kwh": float(soc_values.iloc[1]),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve Question 1 using continuous LP only")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "附件" / "附件1.xlsx")
    parser.add_argument(
        "--template",
        type=Path,
        default=PROJECT_ROOT / "附件" / "附件5" / "result1.xlsx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "question1" / "result1.xlsx",
    )
    parser.add_argument("--no-workbook", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    inputs = load_question1_inputs(args.input)
    solution = solve_question1(inputs)
    output_dir = args.output.parent
    schedule_path, summary_path, payload_path = _write_csv_and_json(solution, output_dir)
    sensitivity_csv, sensitivity_figure = _write_battery_ramp_sensitivity(
        inputs, output_dir
    )
    figures = _write_figures_cn(solution, output_dir)
    word_combined_figure = _write_word_combined_figure(solution, output_dir)
    if not args.no_workbook:
        _run_workbook_writer(
            args.template,
            payload_path,
            args.output,
            output_dir / "workbook_preview",
        )
    result = {
        "workbook": None if args.no_workbook else str(args.output),
        "schedule": str(schedule_path),
        "summary": str(summary_path),
        "battery_ramp_sensitivity": str(sensitivity_csv),
        "battery_ramp_sensitivity_figure": str(sensitivity_figure),
        "figures": [str(path) for path in figures],
        "word_combined_figure": str(word_combined_figure),
        "total_purchase_kwh": solution.summary["total_purchase_kwh"],
        "total_cost_yuan": solution.summary["total_cost_yuan"],
        "validation": solution.validation["status"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
