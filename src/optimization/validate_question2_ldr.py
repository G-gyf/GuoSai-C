"""Independent validation of the revised Question 2 LDR result bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.optimization.question2 import ROOT, EMAX, EMIN, load_inputs, validate_schedule
from src.optimization.question2_ldr import DELTA_BOUND, LAMBDA_BOUND, PARAMETER_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/question2/current/ldr")
    args = parser.parse_args()
    out = args.output_dir
    schedule = pd.read_csv(out / "question2_schedule_with_warmup.csv", parse_dates=["date", "interval_start"])
    formal = schedule[schedule.date >= pd.Timestamp("2025-02-01")]
    diagnostics = pd.read_csv(out / "ldr_daily_parameters.csv", parse_dates=["date"])
    paired = pd.read_csv(out / "same_plan_controller_comparison.csv", parse_dates=["date"])
    summaries = json.loads((out / "question2_summary.json").read_text(encoding="utf-8"))
    summary = next(item for item in summaries if item["settings"]["name"] == "ldr_quantile_a08")
    workbook = json.loads((out / "question2_workbook_validation.json").read_text(encoding="utf-8"))
    dates, _, _, prices = load_inputs()

    assert len(schedule) == 365 * 144
    assert len(formal) == 334 * 144
    assert pd.DatetimeIndex(schedule.date.drop_duplicates()).equals(dates)
    assert np.array_equal(schedule.slot.to_numpy(), np.tile(np.arange(144), 365))
    expected_intervals = np.concatenate(
        [date.to_datetime64() + np.arange(144) * np.timedelta64(10, "m") for date in dates]
    )
    assert np.array_equal(schedule.interval_start.to_numpy(dtype="datetime64[ns]"), expected_intervals)
    np.testing.assert_allclose(schedule.price.to_numpy(), np.tile(prices, 365), atol=0, rtol=0)

    physical = validate_schedule(schedule)
    independently_planned = float(np.dot(formal.grid_kwh, formal.price))
    independently_emergency = float(np.dot(formal.emergency_kwh, 5.0 * formal.price))
    independently_total = independently_planned + independently_emergency
    assert abs(independently_planned - summary["planned_cost"]) < 1e-6
    assert abs(independently_emergency - summary["emergency_cost"]) < 1e-6
    assert abs(independently_total - summary["total_cost"]) < 1e-6

    assert len(diagnostics) == 365
    expected_counts = np.array([max(0, d - max(7, d - 21)) for d in range(365)])
    np.testing.assert_array_equal(diagnostics.residual_count.to_numpy(), expected_counts)
    parameter_values = diagnostics[PARAMETER_NAMES].to_numpy(float)
    warmup = diagnostics.residual_count < 21
    assert np.max(np.abs(parameter_values[warmup])) == 0.0
    assert np.max(np.abs(parameter_values[:, :4])) <= DELTA_BOUND + 1e-9
    assert np.max(np.abs(parameter_values[:, 4:])) <= LAMBDA_BOUND + 1e-9
    assert (diagnostics.selected_scenario_score <= diagnostics.zero_scenario_score + 1e-8).all()
    assert np.max(np.abs(schedule.loc[schedule.stage == 1, "ldr_lambda"])) == 0.0
    assert np.array_equal(schedule.calibration_used.to_numpy(), schedule.residual_count.to_numpy() >= 21)
    assert schedule.soc_start_kwh.iloc[0] == 6000.0
    assert schedule.soc_start_kwh.min() >= EMIN - 1e-7
    assert schedule.soc_end_kwh.max() <= EMAX + 1e-7

    assert len(paired) == 365
    formal_paired = paired[paired.date >= pd.Timestamp("2025-02-01")]
    assert abs(formal_paired.ldr_emergency_cost.sum() - formal.emergency_cost.sum()) < 1e-6
    assert workbook["passed"] is True
    assert abs(workbook["independent_total_bill"] - independently_total) < 1e-6

    result = {
        "passed": True,
        "schedule_rows_full_year": len(schedule),
        "schedule_rows_formal_period": len(formal),
        "dates_full_year": int(schedule.date.nunique()),
        "dates_formal_period": int(formal.date.nunique()),
        "physical_validation": physical,
        "causal_residual_count_sequence": True,
        "warmup_parameters_are_zero": True,
        "stage_one_lambda_is_zero": True,
        "selected_scenario_score_never_worse_than_zero": True,
        "parameter_bounds_passed": True,
        "independent_planned_bill": independently_planned,
        "independent_emergency_bill": independently_emergency,
        "independent_total_bill": independently_total,
        "workbook_validation_passed": True,
    }
    (out / "ldr_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
