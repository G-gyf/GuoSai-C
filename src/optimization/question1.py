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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src.data_pipeline.build_timeline import parse_time_label, validate_source_headers


matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def _solve_two_stage_lp(inputs: pd.DataFrame, parameters: Q1Parameters) -> tuple[np.ndarray, dict[str, Any]]:
    n = len(inputs)
    price = inputs["price_yuan_per_kwh"].to_numpy(dtype=float)
    load_kwh = inputs["load_kw"].to_numpy(dtype=float) * parameters.interval_hours
    pv_kwh = inputs["pv_kw"].to_numpy(dtype=float) * parameters.interval_hours
    equalities, equality_rhs = _build_equalities(load_kwh, pv_kwh, parameters)
    bounds = _variable_bounds(n, parameters)

    cost_objective = np.zeros(5 * n, dtype=float)
    cost_objective[:n] = price
    first = linprog(
        cost_objective,
        A_eq=equalities,
        b_eq=equality_rhs,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    if not first.success:
        raise RuntimeError(f"Question 1 primary LP failed: {first.message}")

    cost_tolerance = max(1e-7, 1e-10 * abs(float(first.fun)))
    throughput_objective = np.zeros(5 * n, dtype=float)
    throughput_objective[n : 3 * n] = 1.0
    # A tiny curtailment tie-break is subordinate to total battery throughput.
    throughput_objective[3 * n : 4 * n] = 1e-9
    second = linprog(
        throughput_objective,
        A_ub=cost_objective.reshape(1, -1),
        b_ub=np.array([float(first.fun) + cost_tolerance]),
        A_eq=equalities,
        b_eq=equality_rhs,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    if not second.success:
        raise RuntimeError(f"Question 1 tie-break LP failed: {second.message}")

    metadata = {
        "solver": "scipy.optimize.linprog(method='highs')",
        "model_class": "continuous_linear_programming",
        "primary_status": int(first.status),
        "primary_message": first.message,
        "primary_optimal_cost_yuan": float(first.fun),
        "secondary_status": int(second.status),
        "secondary_message": second.message,
        "cost_tolerance_yuan": float(cost_tolerance),
        "secondary_cost_yuan": float(cost_objective @ second.x),
        "secondary_throughput_objective": float(throughput_objective @ second.x),
    }
    return second.x, metadata


def _block_name(clock_minute: int) -> str:
    start_hour = (clock_minute // 240) * 4
    return f"{start_hour}:00-{start_hour + 4}:00"


def _build_outputs(
    inputs: pd.DataFrame,
    vector: np.ndarray,
    solver_metadata: dict[str, Any],
    parameters: Q1Parameters,
) -> Q1Solution:
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
        "cost_recalculation_error_yuan": float(
            abs(
                summary["total_cost_yuan"]
                - np.dot(schedule["price_yuan_per_kwh"], schedule["grid_purchase_kwh"])
            )
        ),
        "secondary_cost_gap_yuan": float(
            summary["secondary_cost_yuan"] - summary["primary_optimal_cost_yuan"]
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
    if checks["cost_recalculation_error_yuan"] > tol:
        failures.append("cost_recalculation")
    if checks["secondary_cost_gap_yuan"] > summary["cost_tolerance_yuan"] + tol:
        failures.append("lexicographic_cost")
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
    vector, metadata = _solve_two_stage_lp(inputs, parameters)
    return _build_outputs(inputs, vector, metadata, parameters)


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
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, len(solution.schedule) + 1)
    tick_positions = np.arange(0, SLOTS_PER_DAY, 24)
    tick_labels = [f"{hour:02d}:00" for hour in range(0, 24, 4)]

    dispatch_path = figure_dir / "question1_dispatch.png"
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(x, solution.schedule["load_kwh"], label="Load", color="#1F2937", linewidth=1.4)
    ax.plot(x, solution.schedule["pv_kwh"], label="PV forecast", color="#F59E0B", linewidth=1.4)
    ax.plot(x, solution.schedule["grid_purchase_kwh"], label="Grid purchase", color="#2563EB", linewidth=1.4)
    ax.plot(x, solution.schedule["charge_kwh"], label="Charge", color="#10B981", linewidth=1.0)
    ax.plot(x, -solution.schedule["discharge_kwh"], label="Discharge", color="#DC2626", linewidth=1.0)
    ax.axhline(0, color="#9CA3AF", linewidth=0.7)
    ax.set_xticks(tick_positions + 1, tick_labels)
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy per 10-minute interval (kWh)")
    ax.set_title("Question 1 dispatch")
    ax.legend(ncol=5, frameon=False, loc="upper center")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(dispatch_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    soc_path = figure_dir / "question1_soc_price.png"
    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    ax1.plot(x, solution.schedule["soc_end_kwh"], color="#059669", linewidth=1.5, label="SOC")
    ax1.axhline(1200, color="#9CA3AF", linewidth=0.8, linestyle="--")
    ax1.axhline(10800, color="#9CA3AF", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Stored energy (kWh)", color="#059669")
    ax1.set_xticks(tick_positions + 1, tick_labels)
    ax1.set_xlabel("Time")
    ax2 = ax1.twinx()
    ax2.plot(x, solution.schedule["price_yuan_per_kwh"], color="#7C3AED", linewidth=1.1, label="Price")
    ax2.set_ylabel("Price (yuan/kWh)", color="#7C3AED")
    ax1.set_title("Question 1 storage state and electricity price")
    ax1.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    lines = ax1.get_lines()[:1] + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(soc_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [dispatch_path, soc_path]


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
    figures = _write_figures(solution, output_dir)
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
        "figures": [str(path) for path in figures],
        "total_purchase_kwh": solution.summary["total_purchase_kwh"],
        "total_cost_yuan": solution.summary["total_cost_yuan"],
        "validation": solution.validation["status"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
