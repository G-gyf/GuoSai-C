"""Question 2 clipped-affine reserve controller (LDR upgrade).

The day-ahead purchase plan is *not* re-optimised in this layer. It is the
fixed alpha=0.8 quantile-LP plan from :mod:`src.optimization.question2`.
For each day with a complete 21-day residual window, seven controller
parameters are calibrated on the same historical scenarios:

    R[t,w] = clip(Ep[t] + delta[k] + lambda[k] * a[k,w], EMIN, EMAX)

Here k indexes four six-hour stages, lambda[0] is fixed to zero, and a[k]
uses only intervals completed before the stage starts. Scenario evaluation
and actual execution share ``rule_transition`` exactly.

Run the complete causal year and write an independent result bundle with:

    python -m src.optimization.question2_ldr
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from src.optimization.question2 import (
    ROOT,
    T,
    ETA,
    EMIN,
    EMAX,
    S,
    Settings,
    forecasts,
    load_inputs,
    planning_net,
    run_case,
    solve_plan,
    validate_schedule,
    write_outputs,
)
from src.optimization.question2_g_search import scenario_net_matrix


STAGE_LENGTH = 36
STAGE_STARTS = np.array([0, 36, 72, 108], dtype=int)
DELTA_BOUND = 9600.0
LAMBDA_BOUND = 2.0
PARAMETER_NAMES = [
    "delta_00_06_kwh",
    "delta_06_12_kwh",
    "delta_12_18_kwh",
    "delta_18_24_kwh",
    "lambda_06_12",
    "lambda_12_18",
    "lambda_18_24",
]


@dataclass(frozen=True)
class LDRSettings:
    name: str = "ldr_quantile_a08"
    alpha: float = 0.8
    forecast_days: int = 7
    residual_days: int = 21
    stages: int = 4
    search_seed: int = 20250912
    search_maxiter: int = 8
    search_popsize: int = 5
    delta_bound_kwh: float = DELTA_BOUND
    lambda_bound: float = LAMBDA_BOUND


@dataclass(frozen=True)
class CalibrationResult:
    theta: np.ndarray
    zero_score: float
    selected_score: float
    evaluations: int
    seconds: float
    method: str
    accepted_search: bool
    solver_message: str


def unpack_theta(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return four deltas and four lambdas, with lambda for stage one zero."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (7,):
        raise ValueError(f"theta must have shape (7,), got {theta.shape}")
    return theta[:4], np.r_[0.0, theta[4:]]


def stage_error_means(net_paths: np.ndarray, forecast_net: np.ndarray) -> np.ndarray:
    """Information available at each stage start for one or more daily paths.

    Stage 1 begins at 00:00 and therefore has signal zero. Signals for later
    stages use only completed intervals [0, start), never the current or a
    future interval.
    """
    paths = np.asarray(net_paths, dtype=float)
    if paths.ndim == 1:
        paths = paths[None, :]
    forecast = np.asarray(forecast_net, dtype=float)
    if paths.ndim != 2 or paths.shape[1] != forecast.size:
        raise ValueError("net paths and forecast net have incompatible shapes")
    if forecast.size < STAGE_STARTS[-1]:
        raise ValueError("a four-stage day requires at least 108 intervals")
    signals = np.zeros((paths.shape[0], 4), dtype=float)
    errors = paths - forecast[None, :]
    for k, start in enumerate(STAGE_STARTS[1:], start=1):
        signals[:, k] = errors[:, :start].mean(axis=1)
    return signals


def rule_transition(net, grid, energy, reserve):
    """One physical transition shared by scenario scoring and live execution."""
    r = np.asarray(net, dtype=float) - np.asarray(grid, dtype=float)
    e = np.asarray(energy, dtype=float)
    target = np.asarray(reserve, dtype=float)
    charge = np.minimum(np.minimum(np.maximum(-r, 0.0), S), np.maximum((EMAX - e) / ETA, 0.0))
    discharge = np.minimum(
        np.minimum(np.maximum(r, 0.0), S),
        ETA * np.maximum(e - target, 0.0),
    )
    emergency = np.maximum(r - discharge, 0.0)
    unused = np.maximum(-r - charge, 0.0)
    end = e + ETA * charge - discharge / ETA
    return charge, discharge, emergency, unused, end


class ScenarioObjective:
    """Vectorised empirical objective used by differential evolution."""

    def __init__(self, net_paths, forecast_net, grid, plan_soc, prices, initial, terminal_value):
        self.net = np.asarray(net_paths, dtype=float)
        self.forecast = np.asarray(forecast_net, dtype=float)
        self.grid = np.asarray(grid, dtype=float)
        self.plan_soc = np.asarray(plan_soc, dtype=float)
        self.prices = np.asarray(prices, dtype=float)
        if self.net.ndim != 2:
            raise ValueError("net_paths must be a two-dimensional scenario matrix")
        if not all(x.shape == (self.net.shape[1],) for x in [self.forecast, self.grid, self.plan_soc, self.prices]):
            raise ValueError("daily vectors must match the scenario horizon")
        self.initial = float(initial)
        self.terminal_value = float(terminal_value)
        self.signals = stage_error_means(self.net, self.forecast)
        self.stage = np.minimum(np.arange(self.net.shape[1]) // STAGE_LENGTH, 3)
        self.evaluations = 0

    def __call__(self, theta):
        raw = np.asarray(theta, dtype=float)
        scalar = raw.ndim == 1
        if scalar:
            params = raw.reshape(1, -1)
        elif raw.ndim == 2 and raw.shape[0] == 7:
            # scipy differential_evolution(vectorized=True) supplies (N, S).
            params = raw.T
        elif raw.ndim == 2 and raw.shape[1] == 7:
            params = raw
        else:
            raise ValueError(f"unexpected parameter shape {raw.shape}")
        if params.shape[1] != 7:
            raise ValueError(f"expected seven parameters, got {params.shape[1]}")

        count = params.shape[0]
        scenarios = self.net.shape[0]
        delta = params[:, :4]
        lambdas = np.c_[np.zeros(count), params[:, 4:]]
        energy = np.full((count, scenarios), self.initial, dtype=float)
        emergency_cost = np.zeros_like(energy)
        for t, k in enumerate(self.stage):
            reserve = np.clip(
                self.plan_soc[t]
                + delta[:, k, None]
                + lambdas[:, k, None] * self.signals[None, :, k],
                EMIN,
                EMAX,
            )
            _, _, emergency, _, energy = rule_transition(
                self.net[None, :, t], self.grid[t], energy, reserve
            )
            emergency_cost += 5.0 * self.prices[t] * emergency
        score = np.mean(emergency_cost - self.terminal_value * energy, axis=1)
        self.evaluations += count
        return float(score[0]) if scalar else score


def calibrate_rule(
    net_paths: np.ndarray,
    forecast_net: np.ndarray,
    grid: np.ndarray,
    plan_soc: np.ndarray,
    prices: np.ndarray,
    initial: float,
    terminal_value: float,
    residual_count: int,
    settings: LDRSettings,
    seed: int,
) -> CalibrationResult:
    """Calibrate seven parameters and retain zero whenever search is worse."""
    started = time.perf_counter()
    objective = ScenarioObjective(
        net_paths, forecast_net, grid, plan_soc, prices, initial, terminal_value
    )
    zero = np.zeros(7, dtype=float)
    zero_score = objective(zero)
    if residual_count < settings.residual_days:
        return CalibrationResult(
            theta=zero,
            zero_score=zero_score,
            selected_score=zero_score,
            evaluations=objective.evaluations,
            seconds=time.perf_counter() - started,
            method="zero_parameter_warmup",
            accepted_search=False,
            solver_message="complete 21-day residual window not yet available",
        )

    bounds = [(-settings.delta_bound_kwh, settings.delta_bound_kwh)] * 4
    bounds += [(-settings.lambda_bound, settings.lambda_bound)] * 3
    result = differential_evolution(
        objective,
        bounds,
        strategy="best1bin",
        maxiter=settings.search_maxiter,
        popsize=settings.search_popsize,
        tol=0.0,
        atol=0.0,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=seed,
        polish=False,
        init="latinhypercube",
        updating="deferred",
        workers=1,
        vectorized=True,
        x0=zero,
    )
    candidate = np.asarray(result.x, dtype=float)
    candidate_score = float(result.fun)
    accepted = candidate_score <= zero_score + 1e-8
    theta = candidate if accepted else zero
    selected_score = candidate_score if accepted else zero_score
    return CalibrationResult(
        theta=theta,
        zero_score=zero_score,
        selected_score=selected_score,
        evaluations=objective.evaluations,
        seconds=time.perf_counter() - started,
        method="differential_evolution_bounded_direct_search",
        accepted_search=bool(accepted),
        solver_message=str(result.message),
    )


def execute_actual_day(load, pv, grid, plan_soc, prices, initial, forecast_net, theta):
    """Execute a day causally, locking the stage signal at each stage start."""
    load = np.asarray(load, dtype=float)
    pv = np.asarray(pv, dtype=float)
    grid = np.asarray(grid, dtype=float)
    plan_soc = np.asarray(plan_soc, dtype=float)
    prices = np.asarray(prices, dtype=float)
    forecast_net = np.asarray(forecast_net, dtype=float)
    if not all(x.shape == (T,) for x in [load, pv, grid, plan_soc, prices, forecast_net]):
        raise ValueError("actual-day inputs must all contain 144 intervals")
    delta, lambdas = unpack_theta(theta)
    energy = float(initial)
    error_sum = 0.0
    signal = 0.0
    names = [
        "reserve_kwh",
        "stage_error_mean_kwh",
        "soc_start_kwh",
        "charge_kwh",
        "discharge_kwh",
        "emergency_kwh",
        "unused_kwh",
        "soc_end_kwh",
        "emergency_cost",
    ]
    out = {name: np.zeros(T, dtype=float) for name in names}
    actual_net = load - pv
    for t in range(T):
        stage = min(t // STAGE_LENGTH, 3)
        if t % STAGE_LENGTH == 0:
            signal = 0.0 if t == 0 else error_sum / t
        reserve = float(np.clip(plan_soc[t] + delta[stage] + lambdas[stage] * signal, EMIN, EMAX))
        start = energy
        charge, discharge, emergency, unused, end = rule_transition(
            actual_net[t], grid[t], start, reserve
        )
        charge, discharge, emergency, unused, end = map(
            float, [charge, discharge, emergency, unused, end]
        )
        out["reserve_kwh"][t] = reserve
        out["stage_error_mean_kwh"][t] = signal
        out["soc_start_kwh"][t] = start
        out["charge_kwh"][t] = charge
        out["discharge_kwh"][t] = discharge
        out["emergency_kwh"][t] = emergency
        out["unused_kwh"][t] = unused
        out["soc_end_kwh"][t] = end
        out["emergency_cost"][t] = 5.0 * prices[t] * emergency
        energy = end
        error_sum += actual_net[t] - forecast_net[t]
    return out


def _period_metrics(frame: pd.DataFrame) -> dict:
    return {
        "days": int(frame["date"].nunique()),
        "intervals": int(len(frame)),
        **{
            key: float(frame[key].sum())
            for key in [
                "grid_kwh",
                "charge_kwh",
                "discharge_kwh",
                "emergency_kwh",
                "unused_kwh",
                "planned_cost",
                "emergency_cost",
                "total_cost",
            ]
        },
        "initial_soc_kwh": float(frame.soc_start_kwh.iloc[0]),
        "final_soc_kwh": float(frame.soc_end_kwh.iloc[-1]),
        "emergency_intervals": int((frame.emergency_kwh > 1e-6).sum()),
    }


def run_ldr(dates, load, pv, prices, settings: LDRSettings, limit: int | None = None):
    """Run the LDR strategy continuously from 1 January at 6000 kWh."""
    started = time.perf_counter()
    if limit is None:
        limit = len(dates)
    if not 1 <= limit <= len(dates):
        raise ValueError("limit must be between 1 and the number of input days")
    dates = dates[:limit]
    load = load[:limit]
    pv = pv[:limit]
    fl, fv = forecasts(load, pv)
    nu = float(prices.min() / ETA)
    energy = 6000.0
    records: list[tuple] = []
    diagnostics: list[dict] = []
    paired: list[dict] = []

    for d, date in enumerate(dates):
        day_started = time.perf_counter()
        if d == 0:
            forecast_net = np.zeros(T)
            risk_net = np.zeros(T)
            residual_count = 0
            plan_proxy_objective = 0.0
            plan = np.zeros((5, T))
            plan[4] = EMIN
            calibration = CalibrationResult(
                theta=np.zeros(7),
                zero_score=-nu * EMIN,
                selected_score=-nu * EMIN,
                evaluations=0,
                seconds=0.0,
                method="zero_plan_cold_start",
                accepted_search=False,
                solver_message="1 January cold start",
            )
        else:
            forecast_net = fl[d] - fv[d]
            risk_net, residual_count = planning_net(
                d,
                load,
                pv,
                fl,
                fv,
                Settings(alpha=settings.alpha),
            )
            plan, plan_proxy_objective = solve_plan(risk_net, prices, energy, nu)
            scenarios, scenario_count = scenario_net_matrix(
                d, load, pv, fl, fv, residual_days=settings.residual_days
            )
            if scenario_count != residual_count:
                raise AssertionError("risk curve and LDR scenario windows disagree")
            if scenarios is None:
                scenarios = forecast_net[None, :]
            calibration = calibrate_rule(
                scenarios,
                forecast_net,
                plan[0],
                plan[4],
                prices,
                energy,
                nu,
                residual_count,
                settings,
                settings.search_seed + d,
            )

        theta = calibration.theta
        delta, lambdas = unpack_theta(theta)
        initial_day_soc = energy
        actual = execute_actual_day(
            load[d], pv[d], plan[0], plan[4], prices, energy, forecast_net, theta
        )
        beta1 = execute_actual_day(
            load[d], pv[d], plan[0], plan[4], prices, energy, forecast_net, np.zeros(7)
        )
        beta0 = execute_actual_day(
            load[d], pv[d], plan[0], np.full(T, EMIN), prices, energy, forecast_net, np.zeros(7)
        )
        planned_cost_day = float(np.dot(prices, plan[0]))
        for t in range(T):
            stage = min(t // STAGE_LENGTH, 3)
            records.append(
                (
                    date,
                    t,
                    date + pd.Timedelta(minutes=10 * t),
                    load[d, t],
                    pv[d, t],
                    prices[t],
                    fl[d, t],
                    fv[d, t],
                    risk_net[t],
                    residual_count,
                    plan[0, t],
                    plan[1, t],
                    plan[2, t],
                    plan[4, t],
                    actual["reserve_kwh"][t],
                    actual["soc_start_kwh"][t],
                    actual["charge_kwh"][t],
                    actual["discharge_kwh"][t],
                    actual["emergency_kwh"][t],
                    actual["unused_kwh"][t],
                    actual["soc_end_kwh"][t],
                    prices[t] * plan[0, t],
                    actual["emergency_cost"][t],
                    stage + 1,
                    actual["stage_error_mean_kwh"][t],
                    delta[stage],
                    lambdas[stage],
                    residual_count >= settings.residual_days,
                )
            )
        energy = float(actual["soc_end_kwh"][-1])
        paired.append(
            {
                "date": str(date.date()),
                "same_plan_initial_soc_kwh": initial_day_soc,
                "planned_cost": planned_cost_day,
                "ldr_emergency_cost": float(actual["emergency_cost"].sum()),
                "beta1_emergency_cost": float(beta1["emergency_cost"].sum()),
                "beta0_emergency_cost": float(beta0["emergency_cost"].sum()),
                "ldr_total_cost": planned_cost_day + float(actual["emergency_cost"].sum()),
                "beta1_total_cost": planned_cost_day + float(beta1["emergency_cost"].sum()),
                "beta0_total_cost": planned_cost_day + float(beta0["emergency_cost"].sum()),
                "ldr_final_soc_kwh": energy,
                "beta1_final_soc_kwh": float(beta1["soc_end_kwh"][-1]),
                "beta0_final_soc_kwh": float(beta0["soc_end_kwh"][-1]),
            }
        )
        diagnostics.append(
            {
                "date": str(date.date()),
                "residual_count": residual_count,
                "initial_soc_kwh": initial_day_soc,
                "plan_proxy_objective": float(plan_proxy_objective),
                "planned_cost": planned_cost_day,
                **{name: float(value) for name, value in zip(PARAMETER_NAMES, theta)},
                "zero_scenario_score": calibration.zero_score,
                "selected_scenario_score": calibration.selected_score,
                "zero_scenario_full_objective": planned_cost_day + calibration.zero_score,
                "selected_scenario_full_objective": planned_cost_day + calibration.selected_score,
                "scenario_improvement": calibration.zero_score - calibration.selected_score,
                "search_evaluations": calibration.evaluations,
                "calibration_seconds": calibration.seconds,
                "calibration_method": calibration.method,
                "accepted_search": calibration.accepted_search,
                "solver_message": calibration.solver_message,
                "actual_emergency_cost": float(actual["emergency_cost"].sum()),
                "final_soc_kwh": energy,
                "day_seconds": time.perf_counter() - day_started,
            }
        )
        if (d + 1) % 30 == 0 or d + 1 == limit:
            print(
                f"LDR {d + 1}/{limit} days; date={date.date()} "
                f"residuals={residual_count} search={calibration.seconds:.2f}s",
                flush=True,
            )

    columns = [
        "date",
        "slot",
        "interval_start",
        "load_kwh",
        "pv_kwh",
        "price",
        "forecast_load_kwh",
        "forecast_pv_kwh",
        "planning_net_kwh",
        "residual_count",
        "grid_kwh",
        "plan_charge_kwh",
        "plan_discharge_kwh",
        "plan_soc_kwh",
        "reserve_kwh",
        "soc_start_kwh",
        "charge_kwh",
        "discharge_kwh",
        "emergency_kwh",
        "unused_kwh",
        "soc_end_kwh",
        "planned_cost",
        "emergency_cost",
        "stage",
        "stage_error_mean_kwh",
        "ldr_delta_kwh",
        "ldr_lambda",
        "calibration_used",
    ]
    frame = pd.DataFrame.from_records(records, columns=columns)
    frame["total_cost"] = frame.planned_cost + frame.emergency_cost
    validation = validate_schedule(frame)
    formal = frame[frame.date >= pd.Timestamp("2025-02-01")].copy()
    daily_all = frame.groupby("date").agg(
        grid_kwh=("grid_kwh", "sum"),
        charge_kwh=("charge_kwh", "sum"),
        discharge_kwh=("discharge_kwh", "sum"),
        emergency_kwh=("emergency_kwh", "sum"),
        unused_kwh=("unused_kwh", "sum"),
        planned_cost=("planned_cost", "sum"),
        emergency_cost=("emergency_cost", "sum"),
        total_cost=("total_cost", "sum"),
        soc_start_kwh=("soc_start_kwh", "first"),
        soc_end_kwh=("soc_end_kwh", "last"),
    )
    daily = daily_all[daily_all.index >= pd.Timestamp("2025-02-01")]
    diag_frame = pd.DataFrame(diagnostics)
    pair_frame = pd.DataFrame(paired)
    full_metrics = _period_metrics(frame)
    formal_metrics = _period_metrics(formal) if len(formal) else {}
    summary = {
        "settings": {
            **asdict(settings),
            "plan_generator": "fixed_alpha_0.8_quantile_LP",
            "controller": "four_stage_clipped_affine_reserve_rule",
            "lambda_stage_1": 0.0,
            "parameter_order": PARAMETER_NAMES,
            "search_acceptance": "selected empirical objective must not exceed zero-parameter objective",
        },
        "terminal_value": nu,
        "seconds": time.perf_counter() - started,
        "planner_seconds": float(diag_frame.day_seconds.sum() - diag_frame.calibration_seconds.sum()),
        "calibration_seconds": float(diag_frame.calibration_seconds.sum()),
        "calibrated_days": int((diag_frame.residual_count >= settings.residual_days).sum()),
        "accepted_search_days": int(diag_frame.accepted_search.sum()),
        "search_evaluations": int(diag_frame.search_evaluations.sum()),
        "empirical_scenario_improvement": float(diag_frame.scenario_improvement.sum()),
        "result_days": int(formal_metrics.get("days", 0)),
        "result_intervals": int(formal_metrics.get("intervals", 0)),
        **{
            key: formal_metrics.get(key)
            for key in [
                "grid_kwh",
                "charge_kwh",
                "discharge_kwh",
                "emergency_kwh",
                "unused_kwh",
                "planned_cost",
                "emergency_cost",
                "total_cost",
            ]
        },
        "initial_result_soc_kwh": formal_metrics.get("initial_soc_kwh"),
        "final_soc_kwh": formal_metrics.get("final_soc_kwh"),
        "emergency_intervals": formal_metrics.get("emergency_intervals"),
        "formal_period": formal_metrics,
        "full_year": full_metrics,
        "validation": validation,
    }
    return frame, daily, daily_all, summary, diag_frame, pair_frame


def _summary_row(label: str, period: str, metrics: dict) -> dict:
    return {"strategy": label, "period": period, **metrics}


def write_ldr_report(out: Path, summary: dict, baselines: list[dict], paired: pd.DataFrame) -> None:
    formal = summary["formal_period"]
    full = summary["full_year"]
    by_name = {item["settings"]["name"]: item for item in baselines}
    beta1 = by_name["risk_reserve"]
    beta0 = by_name["risk_greedy"]
    formal_pair = paired[pd.to_datetime(paired.date) >= pd.Timestamp("2025-02-01")]
    paired_totals = formal_pair[["ldr_total_cost", "beta1_total_cost", "beta0_total_cost"]].sum()
    delta_beta1 = formal["total_cost"] - beta1["total_cost"]
    delta_beta0 = formal["total_cost"] - beta0["total_cost"]
    scenario_reference = None
    scenario_path = ROOT / "outputs/question2/archive/scenarios/question2_summary.json"
    if scenario_path.exists():
        for item in json.loads(scenario_path.read_text(encoding="utf-8")):
            if item["settings"]["name"] == "scenario_value_no_NAC":
                scenario_reference = item
                break
    verdict = (
        "LDR费用低于两个独立全年基线，可作为主方案候选。"
        if delta_beta1 < 0 and delta_beta0 < 0
        else "LDR未同时优于两个独立全年基线，因此按修订方案不自动替代原方案。"
    )
    text = f"""# 问题二 LDR 全年实施与结果说明

## 结论

本次按《问题二完整方案与审查修订》实施固定 α=0.8 分位数日前计划，并在每天零点用最近21个完整历史残差日校准四阶段截断仿射保留阈值。1月1日从6000 kWh连续运行至12月31日；2月1日至12月31日为正式期。{verdict}

## 正式期结果（2025-02-01—2025-12-31）

| 指标 | LDR | β=1基线 | β=0基线 |
|---|---:|---:|---:|
| 实际总费用（元） | {formal['total_cost']:.2f} | {beta1['total_cost']:.2f} | {beta0['total_cost']:.2f} |
| 计划购电费用（元） | {formal['planned_cost']:.2f} | {beta1['planned_cost']:.2f} | {beta0['planned_cost']:.2f} |
| 紧急购电费用（元） | {formal['emergency_cost']:.2f} | {beta1['emergency_cost']:.2f} | {beta0['emergency_cost']:.2f} |
| 紧急购电量（kWh） | {formal['emergency_kwh']:.2f} | {beta1['emergency_kwh']:.2f} | {beta0['emergency_kwh']:.2f} |
| 未利用供能（kWh） | {formal['unused_kwh']:.2f} | {beta1['unused_kwh']:.2f} | {beta0['unused_kwh']:.2f} |
| 正式期期初SOC（kWh） | {formal['initial_soc_kwh']:.2f} | {beta1['initial_result_soc_kwh']:.2f} | {beta0['initial_result_soc_kwh']:.2f} |
| 年末SOC（kWh） | {formal['final_soc_kwh']:.2f} | {beta1['final_soc_kwh']:.2f} | {beta0['final_soc_kwh']:.2f} |

相对β=1，LDR费用变化为{delta_beta1:+.2f}元；相对β=0，变化为{delta_beta0:+.2f}元。负值表示节省。

{f"作为不同日前计划与控制器构成的整套策略参考，现有情景价值控制方案正式期费用为{scenario_reference['total_cost']:.2f}元；LDR低{scenario_reference['total_cost'] - formal['total_cost']:.2f}元。该比较不用于拆分纯控制器贡献。" if scenario_reference else "未读取到现有情景价值控制方案结果，因此本说明不列该项参考。"}

## 全年连续账本（2025-01-01—2025-12-31）

- 实际总费用：{full['total_cost']:.2f}元
- 计划购电费用：{full['planned_cost']:.2f}元
- 紧急购电费用：{full['emergency_cost']:.2f}元
- 紧急购电量：{full['emergency_kwh']:.2f} kWh
- 年初SOC：{full['initial_soc_kwh']:.2f} kWh；年末SOC：{full['final_soc_kwh']:.2f} kWh

## 同计划、同日初SOC控制器配对

该对照每天使用LDR账本的同一日初SOC和同一购电计划，只隔离日内控制规则，不代表三个方案各自连续运行的全年账单。正式期配对总费用为：LDR {paired_totals['ldr_total_cost']:.2f}元、β=1 {paired_totals['beta1_total_cost']:.2f}元、β=0 {paired_totals['beta0_total_cost']:.2f}元。

## 方法与信息边界

- 每日购电计划由七天均值预测、最近21天成对残差的80%分位数风险曲线和原线性规划产生，LDR层不重新优化购电量。
- 阶段为0—6、6—12、12—18、18—24时。阶段开始时仅使用此前已完成区间的净负荷预测误差均值；第一阶段固定为0。
- 参数范围预先固定为δ∈[-9600,9600] kWh、λ∈[-2,2]，每天使用固定种子和固定搜索预算。零参数始终在候选集中，经验目标变差时回退。
- 实际账单只计计划购电费与五倍紧急购电费，不扣除日前校准使用的期末库存价值。

## 校验

逐时供需平衡、SOC递推、跨日连续、功率与库存边界、充放电互斥、无紧急购电充电均已通过。全年共有{summary['calibrated_days']}天具备完整21日残差窗，搜索结果被接受{summary['accepted_search_days']}天，总评价参数组数{summary['search_evaluations']}。

## 解释限制

这是“截断仿射保留阈值＋分段执行规则”，不是所有动作均为纯线性函数的LDR。每日直接搜索只保证在给定预算内不劣于零参数候选，不保证七维非凸经验目标的全局最优。历史情景改善也不等同于样本外账单必然改善，方案选择以独立连续全年费用为准。
"""
    (out / "LDR全年结果说明.md").write_text(text, encoding="utf-8")


def write_ldr_outputs(
    out: Path,
    frame: pd.DataFrame,
    daily: pd.DataFrame,
    daily_all: pd.DataFrame,
    summary: dict,
    diagnostics: pd.DataFrame,
    paired: pd.DataFrame,
    baselines: list[dict],
    prices: np.ndarray,
) -> None:
    write_outputs(out, frame, daily, [summary, *baselines], prices)
    daily_all.to_csv(out / "question2_daily_with_warmup.csv", encoding="utf-8-sig")
    diagnostics.to_csv(out / "ldr_daily_parameters.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(out / "same_plan_controller_comparison.csv", index=False, encoding="utf-8-sig")
    rows = [
        _summary_row("LDR", "2025-01-01/2025-12-31", summary["full_year"]),
        _summary_row("LDR", "2025-02-01/2025-12-31", summary["formal_period"]),
    ]
    pd.DataFrame(rows).to_csv(out / "ldr_period_summary.csv", index=False, encoding="utf-8-sig")
    metadata_path = out / "question2_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    plan_path = ROOT / "问题二完整方案与审查修订.md"
    metadata.update(
        {
            "implementation": "fixed alpha=0.8 quantile-LP plan plus four-stage clipped-affine reserve calibration",
            "stage_starts": STAGE_STARTS.tolist(),
            "parameter_names": PARAMETER_NAMES,
            "parameter_bounds": {
                "delta_kwh": [-DELTA_BOUND, DELTA_BOUND],
                "lambda": [-LAMBDA_BOUND, LAMBDA_BOUND],
            },
            "revision_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "actual_execution_information": "At each stage start, the signal uses only errors from completed intervals; parameters and grid plan are locked at 00:00.",
        }
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_ldr_report(out, summary, baselines, paired)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/question2/current/ldr")
    parser.add_argument("--search-seed", type=int, default=LDRSettings.search_seed)
    parser.add_argument("--search-maxiter", type=int, default=LDRSettings.search_maxiter)
    parser.add_argument("--search-popsize", type=int, default=LDRSettings.search_popsize)
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()
    settings = LDRSettings(
        search_seed=args.search_seed,
        search_maxiter=args.search_maxiter,
        search_popsize=args.search_popsize,
    )
    dates, load, pv, prices = load_inputs()
    frame, daily, daily_all, summary, diagnostics, paired = run_ldr(
        dates, load, pv, prices, settings, limit=args.days
    )
    if args.days != 365:
        args.output.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output / "pilot_schedule.csv", index=False, encoding="utf-8-sig")
        diagnostics.to_csv(args.output / "pilot_ldr_parameters.csv", index=False, encoding="utf-8-sig")
        (args.output / "pilot_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return

    baselines: list[dict] = []
    if not args.skip_baselines:
        for setting in [Settings(), Settings(name="risk_greedy", beta=0.0)]:
            _, _, baseline_summary = run_case(dates, load, pv, prices, setting)
            baselines.append(baseline_summary)
    else:
        baseline_path = ROOT / "outputs/question2/archive/baseline/question2_summary.json"
        cached = json.loads(baseline_path.read_text(encoding="utf-8"))
        baselines = [s for s in cached if s["settings"]["name"] in {"risk_reserve", "risk_greedy"}]
    write_ldr_outputs(
        args.output,
        frame,
        daily,
        daily_all,
        summary,
        diagnostics,
        paired,
        baselines,
        prices,
    )
    print(json.dumps([summary, *baselines], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
