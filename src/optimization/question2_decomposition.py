"""Gap decomposition for Question 2: forecast error vs. controller causality.

Decomposes the causal policies' gap to the perfect-information lower bound
into two counterfactuals, all over the full year 2025-01-01..12-31 with a
single 6000 kWh opening inventory and day-to-day SOC continuity:

  LB1 : perfect g  + perfect dispatch   (min sum c*g; already known)
  LB2 : causal forecast g + perfect dispatch  (min sum 5c*b given fixed g)
  LB3 : perfect g  + causal greedy dispatch   (simulation with fixed g)

These are retrospective counterfactuals, not causal attribution. LB2 relaxes
the operational direction rules and allows emergency charging. LB3 embeds
future information in its perfect plan and cannot bound general causal
controller improvements. Use audit_question2_gap for the direction-constrained
and formal-period matched review. Existing field names are legacy labels.

Run:  python -m src.optimization.question2_decomposition
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src.optimization.question2 import (
    ROOT, T, ETA, EMIN, EMAX, S, equality_matrix, load_inputs, Settings,
    run_case, execute_slot,
)

FULL = 365
INITIAL = 6000.0


def tile_prices(prices, days=FULL):
    return np.tile(prices, days)


def perfect_g(dates, load, pv, prices):
    """LB1: jointly optimal day-ahead plan with full information. Returns (g, cost)."""
    n = len(load.ravel())
    l, v = load.ravel(), pv.ravel()
    p = tile_prices(prices, len(dates))
    a = equality_matrix(n)
    rhs = np.r_[l - v, INITIAL, np.zeros(n - 1)]
    obj = np.r_[p, np.zeros(4 * n)]
    bounds = [(0, None)] * n + [(0, S)] * (2 * n) + [(0, None)] * n + [(EMIN, EMAX)] * n
    res = linprog(obj, A_eq=a, b_eq=rhs, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return np.maximum(res.x[:n], 0), float(res.fun)


def perfect_dispatch_given_g(g, dates, load, pv, prices, initial=INITIAL):
    """Relaxed perfect-future lower bound; allows emergency charging.

    Freezing the full plan also freezes later plans generated from the old
    controller's SOC path. This is not a new closed-loop policy simulation.
    """
    n = len(g)
    l, v = load.ravel(), pv.ravel()
    p = tile_prices(prices, len(dates))
    r = l - v - g
    a = equality_matrix(n)
    rhs = np.r_[r, initial, np.zeros(n - 1)]
    obj = np.r_[5 * p, np.zeros(4 * n)]
    bounds = [(0, None)] * n + [(0, S)] * (2 * n) + [(0, None)] * n + [(EMIN, EMAX)] * n
    res = linprog(obj, A_eq=a, b_eq=rhs, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return float(res.fun)


def greedy_dispatch_given_g(g, dates, load, pv, prices, initial=INITIAL):
    """LB3: causal greedy dispatch with a fixed plan g. Returns emergency cost."""
    l, v = load.ravel(), pv.ravel()
    p = tile_prices(prices, len(dates))
    energy, emergency = float(initial), 0.0
    for t in range(len(g)):
        _, _, b, _, end = execute_slot(l[t], v[t], g[t], energy, EMIN)
        emergency += 5 * p[t] * b
        energy = end
    return emergency


def full_year_from_csv(path: Path):
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=["total_cost"])
    return float(df.total_cost.sum())


def main():
    out = ROOT / "outputs" / "question2" / "analysis" / "decomposition"
    out.mkdir(parents=True, exist_ok=True)
    dates, load, pv, prices = load_inputs()
    p_tile = tile_prices(prices)

    # --- causal forecast plan (mean_greedy) and its full-year cost ---
    mg_frame, _, _ = run_case(dates, load, pv, prices, Settings(name="mean_greedy", risk=False, beta=0))
    g_mean = mg_frame.grid_kwh.to_numpy(float)
    mean_plan_cost = float((p_tile * g_mean).sum())
    mean_full_cost = float(mg_frame.total_cost.sum())

    # --- main scheme (risk_reserve) plan g from its saved schedule ---
    main_csv = ROOT / "outputs/question2/archive/baseline/question2_schedule_with_warmup.csv"
    main_df = pd.read_csv(main_csv, encoding="utf-8-sig", usecols=["grid_kwh", "total_cost"])
    g_main = main_df.grid_kwh.to_numpy(float)
    main_plan_cost = float((p_tile * g_main).sum())
    main_full_cost = float(main_df.total_cost.sum())

    # --- scenario value-function scheme full-year cost ---
    scen_csv = ROOT / "outputs/question2/archive/scenarios/question2_schedule_with_warmup.csv"
    scen_full_cost = full_year_from_csv(scen_csv)

    # --- LB1: perfect g + perfect dispatch ---
    t0 = time.perf_counter()
    g_perfect, lb1 = perfect_g(dates, load, pv, prices)
    lb1_sec = time.perf_counter() - t0

    # --- LB2: causal forecast g + perfect dispatch (two plan variants) ---
    t0 = time.perf_counter()
    lb2_mean_emerg = perfect_dispatch_given_g(g_mean, dates, load, pv, prices)
    lb2_mean = mean_plan_cost + lb2_mean_emerg
    lb2_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    lb2_main_emerg = perfect_dispatch_given_g(g_main, dates, load, pv, prices)
    lb2_main = main_plan_cost + lb2_main_emerg
    lb2_main_sec = time.perf_counter() - t0

    # --- LB3: perfect g + causal greedy dispatch ---
    t0 = time.perf_counter()
    lb3_emerg = greedy_dispatch_given_g(g_perfect, dates, load, pv, prices)
    lb3 = lb1 + lb3_emerg
    lb3_sec = time.perf_counter() - t0

    rows = [
        {"case": "LB1_perfect_g_perfect_dispatch", "g_info": "perfect", "controller": "perfect",
         "cost": lb1, "emergency": 0.0, "seconds": lb1_sec},
        {"case": "LB2_mean_forecast_g_perfect_dispatch", "g_info": "7-day mean (deterministic LP)",
         "controller": "perfect", "cost": lb2_mean, "emergency": lb2_mean_emerg, "seconds": lb2_sec},
        {"case": "LB2_main_forecast_g_perfect_dispatch", "g_info": "risk quantile (main plan)",
         "controller": "perfect", "cost": lb2_main, "emergency": lb2_main_emerg, "seconds": lb2_main_sec},
        {"case": "LB3_perfect_g_greedy_dispatch", "g_info": "perfect", "controller": "greedy causal (beta=0)",
         "cost": lb3, "emergency": lb3_emerg, "seconds": lb3_sec},
        {"case": "mean_greedy_actual", "g_info": "7-day mean (deterministic LP)", "controller": "greedy causal (beta=0)",
         "cost": mean_full_cost, "emergency": mean_full_cost - mean_plan_cost, "seconds": None},
        {"case": "risk_reserve_actual", "g_info": "risk quantile (main plan)", "controller": "greedy causal (beta=1)",
         "cost": main_full_cost, "emergency": main_full_cost - main_plan_cost, "seconds": None},
        {"case": "scenario_value_function_actual", "g_info": "scenario 2-stage LP", "controller": "approx value function",
         "cost": scen_full_cost, "emergency": None, "seconds": None},
    ]

    forecast_gap_mean = lb2_mean - lb1
    forecast_gap_main = lb2_main - lb1
    controller_gap = lb3 - lb1
    result = {
        "horizon": "2025-01-01..2025-12-31",
        "intervals": FULL * T,
        "initial_soc_kwh": INITIAL,
        "lb1_perfect": lb1,
        "lb2_mean_forecast": lb2_mean,
        "lb2_main_forecast": lb2_main,
        "lb3_perfect_g_greedy": lb3,
        "forecast_gap_mean_yuan": forecast_gap_mean,
        "forecast_gap_main_yuan": forecast_gap_main,
        "controller_gap_greedy_yuan": controller_gap,
        "forecast_vs_controller_ratio_mean": forecast_gap_mean / controller_gap if controller_gap else float("inf"),
        "rows": rows,
    }
    (out / "decomposition_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n--- TABLE (full year, yuan) ---")
    for r in rows:
        e = "" if r["emergency"] is None else f'{r["emergency"]:>14,.2f}'
        print(f'{r["case"]:<42} {r["cost"]:>15,.2f}  {e}')

    # Retire the old narrative: its prediction/causality attribution is invalid.
    report = """# 差额拆分实验：旧结论已撤回

原数值保留在decomposition_summary.json；字段名forecast_gap、controller_gap是历史名称，不能作为纯预测误差或纯因果性贡献解释。

原LB2允许紧急购电充电，是放宽实际方向规则的下界。完美g包含未来信息，LB3=LB1不证明一般控制器免费。

同时间域、同起始库存、增加方向约束的更新复核见[差额复核说明](../gap_audit/差额复核说明.md)。复现更新复核：python -m src.optimization.audit_question2_gap。
"""
    (out / "差额拆分实验说明.md").write_text(report, encoding="utf-8")
    print("\nReport written to", out)


if __name__ == "__main__":
    main()
