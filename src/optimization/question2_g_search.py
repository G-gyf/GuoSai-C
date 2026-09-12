"""Search over day-ahead plan generators using LB2 as the ruler.

Each candidate generates a causal day-ahead plan g (full year), then LB2 =
sum c*g + minimum 5x emergency cost with a PERFECT controller is computed.
A lower LB2 means a lower relaxed retrospective bound for that fixed plan,
not necessarily a lower realizable bill. The bound permits emergency
charging. Generators use the same greedy rule but develop different SOC
paths; this is not pure causal attribution to the plan generator.

Methods:
  deterministic : 7-day mean L & V, plain LP (no risk)
  quantile      : 7-day mean L & V, 80% quantile risk curve
  cvar          : 7-day mean L & V, two-stage SP with CVaR of emergency cost
  pv_*          : same risk policy but PV forecast from the晴空包络 module

Run:  python -m src.optimization.question2_g_search
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from src.optimization.question2 import (
    ROOT, T, ETA, EMIN, EMAX, S, load_inputs, forecasts, solve_plan, execute_slot,
)
from src.optimization.question2_decomposition import (
    perfect_dispatch_given_g, tile_prices, FULL, INITIAL,
)


def load_better_pv_forecast() -> np.ndarray:
    """Causal PV point forecast from the clear-sky module, aligned to (365,144) kWh."""
    df = pd.read_parquet(ROOT / "data/processed/pv_day_ahead_10min.parquet")
    uniq = pd.DatetimeIndex(pd.to_datetime(pd.Series(df.plan_date.unique()).sort_values()))
    assert uniq.equals(pd.date_range("2025-01-01", "2025-12-31"))
    assert set(df.slot_index) == set(range(1, T + 1))
    wide = df.pivot_table(index="plan_date", columns="slot_index", values="pv_point_kw",
                          aggfunc="first").to_numpy(float)
    assert wide.shape == (FULL, T) and np.isfinite(wide).all()
    return np.clip(wide, 0, None) / 6.0  # kW -> kWh


def scenario_net_matrix(d, load, pv, fl, fv, residual_days=21):
    """Causal (count, 144) net-load scenarios for day d (paired same-day residuals)."""
    first = max(7, d - residual_days)
    count = d - first
    if count <= 0:
        return None, 0
    sl = np.maximum(0, fl[d] + load[first:d] - fl[first:d])
    sv = np.maximum(0, fv[d] + pv[first:d] - fv[first:d])
    return sl - sv, count


def solve_plan_cvar(net_scenarios, prices, initial, terminal_value, alpha, rho):
    """Two-stage SP for the day-ahead g: E[cost] + rho*CVaR(emergency cost)."""
    M, n = net_scenarios.shape
    c = prices
    # Variable offsets
    G, XI = 0, n
    W = n + 1
    SC = n + 1 + M
    step = 5 * n
    N = SC + M * step  # total variables

    def cidx(w, t): return SC + w * step + t
    def didx(w, t): return SC + w * step + n + t
    def bidx(w, t): return SC + w * step + 2 * n + t
    def uidx(w, t): return SC + w * step + 3 * n + t
    def eidx(w, t): return SC + w * step + 4 * n + t

    A_eq = lil_matrix((2 * M * n, N))
    b_eq = np.zeros(2 * M * n)
    # Balance: g + b + D - C - U = N
    for w in range(M):
        base = w * n
        for t in range(n):
            r = base + t
            A_eq[r, t] = 1
            A_eq[r, bidx(w, t)] = 1
            A_eq[r, didx(w, t)] = 1
            A_eq[r, cidx(w, t)] = -1
            A_eq[r, uidx(w, t)] = -1
            b_eq[r] = net_scenarios[w, t]
    # SOC: E_t - eta*C + D/eta = E_{t-1}  (E_{-1}=initial)
    for w in range(M):
        base = M * n + w * n
        for t in range(n):
            r = base + t
            A_eq[r, eidx(w, t)] = 1
            A_eq[r, cidx(w, t)] = -ETA
            A_eq[r, didx(w, t)] = 1.0 / ETA
            if t:
                A_eq[r, eidx(w, t - 1)] = -1
            b_eq[r] = initial if t == 0 else 0.0

    # CVaR inequality: sum 5c*b - w - xi <= 0
    A_ub = lil_matrix((M, N))
    b_ub = np.zeros(M)
    for w in range(M):
        for t in range(n):
            A_ub[w, bidx(w, t)] = 5 * c[t]
        A_ub[w, W + w] = -1
        A_ub[w, XI] = -1

    bounds = [(0, None)] * N
    for w in range(M):
        for t in range(n):
            bounds[cidx(w, t)] = (0, S)
            bounds[didx(w, t)] = (0, S)
            bounds[bidx(w, t)] = (0, None)
            bounds[uidx(w, t)] = (0, None)
            bounds[eidx(w, t)] = (EMIN, EMAX)
    bounds[XI] = (None, None)

    obj = np.zeros(N)
    obj[:n] = c
    obj[XI] = rho
    obj[W:W + M] = rho / ((1 - alpha) * M)
    for w in range(M):
        for t in range(n):
            obj[bidx(w, t)] += (1.0 / M) * 5 * c[t]
        obj[eidx(w, n - 1)] = -terminal_value / M

    res = linprog(obj, A_eq=A_eq.tocsr(), b_eq=b_eq, A_ub=A_ub.tocsr(), b_ub=b_ub,
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    g = np.maximum(res.x[:n], 0)
    return g


def generate_plan(d, load, pv, fl, fv, prices, energy, nu, method, alpha, rho):
    net = fl[d] - fv[d]
    scen, count = scenario_net_matrix(d, load, pv, fl, fv)
    if method == "deterministic":
        plan, _ = solve_plan(net, prices, energy, nu)
        return plan[0]
    if method == "quantile":
        if count > 0:
            q = np.quantile(scen, alpha, axis=0, method="linear")
            net = np.maximum(net, q)
        plan, _ = solve_plan(net, prices, energy, nu)
        return plan[0]
    if method == "cvar":
        if count == 0:
            plan, _ = solve_plan(net, prices, energy, nu)
            return plan[0]
        return solve_plan_cvar(scen, prices, energy, nu, alpha, rho)
    raise ValueError(method)


def gen_schedule(dates, load, pv, prices, method, alpha=0.8, rho=1.0, fv_better=None):
    fl, fv = forecasts(load, pv)
    if fv_better is not None:
        fv = fv_better
    nu = float(prices.min() / ETA)
    energy = INITIAL
    g_sched = np.zeros((len(dates), T))
    for d in range(len(dates)):
        if d == 0:
            g = np.zeros(T)
        else:
            g = generate_plan(d, load, pv, fl, fv, prices, energy, nu, method, alpha, rho)
        g_sched[d] = g
        for t in range(T):
            _, _, _, _, energy = execute_slot(load[d, t], pv[d, t], g[t], energy, EMIN)
    return g_sched


def lb2_of(g, dates, load, pv, prices):
    planned = float((tile_prices(prices) * g.ravel()).sum())
    emergency = perfect_dispatch_given_g(g.ravel(), dates, load, pv, prices, INITIAL)
    return planned + emergency, emergency


def main():
    out = ROOT / "outputs" / "question2" / "analysis" / "g_search"
    out.mkdir(parents=True, exist_ok=True)
    dates, load, pv, prices = load_inputs()
    fv_better = load_better_pv_forecast()

    candidates = [
        dict(method="deterministic", label="deterministic_7day_mean"),
        dict(method="quantile", label="quantile_7day_mean"),
        dict(method="cvar", label="cvar_a08_r1_7day_mean", alpha=0.8, rho=1.0),
        dict(method="quantile", label="quantile_pv_clear_sky", fv_better=fv_better),
        dict(method="cvar", label="cvar_a08_r1_pv_clear_sky", alpha=0.8, rho=1.0, fv_better=fv_better),
    ]

    rows = []
    for cand in candidates:
        label = cand["label"]
        t0 = time.perf_counter()
        g = gen_schedule(dates, load, pv, prices, **{k: v for k, v in cand.items() if k != "label"})
        gen_sec = time.perf_counter() - t0
        t0 = time.perf_counter()
        cost, emergency = lb2_of(g, dates, load, pv, prices)
        lb2_sec = time.perf_counter() - t0
        rows.append({"label": label, "planned": float((tile_prices(prices) * g.ravel()).sum()),
                     "emergency": emergency, "lb2": cost, "gen_sec": gen_sec, "lb2_sec": lb2_sec})
        print(f"{label}: LB2={cost:,.2f} (emergency {emergency:,.2f}, gen {gen_sec:.1f}s, lb2 {lb2_sec:.1f}s)", flush=True)

    # Reference values already computed by the decomposition experiment.
    lb1 = 13768559.688285146
    rows = [
        {"label": "LB1_perfect", "planned": lb1, "emergency": 0.0, "lb2": lb1, "gen_sec": None, "lb2_sec": None},
        {"label": "LB2_mean_7day(det)", "planned": None, "emergency": None, "lb2": 24862755.137311317, "gen_sec": None, "lb2_sec": None},
        *rows,
    ]
    (out / "g_search_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n--- LB2 comparison (full year, yuan) ---")
    for r in rows:
        print(f'{r["label"]:<30} LB2={r["lb2"]:>15,.2f}  (vs LB1 gap {r["lb2"]-lb1:>13,.2f})')


if __name__ == "__main__":
    main()
