"""Question 3: rolling forecast updates + MPC purchase adjustments + staged LDR.

Main strategy ``M612`` (0:00 plan, 6:00/12:00 purchase adjustments, 0/6/12
three staged LDR calibrations, 18:00 signal-only update):

* 0:00  q0 from the alpha=0.8 quantile LP on the 0:00-issuance forecast;
       calibrate delta0 (1-D) for the 0-6 h stage only (a0 = 0 by definition).
* 6:00  re-solve purchases for 06:00-24:00 with segment net settlement
       (口径说明 §3.16) against q0, lock 06:00-12:00; calibrate
       (delta6, lambda6) for stage 6-12 h with the observed a6 signal.
* 12:00 re-solve 12:00-24:00, lock; jointly calibrate
       (delta12, lambda12, delta18, lambda18) for the 12-18 h and 18-24 h
       stages; a18 stays scenario-computed inside the search.
* 18:00 no purchase adjustment, no new forecast, no recalibration: only
       the observed a18 (errors of 12:00-18:00 vs the 12:00 forecast)
       updates the locked reserve rule.

Risk levels are fixed a priori from the settlement structure, not from
backtests.  The single-slot no-storage newsvendor condition
R'(a) = 5c*P(D > a) with the segment net settlement R(a) =
c*min(q0,a) + 0.5c(q0-a)+ + 1.5c(a-q0)+ gives the coverage levels
F = 1 - R'/(5c):

* 0:00 plan:       R' = c    -> F = 1 - 1/5     = 0.8  (Q80 curve);
* down region:     R' = 0.5c -> F = 1 - 0.5/5   = 0.9  (Q90);
* up region:       R' = 1.5c -> F = 1 - 1.5/5   = 0.7  (Q70);

hence the adjustment curve is median(Q70, Q90, q0) with the point
forecast as floor.  Q50/Q80/Q90 up-quantile variants are sensitivity
controls only and are never selected from backtest costs.

Ablations: M0 (0:00 only, PV forecast = 0:00 issuance, Q2-style 7-parameter
daily calibration), M6 (0:00 + 6:00), M61218-S (18:00 purchase
re-optimisation on the 12:00 forecast + current SOC), M61218-F (same with
the 18:00 forecast).  All strategies run causally and continuously from
1 January at 6000 kWh; 1 Feb-31 Dec is the formal period.

Run:  python -m src.optimization.question3 [--days N] [--strategies ...]
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
from scipy.optimize import differential_evolution, linprog
from scipy.sparse import csr_matrix, lil_matrix

from src.optimization.question2 import (
    ROOT,
    T,
    ETA,
    EMIN,
    EMAX,
    S,
    KEY_DATES,
    load_inputs,
    load_forecast_weekly_persist,
    solve_plan,
    time_label,
)
from src.optimization.question2_ldr import (
    DELTA_BOUND,
    LAMBDA_BOUND,
    LDRSettings,
    calibrate_rule,
    execute_actual_day,
    rule_transition,
    unpack_theta,
)
from src.data_pipeline.question3_forecasts import build_issuance_curves

STAGE_STARTS = [0, 36, 72, 108]
RESIDUAL_DAYS = 21

STRATEGIES = {
    "M0": {"updates": [0], "issuance": {0: 0}},
    "M6": {"updates": [0, 36], "issuance": {0: 0, 36: 1}},
    "M612": {"updates": [0, 36, 72], "issuance": {0: 0, 36: 1, 72: 2}},
    "M61218-S": {"updates": [0, 36, 72, 108], "issuance": {0: 0, 36: 1, 72: 2, 108: 2}},
    "M61218-F": {"updates": [0, 36, 72, 108], "issuance": {0: 0, 36: 1, 72: 2, 108: 3}},
}
MAIN_STRATEGY = "M612"
STRATEGY_ORDER = ["M0", "M6", "M612", "M61218-S", "M61218-F"]


@dataclass(frozen=True)
class Q3Settings:
    name: str = "question3"
    strategy: str = MAIN_STRATEGY
    alpha: float = 0.8
    adjustment_up_quantile: float = 0.7
    residual_days: int = RESIDUAL_DAYS
    search_seed: int = 20250912
    search_maxiter: int = 8
    search_popsize: int = 5
    delta_bound_kwh: float = DELTA_BOUND
    lambda_bound: float = LAMBDA_BOUND


# --------------------------------------------------------------------------
# settlement (口径说明 §3.16 segment net settlement)
# --------------------------------------------------------------------------

def settle(q0, qA, prices):
    """Return retained, down penalty, up premium cost arrays."""
    q0 = np.asarray(q0, float)
    qA = np.asarray(qA, float)
    prices = np.asarray(prices, float)
    retained = prices * np.minimum(q0, qA)
    down = 0.5 * prices * np.maximum(q0 - qA, 0.0)
    up = 1.5 * prices * np.maximum(qA - q0, 0.0)
    return retained, down, up


# --------------------------------------------------------------------------
# causal scenarios and risk curves
# --------------------------------------------------------------------------

def scenario_matrix(d, fl, fc, load, pv, k, h0, residual_days=RESIDUAL_DAYS):
    """Causal (M, 144-h0) net scenarios for issuance k over slots [h0, 144)."""
    first = max(7, d - residual_days)
    count = d - first
    if count <= 0:
        return None, 0
    eps_l = load[first:d] - fl[first:d]
    eps_v = pv[first:d] - fc[first:d, k]
    sl = np.maximum(0.0, fl[d, h0:] + eps_l[:, h0:])
    sv = np.maximum(0.0, fc[d, k, h0:] + eps_v[:, h0:])
    return sl - sv, count


def risk_curve(fl_d, fc_dk, scen, h0, alpha=0.8):
    """max(point net forecast, empirical alpha quantile of net scenarios)."""
    net = fl_d[h0:] - fc_dk[h0:]
    if scen is None:
        return net
    q = np.quantile(scen, alpha, axis=0)
    return np.maximum(net, q)


def adjustment_curve(fl_d, fc_dk, scen, q0, h0, q_up=0.7, q_down=0.9):
    """Two-sided newsvendor curve for an adjustment stage.

    Under the segment net settlement R(a) = c*min(q0,a) + 0.5c(q0-a)+
    + 1.5c(a-q0)+, the marginal cost of the adjusted purchase is 0.5c in
    the down region (a < q0) and 1.5c in the up region (a > q0); the
    single-slot no-storage newsvendor condition R'(a) = 5c*P(D > a) then
    gives the coverage levels

        F_down = 1 - 0.5c/(5c) = 0.9   ->  Q90
        F_up   = 1 - 1.5c/(5c) = 0.7   ->  Q70

    and the piecewise optimum against the locked plan is the median of
    Q_up, Q_down and q0 itself (the kink).  The point forecast acts as a
    floor exactly as in the 0:00 curve.  ``q_up`` is a parameter: 0.7 is
    the derived value; other levels (0.5/0.8/0.9) are sensitivity
    variants only and must never be selected from backtest costs.
    Without scenarios the locked plan is kept and the point forecast may
    only raise it.
    """
    net = fl_d[h0:] - fc_dk[h0:]
    if scen is None:
        return np.maximum(net, q0)
    q_hi = np.quantile(scen, q_down, axis=0)
    q_lo = np.quantile(scen, q_up, axis=0)
    target = np.median(np.stack([q_lo, q_hi, np.asarray(q0, float)]), axis=0)
    return np.maximum(net, target)


# --------------------------------------------------------------------------
# adjustment LP: segment net settlement on the remaining horizon
# --------------------------------------------------------------------------

def solve_adjustment(q0, risk_net, prices, initial, terminal_value):
    """Re-solve purchases a over a horizon; deviations settle against q0.

    Variables a, C, D, U, E, p, q (7n).  Objective
    sum c*(a + 0.5*(p+q)) - nu*E_end equals the §3.16 settlement at the
    optimum; a throughput tie-break removes degenerate cycling.
    """
    n = len(risk_net)
    q0 = np.asarray(q0, float)

    def idx(block, t):
        return block * n + t

    a_eq = lil_matrix((2 * n, 7 * n))
    b_eq = np.zeros(2 * n)
    for t in range(n):
        a_eq[t, idx(0, t)] = 1
        a_eq[t, idx(1, t)] = -1
        a_eq[t, idx(2, t)] = 1
        a_eq[t, idx(3, t)] = -1
        b_eq[t] = risk_net[t]
        a_eq[n + t, idx(4, t)] = 1
        a_eq[n + t, idx(1, t)] = -ETA
        a_eq[n + t, idx(2, t)] = 1.0 / ETA
        if t:
            a_eq[n + t, idx(4, t - 1)] = -1
        b_eq[n + t] = initial if t == 0 else 0.0

    a_ub = lil_matrix((2 * n, 7 * n))
    b_ub = np.zeros(2 * n)
    for t in range(n):
        # p >= q0 - a  ->  -p - a <= -q0
        a_ub[2 * t, idx(5, t)] = -1
        a_ub[2 * t, idx(0, t)] = -1
        b_ub[2 * t] = -q0[t]
        # q >= a - q0  ->  -q + a <= q0
        a_ub[2 * t + 1, idx(6, t)] = -1
        a_ub[2 * t + 1, idx(0, t)] = 1
        b_ub[2 * t + 1] = q0[t]

    bounds = (
        [(0, None)] * n
        + [(0, S)] * n
        + [(0, S)] * n
        + [(0, None)] * n
        + [(EMIN, EMAX)] * n
        + [(0, None)] * n
        + [(0, None)] * n
    )
    obj = np.zeros(7 * n)
    obj[:n] = prices
    obj[idx(5, 0): idx(5, 0) + n] = 0.5 * prices
    obj[idx(6, 0): idx(6, 0) + n] = 0.5 * prices
    obj[idx(4, n - 1)] = -terminal_value

    first = linprog(obj, A_eq=a_eq.tocsr(), b_eq=b_eq, A_ub=a_ub.tocsr(),
                    b_ub=b_ub, bounds=bounds, method="highs")
    if not first.success:
        raise RuntimeError(first.message)
    throughput = np.zeros(7 * n)
    throughput[idx(1, 0): idx(1, 0) + n] = 1
    throughput[idx(2, 0): idx(2, 0) + n] = 1
    second = linprog(
        throughput,
        A_eq=a_eq.tocsr(),
        b_eq=b_eq,
        A_ub=csr_matrix(np.vstack([a_ub.toarray(), obj])),
        b_ub=np.r_[b_ub, first.fun + 1e-6],
        bounds=bounds,
        method="highs",
    )
    if not second.success:
        raise RuntimeError(second.message)
    x = second.x.copy()
    x[np.abs(x) < 1e-8] = 0
    a = x[:n]
    e = x[idx(4, 0): idx(4, 0) + n]
    return a, e, float(first.fun)


# --------------------------------------------------------------------------
# shared rule execution (scenario evaluation and live execution alike)
# --------------------------------------------------------------------------

def run_rule_horizon(net, q, ref, prices, initial, forecast_in_effect, stages,
                     observed, theta):
    """Per-stage clipped-affine reserve simulation over one horizon.

    ``stages`` is a list of (start, end, has_lambda) with local indices;
    ``observed`` lists per-stage observed mean errors (NaN means computed
    from the path: stage-local mean of the previous stage's errors against
    ``forecast_in_effect``).  Returns (emergency_cost_per_path,
    end_energy_per_path).
    """
    net = np.asarray(net, float)
    single = net.ndim == 1
    if single:
        net = net[None, :]
    m, h = net.shape
    q = np.asarray(q, float)
    ref = np.asarray(ref, float)
    prices = np.asarray(prices, float)
    forecast_in_effect = np.asarray(forecast_in_effect, float)
    energy = np.full(m, float(initial))
    emergency_cost = np.zeros(m)
    pos = 0
    for j, (s0, s1, has_lam) in enumerate(stages):
        delta = float(theta[pos])
        pos += 1
        lam = float(theta[pos]) if has_lam else 0.0
        pos += 1 if has_lam else 0
        obs = observed[j]
        if not np.isnan(obs):
            a = np.full(m, float(obs))
        elif j == 0:
            a = np.zeros(m)
        else:
            prev_start = stages[j - 1][0]
            a = (net[:, prev_start:s0] - forecast_in_effect[prev_start:s0]).mean(axis=1)
        for t in range(s0, s1):
            reserve = np.clip(ref[t] + delta + lam * a, EMIN, EMAX)
            charge, discharge, emergency, unused, energy = rule_transition(
                net[:, t], q[t], energy, reserve
            )
            emergency_cost += 5.0 * prices[t] * emergency
    if single:
        return emergency_cost[0], energy[0]
    return emergency_cost, energy


def execute_segment(net, q, ref, prices, initial, a_signal, theta):
    """Execute one stage with a fixed observed signal; returns slot arrays."""
    n = len(net)
    delta = float(theta[0])
    lam = float(theta[1]) if len(theta) > 1 else 0.0
    out = {
        "reserve_kwh": np.zeros(n),
        "soc_start_kwh": np.zeros(n),
        "charge_kwh": np.zeros(n),
        "discharge_kwh": np.zeros(n),
        "emergency_kwh": np.zeros(n),
        "unused_kwh": np.zeros(n),
        "soc_end_kwh": np.zeros(n),
    }
    energy = float(initial)
    for t in range(n):
        reserve = float(np.clip(ref[t] + delta + lam * a_signal, EMIN, EMAX))
        start = energy
        charge, discharge, emergency, unused, end = rule_transition(
            float(net[t]), float(q[t]), start, reserve
        )
        charge, discharge, emergency, unused, end = map(
            float, [charge, discharge, emergency, unused, end]
        )
        out["reserve_kwh"][t] = reserve
        out["soc_start_kwh"][t] = start
        out["charge_kwh"][t] = charge
        out["discharge_kwh"][t] = discharge
        out["emergency_kwh"][t] = emergency
        out["unused_kwh"][t] = unused
        out["soc_end_kwh"][t] = end
        energy = end
    return out


def calibrate_staged(scenarios, q, ref, prices, initial, terminal_value,
                     forecast_in_effect, stages, observed, residual_count,
                     settings: Q3Settings, seed: int):
    """Bounded DE with explicit beta=1 (zero) and beta=0 (R=Emin) floors."""
    started = time.perf_counter()
    has_lam = [s[2] for s in stages]
    n_params = sum(1 + int(hl) for hl in has_lam)
    theta0 = np.zeros(n_params)
    theta_min = np.zeros(n_params)
    pos = 0
    for hl in has_lam:
        theta_min[pos] = -settings.delta_bound_kwh
        pos += 1 + int(hl)
    if scenarios is None:
        scenarios = np.asarray(forecast_in_effect, float)[None, :]

    def score(theta):
        ec, e_end = run_rule_horizon(
            scenarios, q, ref, prices, initial, forecast_in_effect,
            stages, observed, theta,
        )
        return float(np.mean(ec - terminal_value * e_end))

    beta1 = score(theta0)
    beta0 = score(theta_min)
    if residual_count < settings.residual_days:
        return {
            "update_slot": stages[0][0],
            "stages_covered": [list(s[:2]) for s in stages],
            "theta": theta0,
            "beta1_score": beta1,
            "beta0_score": beta0,
            "selected_score": beta1,
            "evaluations": 2,
            "seconds": time.perf_counter() - started,
            "method": "zero_parameter_warmup",
            "accepted_search": False,
            "nit": None,
            "success": False,
            "termination": "warmup_zero_parameter",
            "solver_message": "complete 21-day residual window not yet available",
        }

    bounds = []
    for hl in has_lam:
        bounds.append((-settings.delta_bound_kwh, settings.delta_bound_kwh))
        if hl:
            bounds.append((-settings.lambda_bound, settings.lambda_bound))
    result = differential_evolution(
        score,
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
        x0=theta0,
    )
    candidate = np.asarray(result.x, float)
    candidate_score = float(result.fun)
    floor = min(beta1, beta0)
    accepted = candidate_score <= floor + 1e-8
    if accepted:
        theta, selected = candidate, candidate_score
    else:
        theta, selected = (theta0, beta1) if beta1 <= beta0 else (theta_min, beta0)
    return {
        "update_slot": stages[0][0],
        "stages_covered": [list(s[:2]) for s in stages],
        "theta": theta,
        "beta1_score": beta1,
        "beta0_score": beta0,
        "selected_score": selected,
        "evaluations": 2 + int(result.nfev),
        "seconds": time.perf_counter() - started,
        "method": "differential_evolution_staged_direct_search",
        "accepted_search": bool(accepted),
        "nit": int(result.nit),
        "success": bool(result.success),
        "termination": (
            "maxiter_budget" if int(result.nit) >= settings.search_maxiter
            else "population_collapse_convergence"
        ),
        "solver_message": str(result.message),
    }


# --------------------------------------------------------------------------
# record assembly
# --------------------------------------------------------------------------

def settle_of(q0, qA, prices):
    q0 = np.asarray(q0, float)
    qA = np.asarray(qA, float)
    prices = np.asarray(prices, float)
    retained = prices * np.minimum(q0, qA)
    down = 0.5 * prices * np.maximum(q0 - qA, 0.0)
    up = 1.5 * prices * np.maximum(qA - q0, 0.0)
    return retained, down, up


COLUMNS = [
    "date", "slot", "interval_start",
    "load_kwh", "pv_kwh", "price",
    "forecast_load_kwh", "fc_in_effect_kwh", "forecast_net_kwh", "risk_net_kwh",
    "residual_count",
    "q0_kwh", "qA_kwh", "down_kwh", "up_kwh",
    "plan_ref_soc_kwh", "reserve_kwh", "stage_signal_kwh",
    "ldr_delta_kwh", "ldr_lambda",
    "soc_start_kwh", "charge_kwh", "discharge_kwh", "emergency_kwh", "unused_kwh",
    "soc_end_kwh",
    "retained_cost", "down_cost", "up_cost", "emergency_cost", "total_cost",
    "stage", "calibration_used",
]


def _segment_records(seg_start, seg, d, date, load_d, pv_d, prices, fl_d,
                     fc_in_effect, n_hat, risk_by_slot, q0, qA, ref, deltas,
                     lambdas, signals, residual_count, settings):
    rows = []
    n = len(seg["charge_kwh"])
    for i in range(n):
        t = seg_start + i
        stage = min(t // 36, 3)
        retained, down, up = settle_of(q0[t], qA[t], prices[t])
        emergency_cost = 5.0 * prices[t] * seg["emergency_kwh"][i]
        rows.append(
            (
                date, t, date + pd.Timedelta(minutes=10 * t),
                load_d[t], pv_d[t], prices[t],
                fl_d[t], fc_in_effect[t], n_hat[t], risk_by_slot[t],
                residual_count,
                q0[t], qA[t], max(q0[t] - qA[t], 0.0), max(qA[t] - q0[t], 0.0),
                ref[t], seg["reserve_kwh"][i], signals[stage],
                deltas[stage], lambdas[stage],
                seg["soc_start_kwh"][i], seg["charge_kwh"][i],
                seg["discharge_kwh"][i], seg["emergency_kwh"][i],
                seg["unused_kwh"][i], seg["soc_end_kwh"][i],
                retained, down, up, emergency_cost,
                retained + down + up + emergency_cost,
                stage + 1,
                residual_count >= settings.residual_days,
            )
        )
    return rows


# --------------------------------------------------------------------------
# one causal day
# --------------------------------------------------------------------------

def run_day(d, date, load_d, pv_d, prices, fl, fc, energy, settings, load, pv):
    """Return (rows, segments, diagnostics, end_energy, q0, qA)."""
    strategy = settings.strategy
    if strategy == "M0":
        return _run_day_m0(d, date, load_d, pv_d, prices, fl, fc, energy,
                           settings, load, pv)
    return _run_day_staged(d, date, load_d, pv_d, prices, fl, fc, energy,
                           settings, load, pv)


def _run_day_m0(d, date, load_d, pv_d, prices, fl, fc, energy, settings, load, pv):
    """M0 = question-2-style strategy with the 0:00-issuance PV forecast."""
    nu = float(prices.min() / ETA)
    diagnostics = {"calibrations": []}
    if d == 0:
        q0 = np.zeros(T)
        plan_ref = np.full(T, EMIN)
        theta = np.zeros(7)
        residual_count = 0
        n_hat = np.zeros(T)
        risk_by_slot = np.zeros(T)
        diagnostics["planner"] = "zero_plan_cold_start"
        diagnostics["calibrations"].append(
            {"update_slot": 0, "stages_covered": [[0, 36], [36, 72], [72, 108], [108, 144]],
             "theta": np.zeros(7).tolist(), "beta1_score": None, "beta0_score": None,
             "selected_score": None, "evaluations": 0, "seconds": 0.0,
             "method": "zero_plan_cold_start", "accepted_search": False,
             "solver_message": "1 January cold start"}
        )
    else:
        n_hat = fl[d] - fc[d, 0]
        scen0, residual_count = scenario_matrix(d, fl, fc, load, pv, 0, 0,
                                                settings.residual_days)
        risk0 = risk_curve(fl[d], fc[d, 0], scen0, 0, settings.alpha)
        risk_by_slot = risk0
        plan, _ = solve_plan(risk0, prices, energy, nu)
        q0 = plan[0]
        plan_ref = plan[4]
        if scen0 is None:
            scen0 = n_hat[None, :]
        q2_settings = LDRSettings(search_seed=settings.search_seed,
                                  search_maxiter=settings.search_maxiter,
                                  search_popsize=settings.search_popsize,
                                  load_forecast="weekly_persist")
        cal = calibrate_rule(scen0, n_hat, q0, plan_ref, prices, energy, nu,
                             residual_count, q2_settings,
                             settings.search_seed + d)
        theta = cal.theta
        diagnostics["planner"] = "quantile_LP_alpha_0.8"
        diagnostics["residual_count"] = residual_count
        diagnostics["calibrations"].append(
            {"update_slot": 0, "stages_covered": [[0, 36], [36, 72], [72, 108], [108, 144]],
             "theta": theta.tolist(), "beta1_score": cal.zero_score,
             "beta0_score": None, "selected_score": cal.selected_score,
             "evaluations": cal.evaluations, "seconds": cal.seconds,
             "method": cal.method, "accepted_search": cal.accepted_search,
             "solver_message": cal.solver_message}
        )

    initial_soc = float(energy)
    out = execute_actual_day(load_d, pv_d, q0, plan_ref, prices, energy, n_hat, theta)
    deltas, lambdas = unpack_theta(theta)
    signals = out["stage_error_mean_kwh"]
    qA = q0.copy()
    fc_in_effect = fc[d, 0]
    rows = []
    for t in range(T):
        stage = min(t // 36, 3)
        retained, down, up = settle_of(q0[t], qA[t], prices[t])
        emergency_cost = 5.0 * prices[t] * out["emergency_kwh"][t]
        rows.append(
            (
                date, t, date + pd.Timedelta(minutes=10 * t),
                load_d[t], pv_d[t], prices[t],
                fl[d, t], fc_in_effect[t], n_hat[t], risk_by_slot[t],
                residual_count,
                q0[t], qA[t], max(q0[t] - qA[t], 0.0), max(qA[t] - q0[t], 0.0),
                plan_ref[t], out["reserve_kwh"][t], signals[t],
                deltas[stage], lambdas[stage],
                out["soc_start_kwh"][t], out["charge_kwh"][t],
                out["discharge_kwh"][t], out["emergency_kwh"][t],
                out["unused_kwh"][t], out["soc_end_kwh"][t],
                retained, down, up, emergency_cost,
                retained + down + up + emergency_cost,
                stage + 1,
                residual_count >= settings.residual_days,
            )
        )
    segments = [
        dict(net=load_d - pv_d, q=q0, ref=plan_ref, n_hat=n_hat, prices=prices,
             initial=initial_soc, signal=0.0, delta=deltas, lam=lambdas)
    ]
    diagnostics["initial_soc_kwh"] = initial_soc
    end_energy = float(out["soc_end_kwh"][-1])
    return rows, segments, diagnostics, end_energy, q0, qA


def _run_day_staged(d, date, load_d, pv_d, prices, fl, fc, energy, settings, load, pv):
    strategy = settings.strategy
    spec = STRATEGIES[strategy]
    actual_net = load_d - pv_d
    segments = []
    rows = []
    diagnostics = {"calibrations": []}
    nu = float(prices.min() / ETA)

    q0 = None
    qA = np.zeros(T)
    ref = np.zeros(T)
    deltas = np.zeros(4)
    lambdas = np.zeros(4)
    signals = np.zeros(4)
    fc_in_effect = np.zeros(T)
    n_hat = np.zeros(T)
    risk_by_slot = np.zeros(T)
    residual_count = 0

    if d == 0:
        q0 = np.zeros(T)
        qA[:] = q0
        ref[:] = EMIN
        diagnostics["planner"] = "zero_plan_cold_start"
        diagnostics["calibrations"].append(
            {"update_slot": 0, "stages_covered": [[0, 36]], "theta": [0.0],
             "beta1_score": None, "beta0_score": None, "selected_score": None,
             "evaluations": 0, "seconds": 0.0, "method": "zero_plan_cold_start",
             "accepted_search": False, "solver_message": "1 January cold start"}
        )
    else:
        # ---- 0:00 ----
        scen0, residual_count = scenario_matrix(d, fl, fc, load, pv, 0, 0,
                                                settings.residual_days)
        risk0 = risk_curve(fl[d], fc[d, 0], scen0, 0, settings.alpha)
        plan, _ = solve_plan(risk0, prices, energy, nu)
        q0 = plan[0]
        qA[:] = q0
        ref[:] = plan[4]
        risk_by_slot[:] = risk0
        fc_in_effect[:] = fc[d, 0]
        n_hat[:] = fl[d] - fc[d, 0]
        diagnostics["planner"] = "quantile_LP_alpha_0.8"
        diagnostics["residual_count"] = residual_count
        cal0 = calibrate_staged(
            scen0[:, :36] if scen0 is not None else None,
            q0[:36], plan[4][:36], prices[:36], energy, nu, n_hat[:36],
            stages=[(0, 36, False)], observed=[np.nan],
            residual_count=residual_count, settings=settings,
            seed=settings.search_seed + 10 * d,
        )
        deltas[0] = float(cal0["theta"][0])
        lambdas[0] = 0.0
        diagnostics["calibrations"].append(cal0)

    # ---- execute stage 0 (0:00-6:00) ----
    seg0 = execute_segment(actual_net[:36], qA[:36], ref[:36], prices[:36],
                           energy, 0.0, np.array([deltas[0], lambdas[0]]))
    initial_soc = float(seg0["soc_start_kwh"][0])
    energy = float(seg0["soc_end_kwh"][-1])
    segments.append(dict(net=actual_net[:36], q=qA[:36], ref=ref[:36], prices=prices[:36],
                         initial=initial_soc, signal=0.0, delta=deltas[0], lam=lambdas[0]))
    rows.extend(_segment_records(0, seg0, d, date, load_d, pv_d, prices, fl[d],
                                 fc_in_effect, n_hat, risk_by_slot, q0, qA, ref,
                                 deltas, lambdas, signals, residual_count, settings))
    diagnostics["initial_soc_kwh"] = initial_soc

    if d > 0:
        # ---- 6:00 ----
        a6 = float((actual_net[:36] - n_hat[:36]).mean())
        signals[1] = a6
        scen6, _ = scenario_matrix(d, fl, fc, load, pv, 1, 36, settings.residual_days)
        risk6 = adjustment_curve(fl[d], fc[d, 1], scen6, q0[36:], 36,
                                 q_up=settings.adjustment_up_quantile)
        a6plan, ep6, _ = solve_adjustment(q0[36:], risk6, prices[36:], energy, nu)
        qA[36:] = a6plan
        ref[36:] = ep6
        risk_by_slot[36:] = risk6
        fc_in_effect[36:] = fc[d, 1, 36:]
        n_hat[36:] = fl[d, 36:] - fc[d, 1, 36:]
        if strategy == "M6":
            stages6 = [(0, 36, True), (36, 72, True), (72, 108, True)]
            cal6 = calibrate_staged(
                scen6, a6plan, ep6, prices[36:], energy, nu, n_hat[36:],
                stages=stages6, observed=[a6, np.nan, np.nan],
                residual_count=residual_count, settings=settings,
                seed=settings.search_seed + 10 * d + 1,
            )
            pos = 0
            for j, (_, _, hl) in enumerate(stages6):
                deltas[j + 1] = float(cal6["theta"][pos])
                lambdas[j + 1] = float(cal6["theta"][pos + 1]) if hl else 0.0
                pos += 1 + int(hl)
        else:
            cal6 = calibrate_staged(
                scen6[:, :36] if scen6 is not None else None,
                a6plan[:36], ep6[:36], prices[36:72], energy, nu, n_hat[36:72],
                stages=[(0, 36, True)], observed=[a6],
                residual_count=residual_count, settings=settings,
                seed=settings.search_seed + 10 * d + 1,
            )
            deltas[1] = float(cal6["theta"][0])
            lambdas[1] = float(cal6["theta"][1])
        diagnostics["calibrations"].append(cal6)
        cal6["update_slot"] = 36

    # ---- execute stage 1 (6:00-12:00) ----
    seg1 = execute_segment(actual_net[36:72], qA[36:72], ref[36:72], prices[36:72],
                           energy, signals[1], np.array([deltas[1], lambdas[1]]))
    energy = float(seg1["soc_end_kwh"][-1])
    segments.append(dict(net=actual_net[36:72], q=qA[36:72], ref=ref[36:72],
                         prices=prices[36:72], initial=float(seg1["soc_start_kwh"][0]),
                         signal=signals[1], delta=deltas[1], lam=lambdas[1]))
    rows.extend(_segment_records(36, seg1, d, date, load_d, pv_d, prices, fl[d],
                                 fc_in_effect, n_hat, risk_by_slot, q0, qA, ref,
                                 deltas, lambdas, signals, residual_count, settings))

    if d > 0 and strategy in ("M612", "M61218-S", "M61218-F"):
        # ---- 12:00 ----
        a12 = float((actual_net[36:72] - n_hat[36:72]).mean())
        signals[2] = a12
        scen12, _ = scenario_matrix(d, fl, fc, load, pv, 2, 72, settings.residual_days)
        risk12 = adjustment_curve(fl[d], fc[d, 2], scen12, q0[72:], 72,
                                  q_up=settings.adjustment_up_quantile)
        a12plan, ep12, _ = solve_adjustment(q0[72:], risk12, prices[72:], energy, nu)
        qA[72:] = a12plan
        ref[72:] = ep12
        risk_by_slot[72:] = risk12
        fc_in_effect[72:] = fc[d, 2, 72:]
        n_hat[72:] = fl[d, 72:] - fc[d, 2, 72:]
        cal12 = calibrate_staged(
            scen12, a12plan, ep12, prices[72:], energy, nu, n_hat[72:],
            stages=[(0, 36, True), (36, 72, True)], observed=[a12, np.nan],
            residual_count=residual_count, settings=settings,
            seed=settings.search_seed + 10 * d + 2,
        )
        deltas[2] = float(cal12["theta"][0])
        lambdas[2] = float(cal12["theta"][1])
        deltas[3] = float(cal12["theta"][2])
        lambdas[3] = float(cal12["theta"][3])
        diagnostics["calibrations"].append(cal12)
        cal12["update_slot"] = 72

    # M6: no 12:00 purchase update; the stage-2 signal is the observed
    # 6:00-12:00 error against the 6:00 forecast (needed before seg2 runs)
    if d > 0 and strategy == "M6":
        signals[2] = float((actual_net[36:72] - n_hat[36:72]).mean())

    # ---- execute stage 2 (12:00-18:00) ----
    seg2 = execute_segment(actual_net[72:108], qA[72:108], ref[72:108],
                           prices[72:108], energy, signals[2],
                           np.array([deltas[2], lambdas[2]]))
    energy = float(seg2["soc_end_kwh"][-1])
    segments.append(dict(net=actual_net[72:108], q=qA[72:108], ref=ref[72:108],
                         prices=prices[72:108], initial=float(seg2["soc_start_kwh"][0]),
                         signal=signals[2], delta=deltas[2], lam=lambdas[2]))
    rows.extend(_segment_records(72, seg2, d, date, load_d, pv_d, prices, fl[d],
                                 fc_in_effect, n_hat, risk_by_slot, q0, qA, ref,
                                 deltas, lambdas, signals, residual_count, settings))

    if d > 0 and strategy in ("M61218-S", "M61218-F"):
        # ---- 18:00 purchase re-optimisation, no recalibration ----
        a18 = float((actual_net[72:108] - n_hat[72:108]).mean())
        signals[3] = a18
        k18 = spec["issuance"][108]
        scen18, _ = scenario_matrix(d, fl, fc, load, pv, k18, 108,
                                    settings.residual_days)
        risk18 = adjustment_curve(fl[d], fc[d, k18], scen18, q0[108:], 108,
                                  q_up=settings.adjustment_up_quantile)
        a18plan, ep18, _ = solve_adjustment(q0[108:], risk18, prices[108:],
                                            energy, nu)
        qA[108:] = a18plan
        ref[108:] = ep18
        risk_by_slot[108:] = risk18
        fc_in_effect[108:] = fc[d, k18, 108:]
        n_hat[108:] = fl[d, 108:] - fc[d, k18, 108:]
        diagnostics["calibrations"].append(
            {"update_slot": 108, "stages_covered": [], "theta": [],
             "beta1_score": None, "beta0_score": None, "selected_score": None,
             "evaluations": 0, "seconds": 0.0,
             "method": "purchase_reoptimisation_only_no_recalibration",
             "accepted_search": False,
             "solver_message": "18:00 no recalibration, no new forecast in M61218-S"}
        )
    elif d > 0:
        # M612: stage-2 signal was set at 12:00; M6's was set before seg2.
        # Both compute a18 from the executed 12:00-18:00 errors.
        a18 = float((actual_net[72:108] - n_hat[72:108]).mean())
        signals[3] = a18

    # ---- execute stage 3 (18:00-24:00) ----
    seg3 = execute_segment(actual_net[108:], qA[108:], ref[108:], prices[108:],
                           energy, signals[3], np.array([deltas[3], lambdas[3]]))
    energy = float(seg3["soc_end_kwh"][-1])
    segments.append(dict(net=actual_net[108:], q=qA[108:], ref=ref[108:], prices=prices[108:],
                         initial=float(seg3["soc_start_kwh"][0]), signal=signals[3],
                         delta=deltas[3], lam=lambdas[3]))
    rows.extend(_segment_records(108, seg3, d, date, load_d, pv_d, prices, fl[d],
                                 fc_in_effect, n_hat, risk_by_slot, q0, qA, ref,
                                 deltas, lambdas, signals, residual_count, settings))
    return rows, segments, diagnostics, energy, q0, qA


def replay_controller(segments, controller):
    """Same locked plans, same initial SOC, different controllers."""
    emergency_total = 0.0
    for seg in segments:
        if controller == "beta0":
            ref = np.full(len(seg["ref"]), EMIN)
            theta = np.array([0.0, 0.0])
        elif controller == "beta1":
            ref = seg["ref"]
            theta = np.array([0.0, 0.0])
        else:
            ref = seg["ref"]
            theta = np.array([seg["delta"], seg["lam"]])
        out = execute_segment(seg["net"], seg["q"], ref, seg["prices"],
                              seg["initial"], seg["signal"], theta)
        emergency_total += float(np.sum(5.0 * seg["prices"] * out["emergency_kwh"]))
    return emergency_total


def replay_m0(seg, load_d, pv_d, controller):
    theta = np.zeros(7)
    ref = seg["ref"] if controller == "beta1" else np.full(T, EMIN)
    out = execute_actual_day(load_d, pv_d, seg["q"], ref, seg["prices"],
                             seg["initial"], seg["n_hat"], theta)
    return float(np.sum(5.0 * seg["prices"] * out["emergency_kwh"]))


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def validate_question3(f):
    balance = f.qA_kwh + f.emergency_kwh + f.pv_kwh + f.discharge_kwh - f.load_kwh - f.charge_kwh - f.unused_kwh
    state = f.soc_start_kwh + ETA * f.charge_kwh - f.discharge_kwh / ETA - f.soc_end_kwh
    retained = f.price * np.minimum(f.q0_kwh, f.qA_kwh)
    down = 0.5 * f.price * np.maximum(f.q0_kwh - f.qA_kwh, 0)
    up = 1.5 * f.price * np.maximum(f.qA_kwh - f.q0_kwh, 0)
    total = retained + down + up + f.emergency_cost
    checks = {
        "max_balance_error_kwh": float(np.abs(balance).max()),
        "max_state_error_kwh": float(np.abs(state).max()),
        "max_continuity_error_kwh": float(np.abs(f.soc_start_kwh.to_numpy()[1:] - f.soc_end_kwh.to_numpy()[:-1]).max()),
        "min_soc_kwh": float(min(f.soc_start_kwh.min(), f.soc_end_kwh.min())),
        "max_soc_kwh": float(max(f.soc_start_kwh.max(), f.soc_end_kwh.max())),
        "max_charge_kwh": float(f.charge_kwh.max()),
        "max_discharge_kwh": float(f.discharge_kwh.max()),
        "simultaneous_charge_discharge": int(((f.charge_kwh > 1e-6) & (f.discharge_kwh > 1e-6)).sum()),
        "emergency_while_charging": int(((f.charge_kwh > 1e-6) & (f.emergency_kwh > 1e-6)).sum()),
        "max_settlement_recompute_error": float(np.abs(total - f.total_cost).max()),
        "negative_q0": int((f.q0_kwh < -1e-7).sum()),
        "negative_qA": int((f.qA_kwh < -1e-7).sum()),
        "pre_six_executed_adjusted": int((f.slot < 36).sum() - ((f.slot < 36) & (np.abs(f.qA_kwh - f.q0_kwh) < 1e-9)).sum()),
    }
    assert max(checks[k] for k in ["max_balance_error_kwh", "max_state_error_kwh", "max_continuity_error_kwh"]) < 1e-5
    assert checks["min_soc_kwh"] >= EMIN - 1e-5 and checks["max_soc_kwh"] <= EMAX + 1e-5
    assert max(checks["max_charge_kwh"], checks["max_discharge_kwh"]) <= S + 1e-5
    assert checks["simultaneous_charge_discharge"] == checks["emergency_while_charging"] == 0
    assert checks["max_settlement_recompute_error"] < 1e-6
    assert checks["negative_q0"] == checks["negative_qA"] == 0
    checks["passed"] = True
    return checks


# --------------------------------------------------------------------------
# strategy runner
# --------------------------------------------------------------------------

def _period_metrics(frame):
    formal = frame[frame.date >= pd.Timestamp("2025-02-01")]
    if not len(formal):
        return {}
    out = {
        "days": int(formal["date"].nunique()),
        "intervals": int(len(formal)),
        **{key: float(formal[key].sum()) for key in [
            "q0_kwh", "qA_kwh", "down_kwh", "up_kwh", "charge_kwh", "discharge_kwh",
            "emergency_kwh", "unused_kwh", "retained_cost", "down_cost", "up_cost",
            "emergency_cost", "total_cost",
        ]},
        "initial_soc_kwh": float(formal.soc_start_kwh.iloc[0]),
        "final_soc_kwh": float(formal.soc_end_kwh.iloc[-1]),
        "emergency_intervals": int((formal.emergency_kwh > 1e-6).sum()),
        "adjusted_intervals": int((np.abs(formal.qA_kwh - formal.q0_kwh) > 1e-6).sum()),
    }
    return out


def run_strategy(dates, load, pv, prices, fl, fc, settings, limit=None,
                 start_idx=0, initial_energy=6000.0):
    """Continuous causal run from ``start_idx`` at ``initial_energy``.

    When ``start_idx > 0`` the strategy never touches its own January
    execution: all five strategies then fork from one common February
    opening inventory produced by the shared warm-up (the 0:00-only M0
    policy).  History before ``start_idx`` is still available for
    residuals, so 1 February keeps a complete 21-day residual window.
    """
    started = time.perf_counter()
    if limit is None:
        limit = len(dates)
    strategy = settings.strategy
    energy = float(initial_energy)
    records = []
    day_diags = []
    paired = []
    for d in range(start_idx, limit):
        date = dates[d]
        day_started = time.perf_counter()
        rows, segments, diag, energy, q0, qA = run_day(
            d, date, load[d], pv[d], prices, fl, fc, energy, settings, load, pv
        )
        records.extend(rows)
        retained, down, up = settle_of(q0, qA, prices)
        settlement_day = float((retained + down + up).sum())
        if strategy == "M0":
            seg = segments[0]
            ldr_em = float(np.sum(5.0 * prices * np.asarray([r[23] for r in rows])))
            beta1_em = replay_m0(seg, load[d], pv[d], "beta1")
            beta0_em = replay_m0(seg, load[d], pv[d], "beta0")
        else:
            ldr_em = replay_controller(segments, "ldr")
            beta1_em = replay_controller(segments, "beta1")
            beta0_em = replay_controller(segments, "beta0")
        initial_soc = float(diag["initial_soc_kwh"])
        paired.append({
            "date": str(date.date()),
            "same_plan_initial_soc_kwh": initial_soc,
            "settlement_cost": settlement_day,
            "ldr_emergency_cost": ldr_em,
            "beta1_emergency_cost": beta1_em,
            "beta0_emergency_cost": beta0_em,
            "ldr_total_cost": settlement_day + ldr_em,
            "beta1_total_cost": settlement_day + beta1_em,
            "beta0_total_cost": settlement_day + beta0_em,
        })
        day_diags.append({
            "date": str(date.date()),
            "strategy": strategy,
            "initial_soc_kwh": initial_soc,
            "final_soc_kwh": float(energy),
            "residual_count": diag.get("residual_count", 0),
            "q0_cost_face": float(np.dot(prices, q0)),
            "settlement_cost": settlement_day,
            "retained_cost": float(retained.sum()),
            "down_cost": float(down.sum()),
            "up_cost": float(up.sum()),
            "emergency_cost": ldr_em,
            "total_cost": settlement_day + ldr_em,
            "down_kwh": float(np.maximum(q0 - qA, 0).sum()),
            "up_kwh": float(np.maximum(qA - q0, 0).sum()),
            "calibration_count": len(diag["calibrations"]),
            "calibration_seconds": float(sum(c.get("seconds", 0.0) for c in diag["calibrations"])),
            "day_seconds": time.perf_counter() - day_started,
            **{f"cal_{i}_{key}": (c.get(key) if not isinstance(c.get(key), (list, np.ndarray)) else json.dumps(c.get(key))) for i, c in enumerate(diag["calibrations"]) for key in ["update_slot", "method", "accepted_search", "evaluations", "selected_score", "nit", "success", "termination"]},
        })
        if (d - start_idx + 1) % 30 == 0 or d + 1 == limit:
            print(f"{strategy} {d - start_idx + 1}/{limit - start_idx} days; date={date.date()} "
                  f"calib={day_diags[-1]['calibration_seconds']:.2f}s", flush=True)

    frame = pd.DataFrame.from_records(records, columns=COLUMNS)
    validation = validate_question3(frame)
    formal_metrics = _period_metrics(frame)
    full = frame
    full_metrics = {
        "days": int(full["date"].nunique()),
        "intervals": int(len(full)),
        **{key: float(full[key].sum()) for key in [
            "q0_kwh", "qA_kwh", "down_kwh", "up_kwh", "charge_kwh", "discharge_kwh",
            "emergency_kwh", "unused_kwh", "retained_cost", "down_cost", "up_cost",
            "emergency_cost", "total_cost",
        ]},
        "initial_soc_kwh": float(full.soc_start_kwh.iloc[0]),
        "final_soc_kwh": float(full.soc_end_kwh.iloc[-1]),
        "emergency_intervals": int((full.emergency_kwh > 1e-6).sum()),
    }
    diag_frame = pd.DataFrame(day_diags)
    pair_frame = pd.DataFrame(paired)
    summary = {
        "settings": {
            **asdict(settings),
            "plan_generator": "fixed_alpha_0.8_quantile_LP",
            "controller": "staged_clipped_affine_reserve_rule",
            "updates": STRATEGIES[strategy]["updates"],
            "issuance_by_update": STRATEGIES[strategy]["issuance"],
            "settlement": "segment_net_settlement_s316",
            "adjustment_curve": "median(Q_up, Q90, q0) with point-forecast floor; "
            "Q_up=0.7 derived from R'=1.5c vs 5c emergency",
            "calibration_acceptance": "candidate must not exceed min(beta1, beta0) floor",
            "start_idx": start_idx,
            "run_start_date": str(dates[start_idx].date()),
            "common_initial_soc_kwh": float(initial_energy),
            "warmup": "January executed once by the shared 0:00-only policy; "
            "all strategies fork from its 1 February inventory",
        },
        "terminal_value": float(prices.min() / ETA),
        "seconds": time.perf_counter() - started,
        "result_days": formal_metrics.get("days"),
        "result_intervals": formal_metrics.get("intervals"),
        **{key: formal_metrics.get(key) for key in [
            "q0_kwh", "qA_kwh", "down_kwh", "up_kwh", "charge_kwh", "discharge_kwh",
            "emergency_kwh", "unused_kwh", "retained_cost", "down_cost", "up_cost",
            "emergency_cost", "total_cost", "adjusted_intervals",
        ]},
        "initial_result_soc_kwh": formal_metrics.get("initial_soc_kwh"),
        "final_soc_kwh": formal_metrics.get("final_soc_kwh"),
        "emergency_intervals": formal_metrics.get("emergency_intervals"),
        "calibrated_days": int((diag_frame.residual_count >= settings.residual_days).sum()),
        "search_evaluations": int(diag_frame[[c for c in diag_frame.columns if c.startswith("cal_") and c.endswith("_evaluations")]].sum().sum()),
        "formal_period": formal_metrics,
        "full_year": full_metrics,
        "validation": validation,
    }
    return frame, diag_frame, pair_frame, summary


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------

def write_strategy_outputs(out: Path, strategy: str, frame, diag_frame, pair_frame, summary, prices):
    out.mkdir(parents=True, exist_ok=True)
    formal = frame[frame.date >= pd.Timestamp("2025-02-01")].copy()
    frame.to_csv(out / "question3_schedule_with_warmup.csv", index=False, encoding="utf-8-sig")
    formal.to_csv(out / "question3_schedule.csv", index=False, encoding="utf-8-sig")
    daily_all = frame.groupby("date").agg(
        q0_kwh=("q0_kwh", "sum"),
        qA_kwh=("qA_kwh", "sum"),
        down_kwh=("down_kwh", "sum"),
        up_kwh=("up_kwh", "sum"),
        charge_kwh=("charge_kwh", "sum"),
        discharge_kwh=("discharge_kwh", "sum"),
        emergency_kwh=("emergency_kwh", "sum"),
        unused_kwh=("unused_kwh", "sum"),
        retained_cost=("retained_cost", "sum"),
        down_cost=("down_cost", "sum"),
        up_cost=("up_cost", "sum"),
        emergency_cost=("emergency_cost", "sum"),
        total_cost=("total_cost", "sum"),
        soc_start_kwh=("soc_start_kwh", "first"),
        soc_end_kwh=("soc_end_kwh", "last"),
    )
    daily = daily_all[daily_all.index >= pd.Timestamp("2025-02-01")]
    daily_all.to_csv(out / "question3_daily_with_warmup.csv", encoding="utf-8-sig")
    daily.to_csv(out / "question3_daily.csv", encoding="utf-8-sig")
    diag_frame.to_csv(out / "question3_daily_parameters.csv", index=False, encoding="utf-8-sig")
    pair_frame.to_csv(out / "question3_paired_controller_comparison.csv", index=False, encoding="utf-8-sig")
    (out / "question3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "strategy": strategy,
        "source_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in ["附件/附件1.xlsx", "附件/附件2.xlsx", "附件/附件3.xlsx", "附件/附件5/result3.xlsx"]
        },
        "settlement": "segment net settlement: c*min(q0,qA) + 0.5c*(q0-qA)+ + 1.5c*(qA-q0)+ + 5c*b",
        "information_cutoff": "each update uses only the issuance forecast and residuals available before its clock time",
        "template_last_purchase_column": "0:00-0:10+1 is the NEXT day 00:00-00:10 (same mapping as question 2 appendix B)",
        "reported_bill": "retained + down penalty + up premium + 5x emergency; no terminal credit deducted",
    }
    (out / "question3_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def summarize_emergency(frame):
    rows = []
    for date, day in frame.groupby("date"):
        q = day.emergency_kwh.to_numpy()
        t = 0
        found = False
        while t < T:
            if q[t] <= 1e-6:
                t += 1
                continue
            found = True
            begin = t
            while t < T and q[t] > 1e-6:
                t += 1
            rows.append([str(date.date()), f"{time_label(begin)}-{time_label(t)}", float(q[begin:t].sum())])
        if not found:
            rows.append([str(date.date()), "无", 0.0])
    return rows


def write_main_outputs(out: Path, frame, prices):
    """Key-date tables and the official result3 workbook payload for M612."""
    formal = frame[frame.date >= pd.Timestamp("2025-02-01")].copy()
    daily = formal.groupby("date").agg(
        q0_kwh=("q0_kwh", "sum"), qA_kwh=("qA_kwh", "sum"),
        retained_cost=("retained_cost", "sum"), down_cost=("down_cost", "sum"),
        up_cost=("up_cost", "sum"), emergency_cost=("emergency_cost", "sum"),
        total_cost=("total_cost", "sum"),
        soc_start_kwh=("soc_start_kwh", "first"), soc_end_kwh=("soc_end_kwh", "last"),
    )
    selected = []
    for date in KEY_DATES:
        day = frame[frame.date == pd.Timestamp(date)]
        d = daily.loc[pd.Timestamp(date)]
        row = {"date": date}
        for h in [10, 12, 14, 16, 18, 20]:
            g = day[day.slot == h * 6]
            row[f"q0_{h:02d}_00"] = float(g.q0_kwh.iloc[0])
            row[f"qA_{h:02d}_00"] = float(g.qA_kwh.iloc[0])
        row.update({
            "day_q0_kwh": float(d.q0_kwh), "day_qA_kwh": float(d.qA_kwh),
            "face_cost": float(np.dot(prices, day.q0_kwh.to_numpy())),
            "settlement_cost": float(d.retained_cost + d.down_cost + d.up_cost),
            "retained_cost": float(d.retained_cost),
            "down_cost": float(d.down_cost),
            "up_cost": float(d.up_cost),
            "emergency_cost": float(d.emergency_cost),
            "total_cost": float(d.total_cost),
            "soc_start_kwh": float(d.soc_start_kwh),
            "soc_end_kwh": float(d.soc_end_kwh),
        })
        selected.append(row)
    pd.DataFrame(selected).to_csv(out / "question3_key_dates.csv", index=False, encoding="utf-8-sig")

    keyblocks = []
    for date in KEY_DATES:
        day = frame[frame.date == pd.Timestamp(date)]
        for block in range(6):
            f = day.iloc[block * 24: (block + 1) * 24]
            keyblocks.append([
                date, f"{block*4}:00-{(block+1)*4}:00",
                float(f.charge_kwh.sum()), float(f.discharge_kwh.sum()),
                float(day.soc_start_kwh.iloc[0]) if block == 0 else (float(day.soc_end_kwh.iloc[-1]) if block == 1 else None),
            ])
    pd.DataFrame(keyblocks, columns=["date", "block", "charge_kwh", "discharge_kwh", "boundary_soc_kwh"]).to_csv(
        out / "question3_key_battery.csv", index=False, encoding="utf-8-sig"
    )

    # official template payload (template mapping identical to question 2)
    grid_by_date = {d: day.q0_kwh.to_numpy() for d, day in frame.groupby("date")}
    adj_by_date = {d: day.qA_kwh.to_numpy() for d, day in frame.groupby("date")}
    purchases, adjusted, batteries, emergency = [], [], [], []
    for date, day in formal.groupby("date"):
        g0 = day.q0_kwh.to_numpy()
        ga = day.qA_kwh.to_numpy()
        next_date = date + pd.Timedelta(days=1)
        next_first_q0 = float(grid_by_date[next_date][0]) if next_date in grid_by_date else 0.0
        next_first_qA = float(adj_by_date[next_date][0]) if next_date in adj_by_date else 0.0
        purchases.append([str(date.date()), *np.r_[g0[1:], next_first_q0].tolist(),
                          float(g0.sum()), float(np.dot(prices, g0))])
        settlement = float(day.retained_cost.sum() + day.down_cost.sum() + day.up_cost.sum())
        adjusted.append([str(date.date()), *np.r_[ga[1:], next_first_qA].tolist(),
                         float(ga.sum()), settlement])
        for block in range(6):
            f = day.iloc[block * 24: (block + 1) * 24]
            batteries.append([
                str(date.date()) if block == 0 else None,
                f"{block*4}:00-{(block+1)*4}:00",
                float(f.charge_kwh.sum()), float(f.discharge_kwh.sum()),
                "00:00:00" if block == 0 else ("24:00" if block == 1 else None),
                float(day.soc_start_kwh.iloc[0]) if block == 0 else (float(day.soc_end_kwh.iloc[-1]) if block == 1 else None),
            ])
    emergency = summarize_emergency(formal)
    payload = {
        "purchases": purchases,
        "adjusted": adjusted,
        "batteries": batteries,
        "emergency": emergency,
        "prices_template_order": np.r_[prices[1:], prices[0]].tolist(),
    }
    (out / "question3_workbook_payload.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(emergency, columns=["date", "interval", "emergency_kwh"]).to_csv(
        out / "question3_emergency.csv", index=False, encoding="utf-8-sig"
    )


def write_report(out: Path, summaries: list, frame, diag_frame, pair_frame):
    by_name = {s["settings"]["strategy"]: s for s in summaries}
    m0, m6 = by_name["M0"], by_name["M6"]
    m612, ms, mf = by_name["M612"], by_name["M61218-S"], by_name["M61218-F"]
    v6 = m0["total_cost"] - m6["total_cost"]
    v12 = m6["total_cost"] - m612["total_cost"]
    v18_state = m612["total_cost"] - ms["total_cost"]
    v18_forecast = ms["total_cost"] - mf["total_cost"]
    lines = [
        "# 问题三 全年实施与结果说明（主策略 M612）",
        "",
        "## 结论摘要",
        "",
        "| 策略 | 常规结算费 | 下调违约费 | 上调新增费 | 紧急购电费 | 实际总费 | 紧急电量(kWh) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        m = s["formal_period"]
        lines.append(
            f"| {s['settings']['strategy']} | {m['retained_cost']:,.2f} | {m['down_cost']:,.2f} "
            f"| {m['up_cost']:,.2f} | {m['emergency_cost']:,.2f} | {m['total_cost']:,.2f} "
            f"| {m['emergency_kwh']:,.2f} |"
        )
    lines += [
        "",
        "## 预报时刻价值分解（正式期，元）",
        "",
        f"- 6:00 更新价值 V6 = C_M0 − C_M6 = {v6:,.2f}",
        f"- 12:00 更新价值 V12 = C_M6 − C_M612 = {v12:,.2f}",
        f"- 18:00 状态重优化价值 V18_state = C_M612 − C_M61218-S = {v18_state:,.2f}",
        f"- 18:00 新光伏预报纯增量 V18_forecast = C_M61218-S − C_M61218-F = {v18_forecast:,.2f}",
        "",
        "M61218-S 在 18:00 用 12:00 预报与当前 SOC 重解 18:00—24:00 购电计划；"
        "M61218-F 在此基础上改用 18:00 新预报。三套方案的 18:00 LDR 处理完全相同"
        "（12:00 锁定参数、18:00 仅更新 a18），因此 V18_state 隔离的是购电重优化价值，"
        "V18_forecast 才是 18:00 新预报的纯增量。",
        "",
        "## 主策略 M612 结构",
        "",
        "- 0:00：周持久化负载 + 附件3 0:00 预报（PCHIP 10 分钟化）→ 80% 分位数 LP 得到 q0 与 E^p,0；校准 δ0（1 维，0—6 时）。",
        "- 6:00：按分段净结算对 q0 重解 6:00—24:00（双面报童曲线 median(Q70,Q90,q0)），锁定 6:00—12:00；校准 (δ6, λ6)（2 维），信号 a6 为 0—6 时已实现误差均值。",
        "- 12:00：重解并锁定 12:00—24:00（同双面曲线）；联合校准 (δ12, λ12, δ18, λ18)（4 维），a18 在情景内逐路径计算、参数在 12:00 锁定。",
        "- 18:00：不调整购电、不使用 18:00 预报、不重新校准；仅代入实测 a18 更新保留阈值。",
        "- 风险水平按结算结构先验固定（单时段无储能报童条件 R'(a)=5c·P(D>a)）：0:00 计划 R'=c → F=1−1/5=0.8；调整下调区 R'=0.5c → F=1−0.5/5=0.9；调整上调区 R'=1.5c → F=1−1.5/5=0.7；曲线为 median(Q70,Q90,q0)，点预测作下限。Q50/Q80/Q90 仅作敏感性对照，不按回测费用选参。",
        "- 校准均以最近 21 个完整历史日为情景、β=1 与 β=0 为显式保底候选；不劣于两者较优者才接受搜索结果。",
        "- 共同起点：1月仅由 M0（仅0:00）从6000 kWh 预热一次，五种策略在2月1日以同一库存分叉，此后状态差异为策略真实结果。",
        "",
        "## 校验",
        "",
    ]
    for key, value in m612["validation"].items():
        lines.append(f"- {key}: {value}")
    lines += [
        "",
        "## 解释限制",
        "",
        "0:00 计划未显式建模 6:00 可调整的期权价值（近视滚动 MPC），其价值由 M0→M6→M612 的对照量化；"
        "计划层 LP 是生成计划的代理，真实费用以全年仿真回放为准；每阶段直接搜索不宣称全局最优。",
    ]
    (out / "问题三实施与结果说明.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/question3/current")
    parser.add_argument("--strategies", nargs="*", default=None,
                        choices=STRATEGY_ORDER)
    parser.add_argument("--search-seed", type=int, default=Q3Settings.search_seed)
    parser.add_argument("--search-maxiter", type=int, default=Q3Settings.search_maxiter)
    parser.add_argument("--search-popsize", type=int, default=Q3Settings.search_popsize)
    args = parser.parse_args()
    strategies = args.strategies or STRATEGY_ORDER

    dates, load, pv, prices = load_inputs()
    representative = (
        pd.read_excel(ROOT / "附件/附件1.xlsx", sheet_name=0).iloc[:, 2].to_numpy(float) / 6.0
    )
    assert representative.shape == (T,)
    fl = load_forecast_weekly_persist(load, representative)
    fc = build_issuance_curves(ROOT)

    if args.days != 365:
        pilot_out = args.output / "pilot"
        pilot_out.mkdir(parents=True, exist_ok=True)
        for name in strategies:
            settings = Q3Settings(strategy=name, search_seed=args.search_seed,
                                  search_maxiter=args.search_maxiter,
                                  search_popsize=args.search_popsize)
            frame, diag, pair, summary = run_strategy(
                dates, load, pv, prices, fl, fc, settings, limit=args.days
            )
            (pilot_out / f"pilot_{name}_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return

    summaries = []
    frames = {}

    # ---- shared January warm-up (0:00-only M0 policy) ----
    args.output.mkdir(parents=True, exist_ok=True)
    warmup_settings = Q3Settings(strategy="M0", search_seed=args.search_seed,
                                 search_maxiter=args.search_maxiter,
                                 search_popsize=args.search_popsize)
    warmup_frame, _, _, warmup_summary = run_strategy(
        dates, load, pv, prices, fl, fc, warmup_settings, limit=31
    )
    common_feb1_soc = float(warmup_frame.soc_end_kwh.iloc[-1])
    warmup_frame.to_csv(args.output / "warmup_january_schedule.csv",
                        index=False, encoding="utf-8-sig")
    print(f"common warm-up done: 1 February opening SOC = {common_feb1_soc:.6f} kWh", flush=True)

    for name in strategies:
        settings = Q3Settings(strategy=name, search_seed=args.search_seed,
                              search_maxiter=args.search_maxiter,
                              search_popsize=args.search_popsize)
        frame, diag, pair, summary = run_strategy(
            dates, load, pv, prices, fl, fc, settings,
            start_idx=31, initial_energy=common_feb1_soc,
        )
        write_strategy_outputs(args.output / name, name, frame, diag, pair, summary, prices)
        summaries.append(summary)
        frames[name] = frame
        print(f"{name} total={summary['formal_period']['total_cost']:,.2f} "
              f"emergency={summary['formal_period']['emergency_cost']:,.2f}", flush=True)

    comparison = pd.DataFrame([
        {"strategy": s["settings"]["strategy"],
         **{k: v for k, v in s["formal_period"].items() if not isinstance(v, dict)}}
        for s in summaries
    ])
    comparison.to_csv(args.output / "question3_comparison.csv", index=False, encoding="utf-8-sig")
    (args.output / "question3_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    main_frame = frames[MAIN_STRATEGY]
    main_diag = pd.read_csv(args.output / MAIN_STRATEGY / "question3_daily_parameters.csv")
    main_pair = pd.read_csv(args.output / MAIN_STRATEGY / "question3_paired_controller_comparison.csv")
    write_main_outputs(args.output, main_frame, prices)
    write_report(args.output, summaries, main_frame, main_diag, main_pair)
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
