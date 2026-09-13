# -*- coding: utf-8 -*-
"""Shared machinery for question 4 v2 (docs/问题四/问题四优化实施方案.md).

Five-layer plan:
  1. strictly causal day-ahead price forecasts (weekly persistence WP or
     four-day-type similar-day Gaussian kernel GK);
  2. joint load/PV/price scenarios from the SAME historical days
     (42-day window with time-decay weights; 21/63 as sensitivity);
  3. price-weighted quantile risk curves (Q80 plan, Q70/Q90 adjustment);
  4. plan/adjustment LPs with scenario-mean decision prices and next-day
     terminal value; regularised clipped-affine LDR with rolling
     validation-day acceptance;
  5. settlement on the realized attachment-4 prices.

Conventions of the optimized plan (this module):
  * issue ledger / execution ledger: day d's 0:00 plan covers the issue row
    [00:10_d, 00:10_{d+1}) = the 144 template columns; execution interval
    g_exec[d,0] = g_issue[d-1,143] and g_exec[d,t] = g_issue[d,t-1].
  * 1 January is a REAL execution day (zero issue row, battery acts, SOC
    evolves from 6000 kWh); the formal period stays 2/1-12/31 (334 days).
  * adjustments take effect from the first full interval after the update
    instant: 6:10 / 12:10 / 18:10; no realized-price replacement and no
    scenario-price determinization in decision rows (prices are interval
    starts revealed when the interval starts).
  * terminal value nu_d = min_t p_hat[d+1,t] / eta (next-day predicted
    minimum recharge cost), fallback same-day; sensitivity nu = 0.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
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
    load_inputs,
    load_forecast_weekly_persist,
    solve_plan,
    time_label,
    validate_schedule,
)
from src.optimization.question2_ldr import rule_transition
from src.optimization.question3 import settle, solve_adjustment
from src.data_pipeline.question3_forecasts import (
    build_issuance_curves,
    load_hourly_issuances,
    load_pv_actuals,
)
from src.data_pipeline.question4_prices import load_price_actual
from src.forecasting.load_day_ahead import (
    KERNEL_H,
    KERNEL_K,
    KERNEL_TAU,
    build_day_type_frame,
    gaussian_kernel_forecasts,
)
from src.forecasting.price_analysis import PRICE_SIGMA_FLOOR

STAGE_EDGES = [0, 36, 72, 108, 144]  # natural-day four-hour stage starts
DELTA_BOUND = 9600.0
LAMBDA_BOUND = 2.0
DECAY_TAU = 14.0  # days, exponential time-decay of scenario weights
GAMMA_GRID = [0.0, 1e-5, 1e-4, 1e-3]  # regularization strengths for pre-evaluation
FULL_WINDOW = 42
FULL_SAMPLE_FLOOR = 14  # M < 14: no calibration
KEY_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
ISSUE_HOURS = [0, 6, 12, 18]


@dataclass(frozen=True)
class V2Settings:
    """Question-4 v2 configuration (all fields fixed before 2/1)."""
    name: str = "q4_v2"
    price_model: str = "wp"            # "wp" | "gk"
    quantile_kind: str = "price_weighted"  # "price_weighted" | "ordinary"
    window: int = FULL_WINDOW          # scenario window (21/42/63)
    decay_tau: float = DECAY_TAU       # exponential decay time constant (days)
    gamma: float = 0.0                 # L2 regularization strength
    terminal_value_rule: str = "next_day"  # "next_day" | "same_day" | "zero"
    ldr_kind: str = "regularized"      # "regularized" (validation-day) | "original"
    alpha: float = 0.8
    q_up: float = 0.7
    q_down: float = 0.9
    search_seed: int = 20250912
    search_maxiter: int = 8
    search_popsize: int = 5
    delta_bound_kwh: float = DELTA_BOUND
    lambda_bound: float = LAMBDA_BOUND


# --------------------------------------------------------------------------
# 1. causal price forecasts (WP / GK), shape (366, 144) incl. virtual day 365
# --------------------------------------------------------------------------

def build_price_forecast_wp(actual: np.ndarray, placeholder: np.ndarray) -> np.ndarray:
    """Weekly persistence p_hat[d] = actual[d-7]; d<7 available-history mean.

    Row 0 (1 January) is an inert placeholder (zero-plan day, never enters a
    decision); row 365 is the virtual 2026-01-01 forecast used for the 12/31
    plan horizon's last column and the 12/31 next-day terminal value.
    """
    actual = np.asarray(actual, float)
    assert actual.shape == (365, T)
    out = np.full((366, T), np.nan, float)
    out[0] = placeholder
    for d in range(1, 7):
        out[d] = actual[:d].mean(axis=0)
    for d in range(7, 366):
        out[d] = actual[d - 7]
    assert np.isfinite(out).all()
    return out


def build_price_forecast_gk(actual: np.ndarray, dates: pd.DatetimeIndex,
                            placeholder: np.ndarray) -> np.ndarray:
    """Four-day-type similar-day Gaussian-kernel forecast (366, 144)."""
    actual = np.asarray(actual, float)
    extended = np.full((366, T), np.nan, float)
    extended[:365] = actual
    day_types = build_day_type_frame(
        dates.union(pd.DatetimeIndex([pd.Timestamp("2026-01-01")]))
    )["day_type"].to_numpy()
    gk, _ = gaussian_kernel_forecasts(
        extended, day_types, k=KERNEL_K, h=KERNEL_H, tau=KERNEL_TAU,
        sigma_floor=PRICE_SIGMA_FLOOR,
    )
    out = np.asarray(gk, float)
    for d in range(366):
        if not np.isfinite(out[d]).all():
            out[d] = actual[d - 1] if 0 < d < 365 else placeholder
    out[0] = placeholder
    return out


def build_price_forecast(actual: np.ndarray, dates: pd.DatetimeIndex,
                         model: str, placeholder: np.ndarray) -> np.ndarray:
    if model == "wp":
        return build_price_forecast_wp(actual, placeholder)
    if model == "gk":
        return build_price_forecast_gk(actual, dates, placeholder)
    raise ValueError(model)


def build_load_forecast_extended(load: np.ndarray) -> np.ndarray:
    """Weekly-persistence load forecast incl. the virtual day 365 (366, 144)."""
    fl = load_forecast_weekly_persist(load)  # (365, 144), row 0 NaN
    out = np.full((366, T), np.nan, float)
    out[:365] = fl
    out[365] = load[358]  # d-7 for the virtual 2026-01-01
    return out


def build_pv_forecast_extended(pv: np.ndarray) -> np.ndarray:
    """Seven-day mean PV forecast incl. the virtual day 365 (366, 144)."""
    out = np.full((366, T), np.nan, float)
    for d in range(1, 366):
        out[d] = pv[max(0, d - 7):d].mean(axis=0)
    out[365] = pv[358:365].mean(axis=0)
    return out


# --------------------------------------------------------------------------
# 2. issue-horizon and residual matrices (issue-ledger columns j = 0..143)
# --------------------------------------------------------------------------

def plan_horizon(F: np.ndarray, d: int) -> np.ndarray:
    """Issue row of day d: F[d,1:144] + F[d+1,0] (144 columns)."""
    return np.r_[F[d, 1:144], F[d + 1, 0]]


def residual_columns(actual: np.ndarray, F: np.ndarray, i: int) -> np.ndarray:
    """Out-of-sample residual of historical day i aligned to issue columns.

    Column j<143: actual_i[j+1] - F_i[j+1]; column 143: actual_{i+1}[0]
    - F_{i+1}[0] (the 0:00+1 column).  Requires i <= 363.
    """
    out = np.empty(T, float)
    out[:143] = actual[i, 1:144] - F[i, 1:144]
    out[143] = actual[i + 1, 0] - F[i + 1, 0]
    return out


def scenario_window(d: int, window: int) -> tuple[int, int]:
    """(first_day, M) of the historical residual window for decision day d."""
    first = max(7, d - window)
    return first, max(0, d - first)


def decay_weights(d: int, first: int, tau: float = DECAY_TAU) -> np.ndarray:
    """w_i = exp(-(d-1-i)/tau), normalized to sum 1, for i in [first, d)."""
    idx = np.arange(first, d, dtype=float)
    w = np.exp(-(d - 1 - idx) / tau)
    return w / w.sum()


# --------------------------------------------------------------------------
# 3. price-weighted quantile and scenario mean price
# --------------------------------------------------------------------------

def price_weighted_quantile(net: np.ndarray, price: np.ndarray, w: np.ndarray,
                            alpha: float) -> np.ndarray:
    """Price-weighted alpha quantile Q^p per column (inf definition, §5.1).

    Q^p_{alpha,t} = inf{x : sum w*price*1(N<=x) / sum w*price >= alpha}.
    With equal prices this reduces to the ordinary weighted quantile.
    """
    net = np.asarray(net, float)
    price = np.asarray(price, float)
    w = np.asarray(w, float)
    m, n = net.shape
    if price.shape != (m, n) or w.shape != (m,):
        raise ValueError("price-weighted quantile: shape mismatch")
    out = np.zeros(n, float)
    for j in range(n):
        order = np.argsort(net[:, j], kind="stable")
        ns, ps, ws = net[order, j], price[order, j], w[order]
        total = float(ws @ ps)
        if total <= 0.0:
            out[j] = float(np.quantile(net[:, j], alpha))
            continue
        cum = np.cumsum(ws * ps)
        k = int(np.searchsorted(cum, alpha * total, side="left"))
        out[j] = ns[min(k, m - 1)]
    return out


def scenario_mean_price(price_scen: np.ndarray, w: np.ndarray) -> np.ndarray:
    """pbar_t = sum w*price_t / sum w over scenarios."""
    price_scen = np.asarray(price_scen, float)
    w = np.asarray(w, float)
    total = float(w.sum())
    if total <= 0.0:
        return price_scen.mean(axis=0)
    return (w[:, None] * price_scen).sum(axis=0) / total


def horizon_net(fl: np.ndarray, pv: np.ndarray) -> np.ndarray:
    return fl - pv


# --------------------------------------------------------------------------
# 4. regularised clipped-affine LDR calibration (validation-day acceptance)
# --------------------------------------------------------------------------

class CalibObjective:
    """Vectorized empirical objective for differential evolution.

    J(theta) = mean_w [5 * sum_t p^w_t b^w_t - nu * E^w_end] + gamma * ||theta||^2
    with the stage signals precomputed per scenario (cumulative or previous
    stage rule, matching the execution information flow).

    The parameter vector is laid out PER STAGE in the order of
    ``stage_specs``: for each stage one delta and, when has_lam, one lambda
    (theta = [delta_1, (lambda_1), delta_2, (lambda_2), ...]).
    """

    def __init__(self, net, forecast, q, ref, prices, initial, terminal_value,
                 stage_specs, observed, gamma, signal_mode):
        self.net = np.asarray(net, float)
        if self.net.ndim != 2:
            raise ValueError("net must be a 2-D scenario matrix")
        m, h = self.net.shape
        self.forecast = np.asarray(forecast, float)
        for name, arr in [("q", q), ("ref", ref), ("forecast", forecast)]:
            if np.asarray(arr).shape != (h,):
                raise ValueError(f"{name} must match the scenario horizon")
        self.q = np.asarray(q, float)
        self.ref = np.asarray(ref, float)
        prices = np.asarray(prices, float)
        if prices.ndim == 1:
            prices = np.broadcast_to(prices, (m, h))
        if prices.shape != (m, h):
            raise ValueError("price paths must match the scenario matrix")
        self.prices = prices
        self.initial = float(initial)
        self.terminal_value = float(terminal_value)
        self.gamma = float(gamma)
        self.stage_specs = [tuple(s) for s in stage_specs]
        self.n_params = sum(1 + int(has_lam) for _, _, has_lam in self.stage_specs)
        # column -> stage index map
        self.col_stage = np.full(h, -1, int)
        for j, (s0, s1, _) in enumerate(self.stage_specs):
            self.col_stage[s0:s1] = j
        assert (self.col_stage >= 0).all(), "stage specs must cover the horizon"
        self.stage_starts = np.array([s[0] for s in self.stage_specs], int)
        self.signals = self._signals(self.stage_specs, observed, signal_mode)
        self.evaluations = 0

    def _signals(self, stage_specs, observed, mode):
        m, h = self.net.shape
        ns = len(stage_specs)
        sig = np.zeros((m, ns), float)
        for j, (s0, _, _) in enumerate(stage_specs):
            obs = observed[j]
            if obs is not None and not np.isnan(obs):
                sig[:, j] = float(obs)
            elif j == 0:
                sig[:, j] = 0.0
            elif mode == "cumulative":
                sig[:, j] = (self.net[:, :s0] - self.forecast[None, :s0]).mean(axis=1)
            else:  # previous_stage
                p0 = int(stage_specs[j - 1][0])
                sig[:, j] = (self.net[:, p0:s0] - self.forecast[None, p0:s0]).mean(axis=1)
        return sig

    def __call__(self, theta_raw):
        raw = np.asarray(theta_raw, float)
        scalar = raw.ndim == 1
        params = raw.reshape(1, -1) if scalar else raw
        if params.shape[1] != self.n_params:
            raise ValueError(
                f"expected {self.n_params} parameters, got shape {raw.shape}")
        count = params.shape[0]
        m = self.net.shape[0]
        energy = np.full((count, m), self.initial, float)
        emergency_cost = np.zeros((count, m), float)
        pos = 0
        for j, s0 in enumerate(self.stage_starts):
            s1 = self.stage_starts[j + 1] if j + 1 < len(self.stage_starts) \
                else self.net.shape[1]
            _, _, has_lam = self.stage_specs[j]
            delta = params[:, pos]
            pos += 1
            lam = params[:, pos] if has_lam else 0.0
            pos += 1 if has_lam else 0
            delta = np.broadcast_to(np.atleast_1d(delta), (count,))[:, None]
            lam = np.broadcast_to(np.atleast_1d(lam), (count,))[:, None]
            sig = self.signals[:, j]
            for t in range(s0, s1):
                reserve = np.clip(
                    self.ref[t] + delta + lam * sig[None, :],
                    EMIN, EMAX,
                )
                _, _, emergency, _, energy = rule_transition(
                    self.net[None, :, t], self.q[t], energy, reserve
                )
                emergency_cost += 5.0 * self.prices[:, t] * emergency
        score = np.mean(emergency_cost - self.terminal_value * energy, axis=1)
        reg = (params**2).sum(axis=1)
        out = score + self.gamma * reg
        self.evaluations += count
        return float(out[0]) if scalar else out


def stage_param_count(stage_specs) -> int:
    return sum(1 + int(has_lam) for _, _, has_lam in stage_specs)


def zero_theta(stage_specs) -> np.ndarray:
    return np.zeros(stage_param_count(stage_specs), float)


def floor_theta_min(stage_specs) -> np.ndarray:
    """beta=0 rule encoded as theta: all deltas = -delta_bound -> reserve EMIN."""
    out = np.zeros(stage_param_count(stage_specs), float)
    pos = 0
    for _, _, has_lam in stage_specs:
        out[pos] = -DELTA_BOUND
        pos += 1
        if has_lam:
            out[pos] = 0.0
            pos += 1
    return out


def unpack_by_stages(theta: np.ndarray, stage_specs, day_stage_offset: int = 0):
    """Return (deltas (4,), lambdas (4,)) with day-stage indexing.

    ``stage_specs`` calibrates day stages [day_stage_offset, ...]; parameters
    are laid out per stage (delta, lambda?)."""
    theta = np.asarray(theta, float)
    deltas = np.zeros(4)
    lambdas = np.zeros(4)
    pos = 0
    for j, (_, _, has_lam) in enumerate(stage_specs):
        k = day_stage_offset + j
        deltas[k] = theta[pos]
        pos += 1
        if has_lam:
            lambdas[k] = theta[pos]
            pos += 1
    return deltas, lambdas


def calibrate_v2(net, forecast, q, ref, prices, initial, terminal_value,
                 stage_specs, observed, settings: V2Settings, seed: int,
                 price_paths=None, signal_mode="cumulative",
                 residual_count=None):
    """Calibrate clipped-affine parameters with the v2 acceptance protocol.

    * sample rules: M < 14 -> zero parameters (no search); 14 <= M < 42 ->
      intercepts only (lambdas forced off); M >= 42 -> full parameters.
    * regularised (settings.ldr_kind == "regularized"): train on M-1
      scenarios (most recent historical day held out), accept only if the
      candidate beats min(beta1, beta0) on BOTH the training set and the
      held-out validation day; otherwise fall back to the better of beta1
      and beta0 judged on the validation day.
    * original (settings.ldr_kind == "original"): no regularization, no
      validation day; accept iff the candidate beats zero on the training set.
    """
    net = np.asarray(net, float)
    m, h = net.shape
    if residual_count is None:
        residual_count = m
    # per-stage parameter layout; lambdas forced off below 42 history days
    # (the theta vector keeps its full per-stage length, inactive lambdas
    # are pinned to zero by (0, 0) bounds so every reader is uniform)
    specs = [tuple(s) for s in stage_specs]
    lambdas_on = residual_count >= 42
    n_params = stage_param_count(specs)
    n_free = 0
    bounds = []
    for _, _, has_lam in specs:
        bounds.append((-settings.delta_bound_kwh, settings.delta_bound_kwh))
        n_free += 1
        if has_lam:
            if lambdas_on:
                bounds.append((-settings.lambda_bound, settings.lambda_bound))
                n_free += 1
            else:
                bounds.append((0.0, 0.0))

    if price_paths is None:
        price_paths = np.broadcast_to(np.asarray(prices, float), (m, h))
    price_paths = np.asarray(price_paths, float)
    if price_paths.shape != (m, h):
        raise ValueError("price paths must match the net scenario matrix")

    def build_objective(net_m, price_m):
        return CalibObjective(
            net_m, forecast, q, ref, price_m, initial, terminal_value,
            specs, observed, settings.gamma, signal_mode,
        )

    if n_free == 0 or residual_count < FULL_SAMPLE_FLOOR:
        zero = zero_theta(specs)
        obj = build_objective(net, price_paths)
        zs = float(obj(zero))
        return {
            "theta": zero, "n_active": 0,
            "beta1_train": zs, "beta0_train": zs,
            "beta1_val": zs, "beta0_val": zs,
            "selected_score": zs, "evaluations": 0, "seconds": 0.0,
            "method": "zero_parameter_warmup", "accepted_search": False,
            "solver_message": "insufficient history for calibration",
        }

    search_started = time.perf_counter()
    train_net, val_net = net[:-1], net[-1:]
    train_price, val_price = price_paths[:-1], price_paths[-1:]
    train_obj = build_objective(train_net, train_price)
    zero = zero_theta(specs)
    floor = floor_theta_min(specs)
    zero_train = float(train_obj(zero))
    floor_train = float(train_obj(floor))
    val_obj = build_objective(val_net, val_price)
    zero_val = float(val_obj(zero))
    floor_val = float(val_obj(floor))

    def batch_map(func, params_iter):
        """Map-like worker: evaluate a whole DE generation in ONE call.

        scipy passes one scaled parameter vector at a time through the
        map-wrapper; stacking them lets the vectorised objective run its
        column loop once per generation instead of once per candidate
        (identical arithmetic, ~30x fewer Python-loop passes)."""
        stacked = np.asarray(list(params_iter), dtype=float)
        return np.atleast_1d(func(stacked))

    result = differential_evolution(
        train_obj,
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
        workers=batch_map,  # batch map: no multiprocessing, one call per generation
        vectorized=False,
        x0=zero,
    )
    candidate = np.asarray(result.x, float)
    candidate_train = float(result.fun)
    candidate_val = float(val_obj(candidate))

    if settings.ldr_kind == "regularized":
        floor_t = min(zero_train, floor_train)
        floor_v = min(zero_val, floor_val)
        accepted = candidate_train <= floor_t + 1e-8 and candidate_val <= floor_v + 1e-8
        if accepted:
            theta, selected = candidate, candidate_train
        elif zero_val <= floor_val:
            theta, selected = zero, zero_train
        else:
            theta, selected = floor, floor_train
    else:  # original: in-sample improvement vs zero only
        accepted = candidate_train <= zero_train + 1e-8
        theta = candidate if accepted else zero
        selected = candidate_train if accepted else zero_train

    return {
        "theta": theta, "n_active": n_free,
        "beta1_train": zero_train, "beta0_train": floor_train,
        "beta1_val": zero_val, "beta0_val": floor_val,
        "selected_score": selected,
        "evaluations": int(train_obj.evaluations + val_obj.evaluations),
        "seconds": time.perf_counter() - search_started,
        "method": "differential_evolution_v2" if settings.ldr_kind == "regularized"
        else "differential_evolution_original",
        "accepted_search": bool(accepted),
        "solver_message": str(result.message),
    }


# --------------------------------------------------------------------------
# 5. natural-day execution (issue-ledger inputs -> execution-ledger actions)
# --------------------------------------------------------------------------

def execute_day(net_act, q_exec, ref_exec, forecast_exec, prices, initial,
                deltas, lambdas):
    """Execute one natural day (144 intervals) with the clipped-affine rule.

    Stage k signal: 0 for stage 1, cumulative mean error over executed
    intervals [0, 36k) for later stages.  ``q_exec``/``ref_exec`` are the
    execution-ledger purchase plan and reference SOC per interval.
    """
    out = {
        "reserve_kwh": np.zeros(T), "stage_signal_kwh": np.zeros(T),
        "soc_start_kwh": np.zeros(T), "charge_kwh": np.zeros(T),
        "discharge_kwh": np.zeros(T), "emergency_kwh": np.zeros(T),
        "unused_kwh": np.zeros(T), "soc_end_kwh": np.zeros(T),
    }
    energy = float(initial)
    error_sum = 0.0
    signal = 0.0
    for t in range(T):
        stage = min(t // 36, 3)
        if t % 36 == 0:
            signal = 0.0 if t == 0 else error_sum / t
        reserve = float(np.clip(
            ref_exec[t] + deltas[stage] + lambdas[stage] * signal, EMIN, EMAX))
        start = energy
        charge, discharge, emergency, unused, end = map(
            float, rule_transition(float(net_act[t]), float(q_exec[t]), start, reserve)
        )
        out["reserve_kwh"][t] = reserve
        out["stage_signal_kwh"][t] = signal
        out["soc_start_kwh"][t] = start
        out["charge_kwh"][t] = charge
        out["discharge_kwh"][t] = discharge
        out["emergency_kwh"][t] = emergency
        out["unused_kwh"][t] = unused
        out["soc_end_kwh"][t] = end
        energy = end
        error_sum += net_act[t] - forecast_exec[t]
    return out


def theta_to_stages(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(7,) -> (deltas (4,), lambdas (4,)) with lambda stage 1 = 0."""
    theta = np.asarray(theta, float)
    return theta[:4], np.r_[0.0, theta[4:]]


# --------------------------------------------------------------------------
# 6. ledger bridge, validators, workbook payloads
# --------------------------------------------------------------------------

def carry_bridge(issue_cost: float, carry_in_cost: float, carry_out_cost: float) -> float:
    """C_exec = C_issue + carry-in - carry-out (§2.4)."""
    return issue_cost + carry_in_cost - carry_out_cost


def formal_execution_cost(frame: pd.DataFrame, start="2025-02-01", end="2025-12-31") -> float:
    f = frame[(frame.date >= start) & (frame.date <= end)]
    return float(f.total_cost.sum())


def emergency_rows(frame: pd.DataFrame) -> list[list]:
    """Merged emergency intervals per natural day (execution ledger)."""
    rows = []
    for date, day in frame.groupby("date"):
        q = day.emergency_kwh.to_numpy()
        t = 0
        while t < T:
            if q[t] <= 1e-6:
                t += 1
                continue
            begin = t
            while t < T and q[t] > 1e-6:
                t += 1
            rows.append([str(date.date()), f"{time_label(begin)}-{time_label(t)}",
                         float(q[begin:t].sum())])
        if not (q > 1e-6).any():
            rows.append([str(date.date()), "无", 0.0])
    return rows


def battery_blocks(frame: pd.DataFrame) -> list[list]:
    """Six four-hour blocks per natural day with 0:00/24:00 SOC."""
    rows = []
    for date, day in frame.groupby("date"):
        for block in range(6):
            f = day.iloc[block * 24:(block + 1) * 24]
            rows.append([
                str(date.date()) if block == 0 else None,
                f"{block * 4}:00-{(block + 1) * 4}:00",
                float(f.charge_kwh.sum()), float(f.discharge_kwh.sum()),
                "0:00" if block == 0 else ("24:00" if block == 1 else None),
                float(day.soc_start_kwh.iloc[0]) if block == 0
                else (float(day.soc_end_kwh.iloc[-1]) if block == 1 else None),
            ])
    return rows


# --------------------------------------------------------------------------
# 7. perfect-foresight lower bound for Q4-2 (execution ledger, actual prices)
# --------------------------------------------------------------------------

def perfect_foresight_bound(load, pv, prices, initial_soc, start_idx=31):
    """Min cost over 2/1-12/31 with full knowledge (variables g,C,D,U,E)."""
    net = (load[start_idx:] - pv[start_idx:]).ravel()
    p = prices[start_idx:].ravel()
    plan, obj = solve_plan(net, p, float(initial_soc), 0.0)
    return {
        "initial_soc_kwh": float(initial_soc),
        "grid_kwh": float(plan[0].sum()),
        "charge_kwh": float(plan[1].sum()),
        "discharge_kwh": float(plan[2].sum()),
        "unused_kwh": float(plan[3].sum()),
        "final_soc_kwh": float(plan[4][-1]),
        "total_cost": float(obj),
    }
