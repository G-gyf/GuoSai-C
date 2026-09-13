# -*- coding: utf-8 -*-
"""Question 4-3 v2: fluctuating prices, rolling updates under
docs/闂鍥?闂鍥涗紭鍖栧疄鏂芥柟妗?md.

* issue/execution ledgers as in question4_2_v2; adjustments take effect from
  the first full interval after the update instant (6:10 / 12:10 / 18:10),
  so issue slots 0..35 / 36..71 / 72..107 / 108..143 are locked by the
  0:00 / 6:00 / 12:00 / 18:00 decisions respectively;
* price-weighted Q80 plan curve (0:00 issuance PV), price-weighted
  Q70/Q90 two-sided adjustment curves (median with the q0 kink), scenario
  mean decision prices, next-day terminal value;
* 42-day decaying-window regularised staged LDR calibration with rolling
  validation-day acceptance; sample rules: <14 days none, 14-41 deltas
  only, >=42 full parameters; M0 calibrates the whole day at 0:00 (as the
  original protocol), M6/M612/M61218 calibrate per stage;
* strategies M0 / M6 / M612 / M61218 fork from the common 1 February SOC;
  1 January is a zero-plan REAL execution day.

Run:  python -m src.optimization.question4_3_v2
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.optimization.question2 import ROOT, T, ETA, EMIN, S, load_inputs, solve_plan
from src.optimization.question3 import settle, solve_adjustment, execute_segment
from src.optimization.q4_v2_common import (
    V2Settings,
    build_load_forecast_extended,
    calibrate_v2,
    decay_weights,
    plan_horizon,
    price_weighted_quantile,
    residual_columns,
    scenario_mean_price,
    scenario_window,
    unpack_by_stages,
)
from src.optimization.question4_2_v2 import (
    ISSUE_COLS,
    build_forecast_bundle as build_q4_2_bundle,
    load_question4_v2_inputs,
    terminal_value_nu,
)
from src.data_pipeline.question3_forecasts import (
    build_issuance_curves,
    load_hourly_issuances,
)

ISSUE_HOURS = [0, 6, 12, 18]
Q_COLS = ["date", *[f"g{j}" for j in range(T)]]
E_COLS = ["date", *[f"e{j}" for j in range(T)]]
STRATEGIES = {
    "M0": {"updates": [0]},
    "M6": {"updates": [0, 6]},
    "M612": {"updates": [0, 6, 12]},
    "M61218": {"updates": [0, 6, 12, 18]},
}
MAIN_STRATEGY = "M612"

FRAME_COLUMNS = [
    "date", "slot", "interval_start", "issue_date", "issue_slot",
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


def validate_question3_v2(f):
    """validate_question3 with the cross-day carry slot exempt from the
    pre-six check (interval 0 executes the previous day's issue column 143,
    which may legitimately have been adjusted at 18:00)."""
    balance = f.qA_kwh + f.emergency_kwh + f.pv_kwh + f.discharge_kwh \
        - f.load_kwh - f.charge_kwh - f.unused_kwh
    state = f.soc_start_kwh + ETA * f.charge_kwh - f.discharge_kwh / ETA - f.soc_end_kwh
    retained = f.price * np.minimum(f.q0_kwh, f.qA_kwh)
    down = 0.5 * f.price * np.maximum(f.q0_kwh - f.qA_kwh, 0)
    up = 1.5 * f.price * np.maximum(f.qA_kwh - f.q0_kwh, 0)
    total = retained + down + up + f.emergency_cost
    pre_six = (f.slot >= 1) & (f.slot < 36)
    checks = {
        "max_balance_error_kwh": float(np.abs(balance).max()),
        "max_state_error_kwh": float(np.abs(state).max()),
        "max_continuity_error_kwh": float(
            np.abs(f.soc_start_kwh.to_numpy()[1:] - f.soc_end_kwh.to_numpy()[:-1]).max()),
        "min_soc_kwh": float(min(f.soc_start_kwh.min(), f.soc_end_kwh.min())),
        "max_soc_kwh": float(max(f.soc_start_kwh.max(), f.soc_end_kwh.max())),
        "max_charge_kwh": float(f.charge_kwh.max()),
        "max_discharge_kwh": float(f.discharge_kwh.max()),
        "simultaneous_charge_discharge": int(
            ((f.charge_kwh > 1e-6) & (f.discharge_kwh > 1e-6)).sum()),
        "emergency_while_charging": int(
            ((f.charge_kwh > 1e-6) & (f.emergency_kwh > 1e-6)).sum()),
        "max_settlement_recompute_error": float(np.abs(total - f.total_cost).max()),
        "negative_q0": int((f.q0_kwh < -1e-7).sum()),
        "negative_qA": int((f.qA_kwh < -1e-7).sum()),
        "pre_six_executed_adjusted": int(
            pre_six.sum() - (pre_six & (np.abs(f.qA_kwh - f.q0_kwh) < 1e-9)).sum()),
    }
    assert max(checks[k] for k in ["max_balance_error_kwh", "max_state_error_kwh",
                                   "max_continuity_error_kwh"]) < 1e-5
    assert checks["min_soc_kwh"] >= EMIN - 1e-5 and checks["max_soc_kwh"] <= EMAX_LIMIT + 1e-5
    assert max(checks["max_charge_kwh"], checks["max_discharge_kwh"]) <= S + 1e-5
    assert checks["simultaneous_charge_discharge"] == checks["emergency_while_charging"] == 0
    assert checks["max_settlement_recompute_error"] < 1e-6
    assert checks["negative_q0"] == checks["negative_qA"] == 0
    checks["passed"] = True
    return checks


EMAX_LIMIT = 10800.0


def build_q4_3_bundle(load, pv, p_act, dates, fixed, root: Path = ROOT):
    base = build_q4_2_bundle(load, pv, p_act, dates, fixed)
    fc = build_issuance_curves(root)          # (365, 4, 144) kWh
    hourly = load_hourly_issuances(root)      # (365, 4, 24) kW
    anchor24 = np.empty((365, 4), float)
    for k, h in enumerate(ISSUE_HOURS):
        anchor24[:, k] = hourly[:, k, 24 - h - 1]
    fl = base["fl"]
    e_l = np.empty((364, T), float)
    e_v = {k: np.empty((364, T), float) for k in range(4)}
    for i in range(364):
        e_l[i] = residual_columns(load, fl, i)
        for k in range(4):
            row = np.empty(T, float)
            row[:143] = pv[i, 1:144] - fc[i, k, 1:144]
            row[143] = pv[i + 1, 0] - anchor24[i, k] / 6.0
            e_v[k][i] = row
    base["fc"] = fc
    base["anchor24"] = anchor24
    base["res_load"] = e_l
    base["res_pv"] = e_v
    return base


def pv_issue_horizon(bundle, d: int, k: int) -> np.ndarray:
    """Issue-horizon PV forecast of issuance k for day d (144-j columns)."""
    fc, anchor24 = bundle["fc"], bundle["anchor24"]
    if k == 0:
        return np.r_[fc[d, 0, 1:144], anchor24[d, 0] / 6.0]
    h = 6 * ISSUE_HOURS[k]
    return np.r_[fc[d, k, h + 1:144], anchor24[d, k] / 6.0]


def adjustment_curve_v2(net_h, scen_net, scen_price, w, q0, settings: V2Settings):
    if scen_net is None:
        return np.maximum(net_h, q0)
    if settings.quantile_kind == "price_weighted":
        q_hi = price_weighted_quantile(scen_net, scen_price, w, settings.q_down)
        q_lo = price_weighted_quantile(scen_net, scen_price, w, settings.q_up)
    else:
        q_hi = np.quantile(scen_net, settings.q_down, axis=0)
        q_lo = np.quantile(scen_net, settings.q_up, axis=0)
    target = np.median(np.stack([q_lo, q_hi, np.asarray(q0, float)]), axis=0)
    return np.maximum(net_h, target)


def _calibrate(scen, forecast, q, ref, prices, initial, nu, stage_specs, observed,
               settings, seed, signal_mode, M):
    if scen is None or M == 0:
        return {"theta": np.zeros(7), "accepted_search": False,
                "evaluations": 0, "seconds": 0.0, "method": "zero_parameter_warmup",
                "solver_message": "no scenarios"}
    return calibrate_v2(
        scen, forecast, q, ref, prices, initial, nu, stage_specs, observed,
        settings, seed, signal_mode=signal_mode, residual_count=M)


def _record(d, date, t, issue_date, issue_slot, load_d, pv_d, p_settle,
            fl_d, fc_t, fc_net_t, risk_t, residual_count, q0_t, qA_t, ref_t,
            seg, i, delta, lam, signal, calibration_used):
    stage = min(t // 36, 3)
    retained, down, up = settle(q0_t, qA_t, p_settle[t])
    emergency_cost = 5.0 * p_settle[t] * seg["emergency_kwh"][i]
    return (
        date, t, date + pd.Timedelta(minutes=10 * t), issue_date, issue_slot,
        load_d[t], pv_d[t], p_settle[t],
        fl_d[t], fc_t, fc_net_t, risk_t,
        residual_count,
        q0_t, qA_t, max(q0_t - qA_t, 0.0), max(qA_t - q0_t, 0.0),
        ref_t, seg["reserve_kwh"][i], signal,
        delta, lam,
        seg["soc_start_kwh"][i], seg["charge_kwh"][i],
        seg["discharge_kwh"][i], seg["emergency_kwh"][i],
        seg["unused_kwh"][i], seg["soc_end_kwh"][i],
        retained, down, up, emergency_cost,
        retained + down + up + emergency_cost,
        stage + 1, calibration_used,
    )


def run_q4_3_range(dates, load, pv, p_act, bundle, settings: V2Settings,
                   start_idx: int, initial_energy: float, limit: int,
                   price_model: str, prev_q0=None, prev_qA=None, prev_ref=None,
                   prev_risk143=0.0):
    """Continuous Q4-3 strategy run from start_idx at initial_energy."""
    fl = bundle["fl"]
    F_p = bundle["price"][price_model]
    res_l = bundle["res_load"]
    res_p = bundle["res_pv"]
    res_price = bundle["res_price"][price_model]
    fc = bundle["fc"]
    anchor24 = bundle["anchor24"]
    strategy = settings.name
    updates = STRATEGIES[strategy]["updates"]
    if prev_q0 is None:
        prev_q0 = np.zeros(T)
    if prev_qA is None:
        prev_qA = np.zeros(T)
    if prev_ref is None:
        prev_ref = np.full(T, EMIN)
    prev_q0 = np.asarray(prev_q0, float)
    prev_qA = np.asarray(prev_qA, float)
    prev_ref = np.asarray(prev_ref, float)
    energy = float(initial_energy)
    records: list[tuple] = []
    issue_q0: list[list] = []
    issue_qA: list[list] = []
    issue_ref: list[list] = []
    diagnostics: list[dict] = []

    for d in range(start_idx, limit):
        date = dates[d]
        day_started = time.perf_counter()
        p_settle = p_act[d]
        actual_net = load[d] - pv[d]
        day_cost = 0.0
        if d == 0:
            q0 = np.zeros(T)
            qA = np.zeros(T)
            ref = np.full(T, EMIN)
            fc_exec = np.zeros(T)
            forecast_exec = np.zeros(T)
            risk_exec = np.zeros(T)
            deltas = np.zeros(4)
            lambdas = np.zeros(4)
            residual_count = 0
            calibration_used = False
            diag = {"calibrations": [], "residual_count": 0}
        else:
            fl_h = plan_horizon(fl, d)
            p_h = plan_horizon(F_p, d)
            first, M = scenario_window(d, settings.window)
            w = decay_weights(d, first, settings.decay_tau) if M > 0 else None
            rows_i = np.arange(first, d) if M > 0 else np.array([], int)
            nu = terminal_value_nu(F_p, d, settings.terminal_value_rule)
            diag = {"calibrations": [], "residual_count": M}
            deltas = np.zeros(4)
            lambdas = np.zeros(4)

            # ---- 0:00 plan (issue j = 0..143, price-weighted Q80) ----
            pv_h0 = pv_issue_horizon(bundle, d, 0)
            net_h0 = fl_h - pv_h0
            scen0 = price0 = None
            if M > 0:
                scen0 = net_h0[None, :] + (res_l[rows_i] - res_p[0][rows_i])
                price0 = np.maximum(0.0, p_h[None, :] + res_price[rows_i])
            if settings.quantile_kind == "price_weighted" and M > 0:
                q = price_weighted_quantile(scen0, price0, w, settings.alpha)
            elif M > 0:
                q = np.quantile(scen0, settings.alpha, axis=0)
            else:
                q = net_h0.copy()
            risk0 = np.maximum(net_h0, q)
            pbar0 = scenario_mean_price(price0, w) if M > 0 else np.maximum(p_h, 0.0)
            plan, _ = solve_plan(risk0, pbar0, energy, nu)
            q0 = plan[0]
            E_p0 = plan[4]
            qA = q0.copy()
            ref = E_p0.copy()

            f0 = fl[d, 0] - (anchor24[d - 1, 0] / 6.0 if d >= 1 else 0.0)
            if strategy == "M0":
                # full-day calibration at 0:00 (original M0 protocol)
                grid_full = np.r_[prev_qA[143], qA]
                ref_full = np.r_[prev_ref[143], ref]
                forecast_full = np.r_[f0, net_h0]
                scen_full = price_full = None
                if M > 0:
                    scen_full = np.c_[
                        f0 + (res_l[rows_i][:, 143] - res_p[0][rows_i][:, 143]),
                        scen0]
                    price_full = np.maximum(0.0, np.c_[
                        F_p[d, 0] + res_price[rows_i][:, 143], price0])
                else:
                    price_full = np.tile(np.r_[F_p[d, 0], p_h], (1, 1))
                cal0 = _calibrate(
                    scen_full, forecast_full, grid_full, ref_full, price_full,
                    energy, nu,
                    [(0, 36, False), (36, 72, True), (72, 108, True), (108, 145, True)],
                    [None] * 4, settings, settings.search_seed + 10 * d,
                    "cumulative", M)
                deltas, lambdas = unpack_by_stages(cal0["theta"], [(0, 36, False), (36, 72, True), (72, 108, True), (108, 145, True)])
            else:
                # stage-1 calibration over t = 0..35
                q1 = np.r_[prev_qA[143], qA[0:35]]
                ref1 = np.r_[prev_ref[143], ref[0:35]]
                forecast1 = np.r_[f0, net_h0[0:35]]
                scen1 = price1 = None
                if M > 0:
                    scen1 = np.c_[f0 + (res_l[rows_i][:, 143] - res_p[0][rows_i][:, 143]),
                                  scen0[:, 0:35]]
                    price1 = np.maximum(0.0, np.c_[
                        F_p[d, 0] + res_price[rows_i][:, 143], price0[:, 0:35]])
                else:
                    price1 = np.tile(np.r_[F_p[d, 0], p_h[0:35]], (1, 1))
                cal0 = _calibrate(
                    scen1, forecast1, q1, ref1, price1, energy, nu,
                    [(0, 36, False)], [None], settings,
                    settings.search_seed + 10 * d, "cumulative", M)
                deltas[0], lambdas[0] = cal0["theta"][0], 0.0
            diag["calibrations"].append(cal0)
            fc_exec = np.zeros(T)
            fc_exec[0] = anchor24[d - 1, 0] / 6.0 if d >= 1 else 0.0
            fc_exec[1:36] = fc[d, 0, 1:36]
            forecast_exec = fl[d] - fc_exec
            risk_exec = np.zeros(T)
            risk_exec[0] = prev_risk143
            risk_exec[1:36] = risk0[0:35]
            residual_count = M
            calibration_used = bool(cal0["accepted_search"])

        # ---- execute stage 1 ----
        q0_exec0 = np.r_[prev_q0[143], q0[0:35]] if d > 0 else np.zeros(36)
        qA_exec0 = np.r_[prev_qA[143], qA[0:35]] if d > 0 else np.zeros(36)
        ref_exec0 = np.r_[prev_ref[143], ref[0:35]] if d > 0 else np.full(36, EMIN)
        seg0 = execute_segment(actual_net[:36], qA_exec0, ref_exec0, p_settle[:36],
                               energy, 0.0, np.array([deltas[0], lambdas[0]]))
        initial_soc = float(seg0["soc_start_kwh"][0])
        energy = float(seg0["soc_end_kwh"][-1])
        rows = []
        for t in range(36):
            r = _record(
                d, date, t, dates[d - 1] if t == 0 else date,
                143 if t == 0 else t - 1,
                load[d], pv[d], p_settle, fl[d] if d > 0 else np.zeros(T),
                fc_exec[t], forecast_exec[t], risk_exec[t], residual_count,
                q0_exec0[t], qA_exec0[t], ref_exec0[t], seg0, t,
                deltas[0], lambdas[0], 0.0, calibration_used)
            rows.append(r)
            day_cost += r[32]
        if d == 0:
            risk143 = 0.0
        else:
            risk143 = risk0[143]

        if d > 0 and 6 in updates:
            # ---- 6:00 adjustment (issue j = 36..143) ----
            a2 = float((actual_net[:36] - forecast_exec[:36]).mean())
            pv_h1 = pv_issue_horizon(bundle, d, 1)
            net_h1 = fl_h[36:144] - pv_h1
            scen1a = price1a = None
            if M > 0:
                scen1a = net_h1[None, :] + (res_l[rows_i][:, 36:144]
                                            - res_p[1][rows_i][:, 36:144])
                price1a = np.maximum(0.0, p_h[36:144][None, :] + res_price[rows_i][:, 36:144])
            risk1 = adjustment_curve_v2(net_h1, scen1a, price1a, w, q0[36:144], settings)
            pbar1 = scenario_mean_price(price1a, w) if M > 0 else np.maximum(p_h[36:144], 0.0)
            a6, ep6, _ = solve_adjustment(q0[36:144], risk1, pbar1, energy, nu)
            qA[36:144] = a6
            ref[36:144] = ep6
            if strategy == "M6":
                q234 = np.r_[q0[35], a6[0:107]]
                ref234 = np.r_[E_p0[35], ep6[0:107]]
                forecast234 = np.r_[net_h0[35], net_h1[0:107]]
                scen234 = price234 = None
                if M > 0:
                    scen234 = np.c_[
                        net_h0[35] + (res_l[rows_i][:, 35] - res_p[0][rows_i][:, 35]),
                        scen1a[:, 0:107]]
                    price234 = np.maximum(0.0, np.c_[
                        p_h[35] + res_price[rows_i][:, 35], price1a[:, 0:107]])
                else:
                    price234 = np.tile(np.r_[p_h[35], p_h[36:143]], (1, 1))
                cal6 = _calibrate(
                    scen234, forecast234, q234, ref234, price234, energy, nu,
                    [(0, 36, True), (36, 72, True), (72, 109, True)],
                    [a2, None, None], settings, settings.search_seed + 10 * d + 1,
                    "previous_stage", M)
                pos = 0
                for j, (_, _, hl) in enumerate([(0, 36, True), (36, 72, True),
                                                (72, 109, True)]):
                    deltas[j + 1] = float(cal6["theta"][pos])
                    lambdas[j + 1] = float(cal6["theta"][pos + 1]) if hl else 0.0
                    pos += 1 + int(hl)
            else:
                q2 = np.r_[q0[35], a6[0:35]]
                ref2 = np.r_[E_p0[35], ep6[0:35]]
                forecast2 = np.r_[net_h0[35], net_h1[0:35]]
                scen2 = price2 = None
                if M > 0:
                    scen2 = np.c_[
                        net_h0[35] + (res_l[rows_i][:, 35] - res_p[0][rows_i][:, 35]),
                        scen1a[:, 0:35]]
                    price2 = np.maximum(0.0, np.c_[
                        p_h[35] + res_price[rows_i][:, 35], price1a[:, 0:35]])
                else:
                    price2 = np.tile(np.r_[p_h[35], p_h[36:71]], (1, 1))
                cal6 = _calibrate(
                    scen2, forecast2, q2, ref2, price2, energy, nu,
                    [(0, 36, True)], [a2], settings,
                    settings.search_seed + 10 * d + 1, "previous_stage", M)
                deltas[1] = float(cal6["theta"][0])
                lambdas[1] = float(cal6["theta"][1])
            diag["calibrations"].append(cal6)
            fc_exec[36:72] = np.r_[fc[d, 0, 36], fc[d, 1, 37:72]]
            forecast_exec[36:72] = fl[d, 36:72] - fc_exec[36:72]
            risk_exec[36:72] = np.r_[risk0[35], risk1[0:35]]
        elif d > 0 and strategy == "M0":
            fc_exec[36:72] = fc[d, 0, 36:72]
            forecast_exec[36:72] = fl[d, 36:72] - fc_exec[36:72]
            risk_exec[36:72] = risk0[35:71]

        # ---- execute stage 2 ----
        a2 = float((actual_net[:36] - forecast_exec[:36]).mean()) if d > 0 else 0.0
        seg1 = execute_segment(actual_net[36:72],
                               qA[35:71] if d > 0 else np.zeros(36),
                               ref[35:71] if d > 0 else np.full(36, EMIN),
                               p_settle[36:72], energy, a2,
                               np.array([deltas[1], lambdas[1]]))
        energy = float(seg1["soc_end_kwh"][-1])
        for t in range(36, 72):
            r = _record(
                d, date, t, date, t - 1, load[d], pv[d], p_settle,
                fl[d] if d > 0 else np.zeros(T),
                fc_exec[t], forecast_exec[t], risk_exec[t], residual_count,
                q0[t - 1] if d > 0 else 0.0, qA[t - 1] if d > 0 else 0.0,
                ref[t - 1] if d > 0 else EMIN, seg1, t - 36,
                deltas[1], lambdas[1], a2, calibration_used)
            rows.append(r)
            day_cost += r[32]

        if d > 0 and 12 in updates:
            # ---- 12:00 adjustment (issue j = 72..143) ----
            a3 = float((actual_net[:72] - forecast_exec[:72]).mean())
            pv_h2 = pv_issue_horizon(bundle, d, 2)
            net_h2 = fl_h[72:144] - pv_h2
            scen2a = price2a = None
            if M > 0:
                scen2a = net_h2[None, :] + (res_l[rows_i][:, 72:144]
                                            - res_p[2][rows_i][:, 72:144])
                price2a = np.maximum(0.0, p_h[72:144][None, :] + res_price[rows_i][:, 72:144])
            risk2 = adjustment_curve_v2(net_h2, scen2a, price2a, w, q0[72:144], settings)
            pbar2 = scenario_mean_price(price2a, w) if M > 0 else np.maximum(p_h[72:144], 0.0)
            a12, ep12, _ = solve_adjustment(q0[72:144], risk2, pbar2, energy, nu)
            qA[72:144] = a12
            ref[72:144] = ep12
            q34 = np.r_[a6[35], a12[0:71]]
            ref34 = np.r_[ep6[35], ep12[0:71]]
            forecast34 = np.r_[net_h1[35], net_h2[0:71]]
            scen34 = price34 = None
            if M > 0:
                scen34 = np.c_[
                    net_h1[35] + (res_l[rows_i][:, 71] - res_p[1][rows_i][:, 71]),
                    scen2a[:, 0:71]]
                price34 = np.maximum(0.0, np.c_[
                    p_h[71] + res_price[rows_i][:, 71], price2a[:, 0:71]])
            else:
                price34 = np.tile(np.r_[p_h[71], p_h[72:143]], (1, 1))
            cal12 = _calibrate(
                scen34, forecast34, q34, ref34, price34, energy, nu,
                [(0, 36, True), (36, 73, True)], [a3, None], settings,
                settings.search_seed + 10 * d + 2, "previous_stage", M)
            deltas[2] = float(cal12["theta"][0])
            lambdas[2] = float(cal12["theta"][1])
            deltas[3] = float(cal12["theta"][2])
            lambdas[3] = float(cal12["theta"][3])
            diag["calibrations"].append(cal12)
            fc_exec[72:108] = np.r_[fc[d, 1, 72], fc[d, 2, 73:108]]
            forecast_exec[72:108] = fl[d, 72:108] - fc_exec[72:108]
            risk_exec[72:108] = np.r_[risk1[35], risk2[0:35]]
        elif d > 0:
            fc_exec[72:108] = fc[d, 0 if strategy == "M0" else 1, 72:108]
            forecast_exec[72:108] = fl[d, 72:108] - fc_exec[72:108]
            risk_exec[72:108] = risk0[71:107] if strategy == "M0" else risk1[35:71]

        # ---- execute stage 3 ----
        a3 = float((actual_net[:72] - forecast_exec[:72]).mean()) if d > 0 else 0.0
        seg2 = execute_segment(actual_net[72:108],
                               qA[71:107] if d > 0 else np.zeros(36),
                               ref[71:107] if d > 0 else np.full(36, EMIN),
                               p_settle[72:108], energy, a3,
                               np.array([deltas[2], lambdas[2]]))
        energy = float(seg2["soc_end_kwh"][-1])
        for t in range(72, 108):
            r = _record(
                d, date, t, date, t - 1, load[d], pv[d], p_settle,
                fl[d] if d > 0 else np.zeros(T),
                fc_exec[t], forecast_exec[t], risk_exec[t], residual_count,
                q0[t - 1] if d > 0 else 0.0, qA[t - 1] if d > 0 else 0.0,
                ref[t - 1] if d > 0 else EMIN, seg2, t - 72,
                deltas[2], lambdas[2], a3, calibration_used)
            rows.append(r)
            day_cost += r[32]

        if d > 0 and 18 in updates:
            # ---- 18:00 adjustment (issue j = 108..143), no recalibration ----
            pv_h3 = pv_issue_horizon(bundle, d, 3)
            net_h3 = fl_h[108:144] - pv_h3
            scen3 = price3 = None
            if M > 0:
                scen3 = net_h3[None, :] + (res_l[rows_i][:, 108:144]
                                           - res_p[3][rows_i][:, 108:144])
                price3 = np.maximum(0.0, p_h[108:144][None, :] + res_price[rows_i][:, 108:144])
            risk3 = adjustment_curve_v2(net_h3, scen3, price3, w, q0[108:144], settings)
            pbar3 = scenario_mean_price(price3, w) if M > 0 else np.maximum(p_h[108:144], 0.0)
            a18, ep18, _ = solve_adjustment(q0[108:144], risk3, pbar3, energy, nu)
            qA[108:144] = a18
            ref[108:144] = ep18
            fc_exec[108:144] = np.r_[fc[d, 2, 108], fc[d, 3, 109:144]]
            forecast_exec[108:144] = fl[d, 108:144] - fc_exec[108:144]
            risk_exec[108:144] = np.r_[risk2[35], risk3[0:35]]
            diag["calibrations"].append({
                "update_slot": 108, "theta": [], "method":
                "purchase_reoptimisation_only_no_recalibration",
                "accepted_search": False, "solver_message":
                "18:00 no recalibration (parameters locked at 12:00)"})
        elif d > 0:
            k_last = 0 if strategy == "M0" else (2 if 12 in updates else 1)
            fc_exec[108:144] = fc[d, k_last, 108:144]
            forecast_exec[108:144] = fl[d, 108:144] - fc_exec[108:144]
            if strategy == "M0":
                risk_exec[108:144] = risk0[107:143]
            elif 12 in updates:
                risk_exec[108:144] = risk2[35:71]
            else:
                risk_exec[108:144] = risk1[71:107]

        # ---- execute stage 4 ----
        a4 = float((actual_net[:108] - forecast_exec[:108]).mean()) if d > 0 else 0.0
        seg3 = execute_segment(actual_net[108:144],
                               qA[107:143] if d > 0 else np.zeros(36),
                               ref[107:143] if d > 0 else np.full(36, EMIN),
                               p_settle[108:144], energy, a4,
                               np.array([deltas[3], lambdas[3]]))
        energy = float(seg3["soc_end_kwh"][-1])
        for t in range(108, 144):
            r = _record(
                d, date, t, date, t - 1, load[d], pv[d], p_settle,
                fl[d] if d > 0 else np.zeros(T),
                fc_exec[t], forecast_exec[t], risk_exec[t], residual_count,
                q0[t - 1] if d > 0 else 0.0, qA[t - 1] if d > 0 else 0.0,
                ref[t - 1] if d > 0 else EMIN, seg3, t - 108,
                deltas[3], lambdas[3], a4, calibration_used)
            rows.append(r)
            day_cost += r[32]
        records.extend(rows)

        if d > 0:
            issue_q0.append([date, *q0.tolist()])
            issue_qA.append([date, *qA.tolist()])
            issue_ref.append([date, *ref.tolist()])
        else:
            issue_q0.append([date, *np.zeros(T).tolist()])
            issue_qA.append([date, *np.zeros(T).tolist()])
            issue_ref.append([date, *np.full(T, EMIN).tolist()])
        prev_q0, prev_qA, prev_ref = q0, qA, ref
        prev_risk143 = risk143
        diag["initial_soc_kwh"] = initial_soc
        diag["final_soc_kwh"] = float(energy)
        diag["calibration_seconds"] = float(
            sum(c.get("seconds", 0.0) for c in diag["calibrations"]))
        diag["day_seconds"] = time.perf_counter() - day_started
        diagnostics.append(diag)
        if (d - start_idx + 1) % 30 == 0 or d + 1 == limit:
            print(f"Q4-3[{strategy}] {d - start_idx + 1}/{limit - start_idx} days "
                  f"date={date.date()} calib={diag['calibration_seconds']:.2f}s "
                  f"cost={day_cost:,.2f}", flush=True)

    frame = pd.DataFrame.from_records(records, columns=FRAME_COLUMNS)
    validation = validate_question3_v2(frame)
    issue_q0_df = pd.DataFrame(issue_q0, columns=Q_COLS)
    issue_qA_df = pd.DataFrame(issue_qA, columns=Q_COLS)
    issue_ref_df = pd.DataFrame(issue_ref, columns=E_COLS)
    diag = pd.DataFrame(diagnostics)
    return frame, issue_q0_df, issue_qA_df, issue_ref_df, diag, validation


def _period_metrics(frame: pd.DataFrame) -> dict:
    formal = frame[frame.date >= pd.Timestamp("2025-02-01")]
    if not len(formal):
        return {}
    return {
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/question4/v2/result4-3")
    parser.add_argument("--preeval-dir", type=Path,
                        default=ROOT / "outputs/question4/v2/pre_evaluation")
    parser.add_argument("--strategies", nargs="*", default=None,
                        choices=list(STRATEGIES))
    parser.add_argument("--search-seed", type=int, default=V2Settings.search_seed)
    parser.add_argument("--search-maxiter", type=int, default=V2Settings.search_maxiter)
    parser.add_argument("--search-popsize", type=int, default=V2Settings.search_popsize)
    parser.add_argument("--stages", nargs="*", default=None,
                        choices=["warmup", "main", "sensitivity"])
    args = parser.parse_args()
    strategies = args.strategies or ["M0", "M6", "M612", "M61218"]
    stages = args.stages or ["warmup", "main", "sensitivity"]
    args.output.mkdir(parents=True, exist_ok=True)

    dates, load, pv, fixed, p_act, p_tpl = load_question4_v2_inputs()
    bundle = build_q4_3_bundle(load, pv, p_act, dates, fixed)

    if "warmup" in stages:
        warm_settings = replace(V2Settings(name="M0", gamma=0.0),
                                search_seed=args.search_seed,
                                search_maxiter=args.search_maxiter,
                                search_popsize=args.search_popsize)
        frame, q0d, qAd, refd, diag, validation = run_q4_3_range(
            dates, load, pv, p_act, bundle, warm_settings, 0, 6000.0, 31, "wp")
        frame.to_csv(args.output / "warmup_schedule.csv", index=False, encoding="utf-8-sig")
        q0d.to_csv(args.output / "warmup_issue_q0.csv", index=False, encoding="utf-8-sig")
        qAd.to_csv(args.output / "warmup_issue_qA.csv", index=False, encoding="utf-8-sig")
        refd.to_csv(args.output / "warmup_issue_ref.csv", index=False, encoding="utf-8-sig")
        feb1_soc = float(frame.soc_end_kwh.iloc[-1])
        (args.output / "warmup_summary.json").write_text(json.dumps({
            "feb1_soc_kwh": feb1_soc,
            "seconds": float(diag.day_seconds.sum()),
        }, ensure_ascii=False, indent=2))
        print(f"Q4-3 warm-up done: 2/1 SOC = {feb1_soc:.6f} kWh", flush=True)

    if args.days != 365:
        print("pilot mode: stopping after the requested stages", flush=True)
        return

    preeval_path = args.preeval_dir / "pre_evaluation.json"
    if not preeval_path.exists():
        raise SystemExit("pre_evaluation.json missing: run question4_2_v2 first")
    preeval = json.loads(preeval_path.read_text(encoding="utf-8"))
    model_star, gamma_star = preeval["chosen_price_model"], preeval["chosen_gamma"]

    warm_q0 = pd.read_csv(args.output / "warmup_issue_q0.csv", parse_dates=["date"])
    warm_qA = pd.read_csv(args.output / "warmup_issue_qA.csv", parse_dates=["date"])
    warm_ref = pd.read_csv(args.output / "warmup_issue_ref.csv", parse_dates=["date"])
    feb1_soc = float(json.loads((args.output / "warmup_summary.json").read_text(
        encoding="utf-8"))["feb1_soc_kwh"])

    def _row(mat, col):
        sub = mat[mat.date == pd.Timestamp("2025-01-31")]
        return float(sub[col].iloc[0]) if len(sub) else 0.0

    carry_q0_g = np.array([_row(warm_q0, f"g{j}") for j in range(T)])
    carry_qA_g = np.array([_row(warm_qA, f"g{j}") for j in range(T)])
    carry_ref_e = np.array([_row(warm_ref, f"e{j}") for j in range(T)])

    results = {}
    if "main" in stages:
        for name in strategies:
            settings = replace(V2Settings(name=name, price_model=model_star,
                                          gamma=gamma_star),
                               search_seed=args.search_seed,
                               search_maxiter=args.search_maxiter,
                               search_popsize=args.search_popsize)
            frame, q0d, qAd, refd, diag, validation = run_q4_3_range(
                dates, load, pv, p_act, bundle, settings, 31, feb1_soc,
                len(dates), model_star,
                prev_q0=carry_q0_g, prev_qA=carry_qA_g, prev_ref=carry_ref_e)
            out = args.output / name
            out.mkdir(parents=True, exist_ok=True)
            frame.to_csv(out / "question3_schedule.csv", index=False, encoding="utf-8-sig")
            jan31_q0 = warm_q0[warm_q0.date == pd.Timestamp("2025-01-31")]
            jan31_qA = warm_qA[warm_qA.date == pd.Timestamp("2025-01-31")]
            jan31_ref = warm_ref[warm_ref.date == pd.Timestamp("2025-01-31")]
            pd.concat([jan31_q0, q0d], ignore_index=True).to_csv(
                out / "question3_issue_q0.csv", index=False, encoding="utf-8-sig")
            pd.concat([jan31_qA, qAd], ignore_index=True).to_csv(
                out / "question3_issue_qA.csv", index=False, encoding="utf-8-sig")
            pd.concat([jan31_ref, refd], ignore_index=True).to_csv(
                out / "question3_issue_ref.csv", index=False, encoding="utf-8-sig")
            diag.to_csv(out / "question3_diagnostics.csv", index=False, encoding="utf-8-sig")
            metrics = _period_metrics(frame)
            metrics["strategy"] = name
            metrics["validation"] = validation
            metrics["seconds"] = float(diag.day_seconds.sum())
            (out / "question3_summary.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2, default=float))
            results[name] = metrics
            print(f"{name} total={metrics['total_cost']:,.2f} "
                  f"emergency={metrics['emergency_cost']:,.2f}", flush=True)

    if "sensitivity" in stages:
        sens_dir = args.output / "sensitivity"
        sens_dir.mkdir(parents=True, exist_ok=True)
        other_model = "gk" if model_star == "wp" else "wp"
        # Three one-at-a-time variants around the M612 main config; the
        # terminal-value and LDR-acceptance dimensions are covered by the
        # Q4-2 sensitivity suite and its beta=1/beta=0/original-LDR baselines.
        variants = [
            ("sens_model_other", replace(V2Settings(), price_model=other_model,
                                         gamma=gamma_star)),
            ("sens_window_21", replace(V2Settings(), window=21)),
            ("sens_ordinary_quantile", replace(V2Settings(), quantile_kind="ordinary")),
        ]
        sens_summaries = {}
        for vname, variant in variants:
            settings = replace(V2Settings(name=MAIN_STRATEGY),
                               price_model=variant.price_model, gamma=variant.gamma,
                               window=variant.window,
                               quantile_kind=variant.quantile_kind,
                               terminal_value_rule=variant.terminal_value_rule,
                               ldr_kind=variant.ldr_kind,
                               search_seed=args.search_seed,
                               search_maxiter=args.search_maxiter,
                               search_popsize=args.search_popsize)
            frame, q0d, qAd, refd, diag, validation = run_q4_3_range(
                dates, load, pv, p_act, bundle, settings, 31, feb1_soc,
                len(dates), settings.price_model,
                prev_q0=carry_q0_g, prev_qA=carry_qA_g, prev_ref=carry_ref_e)
            m = _period_metrics(frame)
            m["variant"] = vname
            m["seconds"] = float(diag.day_seconds.sum())
            sens_summaries[vname] = m
            frame.to_csv(sens_dir / f"{vname}_schedule.csv", index=False, encoding="utf-8-sig")
            print(f"{vname}: total={m['total_cost']:,.2f}", flush=True)
        (sens_dir / "sensitivity_summaries.json").write_text(
            json.dumps(sens_summaries, ensure_ascii=False, indent=2, default=float))
        results["sensitivity"] = sens_summaries

    (args.output / "run_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float))
    return results


if __name__ == "__main__":
    main()

