# -*- coding: utf-8 -*-
"""Question 4-2 v2: fluctuating prices under docs/问题四/问题四优化实施方案.md.

Pipeline:
  1. build causal WP/GK price forecasts (366 rows incl. virtual 2026-01-01);
  2. warm-up 1/1 (zero-plan REAL execution day) + 1/2-1/31 (weekly-persistence
     price forecast, gamma = 0) -> common 1 February SOC;
  3. pre-evaluation on 1/8-1/31 shadow runs over the (model x gamma) grid,
     frozen before 1 February (tie-break prefers WP unless GK is cheaper by
     more than 0.5%);
  4. main model: price-weighted Q80 plan curve, scenario-mean decision prices,
     next-day terminal value, 42-day decaying-window regularised LDR with
     rolling validation-day acceptance;
  5. required baselines: ordinary Q80 + beta=1 / beta=0 / original LDR
     (21-day window, in-sample acceptance, same-day terminal value), each as
     an INDEPENDENT run from the common 1 February SOC;
  6. sensitivities: other price model, window 21/63, ordinary quantile,
     terminal value 0 / same-day;
  7. perfect-foresight lower bound matched to the main 2/1 SOC.

Run:  python -m src.optimization.question4_2_v2
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.optimization.question2 import (
    ROOT, T, ETA, EMIN, load_inputs, solve_plan, validate_schedule,
)
from src.optimization.q4_v2_common import (
    GAMMA_GRID,
    V2Settings,
    build_load_forecast_extended,
    build_price_forecast,
    build_pv_forecast_extended,
    calibrate_v2,
    decay_weights,
    execute_day,
    perfect_foresight_bound,
    plan_horizon,
    price_weighted_quantile,
    residual_columns,
    scenario_mean_price,
    scenario_window,
    unpack_by_stages,
)
from src.data_pipeline.question4_prices import load_price_actual

FRAME_COLUMNS = [
    "date", "slot", "interval_start", "issue_date", "issue_slot",
    "load_kwh", "pv_kwh", "price", "price_forecast",
    "forecast_load_kwh", "forecast_pv_kwh", "planning_net_kwh", "risk_net_kwh",
    "residual_count",
    "grid_kwh", "plan_charge_kwh", "plan_discharge_kwh", "plan_soc_kwh",
    "reserve_kwh", "soc_start_kwh", "charge_kwh", "discharge_kwh",
    "emergency_kwh", "unused_kwh", "soc_end_kwh",
    "planned_cost", "emergency_cost", "total_cost",
    "stage", "stage_signal_kwh", "ldr_delta_kwh", "ldr_lambda", "calibration_used",
]

ISSUE_COLS = ["date", *[f"g{j}" for j in range(T)], *[f"e{j}" for j in range(T)]]


def load_question4_v2_inputs(root: Path = ROOT):
    dates, load, pv, fixed = load_inputs(root)
    _, p_act = load_price_actual(root)
    raw = pd.read_excel(root / "附件/附件4.xlsx", sheet_name=0)
    p_tpl = raw.iloc[:, 1:145].to_numpy(float)  # template order 00:10..0:00+1
    assert p_tpl.shape == (365, T) and np.isfinite(p_tpl).all()
    return dates, load, pv, fixed, p_act, p_tpl


def build_forecast_bundle(load, pv, p_act, dates, fixed):
    fl = build_load_forecast_extended(load)          # (366, 144)
    fv = build_pv_forecast_extended(pv)              # (366, 144)
    p_wp = build_price_forecast(p_act, dates, "wp", fixed)
    p_gk = build_price_forecast(p_act, dates, "gk", fixed)
    e_net = np.empty((364, T), float)
    e_p = {"wp": np.empty((364, T), float), "gk": np.empty((364, T), float)}
    for i in range(364):
        e_net[i] = residual_columns(load, fl, i) - residual_columns(pv, fv, i)
        e_p["wp"][i] = residual_columns(p_act, p_wp, i)
        e_p["gk"][i] = residual_columns(p_act, p_gk, i)
    return {"fl": fl, "fv": fv, "price": {"wp": p_wp, "gk": p_gk},
            "res_net": e_net, "res_price": e_p}


def terminal_value_nu(F_p: np.ndarray, d: int, rule: str) -> float:
    if rule == "zero":
        return 0.0
    if rule == "same_day":
        return float(F_p[d].min() / ETA)
    return float(F_p[d + 1].min() / ETA)  # next_day


def _day_risk_curve(settings: V2Settings, net_h, scen_net, scen_price, w):
    if scen_net is None:
        return net_h.copy()
    if settings.quantile_kind == "price_weighted":
        q = price_weighted_quantile(scen_net, scen_price, w, settings.alpha)
    else:
        q = np.quantile(scen_net, settings.alpha, axis=0)
    return np.maximum(net_h, q)


def _stage_specs() -> list:
    return [(0, 36, False), (36, 72, True), (72, 108, True), (108, 145, True)]


def issue_row_of(issue: pd.DataFrame, date) -> tuple[np.ndarray, np.ndarray]:
    row = issue[issue.date == pd.Timestamp(date)]
    if len(row) == 0:
        return np.zeros(T), np.full(T, EMIN)
    g = row[[f"g{j}" for j in range(T)]].to_numpy(float).ravel()
    e = row[[f"e{j}" for j in range(T)]].to_numpy(float).ravel()
    return g, e


def run_q4_2_range(dates, load, pv, p_act, bundle, settings: V2Settings,
                   start_idx: int, initial_energy: float, limit: int,
                   price_model: str, prev_g: np.ndarray | None = None,
                   prev_E: np.ndarray | None = None):
    """Continuous Q4-2 execution from start_idx at initial_energy.

    Execution interval 0 of ``start_idx`` carries over the previous day's
    issue column 143 (``prev_g``/``prev_E``); defaults to zero/EMIN (1/1).
    """
    fl, fv = bundle["fl"], bundle["fv"]
    F_p = bundle["price"][price_model]
    res_net, res_price = bundle["res_net"], bundle["res_price"][price_model]
    if prev_g is None:
        prev_g = np.zeros(T)
    if prev_E is None:
        prev_E = np.full(T, EMIN)
    prev_g = np.asarray(prev_g, float)
    prev_E = np.asarray(prev_E, float)
    energy = float(initial_energy)
    records: list[tuple] = []
    issue_records: list[list] = []
    diagnostics: list[dict] = []
    for d in range(start_idx, limit):
        date = dates[d]
        day_started = time.perf_counter()
        p_settle = p_act[d]
        grid_full = np.zeros(145)
        ref_full = np.full(145, EMIN)
        if d == 0:
            g_issue = np.zeros(T)
            E_p = np.full(T, EMIN)
            theta = np.zeros(7)
            residual_count = 0
            calibration_used = False
            risk_exec = np.zeros(T)
            forecast_exec = np.zeros(T)
            p_forecast_row = F_p[0].copy()
        else:
            fl_h = plan_horizon(fl, d)
            fv_h = plan_horizon(fv, d)
            net_h = fl_h - fv_h
            p_h = plan_horizon(F_p, d)
            first, M = scenario_window(d, settings.window)
            scen_net = scen_price = w = None
            if M > 0:
                rows_i = np.arange(first, d)
                w = decay_weights(d, first, settings.decay_tau)
                scen_net = net_h[None, :] + res_net[rows_i]
                scen_price = np.maximum(0.0, p_h[None, :] + res_price[rows_i])
            risk = _day_risk_curve(settings, net_h, scen_net, scen_price, w)
            pbar = scenario_mean_price(scen_price, w) if M > 0 else np.maximum(p_h, 0.0)
            nu = terminal_value_nu(F_p, d, settings.terminal_value_rule)
            plan, _ = solve_plan(risk, pbar, energy, nu)
            g_issue = plan[0]
            E_p = plan[4]
            grid_full = np.r_[prev_g[143], g_issue]
            ref_full = np.r_[prev_E[143], E_p]
            forecast_full = np.r_[fl[d, 0] - fv[d, 0], net_h]
            calibration_used = False
            cal = None
            if M > 0:
                rows_i = np.arange(first, d)
                scen_full = np.c_[
                    fl[d, 0] - fv[d, 0] + res_net[rows_i][:, 143], scen_net]
                price_full = np.maximum(
                    0.0, np.c_[F_p[d, 0] + res_price[rows_i][:, 143], scen_price])
                cal = calibrate_v2(
                    scen_full, forecast_full, grid_full, ref_full, price_full,
                    energy, nu, _stage_specs(), [None, None, None, None],
                    settings, settings.search_seed + d,
                    signal_mode="cumulative", residual_count=M,
                )
                calibration_used = bool(cal["accepted_search"]) \
                    or settings.ldr_kind == "original"
                theta = cal["theta"]
            else:
                theta = np.zeros(7)
            residual_count = M
            q_exec = grid_full[:T]
            ref_exec = ref_full[:T]
            forecast_exec = forecast_full[:T]
            risk_exec = risk
            p_forecast_row = p_h

        deltas, lambdas = unpack_by_stages(theta, _stage_specs())
        out = execute_day(
            load[d] - pv[d], grid_full[:T], ref_full[:T],
            forecast_exec, p_settle, energy, deltas, lambdas,
        )
        issue_records.append([date, *g_issue.tolist(), *E_p.tolist()])
        initial_day_soc = energy
        planned_cost_day = float(np.dot(p_settle, grid_full[:T]))
        for t in range(T):
            stage = min(t // 36, 3)
            records.append((
                date, t, date + pd.Timedelta(minutes=10 * t),
                dates[d - 1] if t == 0 else date,
                143 if t == 0 else t - 1,
                load[d, t], pv[d, t], p_settle[t],
                p_forecast_row[t],
                fl[d, t] if d > 0 else 0.0, fv[d, t] if d > 0 else 0.0,
                (fl[d, t] - fv[d, t]) if d > 0 else 0.0,
                risk_exec[t],
                residual_count,
                grid_full[t],
                0.0, 0.0, ref_full[t],
                out["reserve_kwh"][t], out["soc_start_kwh"][t],
                out["charge_kwh"][t], out["discharge_kwh"][t],
                out["emergency_kwh"][t], out["unused_kwh"][t],
                out["soc_end_kwh"][t],
                p_settle[t] * grid_full[t],
                5.0 * p_settle[t] * out["emergency_kwh"][t],
                p_settle[t] * grid_full[t]
                + 5.0 * p_settle[t] * out["emergency_kwh"][t],
                stage + 1, out["stage_signal_kwh"][t], deltas[stage], lambdas[stage],
                calibration_used,
            ))
        energy = float(out["soc_end_kwh"][-1])
        prev_g, prev_E = g_issue, E_p
        diagnostics.append({
            "date": str(date.date()), "issue": d > 0,
            "residual_count": residual_count,
            "initial_soc_kwh": initial_day_soc,
            "final_soc_kwh": energy,
            "planned_cost": planned_cost_day,
            "emergency_cost": float((5.0 * p_settle * out["emergency_kwh"]).sum()),
            "total_cost": planned_cost_day + float((5.0 * p_settle * out["emergency_kwh"]).sum()),
            "calibration_used": calibration_used,
            "theta": theta.tolist(),
            "day_seconds": time.perf_counter() - day_started,
        })
        if (d - start_idx + 1) % 30 == 0 or d + 1 == limit:
            print(f"Q4-2[{price_model}] {d - start_idx + 1}/{limit - start_idx} days "
                  f"date={date.date()} cost={diagnostics[-1]['total_cost']:,.2f}", flush=True)

    frame = pd.DataFrame.from_records(records, columns=FRAME_COLUMNS)
    validation = validate_schedule(frame)
    issue = pd.DataFrame(issue_records, columns=ISSUE_COLS)
    diag = pd.DataFrame(diagnostics)
    return frame, issue, diag, validation


def _period_metrics(frame: pd.DataFrame) -> dict:
    if len(frame) == 0:
        return {}
    return {
        "days": int(frame["date"].nunique()),
        "intervals": int(len(frame)),
        "grid_kwh": float(frame.grid_kwh.sum()),
        "charge_kwh": float(frame.charge_kwh.sum()),
        "discharge_kwh": float(frame.discharge_kwh.sum()),
        "emergency_kwh": float(frame.emergency_kwh.sum()),
        "unused_kwh": float(frame.unused_kwh.sum()),
        "planned_cost": float(frame.planned_cost.sum()),
        "emergency_cost": float(frame.emergency_cost.sum()),
        "total_cost": float(frame.total_cost.sum()),
        "initial_soc_kwh": float(frame.soc_start_kwh.iloc[0]),
        "final_soc_kwh": float(frame.soc_end_kwh.iloc[-1]),
        "emergency_intervals": int((frame.emergency_kwh > 1e-6).sum()),
    }


def run_independent_baseline(dates, load, pv, p_act, bundle, price_model,
                             start_idx, initial_energy, limit, kind, search_seed,
                             prev_g, prev_E):
    """One independent Q4-2 baseline: beta=1, beta=0 or original LDR.

    All use the ordinary Q80 plan curve; beta=1 keeps zero parameters with
    the reference SOC, beta=0 discharges to EMIN, original LDR calibrates a
    21-day-window unregularised rule with in-sample acceptance and same-day
    terminal value.  The purchase plan is re-solved each day from the
    controller's own SOC (independent continuous run)."""
    fl, fv = bundle["fl"], bundle["fv"]
    F_p = bundle["price"][price_model]
    res_net, res_price = bundle["res_net"], bundle["res_price"][price_model]
    energy = float(initial_energy)
    records: list[tuple] = []
    issue_records: list[list] = []
    for d in range(start_idx, limit):
        date = dates[d]
        p_settle = p_act[d]
        grid_full = np.zeros(145)
        ref_full = np.full(145, EMIN)
        if d == 0:
            g_issue = np.zeros(T)
            E_p = np.full(T, EMIN)
            theta = np.zeros(7)
            net_h = np.zeros(T)
            p_h = np.zeros(T)
            forecast_full = np.zeros(145)
        else:
            fl_h = plan_horizon(fl, d)
            fv_h = plan_horizon(fv, d)
            net_h = fl_h - fv_h
            p_h = plan_horizon(F_p, d)
            first, M = scenario_window(d, 21)
            scen_net = scen_price = w = None
            if M > 0:
                rows_i = np.arange(first, d)
                w = decay_weights(d, first, 1e9)  # ~equal weights (original protocol)
                scen_net = net_h[None, :] + res_net[rows_i]
                scen_price = np.maximum(0.0, p_h[None, :] + res_price[rows_i])
            risk = _day_risk_curve(V2Settings(quantile_kind="ordinary"), net_h,
                                   scen_net, scen_price, w)
            pbar = scenario_mean_price(scen_price, w) if M > 0 else np.maximum(p_h, 0.0)
            nu = terminal_value_nu(F_p, d, "same_day")
            plan, _ = solve_plan(risk, pbar, energy, nu)
            g_issue = plan[0]
            E_p = plan[4]
            grid_full = np.r_[prev_g[143], g_issue]
            ref_full = np.r_[prev_E[143], E_p]
            forecast_full = np.r_[fl[d, 0] - fv[d, 0], net_h]
            if kind == "orig" and M > 0:
                rows_i = np.arange(first, d)
                scen_full = np.c_[
                    fl[d, 0] - fv[d, 0] + res_net[rows_i][:, 143], scen_net]
                price_full = np.maximum(
                    0.0, np.c_[F_p[d, 0] + res_price[rows_i][:, 143], scen_price])
                orig_settings = V2Settings(window=21, gamma=0.0, ldr_kind="original",
                                           terminal_value_rule="same_day",
                                           quantile_kind="ordinary",
                                           search_seed=search_seed)
                cal = calibrate_v2(
                    scen_full, forecast_full, grid_full, ref_full, price_full,
                    energy, nu, _stage_specs(), [None] * 4, orig_settings,
                    search_seed + d, signal_mode="cumulative", residual_count=M,
                )
                theta = cal["theta"]
            else:
                theta = np.zeros(7)
        issue_records.append([date, *g_issue.tolist(), *E_p.tolist()])
        ref_exec = ref_full[:T] if kind != "beta0" else np.full(T, EMIN)
        deltas, lambdas = unpack_by_stages(theta, _stage_specs())
        out = execute_day(load[d] - pv[d], grid_full[:T], ref_exec,
                          forecast_full[:T], p_settle, energy, deltas, lambdas)
        for t in range(T):
            stage = min(t // 36, 3)
            records.append((
                date, t, date + pd.Timedelta(minutes=10 * t), kind,
                grid_full[t], ref_exec[t],
                out["soc_start_kwh"][t], out["charge_kwh"][t],
                out["discharge_kwh"][t], out["emergency_kwh"][t],
                out["unused_kwh"][t], out["soc_end_kwh"][t],
                p_settle[t] * grid_full[t],
                5.0 * p_settle[t] * out["emergency_kwh"][t],
            ))
        energy = float(out["soc_end_kwh"][-1])
        prev_g, prev_E = g_issue, E_p
        if (d - start_idx + 1) % 60 == 0 or d + 1 == limit:
            print(f"Q4-2 baseline {kind} {d - start_idx + 1}/{limit - start_idx} days",
                  flush=True)
    cols = ["date", "slot", "interval_start", "kind", "grid_kwh", "ref_kwh",
            "soc_start_kwh", "charge_kwh", "discharge_kwh", "emergency_kwh",
            "unused_kwh", "soc_end_kwh", "planned_cost", "emergency_cost"]
    frame = pd.DataFrame.from_records(records, columns=cols)
    frame["total_cost"] = frame.planned_cost + frame.emergency_cost
    issue = pd.DataFrame(issue_records, columns=ISSUE_COLS)
    return frame, issue


def pre_evaluate(dates, load, pv, p_act, bundle, warmup_frame, warmup_issue,
                 search_seed, out_dir: Path):
    """1/8-1/31 shadow runs over the (model x gamma) grid from the common 1/8 SOC."""
    soc_0108 = float(
        warmup_frame[warmup_frame.date == pd.Timestamp("2025-01-08")].soc_start_kwh.iloc[0])
    prev_g, prev_E = issue_row_of(warmup_issue, "2025-01-07")
    table = []
    for model in ("wp", "gk"):
        for gamma in GAMMA_GRID:
            settings = V2Settings(price_model=model, gamma=gamma, search_seed=search_seed)
            frame, issue, diag, validation = run_q4_2_range(
                dates, load, pv, p_act, bundle, settings, 7, soc_0108, 31, model,
                prev_g=prev_g, prev_E=prev_E)
            cost = float(frame.total_cost.sum())
            table.append({"price_model": model, "gamma": gamma,
                          "cost_0108_0131": cost,
                          "emergency_cost": float(frame.emergency_cost.sum()),
                          "seconds": float(diag.day_seconds.sum())})
            print(f"pre-eval {model} gamma={gamma}: cost={cost:,.2f}", flush=True)
    df = pd.DataFrame(table)
    sub_wp = df[df.price_model == "wp"]
    sub_gk = df[df.price_model == "gk"]
    wp_best = sub_wp.loc[sub_wp.cost_0108_0131.idxmin()]
    gk_best = sub_gk.loc[sub_gk.cost_0108_0131.idxmin()]
    if (float(wp_best.cost_0108_0131) - float(gk_best.cost_0108_0131)) \
            / float(wp_best.cost_0108_0131) > 0.005:
        chosen = gk_best
    else:
        chosen = wp_best
    result = {
        "window": "2025-01-08/2025-01-31",
        "grid": table,
        "chosen_price_model": str(chosen.price_model),
        "chosen_gamma": float(chosen.gamma),
        "tie_break_rule": "prefer wp unless gk is cheaper by more than 0.5%",
        "wp_best": wp_best.to_dict(),
        "gk_best": gk_best.to_dict(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pre_evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/question4/v2/result4-2")
    parser.add_argument("--preeval-dir", type=Path,
                        default=ROOT / "outputs/question4/v2/pre_evaluation")
    parser.add_argument("--search-seed", type=int, default=V2Settings.search_seed)
    parser.add_argument("--search-maxiter", type=int, default=V2Settings.search_maxiter)
    parser.add_argument("--search-popsize", type=int, default=V2Settings.search_popsize)
    parser.add_argument("--stages", nargs="*", default=None,
                        choices=["warmup", "preeval", "main", "baselines",
                                 "sensitivity", "bound"])
    args = parser.parse_args()
    stages = args.stages or ["warmup", "preeval", "main", "baselines",
                             "sensitivity", "bound"]
    args.output.mkdir(parents=True, exist_ok=True)

    dates, load, pv, fixed, p_act, p_tpl = load_question4_v2_inputs()
    bundle = build_forecast_bundle(load, pv, p_act, dates, fixed)
    if args.days != 365:
        dates = dates[: args.days]
        load, pv, p_act, p_tpl = load[: args.days], pv[: args.days], \
            p_act[: args.days], p_tpl[: args.days]

    warmup_settings = V2Settings(gamma=0.0, search_seed=args.search_seed,
                                 search_maxiter=args.search_maxiter,
                                 search_popsize=args.search_popsize)

    if "warmup" in stages or "preeval" in stages or "main" in stages \
            or "baselines" in stages:
        started = time.perf_counter()
        warmup_frame, warmup_issue, warmup_diag, _ = run_q4_2_range(
            dates, load, pv, p_act, bundle, warmup_settings, 0, 6000.0, 31, "wp")
        warmup_frame.to_csv(args.output / "warmup_schedule.csv", index=False,
                            encoding="utf-8-sig")
        warmup_issue.to_csv(args.output / "warmup_issue.csv", index=False,
                            encoding="utf-8-sig")
        feb1_soc = float(warmup_frame.soc_end_kwh.iloc[-1])
        jan8_soc = float(
            warmup_frame[warmup_frame.date == pd.Timestamp("2025-01-08")].soc_start_kwh.iloc[0])
        (args.output / "warmup_summary.json").write_text(json.dumps({
            "jan8_soc_kwh": jan8_soc, "feb1_soc_kwh": feb1_soc,
            "seconds": time.perf_counter() - started,
        }, ensure_ascii=False, indent=2))
        print(f"warm-up done: 2/1 SOC = {feb1_soc:.6f} kWh", flush=True)
    else:
        warmup_frame = pd.read_csv(args.output / "warmup_schedule.csv",
                                   parse_dates=["date"])
        warmup_issue = pd.read_csv(args.output / "warmup_issue.csv",
                                   parse_dates=["date"])

    if "preeval" in stages and args.days == 365:
        pre_evaluate(dates, load, pv, p_act, bundle, warmup_frame, warmup_issue,
                     args.search_seed, args.preeval_dir)

    if args.days != 365:
        print("pilot mode: stopping after the requested stages", flush=True)
        return

    preeval_path = args.preeval_dir / "pre_evaluation.json"
    if not preeval_path.exists():
        raise SystemExit("pre_evaluation.json missing: run the pre-evaluation stage first")
    preeval = json.loads(preeval_path.read_text(encoding="utf-8"))
    model_star, gamma_star = preeval["chosen_price_model"], preeval["chosen_gamma"]
    warmup_summary = json.loads(
        (args.output / "warmup_summary.json").read_text(encoding="utf-8"))
    feb1_soc = float(warmup_summary["feb1_soc_kwh"])
    carry_g, carry_e = issue_row_of(warmup_issue, "2025-01-31")

    results = {}

    if "main" in stages:
        main_settings = V2Settings(price_model=model_star, gamma=gamma_star,
                                   search_seed=args.search_seed,
                                   search_maxiter=args.search_maxiter,
                                   search_popsize=args.search_popsize)
        frame, issue, diag, validation = run_q4_2_range(
            dates, load, pv, p_act, bundle, main_settings, 31, feb1_soc,
            len(dates), model_star, prev_g=carry_g, prev_E=carry_e)
        full = pd.concat([warmup_frame, frame], ignore_index=True)
        full_metrics = _period_metrics(full)
        formal = frame[frame.date >= pd.Timestamp("2025-02-01")]
        formal_metrics = _period_metrics(formal)
        issue_cost = 0.0
        for _, row in issue.iterrows():
            d_idx = (pd.Timestamp(row["date"]) - pd.Timestamp("2025-01-01")).days
            row_cost = float(np.dot(p_tpl[d_idx], row[[f"g{j}" for j in range(T)]].to_numpy(float)))
            issue_cost += row_cost
        carry_in = float(p_tpl[30][143] * carry_g[143])
        last = issue.iloc[-1]
        carry_out = float(p_tpl[364][143] * last[f"g143"])
        exec_plan_cost = float(formal.planned_cost.sum())
        bridge_error = abs(issue_cost + carry_in - carry_out - exec_plan_cost)
        summary = {
            "settings": {"price_model": model_star, "gamma": gamma_star,
                         "quantile": "price_weighted_q80", "window": 42,
                         "decay_tau": 14.0, "terminal_value": "next_day",
                         "ldr": "regularized_validation_day"},
            "full_year": full_metrics,
            "formal_period": formal_metrics,
            "issue_cost": issue_cost, "carry_in": carry_in, "carry_out": carry_out,
            "exec_plan_cost": exec_plan_cost, "bridge_error": bridge_error,
            "validation": validation,
            "seconds": float(diag.day_seconds.sum()),
        }
        frame.to_csv(args.output / "question2_schedule_with_warmup.csv", index=False,
                     encoding="utf-8-sig")
        formal.to_csv(args.output / "question2_schedule.csv", index=False,
                      encoding="utf-8-sig")
        # prepend the 1/31 issue row so the carry-in bridge can be recomputed
        jan31 = warmup_issue[warmup_issue.date == pd.Timestamp("2025-01-31")]
        pd.concat([jan31, issue], ignore_index=True).to_csv(
            args.output / "question2_issue.csv", index=False, encoding="utf-8-sig")
        diag.to_csv(args.output / "ldr_daily_parameters.csv", index=False,
                    encoding="utf-8-sig")
        (args.output / "question2_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float))
        results["main"] = summary
        print(json.dumps({k: v for k, v in summary.items() if k != "validation"},
                         ensure_ascii=False, indent=2, default=float), flush=True)

    if "baselines" in stages:
        base_dir = args.output / "baselines"
        base_dir.mkdir(parents=True, exist_ok=True)
        base_summaries = {}
        for kind in ("beta1", "beta0", "orig"):
            frame, issue = run_independent_baseline(
                dates, load, pv, p_act, bundle, model_star, 31, feb1_soc,
                len(dates), kind, args.search_seed, prev_g=carry_g, prev_E=carry_e)
            formal = frame[frame.date >= pd.Timestamp("2025-02-01")]
            m = _period_metrics(formal)
            m["kind"] = kind
            base_summaries[kind] = m
            frame.to_csv(base_dir / f"baseline_{kind}_schedule.csv", index=False,
                         encoding="utf-8-sig")
            issue.to_csv(base_dir / f"baseline_{kind}_issue.csv", index=False,
                         encoding="utf-8-sig")
        (base_dir / "baseline_summaries.json").write_text(
            json.dumps(base_summaries, ensure_ascii=False, indent=2, default=float))
        results["baselines"] = base_summaries
        print(json.dumps(base_summaries, ensure_ascii=False, indent=2, default=float),
              flush=True)

    if "sensitivity" in stages:
        other_model = "gk" if model_star == "wp" else "wp"
        variants = [
            ("sens_model_other",
             replace(V2Settings(), price_model=other_model, gamma=gamma_star)),
            ("sens_window_21",
             replace(V2Settings(), price_model=model_star, gamma=gamma_star, window=21)),
            ("sens_window_63",
             replace(V2Settings(), price_model=model_star, gamma=gamma_star, window=63)),
            ("sens_ordinary_quantile",
             replace(V2Settings(), price_model=model_star, gamma=gamma_star,
                     quantile_kind="ordinary")),
            ("sens_nu_zero",
             replace(V2Settings(), price_model=model_star, gamma=gamma_star,
                     terminal_value_rule="zero")),
            ("sens_nu_same_day",
             replace(V2Settings(), price_model=model_star, gamma=gamma_star,
                     terminal_value_rule="same_day")),
        ]
        sens_dir = args.output / "sensitivity"
        sens_dir.mkdir(parents=True, exist_ok=True)
        sens_summaries = {}
        for name, variant in variants:
            settings = replace(variant, search_seed=args.search_seed,
                               search_maxiter=args.search_maxiter,
                               search_popsize=args.search_popsize)
            frame, issue, diag, validation = run_q4_2_range(
                dates, load, pv, p_act, bundle, settings, 31, feb1_soc,
                len(dates), settings.price_model, prev_g=carry_g, prev_E=carry_e)
            m = _period_metrics(frame[frame.date >= pd.Timestamp("2025-02-01")])
            m["variant"] = name
            m["seconds"] = float(diag.day_seconds.sum())
            sens_summaries[name] = m
            frame.to_csv(sens_dir / f"{name}_schedule.csv", index=False,
                         encoding="utf-8-sig")
            print(f"{name}: total={m['total_cost']:,.2f}", flush=True)
        (sens_dir / "sensitivity_summaries.json").write_text(
            json.dumps(sens_summaries, ensure_ascii=False, indent=2, default=float))
        results["sensitivity"] = sens_summaries

    if "bound" in stages:
        bound = perfect_foresight_bound(load, pv, p_act, feb1_soc, start_idx=31)
        (args.output / "perfect_foresight_bound.json").write_text(
            json.dumps(bound, ensure_ascii=False, indent=2, default=float))
        results["bound"] = bound
        print(json.dumps(bound, ensure_ascii=False, indent=2), flush=True)

    (args.output / "run_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float))
    return results


if __name__ == "__main__":
    main()
