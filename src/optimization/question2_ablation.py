# -*- coding: utf-8 -*-
"""Question 2 ablation & robustness experiments (registered 2026-09-13).

Protocol registered BEFORE running, see
outputs/question2/analysis/ablation_robustness/实验协议.md.  All runs are
report-only: the main configuration (M=21, seed 20250912+d, alpha=0.8,
lambda free, natural warm-up) and outputs/question2/current/ldr/ stay
unchanged.

Subcommands
-----------
python -m src.optimization.question2_ablation delta-only      # exp 2a: independent year
python -m src.optimization.question2_ablation paired-delta-only  # exp 2b: same-plan replay
python -m src.optimization.question2_ablation common-start    # exp 3: beta1/beta0/scenarios
python -m src.optimization.question2_ablation window-seed-grid  # exp 1: 3 windows x 3 seeds
python -m src.optimization.question2_ablation all             # 1 + 2 + 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.optimization.question2 import (
    ETA,
    ROOT,
    Settings,
    forecasts,
    load_inputs,
    load_forecast_weekly_persist,
    run_case,
)
from src.optimization.question2_ldr import (
    LDRSettings,
    calibrate_rule,
    execute_actual_day,
    run_ldr,
)
from src.optimization.question2_g_search import scenario_net_matrix
from src.optimization import question2_scenarios

OUT = ROOT / "outputs/question2/analysis/ablation_robustness"
COMMON_SOC = 2263.2065799206325          # LDR natural 1/31 end-of-day SOC
FORMAL_START = pd.Timestamp("2025-02-01")
MAIN_LDR_JSON = ROOT / "outputs/question2/current/ldr/question2_summary.json"
MAIN_LEDGER = ROOT / "outputs/question2/current/ldr/question2_schedule_with_warmup.csv"


def formal_totals(frame: pd.DataFrame) -> dict:
    g = frame[frame.date >= FORMAL_START]
    out = {}
    for key in ["grid_kwh", "charge_kwh", "discharge_kwh", "emergency_kwh",
                "unused_kwh", "planned_cost", "emergency_cost", "total_cost"]:
        out[key] = float(g[key].sum())
    out["days"] = int(g.date.nunique())
    out["intervals"] = int(len(g))
    out["initial_soc_kwh"] = float(g.soc_start_kwh.iloc[0])
    out["final_soc_kwh"] = float(g.soc_end_kwh.iloc[-1])
    out["emergency_intervals"] = int((g.emergency_kwh > 1e-6).sum())
    return out


def ldr_variant_summary(settings_kwargs: dict, run_kwargs: dict, tag: str) -> dict:
    started = time.perf_counter()
    dates, load, pv, prices = load_inputs()
    fl = load_forecast_weekly_persist(load)
    settings = LDRSettings(load_forecast="weekly_persist", **settings_kwargs)
    frame, daily, daily_all, summary, diag, pair = run_ldr(
        dates, load, pv, prices, settings, fl=fl, **run_kwargs
    )
    rec = {
        "tag": tag,
        "settings": {**summary["settings"],
                     "experiment_settings": settings_kwargs,
                     "experiment_run_kwargs": run_kwargs},
        "formal_period": summary["formal_period"],
        "validation": summary["validation"],
        "seconds": time.perf_counter() - started,
        "calibrated_days": int(summary["calibrated_days"]),
        "accepted_search_days": int(summary["accepted_search_days"]),
        "search_evaluations": int(summary["search_evaluations"]),
    }
    daily.to_csv(OUT / f"{tag}_daily.csv", encoding="utf-8-sig")
    diag.to_csv(OUT / f"{tag}_parameters.csv", index=False, encoding="utf-8-sig")
    return rec


def run_delta_only() -> None:
    print("experiment 2a: delta-only independent year ...", flush=True)
    rec = ldr_variant_summary(
        settings_kwargs={},
        run_kwargs={"lambda_locked": True},
        tag="delta_only",
    )
    rec["settings"]["name"] = "ldr_delta_only"
    (OUT / "delta_only_summary.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(rec["formal_period"], indent=2), flush=True)


def run_paired_delta_only() -> None:
    print("experiment 2b: delta-only same-plan paired replay ...", flush=True)
    started = time.perf_counter()
    dates, load, pv, prices = load_inputs()
    fl = load_forecast_weekly_persist(load)
    _, fv = forecasts(load, pv)
    ledger = pd.read_csv(MAIN_LEDGER, parse_dates=["interval_start", "date"])
    ledger["date"] = pd.to_datetime(ledger["date"])
    nu = float(prices.min() / ETA)
    settings = LDRSettings(load_forecast="weekly_persist")
    rows = []
    for date, day in ledger.groupby("date"):
        idx = (date.date() - dates[0].date()).days
        g = day.grid_kwh.to_numpy(float)
        plan_soc = day.plan_soc_kwh.to_numpy(float)
        e0 = float(day.soc_start_kwh.iloc[0])
        forecast_net = (day.forecast_load_kwh - day.forecast_pv_kwh).to_numpy(float)
        residual_count = int(day.residual_count.iloc[0])
        scenarios, count = scenario_net_matrix(idx, load, pv, fl, fv, residual_days=21)
        scen = forecast_net[None, :] if scenarios is None else scenarios
        cal = calibrate_rule(
            scen, forecast_net, g, plan_soc, prices, e0, nu, residual_count,
            settings, settings.search_seed + idx, lambda_locked=True,
        )
        out = execute_actual_day(
            load[idx], pv[idx], g, plan_soc, prices, e0, forecast_net, cal.theta
        )
        rows.append({
            "date": str(date.date()),
            "planned_cost": float(np.dot(prices, g)),
            "delta_only_emergency_cost": float(out["emergency_cost"].sum()),
            "final_soc_kwh": float(out["soc_end_kwh"][-1]),
            "zero_score": cal.zero_score,
            "selected_score": cal.selected_score,
            "accepted_search": cal.accepted_search,
            "evaluations": cal.evaluations,
        })
    pair = pd.DataFrame(rows)
    pair.to_csv(OUT / "delta_only_paired_daily.csv", index=False, encoding="utf-8-sig")
    formal = pair[pd.to_datetime(pair.date) >= FORMAL_START]
    rec = {
        "tag": "delta_only_paired",
        "note": "daily LDR plan g^Q and daily initial SOC taken from the main ledger; "
                "delta-only parameters re-calibrated each morning with lambda locked; "
                "paired totals are NOT an independent annual strategy bill.",
        "planned_cost": float(formal.planned_cost.sum()),
        "delta_only_emergency_cost": float(formal.delta_only_emergency_cost.sum()),
        "paired_total_cost": float((formal.planned_cost + formal.delta_only_emergency_cost).sum()),
        "ldr_paired_total_cost": 13978077.233121704,
        "ldr_paired_emergency_cost": 645227.7311187075,
        "seconds": time.perf_counter() - started,
        "accepted_search_days": int(formal.accepted_search.sum()),
        "total_evaluations": int(formal.evaluations.sum()),
    }
    (OUT / "delta_only_paired_summary.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(rec, indent=2), flush=True)


def run_common_start() -> None:
    print("experiment 3: common-start backtest ...", flush=True)
    dates, load, pv, prices = load_inputs()
    fl = load_forecast_weekly_persist(load)
    results = {"common_initial_soc_kwh": COMMON_SOC, "runs": {}}

    for tag, setting in [
        ("beta1", Settings()),
        ("beta0", Settings(name="risk_greedy", beta=0.0)),
    ]:
        t0 = time.perf_counter()
        frame, daily, summary = run_case(
            dates, load, pv, prices, setting, fl_override=fl,
            start_index=31, initial_soc=COMMON_SOC,
        )
        results["runs"][tag] = {
            "formal_period": formal_totals(frame),
            "validation": summary["validation"],
            "seconds": time.perf_counter() - t0,
        }
        print(tag, json.dumps(results["runs"][tag]["formal_period"]), flush=True)

    t0 = time.perf_counter()
    f, daily, summary, diag, pair = question2_scenarios.run(
        dates, load, pv, prices, 365, step=25.0, time_limit=5.0, fl=fl,
        load_forecast="weekly_persist", start_index=31, initial_soc=COMMON_SOC,
    )
    results["runs"]["scenarios"] = {
        "formal_period": formal_totals(f),
        "validation": summary["validation"],
        "seconds": time.perf_counter() - t0,
    }
    print("scenarios", json.dumps(results["runs"]["scenarios"]["formal_period"]), flush=True)

    ldr_main = json.loads(MAIN_LDR_JSON.read_text(encoding="utf-8"))[0]
    results["runs"]["ldr_reference"] = {
        "note": "LDR main run already starts 2/1 at the common SOC; ledger reused, not re-run.",
        "formal_period": ldr_main["formal_period"],
    }
    (OUT / "common_start_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


def formal_totals_from_daily(daily: pd.DataFrame) -> dict:
    d = daily[pd.to_datetime(daily.date) >= FORMAL_START]
    out = {"days": int(d.date.nunique()), "intervals": int(len(d) * 144)}
    for key in ["grid_kwh", "charge_kwh", "discharge_kwh", "emergency_kwh",
                "unused_kwh", "planned_cost", "emergency_cost", "total_cost"]:
        out[key] = float(d[key].sum())
    out["initial_soc_kwh"] = float(d.soc_start_kwh.iloc[0])
    out["final_soc_kwh"] = float(d.soc_end_kwh.iloc[-1])
    out["emergency_intervals"] = 0  # not available in daily-only artifacts
    return out


def run_window_seed_grid() -> None:
    print("experiment 1: window x seed grid ...", flush=True)
    dates, load, pv, prices = load_inputs()
    fl = load_forecast_weekly_persist(load)
    windows = [14, 21, 28]
    seeds = [20250912, 20250913, 20250914]
    rows = []
    for m in windows:
        for tag, setting in [
            ("beta1", Settings(residual_days=m)),
            ("beta0", Settings(name="risk_greedy", beta=0.0, residual_days=m)),
        ]:
            frame, daily, summary = run_case(
                dates, load, pv, prices, setting, fl_override=fl
            )
            ft = formal_totals(frame)
            rows.append({
                "window": m, "seed": None, "strategy": tag,
                "planned_cost": ft["planned_cost"],
                "emergency_cost": ft["emergency_cost"],
                "total_cost": ft["total_cost"],
                "emergency_kwh": ft["emergency_kwh"],
                "initial_soc_kwh": ft["initial_soc_kwh"],
                "final_soc_kwh": ft["final_soc_kwh"],
            })
            print(f"M={m} {tag}: {ft['total_cost']:.2f}", flush=True)
        for base in seeds:
            tag = f"grid_M{m}_s{base}"
            if (OUT / f"{tag}_daily.csv").exists():
                print(f"M={m} seed={base} LDR: reuse existing artifact", flush=True)
                ft = formal_totals_from_daily(
                    pd.read_csv(OUT / f"{tag}_daily.csv", parse_dates=["date"])
                )
            else:
                rec = ldr_variant_summary(
                    settings_kwargs={},
                    run_kwargs={"residual_days": m, "seed_base": base},
                    tag=tag,
                )
                ft = rec["formal_period"]
                print(f"M={m} seed={base} LDR: {ft['total_cost']:.2f}", flush=True)
            rows.append({
                "window": m, "seed": base, "strategy": "ldr",
                "planned_cost": ft["planned_cost"],
                "emergency_cost": ft["emergency_cost"],
                "total_cost": ft["total_cost"],
                "emergency_kwh": ft["emergency_kwh"],
                "initial_soc_kwh": ft["initial_soc_kwh"],
                "final_soc_kwh": ft["final_soc_kwh"],
            })
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "window_seed_grid.csv", index=False, encoding="utf-8-sig")
    ldr = grid[grid.strategy == "ldr"]
    main = float(ldr[(ldr.window == 21) & (ldr.seed == 20250912)].total_cost.iloc[0])
    spread = float(ldr.total_cost.max() - ldr.total_cost.min())
    ranking_flips = []
    for m in windows:
        b1 = float(grid[(grid.window == m) & (grid.strategy == "beta1")].total_cost.iloc[0])
        b0 = float(grid[(grid.window == m) & (grid.strategy == "beta0")].total_cost.iloc[0])
        for base in seeds:
            v = float(ldr[(ldr.window == m) & (ldr.seed == base)].total_cost.iloc[0])
            if v >= min(b1, b0):
                ranking_flips.append({"window": m, "seed": base, "ldr": v,
                                      "beta1": b1, "beta0": b0})
    summary = {
        "main_config_total": main,
        "ldr_spread_yuan": spread,
        "ldr_spread_pct": spread / main * 100,
        "max_deviation_from_main_yuan": float(np.max(np.abs(ldr.total_cost - main))),
        "ranking_stable": len(ranking_flips) == 0,
        "ranking_flips": ranking_flips,
        "grid": grid.to_dict(orient="records"),
    }
    (OUT / "window_seed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "grid"},
                     ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["delta-only", "paired-delta-only",
                                        "common-start", "window-seed-grid", "all"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command in ("delta-only", "all"):
        run_delta_only()
    if args.command in ("paired-delta-only", "all"):
        run_paired_delta_only()
    if args.command in ("common-start", "all"):
        run_common_start()
    if args.command in ("window-seed-grid", "all"):
        run_window_seed_grid()


if __name__ == "__main__":
    main()
