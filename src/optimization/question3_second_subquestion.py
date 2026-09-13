"""Question 3 second sub-question: economic value of candidate integer-hour forecasts.

Implements sections 5-6 of
``outputs/question3/current/paper/问题三第二小问_新增整数时点预报完整方案.md``:

* for each candidate hour (main: 10:00, 14:00; sensitivity: 9/11/13/15)
  four matched branches fork from the same M612 path and the same SOC at
  the candidate instant:
  B = M612 base (no operation), S = state-only re-optimisation on the
  latest official issuance, F = re-optimisation on the causal self
  forecast, O = re-optimisation on the actual PV (perfect-information
  upper bound; backtest-only);
* all branches run causally and continuously from the shared February
  opening inventory to 31 December; the terminal inventory is valued once
  at the final boundary (nu * E_Dec31, nu = min price / eta);
* inventory-adjusted values V_state = C~_B - C~_S,
  V_forecast = C~_S - C~_F, V_total = C~_B - C~_F, UB = C~_S - C~_O and
  the capture rate R = V_forecast / UB;
* daily potential-adjusted costs Cbar_d = C_d + nu*E_{d,0} - nu*E_{d,24}
  telescope across days, so daily/monthly paired values and block
  bootstrap confidence intervals are computed on Cbar differences;
* the section-5.8 decision rules decide whether an extra integer-hour
  forecast should enter the main strategy.

Run modes:
  python -m src.optimization.question3_second_subquestion --run --spec 10F [--seed S] [--days N] [--nu-factor F] [--recalibrate]
  python -m src.optimization.question3_second_subquestion --report [--seeds ...] [--include-sensitivity]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.question3_self_forecasts import (
    CANDIDATES,
    block_bootstrap_stat,
    build_self_forecasts,
    pv_hourly_kw,
)
from src.data_pipeline.question3_forecasts import (
    ISSUE_HOURS,
    build_issuance_curves,
    load_hourly_issuances,
    load_pv_actuals,
)
from src.optimization.question2 import (
    ETA,
    load_forecast_weekly_persist,
    load_inputs,
)
from src.optimization.question3 import (
    Q3Settings,
    run_strategy,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "outputs/question3/second_subquestion"
RUNS = OUT_ROOT / "runs"
WARMUP = OUT_ROOT / "warmup"
CURVES = OUT_ROOT / "forecasts/curves"

MAIN_SEED = 20250912
MAIN_SEEDS = [20250912, 20250913, 20250914, 20250915, 20250916]
FORMAL_START_DAY = pd.Timestamp("2025-02-01")

# candidate hour -> (slot, base issuance index)
HOUR_SLOT = {h: 6 * h for h in [9, 10, 11, 13, 14, 15]}
HOUR_BASE_K = {9: 1, 10: 1, 11: 1, 13: 2, 14: 2, 15: 2}

MAIN_TOTAL_Q3 = 13_848_036.20  # M612 formal-period total (fixed-price Q3)


# --------------------------------------------------------------------------
# spec parsing
# --------------------------------------------------------------------------

def parse_spec(spec: str) -> list[dict]:
    """'10F' | '14S' | 'B' | '10F+14S' ... -> op dicts (slot, info)."""
    if spec == "B":
        return []
    ops = []
    for part in spec.split("+"):
        part = part.strip()
        if not part:
            continue
        hour = int(part[:-1])
        info = {"S": "state", "F": "self", "O": "oracle"}[part[-1]]
        if hour not in HOUR_SLOT:
            raise ValueError(f"unsupported candidate hour {hour} in spec {spec!r}")
        ops.append({"slot": HOUR_SLOT[hour], "info": info, "hour": hour})
    return ops


def spec_branch_name(spec: str, seed: int, nu_factor: float,
                     recalibrate: bool) -> str:
    name = spec.replace("+", "p")
    if recalibrate:
        name += "_recal"
    if nu_factor != 1.0:
        name += f"_nu{nu_factor:g}"
    if seed != MAIN_SEED:
        name += f"_s{seed}"
    return name


# --------------------------------------------------------------------------
# self-forecast curve cache
# --------------------------------------------------------------------------

def load_self_curves(hour: int, base_k: int) -> tuple[np.ndarray, np.ndarray]:
    """(self_fc (365,144), self_err (365,144)) for one candidate hour."""
    CURVES.mkdir(parents=True, exist_ok=True)
    cache = CURVES / f"self_{hour:02d}.npz"
    if cache.exists():
        with np.load(cache) as z:
            return z["curve_kwh"], z["err_kwh"]
    pv = load_pv_actuals(ROOT)
    pv_h = pv_hourly_kw(pv)
    hourly = load_hourly_issuances(ROOT)
    next_hour = 12 if base_k == 1 else 18
    spec = build_self_forecasts(hour, base_k, next_hour, pv, pv_h, hourly)
    np.savez_compressed(cache, curve_kwh=spec["curve_kwh"],
                        err_kwh=spec["err_kwh"])
    return spec["curve_kwh"], spec["err_kwh"]


def build_ops(ops: list[dict]) -> list[dict]:
    """Attach self-forecast arrays to the self ops."""
    out = []
    for op in ops:
        op = dict(op)
        if op["info"] == "self":
            fc, err = load_self_curves(op["hour"], HOUR_BASE_K[op["hour"]])
            op["self_fc"] = fc
            op["self_err"] = err
        out.append(op)
    return out


# --------------------------------------------------------------------------
# warm-up cache (common January opening inventory per seed / nu factor)
# --------------------------------------------------------------------------

def warmup_soc(seed: int, nu_factor: float, dates, load, pv, prices, fl, fc,
               force: bool = False) -> float:
    WARMUP.mkdir(parents=True, exist_ok=True)
    cache = WARMUP / f"warmup_seed{seed}_nu{nu_factor:g}.json"
    if cache.exists() and not force:
        return float(json.loads(cache.read_text(encoding="utf-8"))["feb1_soc_kwh"])
    settings = Q3Settings(strategy="M0", search_seed=seed, nu_factor=nu_factor)
    frame, _, _, _ = run_strategy(dates, load, pv, prices, fl, fc, settings,
                                  limit=31)
    soc = float(frame.soc_end_kwh.iloc[-1])
    cache.write_text(json.dumps(
        {"seed": seed, "nu_factor": nu_factor, "feb1_soc_kwh": soc},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"warm-up done: seed={seed} nu={nu_factor:g} -> 2/1 SOC = {soc:.6f}",
          flush=True)
    return soc


# --------------------------------------------------------------------------
# branch runner
# --------------------------------------------------------------------------

def run_branch(spec: str, seed: int, nu_factor: float, recalibrate: bool,
               days: int, out_dir: Path, force: bool = False) -> Path:
    name = spec_branch_name(spec, seed, nu_factor, recalibrate)
    out = out_dir / name
    summary_file = out / "summary.json"
    if summary_file.exists() and not force:
        print(f"{name}: exists, skipped", flush=True)
        return out
    out.mkdir(parents=True, exist_ok=True)

    dates, load, pv, prices = load_inputs()
    fl = load_forecast_weekly_persist(load)
    fc = build_issuance_curves(ROOT)
    soc = warmup_soc(seed, nu_factor, dates, load, pv, prices, fl, fc)
    ops = build_ops(parse_spec(spec))
    for op in ops:
        # First round: no candidate-time LDR recalibration for ANY branch, so
        # the tuning value never contaminates the forecast value.  O uses the
        # same 12:00-locked controller as S/F; its plan layer carries perfect
        # PV information plus the same load scenarios.
        op["recalibrate"] = bool(recalibrate)
    settings = Q3Settings(strategy="M612", search_seed=seed, nu_factor=nu_factor)
    frame, diag, pair, summary = run_strategy(
        dates, load, pv, prices, fl, fc, settings,
        start_idx=31, initial_energy=soc, candidate_ops=ops,
        limit=days if days < 365 else None,
    )
    formal = frame[frame.date >= FORMAL_START_DAY].copy()
    frame.to_csv(out / "schedule_with_warmup.csv", index=False,
                 encoding="utf-8-sig")
    formal.to_csv(out / "schedule.csv", index=False, encoding="utf-8-sig")
    daily = (formal.groupby("date")
             .agg(q0_kwh=("q0_kwh", "sum"), qA_kwh=("qA_kwh", "sum"),
                  down_kwh=("down_kwh", "sum"), up_kwh=("up_kwh", "sum"),
                  charge_kwh=("charge_kwh", "sum"),
                  discharge_kwh=("discharge_kwh", "sum"),
                  emergency_kwh=("emergency_kwh", "sum"),
                  unused_kwh=("unused_kwh", "sum"),
                  retained_cost=("retained_cost", "sum"),
                  down_cost=("down_cost", "sum"), up_cost=("up_cost", "sum"),
                  emergency_cost=("emergency_cost", "sum"),
                  total_cost=("total_cost", "sum"),
                  soc_start_kwh=("soc_start_kwh", "first"),
                  soc_end_kwh=("soc_end_kwh", "last")).reset_index())
    daily.to_csv(out / "daily.csv", index=False, encoding="utf-8-sig")
    diag.to_csv(out / "daily_parameters.csv", index=False, encoding="utf-8-sig")

    meta = {
        "spec": spec, "seed": seed, "nu_factor": nu_factor,
        "recalibrate": recalibrate,
        "branch_name": name,
        "candidate_ops": [{"slot": op["slot"], "info": op["info"],
                           "hour": op["hour"]} for op in ops],
        "warmup_feb1_soc_kwh": soc,
        "validation": summary["validation"],
        "formal_period": summary["formal_period"],
        "run_seconds": summary["seconds"],
        "result_days": summary["result_days"],
    }
    (out / "summary.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{name}: total={summary['formal_period']['total_cost']:,.2f} "
          f"emerg={summary['formal_period']['emergency_cost']:,.2f} "
          f"end_soc={summary['formal_period']['final_soc_kwh']:.4f}", flush=True)
    return out


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def nu_price(prices: np.ndarray) -> float:
    return float(np.asarray(prices).min() / ETA)


def branch_daily(runs_root: Path, spec: str, seed: int, nu_factor: float,
                 recalibrate: bool, saved_m612: Path | None) -> pd.DataFrame:
    name = spec_branch_name(spec, seed, nu_factor, recalibrate)
    if spec == "B" and seed == MAIN_SEED and nu_factor == 1.0 and not recalibrate \
            and saved_m612 is not None and saved_m612.exists():
        return pd.read_csv(saved_m612, parse_dates=["date"])
    path = runs_root / name / "daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing branch result {path}")
    return pd.read_csv(path, parse_dates=["date"])


def value_decomposition(branches: dict[str, pd.DataFrame], nu: float) -> dict:
    """Inventory-adjusted values from four matched daily frames."""
    cost = {}
    end = {}
    start = {}
    for key, daily in branches.items():
        cost[key] = float(daily.total_cost.sum())
        start[key] = float(daily.soc_start_kwh.iloc[0])
        end[key] = float(daily.soc_end_kwh.iloc[-1])
    ctilde = {k: cost[k] - nu * end[k] for k in branches}
    out = {
        "cash_cost": {k: round(cost[k], 2) for k in branches},
        "start_soc_kwh": {k: round(start[k], 6) for k in branches},
        "end_soc_kwh": {k: round(end[k], 6) for k in branches},
        "adjusted_cost": {k: round(ctilde[k], 2) for k in branches},
        "nu": nu,
    }
    for lhs, rhs, label in [
        ("B", "S", "V_state"), ("S", "F", "V_forecast"),
        ("B", "F", "V_total"), ("S", "O", "UB_forecast"),
    ]:
        if lhs in branches and rhs in branches:
            out[label] = round(ctilde[lhs] - ctilde[rhs], 2)
    if "V_forecast" in out and "UB_forecast" in out and out["UB_forecast"] != 0:
        out["capture_rate"] = round(out["V_forecast"] / out["UB_forecast"], 4)
    # settlement decomposition of each branch
    comp = {}
    for key, daily in branches.items():
        comp[key] = {
            "retained_cost": round(float(daily.retained_cost.sum()), 2),
            "down_cost": round(float(daily.down_cost.sum()), 2),
            "up_cost": round(float(daily.up_cost.sum()), 2),
            "emergency_cost": round(float(daily.emergency_cost.sum()), 2),
            "down_kwh": round(float(daily.down_kwh.sum()), 2),
            "up_kwh": round(float(daily.up_kwh.sum()), 2),
            "emergency_kwh": round(float(daily.emergency_kwh.sum()), 2),
            "unused_kwh": round(float(daily.unused_kwh.sum()), 2),
        }
    out["settlement"] = comp
    return out


def potential_daily(daily: pd.DataFrame, nu: float) -> np.ndarray:
    """Cbar_d = C_d + nu*E_{d,0} - nu*E_{d,24}."""
    return (daily.total_cost.to_numpy()
            + nu * daily.soc_start_kwh.to_numpy()
            - nu * daily.soc_end_kwh.to_numpy())


def paired_values(base: pd.DataFrame, other: pd.DataFrame, nu: float,
                  kind: str) -> dict:
    """Paired daily Cbar differences (base - other) with bootstrap CI."""
    dates = base.date.to_numpy()
    diff = potential_daily(base, nu) - potential_daily(other, nu)
    n = len(diff)

    def stat(subset):
        return float(diff[subset].mean())

    lo, hi = block_bootstrap_stat(stat, n, seed=MAIN_SEED)
    monthly = {}
    for month in range(2, 13):
        sel = np.array([d.month == month for d in pd.DatetimeIndex(dates)])
        monthly[month] = float(diff[sel].sum())
    return {
        "kind": kind,
        "value": float(diff.sum()),
        "mean_per_day": float(diff.mean()),
        "bootstrap_ci95": [round(lo, 2), round(hi, 2)],
        "positive_days": int((diff > 1e-9).sum()),
        "negative_days": int((diff < -1e-9).sum()),
        "monthly_sum": {str(m): round(v, 2) for m, v in monthly.items()},
        "positive_months": int(sum(v > 1e-9 for v in monthly.values())),
    }


def evaluate_candidate(runs_root: Path, hour: int, seeds: list[int],
                       nu_factor: float, recalibrate: bool,
                       saved_m612: Path | None, nu: float) -> dict:
    hour_str = f"{hour}"
    per_seed = []
    for seed in seeds:
        specs = {key: ("B" if key == "B" else f"{hour_str}{key}")
                 for key in "BSFO"}
        branches = {}
        for key, spec in specs.items():
            daily = branch_daily(runs_root, spec, seed, nu_factor, recalibrate,
                                 saved_m612)
            branches[key] = daily
        dec = value_decomposition(branches, nu)
        pairs = {
            "V_state_daily": paired_values(branches["B"], branches["S"], nu,
                                           "B-S"),
            "V_forecast_daily": paired_values(branches["S"], branches["F"], nu,
                                              "S-F"),
            "V_total_daily": paired_values(branches["B"], branches["F"], nu,
                                           "B-F"),
            "UB_daily": paired_values(branches["S"], branches["O"], nu, "S-O"),
        }
        per_seed.append({"seed": seed, "decomposition": dec, "paired": pairs})
    return {"hour": hour_str, "seeds": per_seed,
            "seed_count": len(seeds), "nu_factor": nu_factor,
            "recalibrate": recalibrate}


def multi_seed_summary(result: dict, key: str) -> dict:
    values = [s["decomposition"][key] for s in result["seeds"]
              if key in s["decomposition"]]
    if not values:
        return {}
    return {"mean": round(float(np.mean(values)), 2),
            "min": round(float(np.min(values)), 2),
            "max": round(float(np.max(values)), 2),
            "positive_seeds": int(sum(v > 0 for v in values)),
            "values": [round(float(v), 2) for v in values]}


def decision_check(result: dict, screening: pd.DataFrame,
                   monthly: pd.DataFrame, nu: float) -> dict:
    """Section-5.8 decision rules for one candidate."""
    hour = int(result["hour"])
    multi_v = multi_seed_summary(result, "V_forecast")
    multi_ub = multi_seed_summary(result, "UB_forecast")
    multi_state = multi_seed_summary(result, "V_state")
    main_seed = next(s for s in result["seeds"] if s["seed"] == MAIN_SEED)
    dec = main_seed["decomposition"]
    pair = main_seed["paired"]["V_forecast_daily"]
    row = screening[screening.candidate == f"{hour:02d}:00"].iloc[0]
    mrow = monthly[monthly.candidate == f"{hour:02d}:00"]
    months_rmse_pos = int((mrow.rmse_delta > 0).sum())
    months_mae_pos = int((mrow.mae_delta > 0).sum())
    threshold = 0.0005 * MAIN_TOTAL_Q3
    capture = (100.0 * multi_v["mean"] / multi_ub["mean"]
               if multi_ub.get("mean") else None)
    checks = {
        "forecast_effective_rmse_mae_10pct": bool(
            row.rmse_improvement_pct >= 10 and row.mae_improvement_pct >= 10),
        "monthly_stability_7_of_11": bool(
            months_rmse_pos >= 7 and months_mae_pos >= 7),
        "months_rmse_positive": months_rmse_pos,
        "months_mae_positive": months_mae_pos,
        "economic_positive": bool(multi_v["mean"] > 0),
        "bootstrap_ci_lower_positive": bool(pair["bootstrap_ci95"][0] > 0),
        "stable_months_8_positive": bool(pair["positive_months"] >= 8),
        "stable_seeds_4_of_5": bool(multi_v["positive_seeds"] >= 4),
        "material_threshold": bool(multi_v["mean"] > threshold),
        "capture_rate_10pct": bool(capture is not None and capture >= 10.0),
        "state_value": round(multi_state.get("mean", 0.0), 2),
        "forecast_value_mean": round(multi_v.get("mean", 0.0), 2),
        "forecast_value_min": round(multi_v.get("min", 0.0), 2),
        "ub_mean": round(multi_ub.get("mean", 0.0), 2),
        "capture_rate_pct": round(capture, 1) if capture is not None else None,
        "threshold_yuan": round(threshold, 2),
        "emergency_cost_B": dec["settlement"]["B"]["emergency_cost"],
        "emergency_cost_F": dec["settlement"]["F"]["emergency_cost"],
        "emergency_kwh_B": dec["settlement"]["B"]["emergency_kwh"],
        "emergency_kwh_F": dec["settlement"]["F"]["emergency_kwh"],
    }
    core = [checks["forecast_effective_rmse_mae_10pct"],
            checks["economic_positive"], checks["bootstrap_ci_lower_positive"],
            checks["stable_months_8_positive"], checks["stable_seeds_4_of_5"],
            checks["material_threshold"], checks["capture_rate_10pct"]]
    checks["enter_main_strategy"] = bool(all(core))
    if checks["enter_main_strategy"]:
        checks["partial_verdict"] = "enter"
    elif (checks["forecast_effective_rmse_mae_10pct"]
          and checks["economic_positive"]
          and checks["stable_seeds_4_of_5"]
          and checks["material_threshold"]):
        # section-5.8 fallback: statistical value but not every gate passed
        checks["partial_verdict"] = "statistical_value_below_full_gate"
    elif checks["forecast_effective_rmse_mae_10pct"] and checks["economic_positive"]:
        checks["partial_verdict"] = "statistical_only"
    else:
        checks["partial_verdict"] = "reject"
    return checks


def load_screening(root: Path) -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs/question3/second_subquestion/forecasts/"
                       "candidate_screening.csv")


def run_report(runs_root: Path, seeds: list[int], nu_factor: float = 1.0,
               recalibrate: bool = False,
               include_sensitivity: bool = False,
               saved_m612: Path | None = None) -> dict:
    _, _, _, prices = load_inputs()
    nu = nu_price(prices)
    screening = load_screening(ROOT)
    monthly = pd.read_csv(ROOT / "outputs/question3/second_subquestion/"
                          "forecasts/monthly_robustness.csv")
    report = {"nu": nu, "seeds": seeds, "nu_factor": nu_factor}
    for hour in [10, 14]:
        result = evaluate_candidate(runs_root, hour, seeds, nu_factor,
                                    recalibrate, saved_m612, nu)
        report[f"hour_{hour}"] = result
        report[f"hour_{hour}"]["decision"] = decision_check(result, screening,
                                                            monthly, nu)
    if include_sensitivity:
        report["sensitivity"] = {}
        for hour in [9, 11, 13, 15]:
            result = evaluate_candidate(runs_root, hour, [MAIN_SEED],
                                        nu_factor, False, saved_m612, nu)
            report["sensitivity"][f"hour_{hour}"] = result
            report["sensitivity"][f"hour_{hour}"]["decision"] = decision_check(
                result, screening, monthly, nu)

    # ---- combined 10:00+14:00 increments (section 6.2) ----
    report["combo"] = {}
    for spec in ["10F+14F", "10F+14S", "10S+14F"]:
        name = spec_branch_name(spec, MAIN_SEED, nu_factor, False)
        path = runs_root / name / "daily.csv"
        if path.exists():
            d = pd.read_csv(path, parse_dates=["date"])
            report["combo"][spec] = {
                "cash_cost": round(float(d.total_cost.sum()), 2),
                "end_soc_kwh": round(float(d.soc_end_kwh.iloc[-1]), 6),
                "adjusted_cost": round(
                    float(d.total_cost.sum())
                    - nu_factor * nu * float(d.soc_end_kwh.iloc[-1]), 2),
            }
    if {"10F+14F", "10F+14S", "10S+14F"} <= set(report["combo"]):
        a = report["combo"]
        report["combo"]["V14_given_10"] = round(
            a["10F+14S"]["adjusted_cost"] - a["10F+14F"]["adjusted_cost"], 2)
        report["combo"]["V10_given_14"] = round(
            a["10S+14F"]["adjusted_cost"] - a["10F+14F"]["adjusted_cost"], 2)

    # ---- nu sensitivity (section 5.5.3): re-run planning with nu' = f*nu ----
    report["nu_sensitivity"] = {}
    for factor in [0.0, 0.5, 1.0, 1.5]:
        per_hour = {}
        for hour in [10, 14]:
            costs = {}
            for spec in ["B", "S", "F"]:
                name = spec_branch_name(f"{hour}{spec}", MAIN_SEED, factor,
                                        False)
                try:
                    d = branch_daily(runs_root, f"{hour}{spec}", MAIN_SEED,
                                     factor, False, saved_m612)
                except FileNotFoundError:
                    costs[spec] = None
                    continue
                costs[spec] = (float(d.total_cost.sum())
                               - factor * nu * float(d.soc_end_kwh.iloc[-1]))
            per_hour[f"hour_{hour}"] = {
                "V_state": (None if costs["B"] is None or costs["S"] is None
                            else round(costs["B"] - costs["S"], 2)),
                "V_forecast": (None if costs["S"] is None or costs["F"] is None
                               else round(costs["S"] - costs["F"], 2)),
                "V_total": (None if costs["B"] is None or costs["F"] is None
                            else round(costs["B"] - costs["F"], 2)),
            }
        report["nu_sensitivity"][f"nu_factor_{factor:g}"] = per_hour

    # ---- candidate-time LDR recalibration sensitivity (section 7.11) ----
    report["recal_sensitivity"] = {}
    for hour in [10, 14]:
        values = {}
        for tag, rec in [("base", False), ("recal", True)]:
            name = spec_branch_name(f"{hour}F", MAIN_SEED, nu_factor, rec)
            path = runs_root / name / "daily.csv"
            if not path.exists():
                values[tag] = None
                continue
            d = pd.read_csv(path, parse_dates=["date"])
            values[tag] = (float(d.total_cost.sum())
                           - nu_factor * nu * float(d.soc_end_kwh.iloc[-1]))
        report["recal_sensitivity"][f"hour_{hour}"] = {
            "adjusted_F_base": values["base"],
            "adjusted_F_recal": values["recal"],
            "recal_gain": (None if values["base"] is None
                           or values["recal"] is None
                           else round(values["base"] - values["recal"], 2)),
        }
    return report


def write_report_md(report: dict, out: Path) -> None:
    lines = ["# 问题三第二小问：新增整数时点预报的实施与结果",
             "",
             f"- 主种子：{MAIN_SEED}；复核种子：{report['seeds']}",
             f"- 期末库存价值系数 ν = {report['nu']:.5f} 元/kWh（min c/η）；"
             f"LP 终端项与计价统一使用 ν 因子 {report['nu_factor']:g}",
             "",
             "## 一、预测层筛选结果（正式期，整点锚点口径）", ""]
    screening = load_screening(ROOT)
    lines.append("| 候选 | 角色 | 综合相关 | RMSE 改善% | MAE 改善% | 剩余时数 |")
    lines.append("|---|---|---|---:|---:|---:|")
    for _, r in screening.iterrows():
        lines.append(f"| {r.candidate} | {r.role} | {r.correlation} | "
                     f"{r.rmse_improvement_pct} | {r.mae_improvement_pct} | "
                     f"{r.remaining_hours} |")
    lines += ["", "## 二、经济回测（B/S/F/O，全年连续、库存调整）", ""]
    for hour in [10, 14]:
        res = report[f"hour_{hour}"]
        dec = res["decision"]
        main_seed = next(s for s in res["seeds"] if s["seed"] == MAIN_SEED)
        mdec = main_seed["decomposition"]
        lines += [f"### {hour}:00", "",
                  "**分支现金与库存（正式期）**：",
                  "| 分支 | 现金总费 | 期末SOC | 库存调整成本 |",
                  "|---|---:|---:|---:|"]
        for key, label in [("B", "B 基准"), ("S", "S 状态重优化"),
                           ("F", "F 自建预报"), ("O", "O 完美信息")]:
            lines.append(f"| {label} | {mdec['cash_cost'][key]:,.2f} | "
                         f"{mdec['end_soc_kwh'][key]:,.2f} | "
                         f"{mdec['adjusted_cost'][key]:,.2f} |")
        lines += ["", "**结算分解（主种子，元 / kWh）**：",
                  "| 分支 | 保留 | 下调违约 | 上调新增 | 紧急费 | 紧急电量 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for key, label in [("B", "B"), ("S", "S"), ("F", "F"), ("O", "O")]:
            s = mdec["settlement"][key]
            lines.append(f"| {label} | {s['retained_cost']:,.2f} | "
                         f"{s['down_cost']:,.2f} | {s['up_cost']:,.2f} | "
                         f"{s['emergency_cost']:,.2f} | "
                         f"{s['emergency_kwh']:,.2f} |")
        lines += ["", f"**库存调整价值**：V_state={dec['state_value']:,.2f}，"
                  f"V_forecast={dec['forecast_value_mean']:,.2f}",
                  f"（多种子 min {dec['forecast_value_min']:,.2f}，"
                  f"{multi_seed_summary(res, 'V_forecast')['positive_seeds']}/5 正），"
                  f"UB={dec['ub_mean']:,.2f}，捕获率={dec['capture_rate_pct']}%",
                  f"实质门槛（0.05%×M612）={dec['threshold_yuan']:,.2f} 元",
                  f"紧急购电：B {dec['emergency_cost_B']:,.2f} 元/"
                  f"{dec['emergency_kwh_B']:,.2f} kWh → F {dec['emergency_cost_F']:,.2f} 元/"
                  f"{dec['emergency_kwh_F']:,.2f} kWh",
                  ""]
        for label in ["V_forecast_daily", "V_state_daily", "V_total_daily",
                      "UB_daily"]:
            p = main_seed["paired"][label]
            lines.append(f"- {label}（逐日潜势配对）：值 {p['value']:,.2f}，"
                         f"95% 块自助 CI [{p['bootstrap_ci95'][0]:,.2f}, "
                         f"{p['bootstrap_ci95'][1]:,.2f}]，"
                         f"正天数 {p['positive_days']}/{p['positive_days'] + p['negative_days']}，"
                         f"正月份 {p['positive_months']}/11")
        verdict_cn = {
            "enter": "✅ 进入主策略",
            "statistical_value_below_full_gate": "⚠️ 有统计价值、多种子为正且超实质门槛,但未过全部门槛(见括号)",
            "statistical_only": "仅有统计价值,未达实质门槛",
            "reject": "❌ 不新增该时点预报",
        }[dec["partial_verdict"]]
        lines += ["",
                  f"**判定**：{verdict_cn}",
                  f"（预测有效 {dec['forecast_effective_rmse_mae_10pct']}，"
                  f"月度稳定 {dec['months_rmse_positive']}/{dec['months_mae_positive']}，"
                  f"经济为正 {dec['economic_positive']}，"
                  f"CI 下界>0 {dec['bootstrap_ci_lower_positive']}，"
                  f"月稳定≥8 {dec['stable_months_8_positive']}，"
                  f"种子≥4/5 {dec['stable_seeds_4_of_5']}，"
                  f"过门槛 {dec['material_threshold']}，"
                  f"捕获率≥10% {dec['capture_rate_10pct']}）",
                  ""]
    if "sensitivity" in report:
        lines += ["", "## 三、敏感性：相邻整数时点（9/11/13/15，单种子）", "",
                  "| 时点 | V_state | V_forecast | UB | 捕获率% |",
                  "|---|---:|---:|---:|---:|"]
        for hour, res in report["sensitivity"].items():
            dec = res["decision"]
            lines.append(f"| {hour.replace('hour_', '')}:00 | {dec['state_value']:,.2f} | "
                         f"{dec['forecast_value_mean']:,.2f} | "
                         f"{dec['ub_mean']:,.2f} | {dec['capture_rate_pct']} |")
    if "combo" in report and "V14_given_10" in report["combo"]:
        lines += ["", "## 四、组合方案 10:00+14:00", "",
                  f"- V14|10 = C̃(10F+14S) − C̃(10F+14F) = "
                  f"{report['combo']['V14_given_10']:,.2f} 元",
                  f"- V10|14 = C̃(10S+14F) − C̃(10F+14F) = "
                  f"{report['combo']['V10_given_14']:,.2f} 元",
                  f"- 联合方案 10F+14F 库存调整成本 "
                  f"{report['combo']['10F+14F']['adjusted_cost']:,.2f} 元",
                  ""]
    if "nu_sensitivity" in report:
        lines += ["", "## 五、ν 敏感性（0 / 0.5ν / ν / 1.5ν，V_forecast 元）", "",
                  "| ν 因子 | 10:00 V_forecast | 14:00 V_forecast |",
                  "|---|---:|---:|"]
        for factor, per_hour in report["nu_sensitivity"].items():
            v10 = per_hour["hour_10"]["V_forecast"]
            v14 = per_hour["hour_14"]["V_forecast"]
            lines.append(f"| {factor} | {'' if v10 is None else f'{v10:,.2f}'} | "
                         f"{'' if v14 is None else f'{v14:,.2f}'} |")
    if "recal_sensitivity" in report:
        lines += ["", "## 六、候选时点 LDR 重校准敏感性", "",
                  "| 时点 | 不重校准 C̃(F) | 重校准 C̃(F) | 重校准增益 |",
                  "|---|---:|---:|---:|"]
        for hour, row in report["recal_sensitivity"].items():
            base = row["adjusted_F_base"]
            rec = row["adjusted_F_recal"]
            g = row["recal_gain"]
            base_s = "" if base is None else f"{base:,.2f}"
            rec_s = "" if rec is None else f"{rec:,.2f}"
            g_s = "" if g is None else f"{g:,.2f}"
            lines.append(f"| {hour} | {base_s} | {rec_s} | {g_s} |")
    lines += ["", "## 结论口径",
              "",
              "按 §5.8 判定规则逐条核对后的结论见各候选的“判定”行：",
              "预测层与全年经济回测均通过全部门槛的时点建议进入主策略",
              "（M612+10:00+14:00 视组合增量而定）；仅有统计价值或主要来自",
              "状态重优化价值的时点不新增预报。",
              ""]
    out.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--spec", default="B", help="e.g. 10F, 14S, B, 10F+14S")
    parser.add_argument("--seed", type=int, default=MAIN_SEED)
    parser.add_argument("--seeds", type=int, nargs="*", default=MAIN_SEEDS)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--nu-factor", type=float, default=1.0)
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument("--include-sensitivity", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--runs-root", type=Path, default=RUNS)
    args = parser.parse_args()

    saved_m612 = (ROOT / "outputs/question3/current/M612/question3_daily.csv")

    if args.run:
        run_branch(args.spec, args.seed, args.nu_factor, args.recalibrate,
                   args.days, args.runs_root, force=args.force)
    elif args.report:
        report = run_report(args.runs_root, args.seeds,
                            nu_factor=args.nu_factor,
                            recalibrate=args.recalibrate,
                            include_sensitivity=args.include_sensitivity,
                            saved_m612=saved_m612)
        out = args.runs_root / f"report_nu{args.nu_factor:g}.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        write_report_md(report, args.runs_root / "问题三第二小问_实施与结果.md")
        print(json.dumps(report, ensure_ascii=False, indent=2)[:4000], flush=True)


if __name__ == "__main__":
    main()
