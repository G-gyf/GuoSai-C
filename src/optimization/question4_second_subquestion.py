# -*- coding: utf-8 -*-
"""Question 4 last sub-question: whether to introduce ADDITIONAL forecast
releases at extra integer hours under fluctuating prices, and whether the
extended adjustment model stays feasible/solvable.

Analogue of question 3's second sub-question, but on the question-4 v2
structure (issue/execution ledgers, price-weighted quantiles, 42-day
decaying joint scenarios, regularised LDR, attachment-4 settlement):

* candidate integer hours 9:00/10:00/11:00 (between the 6:00 and 12:00
  updates, base = 6:00 issuance) and 13:00/14:00/15:00 (between the 12:00
  update and day end, base = 12:00 issuance); main candidates 10:00/14:00
  as in the question-3 second sub-question;
* at a candidate hour h (issue column m = 6h) the remaining issue columns
  m..143 are re-solved with the same two-sided price-weighted adjustment
  curve as the official 6:00/12:00 updates.  Two NEW forecast components
  can be introduced at that instant:
    - a causal PV self-forecast (latest official issuance + same-day
      dynamic residual correction, reused from the question-3 second
      sub-question forecast layer, data-identical);
    - a causal INTRAday price update (day-level multiplicative level
      correction of the day-ahead WP forecast, OLS-shrunk on the last 42
      complete days, using only prices of intervals that have started);
* six matched branches per candidate hour: B = M612 base (no operation),
  S = state-only re-optimisation (latest official PV, 0:00 price forecast),
  Fpv = self PV forecast, Fp = intraday price update only, Fboth = both,
  O = perfect-information upper bound (actual PV + actual prices,
  backtest-only);
* every branch runs causally and continuously from the shared 1 February
  SOC to 31 December; the terminal inventory is valued once at the final
  boundary, so inventory-adjusted values decompose
  V_state = C~_B - C~_S, V_pv = C~_S - C~_Fpv, V_p = C~_S - C~_Fp,
  V_both = C~_S - C~_Fboth, UB = C~_S - C~_O;
* feasibility of the extended problem is guaranteed BY CONSTRUCTION (the
  adjustment LP always admits the no-battery move a = max(R-V,0), u =
  max(V-R,0) with E constant), audited per solve (HiGHS status recorded,
  fallback = keep the previous plan if a solve ever failed), and verified
  by the question-3 v2 physics validator on every branch frame.

Run modes (mirror of question3_second_subquestion):
  python -m src.optimization.question4_second_subquestion --run --spec 10Fboth [--seed S] [--days N] [--force]
  python -m src.optimization.question4_second_subquestion --report [--seeds ...] [--include-sensitivity]
  python -m src.optimization.question4_second_subquestion --build-price
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.question3_self_forecasts import block_bootstrap_stat
from src.optimization.question2 import (
    ETA,
    EMIN,
    ROOT,
    T,
    solve_plan,
)
from src.optimization.question3 import execute_segment, solve_adjustment
from src.optimization.question3_second_subquestion import (
    HOUR_BASE_K,
    load_self_curves,
)
from src.optimization.q4_v2_common import (
    V2Settings,
    decay_weights,
    plan_horizon,
    price_weighted_quantile,
    scenario_mean_price,
    scenario_window,
)
from src.optimization.question4_2_v2 import (
    load_question4_v2_inputs,
    terminal_value_nu,
)
from src.optimization.question4_3_v2 import (
    FRAME_COLUMNS,
    _calibrate,
    _record,
    adjustment_curve_v2,
    build_q4_3_bundle,
    pv_issue_horizon,
    validate_question3_v2,
)
from src.data_pipeline.question3_forecasts import ISSUE_HOURS

OUT_ROOT = ROOT / "outputs/question4/second_subquestion"
RUNS = OUT_ROOT / "runs"
PRICE_FC_DIR = OUT_ROOT / "forecasts/price_update"
WARM_ROOT = ROOT / "outputs/question4/v2/result4-3"

MAIN_SEED = 20250912
MAIN_SEEDS = [20250912, 20250913, 20250914, 20250915, 20250916]
FORMAL_START_DAY = pd.Timestamp("2025-02-01")

# candidate hour -> issue column of the first adjustable interval (h:10)
HOUR_SLOT = {h: 6 * h for h in [9, 10, 11, 13, 14, 15]}
MAIN_HOURS = [10, 14]
SENSITIVITY_HOURS = [9, 11, 13, 15]
INFO_KIND = {"S": "state", "Fpv": "pv_only", "Fp": "price_only",
             "Fboth": "both", "O": "oracle"}

MAIN_TOTAL_Q4_3 = 14_638_588.62  # M612 formal-period total (Q4-3 v2)
PRICE_WINDOW = 42
PRICE_MIN_SAMPLES = 14
RATIO_CLIP = (0.5, 1.5)
LAM_CLIP = (0.0, 1.0)


# --------------------------------------------------------------------------
# spec parsing
# --------------------------------------------------------------------------

def parse_spec(spec: str) -> list[dict]:
    """'B' | '10Fboth' | '10Fboth+14S' -> ordered op dicts."""
    if spec == "B":
        return []
    ops = []
    for part in spec.split("+"):
        part = part.strip()
        if not part:
            continue
        i = 0
        while i < len(part) and part[i].isdigit():
            i += 1
        hour, tag = int(part[:i]), part[i:]
        if tag not in INFO_KIND:
            raise ValueError(f"unsupported info tag {tag!r} in spec {spec!r}")
        if hour not in HOUR_SLOT:
            raise ValueError(f"unsupported candidate hour {hour} in spec {spec!r}")
        ops.append({"hour": hour, "slot": HOUR_SLOT[hour], "info": INFO_KIND[tag]})
    if len(ops) > 2 or len({op["hour"] for op in ops}) != len(ops):
        raise ValueError(f"spec {spec!r} must have <= 2 distinct candidate hours")
    morning = [o for o in ops if o["hour"] < 12]
    afternoon = [o for o in ops if o["hour"] >= 12]
    if len(morning) > 1 or len(afternoon) > 1:
        raise ValueError("at most one morning and one afternoon candidate op")
    return sorted(ops, key=lambda o: o["hour"])


def spec_branch_name(spec: str, seed: int) -> str:
    name = spec.replace("+", "p")
    if seed != MAIN_SEED:
        name += f"_s{seed}"
    return name


# --------------------------------------------------------------------------
# 1. causal intraday price update layer
# --------------------------------------------------------------------------

def _price_seen_ratio(p_act: np.ndarray, F_p: np.ndarray, d: int, m: int) -> float:
    """r_d = mean realized price of intervals started by hour h over the
    day-ahead forecast for the same slots (natural slots 0..m)."""
    seen_act = p_act[d, : m + 1]
    seen_fc = F_p[d, : m + 1]
    den = float(seen_fc.mean())
    if den <= 1e-9:
        return 1.0
    return float(seen_act.mean() / den)


def _future_ratio(p_act: np.ndarray, F_p: np.ndarray, d: int, m: int) -> float:
    """mean actual / mean day-ahead over the remaining natural slots m+1..143."""
    fut_act = p_act[d, m + 1 : 144]
    fut_fc = F_p[d, m + 1 : 144]
    den = float(fut_fc.mean())
    if den <= 1e-9:
        return 1.0
    return float(fut_act.mean() / den)


def fit_price_shrink(p_act: np.ndarray, F_p: np.ndarray, d: int, m: int,
                     window: int = PRICE_WINDOW,
                     min_samples: int = PRICE_MIN_SAMPLES) -> float:
    """OLS-shrunk lambda: lambda_d = clip(mean(x*y)/mean(x^2), 0, 1) from
    the last ``window`` complete days, where x = seen-ratio - 1 and
    y = future-ratio - 1.  Strictly causal: only days < d are used."""
    first = max(0, d - window)
    idx = np.arange(first, d)
    if idx.size < min_samples:
        return 0.0
    xs = np.array([_price_seen_ratio(p_act, F_p, i, m) - 1.0 for i in idx])
    ys = np.array([_future_ratio(p_act, F_p, i, m) - 1.0 for i in idx])
    den = float((xs * xs).sum())
    if den <= 1e-9:
        return 0.0
    lam = float((xs * ys).sum() / den)
    return float(np.clip(lam, LAM_CLIP[0], LAM_CLIP[1]))


def build_price_update(p_act: np.ndarray, F_p: np.ndarray, hour: int,
                       force: bool = False) -> dict:
    """Intraday price update for candidate hour h over all days.

    updated[d, j] = p_h[d, j] * clip(1 + lam_d * (r_d - 1), 0.5, 1.5)
    for issue columns j >= m = 6h (the decision row of a candidate update),
    and equal to the day-ahead issue-horizon price for j < m.  Row 0 is
    never used (1 January is a zero-plan day).  Cached to .npz.
    """
    PRICE_FC_DIR.mkdir(parents=True, exist_ok=True)
    m = HOUR_SLOT[hour]
    cache = PRICE_FC_DIR / f"upd_{hour:02d}.npz"
    if cache.exists() and not force:
        with np.load(cache) as z:
            return {"updated": z["updated"], "lam": z["lam"], "ratio": z["ratio"]}
    updated = np.full((365, T), np.nan, float)
    lam = np.zeros(365, float)
    ratio = np.ones(365, float)
    for d in range(1, 365):
        p_h = plan_horizon(F_p, d)
        lam_d = fit_price_shrink(p_act, F_p, d, m)
        r_d = _price_seen_ratio(p_act, F_p, d, m)
        mult = float(np.clip(1.0 + lam_d * (r_d - 1.0), RATIO_CLIP[0],
                             RATIO_CLIP[1]))
        row = p_h.copy()
        row[m:] = np.maximum(0.0, p_h[m:] * mult)
        updated[d] = row
        lam[d] = lam_d
        ratio[d] = r_d
    np.savez_compressed(cache, updated=updated, lam=lam, ratio=ratio)
    return {"updated": updated, "lam": lam, "ratio": ratio}


def price_update_metrics(p_act: np.ndarray, F_p: np.ndarray, hour: int,
                         upd: dict) -> dict:
    """Causal out-of-sample metrics of the intraday update on the formal
    period: remaining issue columns m..142 vs realized prices (column 143
    belongs to 2026 and is excluded from the metric window)."""
    m = HOUR_SLOT[hour]
    days = np.arange(31, 365)
    cols = np.arange(m, 143)
    base = np.stack([plan_horizon(F_p, d)[cols] for d in days])
    updt = upd["updated"][days][:, cols]
    act = np.stack([p_act[d, cols + 1] for d in days])
    mae_b = float(np.abs(base - act).mean())
    mae_u = float(np.abs(updt - act).mean())
    rmse_b = float(np.sqrt(((base - act) ** 2).mean()))
    rmse_u = float(np.sqrt(((updt - act) ** 2).mean()))
    monthly = {}
    dates = pd.DatetimeIndex([pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(d))
                              for d in days])
    for month in range(2, 13):
        sel = np.array([d.month == month for d in dates])
        if not sel.any():
            continue
        e_b = float(np.abs(base[sel] - act[sel]).mean())
        e_u = float(np.abs(updt[sel] - act[sel]).mean())
        monthly[month] = float((e_b - e_u) / e_b * 100.0) if e_b > 1e-12 else 0.0
    # block-bootstrap CI of the per-day MAE improvement (formal days)
    daily_delta = np.abs(base - act).mean(axis=1) - np.abs(updt - act).mean(axis=1)

    def stat(subset):
        return float(daily_delta[subset].mean())

    lo, hi = block_bootstrap_stat(stat, len(daily_delta), seed=MAIN_SEED)
    return {
        "candidate": f"{hour:02d}:00",
        "mae_base": round(mae_b, 6), "mae_updated": round(mae_u, 6),
        "mae_improvement_pct": round((mae_b - mae_u) / mae_b * 100.0, 2)
        if mae_b > 1e-12 else 0.0,
        "rmse_base": round(rmse_b, 6), "rmse_updated": round(rmse_u, 6),
        "rmse_improvement_pct": round((rmse_b - rmse_u) / rmse_b * 100.0, 2)
        if rmse_b > 1e-12 else 0.0,
        "monthly_mae_improvement_pct": {str(k): round(v, 2)
                                        for k, v in monthly.items()},
        "positive_months_mae": int(sum(v > 1e-9 for v in monthly.values())),
        "daily_mae_delta_mean": round(float(daily_delta.mean()), 6),
        "daily_mae_delta_ci95": [round(lo, 6), round(hi, 6)],
        "mean_lam": round(float(upd["lam"][31:365].mean()), 4),
        "lam_positive_days": int((upd["lam"][31:365] > 1e-9).sum()),
    }


def build_price_layer(force: bool = False) -> pd.DataFrame:
    """Rebuild the intraday price update for every candidate hour and write
    the causal screening table."""
    dates, load, pv, fixed, p_act, p_tpl = load_question4_v2_inputs()
    from src.optimization.q4_v2_common import build_price_forecast
    F_p = build_price_forecast(p_act, dates, "wp", fixed)
    rows = []
    for hour in HOUR_SLOT:
        upd = build_price_update(p_act, F_p, hour, force=force)
        rows.append(price_update_metrics(p_act, F_p, hour, upd))
    df = pd.DataFrame(rows)
    PRICE_FC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PRICE_FC_DIR / "price_update_screening.csv", index=False,
              encoding="utf-8-sig")
    print(df.to_string(index=False), flush=True)
    return df


def load_price_update(p_act: np.ndarray, F_p: np.ndarray, hour: int) -> dict:
    return build_price_update(p_act, F_p, hour)


# --------------------------------------------------------------------------
# 2. candidate horizons: PV / net scenarios / price scenarios
# --------------------------------------------------------------------------

def candidate_pv_horizon(bundle: dict, pv: np.ndarray, d: int, m: int, k: int,
                         info: str, self_fc=None) -> np.ndarray:
    """PV forecast over issue columns m..143 for one candidate op.

    official (state / price_only): latest issuance curve (6:00 -> k=1 for
    morning candidates, 12:00 -> k=2 for afternoon ones) plus the 24:00
    anchor for the carry column; self: Q3 second-sub-question self curve
    (night carry column = 0); oracle: actual PV (carry column = next-day
    actual, 0.0 on 31 December)."""
    fc, anchor24 = bundle["fc"], bundle["anchor24"]
    if info in ("state", "price_only"):
        return np.r_[fc[d, k, m + 1 : 144], anchor24[d, k] / 6.0]
    if info in ("pv_only", "both"):
        return np.r_[self_fc[d, m + 1 : 144], 0.0]
    if info == "oracle":
        carry = float(pv[d + 1, 0]) if d + 1 < 365 else 0.0
        return np.r_[pv[d, m + 1 : 144], carry]
    raise ValueError(info)


def candidate_scenarios(bundle: dict, pv: np.ndarray, fl_h: np.ndarray,
                        d: int, m: int, k: int, info: str, rows_i, net_point,
                        self_fc=None, self_err=None) -> np.ndarray | None:
    """(M, 144-m) net scenarios for the candidate adjustment risk curve.

    Standard residual convention (same as the Q4-3 v2 official updates):
    scenario = today's point forecast + historical (actual - forecast), so
    the self-forecast scenarios use -self_err = actual - self."""
    res_l = bundle["res_load"]
    res_pv = bundle["res_pv"]
    if info in ("state", "price_only"):
        return net_point[None, :] + (res_l[rows_i][:, m:144]
                                     - res_pv[k][rows_i][:, m:144])
    if info in ("pv_only", "both"):
        eps_self = np.c_[self_err[rows_i][:, m + 1 : 144],
                         np.zeros((len(rows_i), 1))]
        return net_point[None, :] + res_l[rows_i][:, m:144] + eps_self
    if info == "oracle":
        # deterministic actual PV; load uncertainty retained
        pv_o = candidate_pv_horizon(bundle, pv, d, m, k, "oracle")
        return fl_h[m:144][None, :] - pv_o[None, :] + res_l[rows_i][:, m:144]
    raise ValueError(info)


def candidate_price_rows(bundle: dict, p_act: np.ndarray, d: int, m: int,
                         info: str, p_h: np.ndarray, p_upd: dict | None,
                         rows_i, w, price_model: str = "wp") -> tuple:
    """(pbar (144-m,), scen_price (M, 144-m)) for the candidate adjustment.

    state/pv_only: day-ahead forecast and day-ahead residual scenarios;
    price_only/both: intraday-updated point forecast (same residual pool);
    oracle: actual prices, deterministic across scenarios."""
    res_price = bundle["res_price"][price_model]
    horizon_act = np.r_[p_act[d, m + 1 : 144],
                        p_act[d + 1, 0] if d + 1 < 365 else p_h[143]]
    if info in ("state", "pv_only"):
        scen_p = np.maximum(0.0, p_h[m:144][None, :] + res_price[rows_i][:, m:144])
        return scenario_mean_price(scen_p, w), scen_p
    if info in ("price_only", "both"):
        p_tau = p_upd["updated"][d, m:144]
        scen_p = np.maximum(0.0, p_tau[None, :] + res_price[rows_i][:, m:144])
        return scenario_mean_price(scen_p, w), scen_p
    if info == "oracle":
        scen_p = np.tile(horizon_act, (len(rows_i), 1))
        return horizon_act, scen_p
    raise ValueError(info)


def _solve_adjustment_safe(q0_m, risk, pbar, initial, nu):
    """solve_adjustment with an explicit feasibility fallback: if the LP
    ever failed (numerical infeasibility), the previous plan stays locked
    and the event is recorded instead of aborting the year."""
    try:
        a, ep, _ = solve_adjustment(q0_m, risk, pbar, initial, nu)
        return a, ep, "optimal", ""
    except Exception as exc:  # RuntimeError from linprog failure
        return None, None, "failed", str(exc)


# --------------------------------------------------------------------------
# 3. adapted Q4-3 v2 day loop with candidate operations
# --------------------------------------------------------------------------

def run_q4_3_candidate_range(dates, load, pv, p_act, bundle, settings: V2Settings,
                             start_idx: int, initial_energy: float, limit: int,
                             price_model: str, prev_q0=None, prev_qA=None,
                             prev_ref=None, prev_risk143=0.0, ops=None):
    """run_q4_3_range (M612) extended with candidate forecast operations.

    ``ops`` = parse_spec output (<= 1 morning + <= 1 afternoon candidate).
    Empty ops reproduce run_q4_3_range bit-for-bit (guards are inert).
    Every candidate adjustment LP status is recorded; a failed solve keeps
    the previously locked plan (feasible fallback, counted, never silent).
    """
    ops = list(ops or [])
    op_morning = next((o for o in ops if o["hour"] < 12), None)
    op_afternoon = next((o for o in ops if o["hour"] >= 12), None)

    fl = bundle["fl"]
    F_p = bundle["price"][price_model]
    res_l = bundle["res_load"]
    res_pv = bundle["res_pv"]
    res_price = bundle["res_price"][price_model]
    fc = bundle["fc"]
    anchor24 = bundle["anchor24"]
    if prev_q0 is None:
        prev_q0 = np.zeros(T)
    if prev_qA is None:
        prev_qA = np.zeros(T)
    if prev_ref is None:
        prev_ref = np.full(T, EMIN)
    prev_q0 = np.asarray(prev_q0, float)
    prev_qA = np.asarray(prev_qA, float)
    prev_ref = np.asarray(prev_ref, float)

    self_layers = {}
    price_layers = {}
    for op in ops:
        if op["info"] in ("pv_only", "both") and op["hour"] not in self_layers:
            fc_s, err_s = load_self_curves(op["hour"], HOUR_BASE_K[op["hour"]])
            self_layers[op["hour"]] = (fc_s, err_s)
        if op["info"] in ("price_only", "both") and op["hour"] not in price_layers:
            price_layers[op["hour"]] = load_price_update(p_act, F_p, op["hour"])

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
        fl_h = plan_horizon(fl, d)
        p_h = plan_horizon(F_p, d)
        first, M = scenario_window(d, settings.window)
        w = decay_weights(d, first, settings.decay_tau) if M > 0 else None
        rows_i = np.arange(first, d) if M > 0 else np.array([], int)
        nu = terminal_value_nu(F_p, d, settings.terminal_value_rule)
        diag = {"calibrations": [], "residual_count": M, "candidate_ops": []}
        deltas = np.zeros(4)
        lambdas = np.zeros(4)

        # ---- 0:00 plan (issue j = 0..143, price-weighted Q80) ----
        pv_h0 = pv_issue_horizon(bundle, d, 0)
        net_h0 = fl_h - pv_h0
        scen0 = price0 = None
        if M > 0:
            scen0 = net_h0[None, :] + (res_l[rows_i] - res_pv[0][rows_i])
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
        # stage-1 calibration over t = 0..35 (M612 protocol)
        q1 = np.r_[prev_qA[143], qA[0:35]]
        ref1 = np.r_[prev_ref[143], ref[0:35]]
        forecast1 = np.r_[f0, net_h0[0:35]]
        scen1 = price1 = None
        if M > 0:
            scen1 = np.c_[f0 + (res_l[rows_i][:, 143] - res_pv[0][rows_i][:, 143]),
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
        q0_exec0 = np.r_[prev_q0[143], q0[0:35]]
        qA_exec0 = np.r_[prev_qA[143], qA[0:35]]
        ref_exec0 = np.r_[prev_ref[143], ref[0:35]]
        seg0 = execute_segment(actual_net[:36], qA_exec0, ref_exec0,
                               p_settle[:36], energy, 0.0,
                               np.array([deltas[0], lambdas[0]]))
        initial_soc = float(seg0["soc_start_kwh"][0])
        energy = float(seg0["soc_end_kwh"][-1])
        rows = []
        for t in range(36):
            r = _record(
                d, date, t, dates[d - 1] if t == 0 else date,
                143 if t == 0 else t - 1,
                load[d], pv[d], p_settle, fl[d],
                fc_exec[t], forecast_exec[t], risk_exec[t], residual_count,
                q0_exec0[t], qA_exec0[t], ref_exec0[t], seg0, t,
                deltas[0], lambdas[0], 0.0, calibration_used)
            rows.append(r)
            day_cost += r[32]
        risk143 = risk0[143]

        # ---- 6:00 adjustment (issue j = 36..143) ----
        a2 = float((actual_net[:36] - forecast_exec[:36]).mean())
        pv_h1 = pv_issue_horizon(bundle, d, 1)
        net_h1 = fl_h[36:144] - pv_h1
        scen1a = price1a = None
        if M > 0:
            scen1a = net_h1[None, :] + (res_l[rows_i][:, 36:144]
                                        - res_pv[1][rows_i][:, 36:144])
            price1a = np.maximum(0.0, p_h[36:144][None, :] + res_price[rows_i][:, 36:144])
        risk1 = adjustment_curve_v2(net_h1, scen1a, price1a, w, q0[36:144], settings)
        pbar1 = scenario_mean_price(price1a, w) if M > 0 else np.maximum(p_h[36:144], 0.0)
        a6, ep6, _ = solve_adjustment(q0[36:144], risk1, pbar1, energy, nu)
        qA[36:144] = a6
        ref[36:144] = ep6
        q2 = np.r_[q0[35], a6[0:35]]
        ref2 = np.r_[E_p0[35], ep6[0:35]]
        forecast2 = np.r_[net_h0[35], net_h1[0:35]]
        scen2 = price2 = None
        if M > 0:
            scen2 = np.c_[
                net_h0[35] + (res_l[rows_i][:, 35] - res_pv[0][rows_i][:, 35]),
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

        # ---- execute stage 2, split at the morning candidate if present ----
        end1 = 72 if op_morning is None else int(op_morning["slot"]) + 1
        seg1 = execute_segment(actual_net[36:end1], qA[35 : end1 - 1],
                               ref[35 : end1 - 1], p_settle[36:end1], energy,
                               a2, np.array([deltas[1], lambdas[1]]))
        energy = float(seg1["soc_end_kwh"][-1])
        for t in range(36, end1):
            r = _record(
                d, date, t, date, t - 1, load[d], pv[d], p_settle, fl[d],
                fc_exec[t], forecast_exec[t], risk_exec[t], residual_count,
                q0[t - 1], qA[t - 1], ref[t - 1], seg1, t - 36,
                deltas[1], lambdas[1], a2, calibration_used)
            rows.append(r)
            day_cost += r[32]

        if op_morning is not None:
            m = int(op_morning["slot"])
            info = op_morning["info"]
            k = HOUR_BASE_K[op_morning["hour"]]
            self_fc = self_layers.get(op_morning["hour"], (None, None))[0]
            self_err = self_layers.get(op_morning["hour"], (None, None))[1]
            pv_c = candidate_pv_horizon(bundle, pv, d, m, k, info,
                                        self_fc=self_fc)
            net_c = fl_h[m:144] - pv_c
            scen_c = candidate_scenarios(
                bundle, pv, fl_h, d, m, k, info, rows_i, net_c,
                self_fc=self_fc, self_err=self_err) if M > 0 else None
            if M > 0:
                pbar_c, scen_pc = candidate_price_rows(
                    bundle, p_act, d, m, info, p_h,
                    price_layers.get(op_morning["hour"]), rows_i, w,
                    price_model=price_model)
            else:
                scen_pc = None
                if info == "oracle":
                    pbar_c = np.r_[p_act[d, m + 1 : 144],
                                   p_act[d + 1, 0] if d + 1 < 365 else p_h[143]]
                elif info in ("price_only", "both"):
                    pbar_c = np.maximum(
                        price_layers[op_morning["hour"]]["updated"][d, m:144], 0.0)
                else:
                    pbar_c = np.maximum(p_h[m:144], 0.0)
            risk_c = adjustment_curve_v2(net_c, scen_c, scen_pc, w,
                                         q0[m:144], settings)
            a_c, ep_c, lp_status, lp_message = _solve_adjustment_safe(
                q0[m:144], risk_c, pbar_c, energy, nu)
            if lp_status == "optimal":
                qA[m:144] = a_c
                ref[m:144] = ep_c
                fc_exec[m + 1 : 73] = pv_c[0 : 72 - m]
                forecast_exec[m + 1 : 73] = fl[d, m + 1 : 73] - fc_exec[m + 1 : 73]
                risk_exec[m + 1 : 73] = risk_c[0 : 72 - m]
            diag["candidate_ops"].append({
                "update_slot": m, "hour": op_morning["hour"], "info": info,
                "lp_status": lp_status, "lp_message": lp_message,
                "fallback": lp_status != "optimal"})
            seg1b = execute_segment(actual_net[m + 1 : 72], qA[m:71], ref[m:71],
                                    p_settle[m + 1 : 72], energy, a2,
                                    np.array([deltas[1], lambdas[1]]))
            energy = float(seg1b["soc_end_kwh"][-1])
            for t in range(m + 1, 72):
                r = _record(
                    d, date, t, date, t - 1, load[d], pv[d], p_settle, fl[d],
                    fc_exec[t], forecast_exec[t], risk_exec[t], residual_count,
                    q0[t - 1], qA[t - 1], ref[t - 1], seg1b, t - (m + 1),
                    deltas[1], lambdas[1], a2, calibration_used)
                rows.append(r)
                day_cost += r[32]

        # ---- 12:00 adjustment (issue j = 72..143) ----
        a3 = float((actual_net[:72] - forecast_exec[:72]).mean())
        pv_h2 = pv_issue_horizon(bundle, d, 2)
        net_h2 = fl_h[72:144] - pv_h2
        scen2a = price2a = None
        if M > 0:
            scen2a = net_h2[None, :] + (res_l[rows_i][:, 72:144]
                                        - res_pv[2][rows_i][:, 72:144])
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
                net_h1[35] + (res_l[rows_i][:, 71] - res_pv[1][rows_i][:, 71]),
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
        if op_morning is None:
            fc_exec[72:108] = np.r_[fc[d, 1, 72], fc[d, 2, 73:108]]
            risk_exec[72:108] = np.r_[risk1[35], risk2[0:35]]
        else:
            # column 71 (slot 72) keeps the morning-candidate values
            fc_exec[73:108] = fc[d, 2, 73:108]
            risk_exec[73:108] = risk2[0:35]
        forecast_exec[72:108] = fl[d, 72:108] - fc_exec[72:108]

        # ---- execute stage 3, split at the afternoon candidate if present ----
        if op_morning is not None:
            # slot 72 alone: column 71 is morning-candidate-locked
            seg2 = execute_segment(actual_net[72:73], qA[71:72], ref[71:72],
                                   p_settle[72:73], energy, a3,
                                   np.array([deltas[2], lambdas[2]]))
            energy = float(seg2["soc_end_kwh"][-1])
            r = _record(
                d, date, 72, date, 71, load[d], pv[d], p_settle, fl[d],
                fc_exec[72], forecast_exec[72], risk_exec[72], residual_count,
                q0[71], qA[71], ref[71], seg2, 0,
                deltas[2], lambdas[2], a3, calibration_used)
            rows.append(r)
            day_cost += r[32]
            s3_start = 73
        else:
            s3_start = 72
        end2 = 108 if op_afternoon is None else int(op_afternoon["slot"]) + 1
        seg2 = execute_segment(actual_net[s3_start:end2],
                               qA[s3_start - 1 : end2 - 1],
                               ref[s3_start - 1 : end2 - 1],
                               p_settle[s3_start:end2], energy, a3,
                               np.array([deltas[2], lambdas[2]]))
        energy = float(seg2["soc_end_kwh"][-1])
        for t in range(s3_start, end2):
            r = _record(
                d, date, t, date, t - 1, load[d], pv[d], p_settle, fl[d],
                fc_exec[t], forecast_exec[t], risk_exec[t], residual_count,
                q0[t - 1], qA[t - 1], ref[t - 1], seg2, t - s3_start,
                deltas[2], lambdas[2], a3, calibration_used)
            rows.append(r)
            day_cost += r[32]

        if op_afternoon is not None:
            m = int(op_afternoon["slot"])
            info = op_afternoon["info"]
            k = HOUR_BASE_K[op_afternoon["hour"]]
            self_fc = self_layers.get(op_afternoon["hour"], (None, None))[0]
            self_err = self_layers.get(op_afternoon["hour"], (None, None))[1]
            pv_c = candidate_pv_horizon(bundle, pv, d, m, k, info,
                                        self_fc=self_fc)
            net_c = fl_h[m:144] - pv_c
            scen_c = candidate_scenarios(
                bundle, pv, fl_h, d, m, k, info, rows_i, net_c,
                self_fc=self_fc, self_err=self_err) if M > 0 else None
            if M > 0:
                pbar_c, scen_pc = candidate_price_rows(
                    bundle, p_act, d, m, info, p_h,
                    price_layers.get(op_afternoon["hour"]), rows_i, w,
                    price_model=price_model)
            else:
                scen_pc = None
                if info == "oracle":
                    pbar_c = np.r_[p_act[d, m + 1 : 144],
                                   p_act[d + 1, 0] if d + 1 < 365 else p_h[143]]
                elif info in ("price_only", "both"):
                    pbar_c = np.maximum(
                        price_layers[op_afternoon["hour"]]["updated"][d, m:144],
                        0.0)
                else:
                    pbar_c = np.maximum(p_h[m:144], 0.0)
            risk_c = adjustment_curve_v2(net_c, scen_c, scen_pc, w,
                                         q0[m:144], settings)
            a_c, ep_c, lp_status, lp_message = _solve_adjustment_safe(
                q0[m:144], risk_c, pbar_c, energy, nu)
            if lp_status == "optimal":
                qA[m:144] = a_c
                ref[m:144] = ep_c
                fc_exec[m + 1 : 144] = pv_c[0 : 143 - m]
                forecast_exec[m + 1 : 144] = fl[d, m + 1 : 144] - fc_exec[m + 1 : 144]
                risk_exec[m + 1 : 144] = risk_c[0 : 143 - m]
            diag["candidate_ops"].append({
                "update_slot": m, "hour": op_afternoon["hour"], "info": info,
                "lp_status": lp_status, "lp_message": lp_message,
                "fallback": lp_status != "optimal"})
            seg2b = execute_segment(actual_net[m + 1 : 108], qA[m:107], ref[m:107],
                                    p_settle[m + 1 : 108], energy, a3,
                                    np.array([deltas[2], lambdas[2]]))
            energy = float(seg2b["soc_end_kwh"][-1])
            for t in range(m + 1, 108):
                r = _record(
                    d, date, t, date, t - 1, load[d], pv[d], p_settle, fl[d],
                    fc_exec[t], forecast_exec[t], risk_exec[t], residual_count,
                    q0[t - 1], qA[t - 1], ref[t - 1], seg2b, t - (m + 1),
                    deltas[2], lambdas[2], a3, calibration_used)
                rows.append(r)
                day_cost += r[32]

        if op_afternoon is None:
            # M612 without afternoon candidate: stage-4 columns locked at 12:00
            fc_exec[108:144] = fc[d, 2, 108:144]
            forecast_exec[108:144] = fl[d, 108:144] - fc_exec[108:144]
            risk_exec[108:144] = risk2[35:71]
        else:
            # slots 108..143 (cols 107..142) locked by the afternoon candidate
            forecast_exec[108:144] = fl[d, 108:144] - fc_exec[108:144]

        # ---- execute stage 4 ----
        a4 = float((actual_net[:108] - forecast_exec[:108]).mean())
        seg3 = execute_segment(actual_net[108:144], qA[107:143], ref[107:143],
                               p_settle[108:144], energy, a4,
                               np.array([deltas[3], lambdas[3]]))
        energy = float(seg3["soc_end_kwh"][-1])
        for t in range(108, 144):
            r = _record(
                d, date, t, date, t - 1, load[d], pv[d], p_settle, fl[d],
                fc_exec[t], forecast_exec[t], risk_exec[t], residual_count,
                q0[t - 1], qA[t - 1], ref[t - 1], seg3, t - 108,
                deltas[3], lambdas[3], a4, calibration_used)
            rows.append(r)
            day_cost += r[32]
        records.extend(rows)

        issue_q0.append([date, *q0.tolist()])
        issue_qA.append([date, *qA.tolist()])
        issue_ref.append([date, *ref.tolist()])
        prev_q0, prev_qA, prev_ref = q0, qA, ref
        prev_risk143 = risk143
        diag["initial_soc_kwh"] = initial_soc
        diag["final_soc_kwh"] = float(energy)
        diag["calibration_seconds"] = float(
            sum(c.get("seconds", 0.0) for c in diag["calibrations"]))
        diag["day_seconds"] = time.perf_counter() - day_started
        diagnostics.append(diag)
        if (d - start_idx + 1) % 30 == 0 or d + 1 == limit:
            print(f"Q4-2nd[{settings.name}/{','.join(o['info'] for o in ops)}] "
                  f"{d - start_idx + 1}/{limit - start_idx} days "
                  f"date={date.date()} cost={day_cost:,.2f}", flush=True)

    frame = pd.DataFrame.from_records(records, columns=FRAME_COLUMNS)
    validation = validate_question3_v2(frame)
    issue_q0_df = pd.DataFrame(issue_q0, columns=["date", *[f"g{j}" for j in range(T)]])
    issue_qA_df = pd.DataFrame(issue_qA, columns=["date", *[f"g{j}" for j in range(T)]])
    issue_ref_df = pd.DataFrame(issue_ref, columns=["date", *[f"e{j}" for j in range(T)]])
    diag = pd.DataFrame(diagnostics)
    return frame, issue_q0_df, issue_qA_df, issue_ref_df, diag, validation


# --------------------------------------------------------------------------
# 4. branch runner (one spec per run)
# --------------------------------------------------------------------------

def warmup_carry() -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Shared Q4-3 v2 warm-up: 2/1 SOC and the 1/31 carry arrays."""
    warm_q0 = pd.read_csv(WARM_ROOT / "warmup_issue_q0.csv", parse_dates=["date"])
    warm_qA = pd.read_csv(WARM_ROOT / "warmup_issue_qA.csv", parse_dates=["date"])
    warm_ref = pd.read_csv(WARM_ROOT / "warmup_issue_ref.csv", parse_dates=["date"])
    feb1_soc = float(json.loads((WARM_ROOT / "warmup_summary.json").read_text(
        encoding="utf-8"))["feb1_soc_kwh"])

    def _row(mat, col):
        sub = mat[mat.date == pd.Timestamp("2025-01-31")]
        return float(sub[col].iloc[0]) if len(sub) else 0.0

    carry_q0 = np.array([_row(warm_q0, f"g{j}") for j in range(T)])
    carry_qA = np.array([_row(warm_qA, f"g{j}") for j in range(T)])
    carry_ref = np.array([_row(warm_ref, f"e{j}") for j in range(T)])
    return feb1_soc, carry_q0, carry_qA, carry_ref


def chosen_price_settings() -> tuple[str, float]:
    preeval = json.loads((ROOT / "outputs/question4/v2/pre_evaluation/"
                          "pre_evaluation.json").read_text(encoding="utf-8"))
    return preeval["chosen_price_model"], preeval["chosen_gamma"]


def run_branch(spec: str, seed: int, days: int, out_dir: Path,
               force: bool = False, keep_schedule: bool = False) -> Path:
    name = spec_branch_name(spec, seed)
    out = out_dir / name
    summary_file = out / "summary.json"
    if summary_file.exists() and not force:
        print(f"{name}: exists, skipped", flush=True)
        return out
    out.mkdir(parents=True, exist_ok=True)

    dates, load, pv, fixed, p_act, p_tpl = load_question4_v2_inputs()
    bundle = build_q4_3_bundle(load, pv, p_act, dates, fixed)
    feb1_soc, carry_q0, carry_qA, carry_ref = warmup_carry()
    model_star, gamma_star = chosen_price_settings()
    ops = parse_spec(spec)
    settings = V2Settings(name="M612", price_model=model_star, gamma=gamma_star,
                          search_seed=seed)
    frame, q0d, qAd, refd, diag, validation = run_q4_3_candidate_range(
        dates, load, pv, p_act, bundle, settings, 31, feb1_soc,
        len(dates) if days >= 365 else 31 + days, model_star,
        prev_q0=carry_q0, prev_qA=carry_qA, prev_ref=carry_ref, ops=ops)
    formal = frame[frame.date >= FORMAL_START_DAY].copy()
    if keep_schedule:
        frame.to_csv(out / "schedule.csv", index=False, encoding="utf-8-sig")
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
    q0d.to_csv(out / "issue_q0.csv", index=False, encoding="utf-8-sig")
    qAd.to_csv(out / "issue_qA.csv", index=False, encoding="utf-8-sig")
    refd.to_csv(out / "issue_ref.csv", index=False, encoding="utf-8-sig")
    diag.to_csv(out / "daily_parameters.csv", index=False, encoding="utf-8-sig")

    lp_statuses = {}
    if "candidate_ops" in diag.columns:
        for _, row in diag.iterrows():
            if isinstance(row["candidate_ops"], list):
                for op in row["candidate_ops"]:
                    key = f"{op['hour']}:00/{op['info']}"
                    lp_statuses.setdefault(key, {}).setdefault(
                        op["lp_status"], 0)
                    lp_statuses[key][op["lp_status"]] += 1
    period = {
        "days": int(formal["date"].nunique()),
        "total_cost": float(formal.total_cost.sum()),
        "retained_cost": float(formal.retained_cost.sum()),
        "down_cost": float(formal.down_cost.sum()),
        "up_cost": float(formal.up_cost.sum()),
        "emergency_cost": float(formal.emergency_cost.sum()),
        "emergency_kwh": float(formal.emergency_kwh.sum()),
        "down_kwh": float(formal.down_kwh.sum()),
        "up_kwh": float(formal.up_kwh.sum()),
        "unused_kwh": float(formal.unused_kwh.sum()),
        "final_soc_kwh": float(formal.soc_end_kwh.iloc[-1]),
        "start_soc_kwh": float(formal.soc_start_kwh.iloc[0]),
    }
    meta = {
        "spec": spec, "seed": seed, "branch_name": name,
        "candidate_ops": [{"hour": op["hour"], "slot": op["slot"],
                           "info": op["info"]} for op in ops],
        "feb1_soc_kwh": feb1_soc,
        "validation": validation,
        "formal_period": period,
        "candidate_lp_statuses": lp_statuses,
        "run_seconds": float(diag.calibration_seconds.sum()),
        "result_days": int(formal["date"].nunique()),
    }
    (out / "summary.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print(f"{name}: total={period['total_cost']:,.2f} "
          f"emerg={period['emergency_cost']:,.2f} "
          f"end_soc={period['final_soc_kwh']:.4f}", flush=True)
    return out


# --------------------------------------------------------------------------
# 5. evaluation: inventory-adjusted decomposition, paired values, gates
# --------------------------------------------------------------------------

def nu_adj_price(p_act: np.ndarray) -> float:
    return float(p_act[31:365].min() / ETA)


def saved_m612_daily() -> pd.DataFrame:
    path = WARM_ROOT / "M612/question3_schedule.csv"
    f = pd.read_csv(path, parse_dates=["date"])
    formal = f[f.date >= FORMAL_START_DAY].copy()
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
    return daily


def branch_daily(runs_root: Path, spec: str, seed: int,
                 saved_m612: pd.DataFrame | None) -> pd.DataFrame:
    if spec == "B" and seed == MAIN_SEED and saved_m612 is not None:
        return saved_m612
    path = runs_root / spec_branch_name(spec, seed) / "daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing branch result {path}")
    return pd.read_csv(path, parse_dates=["date"])


def value_decomposition(branches: dict[str, pd.DataFrame], nu: float) -> dict:
    """Inventory-adjusted values from matched daily frames."""
    cost, start, end = {}, {}, {}
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
        ("B", "S", "V_state"),
        ("S", "Fpv", "V_forecast_pv"),
        ("S", "Fp", "V_forecast_p"),
        ("S", "Fboth", "V_forecast_both"),
        ("B", "Fboth", "V_total"),
        ("S", "O", "UB_forecast"),
    ]:
        if lhs in branches and rhs in branches:
            out[label] = round(ctilde[lhs] - ctilde[rhs], 2)
    for key in ["V_forecast_pv", "V_forecast_p", "V_forecast_both"]:
        if key in out and "UB_forecast" in out and out["UB_forecast"] > 0:
            out[f"{key}_capture"] = round(out[key] / out["UB_forecast"], 4)
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
    return (daily.total_cost.to_numpy()
            + nu * daily.soc_start_kwh.to_numpy()
            - nu * daily.soc_end_kwh.to_numpy())


def paired_values(base: pd.DataFrame, other: pd.DataFrame, nu: float,
                  kind: str) -> dict:
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


BRANCH_KEYS = ["B", "S", "Fpv", "Fp", "Fboth", "O"]
BRANCH_LABELS = {
    "B": "B 基准(M612)", "S": "S 状态重优化", "Fpv": "Fpv 自建光伏预报",
    "Fp": "Fp 日内电价更新", "Fboth": "Fboth 光伏+电价",
    "O": "O 完美信息",
}


def evaluate_candidate(runs_root: Path, hour: int, seeds: list[int],
                       saved_m612: pd.DataFrame | None, nu: float,
                       branch_keys: list[str] | None = None) -> dict:
    hour_str = f"{hour}"
    keys = list(branch_keys or BRANCH_KEYS)
    per_seed = []
    for seed in seeds:
        # Fpv/Fp decomposition branches exist only for the main seed
        seed_keys = keys if seed == MAIN_SEED else [
            k for k in keys if k in ("B", "S", "Fboth", "O")]
        specs = {key: ("B" if key == "B" else f"{hour_str}{key}")
                 for key in seed_keys}
        branches = {}
        for key, spec in specs.items():
            branches[key] = branch_daily(runs_root, spec, seed, saved_m612)
        dec = value_decomposition(branches, nu)
        pairs = {
            "V_state_daily": paired_values(branches["B"], branches["S"], nu, "B-S"),
            "V_both_daily": paired_values(branches["S"], branches["Fboth"], nu,
                                          "S-Fboth"),
            "V_total_daily": paired_values(branches["B"], branches["Fboth"], nu,
                                           "B-Fboth"),
            "UB_daily": paired_values(branches["S"], branches["O"], nu, "S-O"),
        }
        if "Fpv" in branches:
            pairs["V_pv_daily"] = paired_values(branches["S"], branches["Fpv"],
                                                nu, "S-Fpv")
        if "Fp" in branches:
            pairs["V_p_daily"] = paired_values(branches["S"], branches["Fp"],
                                               nu, "S-Fp")
        per_seed.append({"seed": seed, "decomposition": dec, "paired": pairs})
    return {"hour": hour_str, "seeds": per_seed, "seed_count": len(seeds)}


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


def decision_check(result: dict, pv_screening: pd.DataFrame,
                   price_screening: pd.DataFrame) -> dict:
    """Q4-adapted section-5.8 decision rules for one candidate hour."""
    multi_v = multi_seed_summary(result, "V_forecast_both")
    multi_ub = multi_seed_summary(result, "UB_forecast")
    multi_state = multi_seed_summary(result, "V_state")
    multi_pv = multi_seed_summary(result, "V_forecast_pv")
    multi_p = multi_seed_summary(result, "V_forecast_p")
    main_seed = next(s for s in result["seeds"] if s["seed"] == MAIN_SEED)
    dec = main_seed["decomposition"]
    pair = main_seed["paired"]["V_both_daily"]
    key = f"{result['hour'].zfill(2)}:00"
    prow = pv_screening[pv_screening.candidate == key].iloc[0]
    rrow = price_screening[price_screening.candidate == key].iloc[0]
    threshold = 0.0005 * MAIN_TOTAL_Q4_3
    capture = (100.0 * multi_v["mean"] / multi_ub["mean"]
               if multi_ub.get("mean") else None)
    em_s = dec["settlement"]["S"]["emergency_cost"]
    em_f = dec["settlement"]["Fboth"]["emergency_cost"]
    checks = {
        "pv_forecast_effective_10pct": bool(
            prow.rmse_improvement_pct >= 10 and prow.mae_improvement_pct >= 10),
        "price_update_effective": bool(
            rrow.mae_improvement_pct > 0 and rrow.rmse_improvement_pct >= 5.0
            and rrow.positive_months_mae >= 6),
        "economic_positive": bool(multi_v["mean"] > 0),
        "bootstrap_ci_lower_positive": bool(pair["bootstrap_ci95"][0] > 0),
        "stable_months_8_positive": bool(pair["positive_months"] >= 8),
        "stable_seeds_4_of_5": bool(multi_v["positive_seeds"] >= 4),
        "material_threshold": bool(multi_v["mean"] > threshold),
        "capture_rate_10pct": bool(capture is not None and capture >= 10.0),
        "emergency_not_worse": bool(em_f <= em_s * 1.05 + 1.0),
        "state_value": round(multi_state.get("mean", 0.0), 2),
        "forecast_pv_value": round(multi_pv.get("mean", 0.0), 2),
        "forecast_p_value": round(multi_p.get("mean", 0.0), 2),
        "forecast_both_value": round(multi_v.get("mean", 0.0), 2),
        "forecast_both_min": round(multi_v.get("min", 0.0), 2),
        "ub_mean": round(multi_ub.get("mean", 0.0), 2),
        "capture_rate_pct": round(capture, 1) if capture is not None else None,
        "threshold_yuan": round(threshold, 2),
        "emergency_cost_S": em_s,
        "emergency_cost_Fboth": em_f,
        "emergency_kwh_S": dec["settlement"]["S"]["emergency_kwh"],
        "emergency_kwh_Fboth": dec["settlement"]["Fboth"]["emergency_kwh"],
        "price_mae_improvement_pct": rrow.mae_improvement_pct,
        "price_positive_months_mae": rrow.positive_months_mae,
    }
    core = [checks["pv_forecast_effective_10pct"], checks["price_update_effective"],
            checks["economic_positive"], checks["bootstrap_ci_lower_positive"],
            checks["stable_months_8_positive"], checks["stable_seeds_4_of_5"],
            checks["material_threshold"], checks["capture_rate_10pct"],
            checks["emergency_not_worse"]]
    checks["enter_main_strategy"] = bool(all(core))
    if checks["enter_main_strategy"]:
        checks["partial_verdict"] = "enter"
    elif (checks["economic_positive"] and checks["stable_seeds_4_of_5"]
          and checks["material_threshold"]):
        checks["partial_verdict"] = "statistical_value_below_full_gate"
    elif checks["economic_positive"]:
        checks["partial_verdict"] = "statistical_only"
    else:
        checks["partial_verdict"] = "reject"
    return checks


def load_pv_screening() -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs/question3/second_subquestion/forecasts/"
                       "candidate_screening.csv")


def load_price_screening() -> pd.DataFrame:
    df = pd.read_csv(PRICE_FC_DIR / "price_update_screening.csv")
    import ast
    df["daily_mae_delta_ci95"] = df["daily_mae_delta_ci95"].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) else s)
    return df


def feasibility_audit(runs_root: Path) -> dict:
    """Aggregate the feasibility audit across finished branch runs.

    For every run: candidate adjustment LP statuses (optimal counts), the
    number of fallback events, and the physics validator outcomes.  Also
    checks the B-branch identity against the saved Q4-3 v2 M612 result."""
    audit = {"runs": {}, "fallback_total": 0, "lp_failed_total": 0,
             "lp_optimal_total": 0, "physics_failures": [],
             "max_balance_error_kwh": 0.0, "max_state_error_kwh": 0.0,
             "b_identity": None}
    for path in sorted(runs_root.glob("*/summary.json")):
        name = path.parent.name
        meta = json.loads(path.read_text(encoding="utf-8"))
        statuses = meta.get("candidate_lp_statuses", {})
        failed = optimal = 0
        for key, st in statuses.items():
            failed += sum(v for s, v in st.items() if s != "optimal")
            optimal += sum(v for s, v in st.items() if s == "optimal")
        audit["lp_failed_total"] += failed
        audit["lp_optimal_total"] += optimal
        audit["fallback_total"] += failed  # a failed LP triggers the fallback
        val = meta.get("validation", {})
        if not val.get("passed", True):
            audit["physics_failures"].append(name)
        audit["max_balance_error_kwh"] = max(
            audit["max_balance_error_kwh"],
            float(val.get("max_balance_error_kwh", 0.0)))
        audit["max_state_error_kwh"] = max(
            audit["max_state_error_kwh"],
            float(val.get("max_state_error_kwh", 0.0)))
        audit["runs"][name] = {
            "candidate_lp_statuses": statuses,
            "min_soc_kwh": val.get("min_soc_kwh"),
            "max_soc_kwh": val.get("max_soc_kwh"),
            "max_balance_error_kwh": val.get("max_balance_error_kwh"),
            "simultaneous_charge_discharge":
                val.get("simultaneous_charge_discharge"),
            "emergency_while_charging": val.get("emergency_while_charging"),
            "fallback_total": failed,
        }
    # B branch (main seed) vs the locked Q4-3 v2 M612 run
    b_path = runs_root / "B" / "daily.csv"
    if b_path.exists():
        saved = saved_m612_daily()
        b_run = pd.read_csv(b_path, parse_dates=["date"])
        m = saved.merge(b_run, on="date", suffixes=("_saved", "_run"))
        audit["b_identity"] = {
            "total_cost_diff": round(float(
                m.total_cost_saved.sum() - m.total_cost_run.sum()), 6),
            "max_daily_cost_diff": round(float(
                (m.total_cost_saved - m.total_cost_run).abs().max()), 6),
            "max_daily_soc_diff": round(float(
                (m.soc_end_kwh_saved - m.soc_end_kwh_run).abs().max()), 6),
            "days": int(len(m)),
        }
    return audit


# --------------------------------------------------------------------------
# 6. report
# --------------------------------------------------------------------------

def run_report(runs_root: Path, seeds: list[int],
               include_sensitivity: bool = False) -> dict:
    _, load, pv, fixed, p_act, p_tpl = load_question4_v2_inputs()
    nu = nu_adj_price(p_act)
    pv_screen = load_pv_screening()
    price_screen = load_price_screening()
    saved_m612 = saved_m612_daily()
    report = {"nu": nu, "seeds": seeds, "main_total_q4_3": MAIN_TOTAL_Q4_3}
    for hour in MAIN_HOURS:
        result = evaluate_candidate(runs_root, hour, seeds, saved_m612, nu)
        report[f"hour_{hour}"] = result
        report[f"hour_{hour}"]["decision"] = decision_check(
            result, pv_screen, price_screen)
    if include_sensitivity:
        report["sensitivity"] = {}
        for hour in SENSITIVITY_HOURS:
            result = evaluate_candidate(runs_root, hour, [MAIN_SEED],
                                        saved_m612, nu,
                                        branch_keys=["B", "S", "Fboth", "O"])
            report["sensitivity"][f"hour_{hour}"] = result
            report["sensitivity"][f"hour_{hour}"]["decision"] = decision_check(
                result, pv_screen, price_screen)
    # combined 10:00+14:00 increments (F = Fboth)
    report["combo"] = {}
    for spec in ["10Fboth+14Fboth", "10Fboth+14S", "10S+14Fboth"]:
        name = spec_branch_name(spec, MAIN_SEED)
        path = runs_root / name / "daily.csv"
        if path.exists():
            d = pd.read_csv(path, parse_dates=["date"])
            report["combo"][spec] = {
                "cash_cost": round(float(d.total_cost.sum()), 2),
                "end_soc_kwh": round(float(d.soc_end_kwh.iloc[-1]), 6),
                "adjusted_cost": round(
                    float(d.total_cost.sum())
                    - nu * float(d.soc_end_kwh.iloc[-1]), 2),
            }
    if {"10Fboth+14Fboth", "10Fboth+14S", "10S+14Fboth"} <= set(report["combo"]):
        a = report["combo"]
        report["combo"]["V14_given_10"] = round(
            a["10Fboth+14S"]["adjusted_cost"]
            - a["10Fboth+14Fboth"]["adjusted_cost"], 2)
        report["combo"]["V10_given_14"] = round(
            a["10S+14Fboth"]["adjusted_cost"]
            - a["10Fboth+14Fboth"]["adjusted_cost"], 2)

    # report-time nu robustness: re-weight main-seed frames with f*nu
    report["nu_robustness"] = {}
    for factor in [0.0, 0.5, 1.0, 1.5]:
        per_hour = {}
        for hour in MAIN_HOURS:
            costs = {}
            for key in ["B", "S", "Fboth"]:
                spec = "B" if key == "B" else f"{hour}{key}"
                d = branch_daily(runs_root, spec, MAIN_SEED, saved_m612)
                costs[key] = (float(d.total_cost.sum())
                              - factor * nu * float(d.soc_end_kwh.iloc[-1]))
            per_hour[f"hour_{hour}"] = {
                "V_state": round(costs["B"] - costs["S"], 2),
                "V_forecast_both": round(costs["S"] - costs["Fboth"], 2),
            }
        report["nu_robustness"][f"nu_factor_{factor:g}"] = per_hour
    report["feasibility"] = feasibility_audit(runs_root)
    return report


def write_report_md(report: dict, out: Path) -> None:
    lines = ["# 问题四最后一小问：是否新增预报与求解可行性（波动电价）", "",
             f"- 主种子：{MAIN_SEED}；复核种子：{report['seeds']}",
             f"- 期末库存价值系数 ν = {report['nu']:.5f} 元/kWh（正式期实际最低价/η）",
             f"- M612 主方案正式期总费（Q4-3 v2）＝ {report['main_total_q4_3']:,.2f} 元",
             "", "## 一、预测层筛选结果", "",
             "光伏自建预报沿用问题三第二小问预测层（数据完全相同），筛选表：", ""]
    pv_screen = load_pv_screening()
    lines.append("| 候选 | 角色 | 综合相关 | RMSE 改善% | MAE 改善% | 剩余时数 |")
    lines.append("|---|---|---|---:|---:|---:|")
    for _, r in pv_screen.iterrows():
        lines.append(f"| {r.candidate} | {r.role} | {r.correlation} | "
                     f"{r.rmse_improvement_pct} | {r.mae_improvement_pct} | "
                     f"{r.remaining_hours} |")
    price_screen = load_price_screening()
    lines += ["", "日内电价更新预报（因果：仅用截至候选时点已开始的区间价格，"
              "42 日滚动 OLS 收缩；指标在候选时点后剩余列上计算；判定门：MAE 改善>0、"
              "RMSE 改善≥5%、正月份≥6）：", "",
              "| 候选 | MAE 改善% | RMSE 改善% | 日MAE改善95%CI | 月MAE正月份 | 平均λ |",
              "|---|---:|---:|---:|---:|---:|"]
    for _, r in price_screen.iterrows():
        lines.append(f"| {r.candidate} | {r.mae_improvement_pct} | "
                     f"{r.rmse_improvement_pct} | "
                     f"[{r.daily_mae_delta_ci95[0]:.4f}, {r.daily_mae_delta_ci95[1]:.4f}]"
                     f" | {r.positive_months_mae} | {r.mean_lam} |")
    lines += ["", "## 二、经济回测（B/S/Fpv/Fp/Fboth/O，全年连续、库存调整）", ""]
    for hour in MAIN_HOURS:
        res = report[f"hour_{hour}"]
        dec = res["decision"]
        main_seed = next(s for s in res["seeds"] if s["seed"] == MAIN_SEED)
        mdec = main_seed["decomposition"]
        lines += [f"### {hour}:00", "",
                  "**分支现金与库存（正式期）**：",
                  "| 分支 | 现金总费 | 期末SOC | 库存调整成本 |",
                  "|---|---:|---:|---:|"]
        for key in BRANCH_KEYS:
            lines.append(f"| {BRANCH_LABELS[key]} | {mdec['cash_cost'][key]:,.2f} | "
                         f"{mdec['end_soc_kwh'][key]:,.2f} | "
                         f"{mdec['adjusted_cost'][key]:,.2f} |")
        lines += ["", "**结算分解（主种子，元 / kWh）**：",
                  "| 分支 | 保留 | 下调违约 | 上调新增 | 紧急费 | 紧急电量 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for key in BRANCH_KEYS:
            s = mdec["settlement"][key]
            lines.append(f"| {key} | {s['retained_cost']:,.2f} | "
                         f"{s['down_cost']:,.2f} | {s['up_cost']:,.2f} | "
                         f"{s['emergency_cost']:,.2f} | "
                         f"{s['emergency_kwh']:,.2f} |")
        lines += ["", "**库存调整价值（元）**：",
                  f"- V_state（状态重优化）＝ {dec['state_value']:,.2f}",
                  f"- V_forecast_pv（新增光伏预报）＝ {dec['forecast_pv_value']:,.2f}",
                  f"- V_forecast_p（新增电价更新预报）＝ {dec['forecast_p_value']:,.2f}",
                  f"- V_forecast_both（光伏+电价）＝ {dec['forecast_both_value']:,.2f}",
                  f"  （多种子 min {dec['forecast_both_min']:,.2f}，"
                  f"{multi_seed_summary(res, 'V_forecast_both')['positive_seeds']}/5 正），"
                  f"UB＝ {dec['ub_mean']:,.2f}，捕获率 {dec['capture_rate_pct']}%",
                  f"- 实质门槛（0.05%×M612）＝ {dec['threshold_yuan']:,.2f} 元",
                  f"- 紧急购电：S {dec['emergency_cost_S']:,.2f} 元/"
                  f"{dec['emergency_kwh_S']:,.2f} kWh → Fboth {dec['emergency_cost_Fboth']:,.2f} 元/"
                  f"{dec['emergency_kwh_Fboth']:,.2f} kWh",
                  ""]
        for label in ["V_both_daily", "V_pv_daily", "V_p_daily",
                      "V_state_daily", "V_total_daily", "UB_daily"]:
            p = main_seed["paired"][label]
            lines.append(f"- {label}（逐日潜势配对）：值 {p['value']:,.2f}，"
                         f"95% 块自助 CI [{p['bootstrap_ci95'][0]:,.2f}, "
                         f"{p['bootstrap_ci95'][1]:,.2f}]，"
                         f"正天数 {p['positive_days']}/{p['positive_days'] + p['negative_days']}，"
                         f"正月份 {p['positive_months']}/11")
        verdict_cn = {
            "enter": "✅ 进入主策略（M612+该时点新增预报）",
            "statistical_value_below_full_gate":
                "⚠️ 有统计价值、多种子为正且超实质门槛,但未过全部门槛(见括号)",
            "statistical_only": "仅有统计价值,未达实质门槛",
            "reject": "❌ 不新增该时点预报",
        }[dec["partial_verdict"]]
        lines += ["",
                  f"**判定**：{verdict_cn}",
                  f"（光伏预报有效 {dec['pv_forecast_effective_10pct']}，"
                  f"电价更新有效 {dec['price_update_effective']}，"
                  f"经济为正 {dec['economic_positive']}，"
                  f"CI 下界>0 {dec['bootstrap_ci_lower_positive']}，"
                  f"月稳定≥8 {dec['stable_months_8_positive']}，"
                  f"种子≥4/5 {dec['stable_seeds_4_of_5']}，"
                  f"过门槛 {dec['material_threshold']}，"
                  f"捕获率≥10% {dec['capture_rate_10pct']}，"
                  f"紧急购电不恶化 {dec['emergency_not_worse']}）",
                  ""]
    if "sensitivity" in report:
        lines += ["", "## 三、敏感性：相邻整数时点（9/11/13/15，单种子）", "",
                  "| 时点 | V_state | V_pv | V_p | V_both | UB | 捕获率% |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for hour, res in report["sensitivity"].items():
            dec = res["decision"]
            lines.append(
                f"| {hour.replace('hour_', '')}:00 | {dec['state_value']:,.2f} | "
                f"{dec['forecast_pv_value']:,.2f} | {dec['forecast_p_value']:,.2f} | "
                f"{dec['forecast_both_value']:,.2f} | {dec['ub_mean']:,.2f} | "
                f"{dec['capture_rate_pct']} |")
    if "combo" in report and "V14_given_10" in report["combo"]:
        lines += ["", "## 四、组合方案 10:00+14:00（F=Fboth）", "",
                  f"- V14|10 = C̃(10F+14S) − C̃(10F+14F) = "
                  f"{report['combo']['V14_given_10']:,.2f} 元",
                  f"- V10|14 = C̃(10S+14F) − C̃(10F+14F) = "
                  f"{report['combo']['V10_given_14']:,.2f} 元",
                  f"- 联合方案 10F+14F 库存调整成本 "
                  f"{report['combo']['10Fboth+14Fboth']['adjusted_cost']:,.2f} 元",
                  ""]
    if "nu_robustness" in report:
        lines += ["", "## 五、ν 稳健性（报告期重新计价，轨迹不变；V_forecast_both 元）", "",
                  "| ν 因子 | 10:00 V_state | 10:00 V_both | 14:00 V_state | 14:00 V_both |",
                  "|---|---:|---:|---:|---:|"]
        for factor, per_hour in report["nu_robustness"].items():
            v10 = per_hour["hour_10"]
            v14 = per_hour["hour_14"]
            lines.append(
                f"| {factor} | {v10['V_state']:,.2f} | {v10['V_forecast_both']:,.2f} | "
                f"{v14['V_state']:,.2f} | {v14['V_forecast_both']:,.2f} |")
    if "feasibility" in report:
        feas = report["feasibility"]
        lines += ["", "## 六、求解可行性审计（“确保求解可行”）", "",
                  "**构造可行性**：候选时点调整 LP 的可行域非空——取 c=d=0、"
                  "a=max(R−V,0)、u=max(V−R,0) 时能量平衡成立、储能恒在"
                  "[1200,10800] 内、p=(q0−a)⁺、q=(a−q0)⁺ 恒可行，故任何候选时点、"
                  "任何信息分支的调整问题按构造必然可行；执行层由紧急购电兜底，"
                  "亦恒可行。**数值审计**（全部已完成回测）：",
                  f"- 候选时点调整 LP 求解：最优 **{feas['lp_optimal_total']}** 次，"
                  f"失败 **{feas['lp_failed_total']}** 次，触发保底（沿用已锁定计划）"
                  f"**{feas['fallback_total']}** 次；",
                  f"- 全部分支物理校验：最大能量平衡误差 "
                  f"{feas['max_balance_error_kwh']:.3e} kWh、最大 SOC 递推误差 "
                  f"{feas['max_state_error_kwh']:.3e} kWh，无同时充放电、"
                  f"无“紧急购电同时充电”，物理校验未通过分支数 "
                  f"{len(feas['physics_failures'])}；",
                  f"- 因果性：电价更新只使用候选时点及以前已开始区间的实际价，"
                  f"光伏自建预报只使用候选时点及以前的观测，均为按构造成立并经测试验证。"]
        if feas.get("b_identity"):
            bi = feas["b_identity"]
            lines += [
                f"- B 分支与已锁定 Q4-3 v2 M612 主结果逐日对照（{bi['days']} 天）："
                f"总费差 {bi['total_cost_diff']:,.6f} 元、单日最大费差 "
                f"{bi['max_daily_cost_diff']:,.6f} 元、单日最大期末 SOC 差 "
                f"{bi['max_daily_soc_diff']:,.6f} kWh（应全为 0，验证扩展实现"
                f"未改变主策略路径）。"]
    lines += ["", "## 结论口径", "",
              "按 Q4 版 §5.8 判定规则（光伏预报有效、电价更新有效、经济为正、"
              "CI 下界>0、月稳定≥8、种子≥4/5、过实质门槛、捕获率≥10%、紧急购电不恶化）"
              "逐条核对后的结论见各候选的“判定”行。新增预报的价值按"
              "B/S/Fpv/Fp/Fboth/O 六分支拆分：状态重优化价值、光伏预报增量价值、"
              "电价更新增量价值分别报告，避免把重优化节费误归因于预报。"
              "求解可行性由构造证明＋数值审计双重保证（见“六”）。",
              ""]
    out.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--build-price", action="store_true")
    parser.add_argument("--spec", default="B")
    parser.add_argument("--seed", type=int, default=MAIN_SEED)
    parser.add_argument("--seeds", type=int, nargs="*", default=MAIN_SEEDS)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--include-sensitivity", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-schedule", action="store_true")
    parser.add_argument("--runs-root", type=Path, default=RUNS)
    args = parser.parse_args()

    if args.build_price:
        build_price_layer(force=args.force)
        return
    if args.run:
        run_branch(args.spec, args.seed, args.days, args.runs_root,
                   force=args.force, keep_schedule=args.keep_schedule)
    elif args.report:
        report = run_report(args.runs_root, args.seeds,
                            include_sensitivity=args.include_sensitivity)
        out = args.runs_root / "report.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                  default=float), encoding="utf-8")
        write_report_md(report, args.runs_root / "问题四第二小问_实施与结果.md")
        print(json.dumps(report, ensure_ascii=False, indent=2,
                         default=float)[:6000], flush=True)


if __name__ == "__main__":
    main()
