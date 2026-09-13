"""Question 3 second sub-question: causal self-forecasts at candidate integer hours.

Implements ``outputs/question3/current/paper/问题三第二小问_新增整数时点预报完整方案.md``
sections 3-4:

* candidate integer hours 7:00-11:00 (base = 6:00 official issuance,
  operative window until 12:00) and 13:00-17:00 (base = 12:00 issuance,
  operative window until 18:00);
* self-forecast = latest official issuance + same-day dynamic residual
  correction fitted on the previous 21 complete days with ridge shrinkage,
  clipped to [0, 1.5], plus the fallback rules of section 4.2;
* PCHIP ten-minute conversion anchored at the observed value at the
  candidate hour (the observation is causal: the interval ending at the
  candidate hour has just completed);
* screening gates of section 3.3: correlation gate, causal out-of-sample
  accuracy gate (RMSE/MAE improvement >= 10% vs the official curve on the
  operative window), monthly stability gate, operability gate.

Strictly causal: the self-forecast for day d uses only days before d; the
scenario-error pool for day d uses the self-forecasts those historical days
would themselves have produced (computed from their own history).

Run ``python -m src.forecasting.question3_self_forecasts`` to rebuild the
parquet curves and all screening tables under
``outputs/question3/second_subquestion/forecasts/``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from src.data_pipeline.question3_forecasts import (
    ISSUE_HOURS,
    build_issuance_curves,
    load_hourly_issuances,
    load_pv_actuals,
)

ROOT = Path(__file__).resolve().parents[2]
T = 144
DAYS = 365
FORMAL_START = 31  # 2025-02-01 (zero-based day index)

# (hour, base issuance index, next key update hour)
MORNING = [(7, 1, 12), (8, 1, 12), (9, 1, 12), (10, 1, 12), (11, 1, 12)]
AFTERNOON = [(13, 2, 18), (14, 2, 18), (15, 2, 18), (16, 2, 18), (17, 2, 18)]
CANDIDATES = MORNING + AFTERNOON

WINDOW_DAYS = 21
MIN_SAMPLES = 14
RIDGE_FRACTION = 0.25  # lambda = RIDGE_FRACTION * sum(x^2)  ->  shrink 0.8
RHO_MAX = 1.5
MIN_CURRENT_VAR_KW2 = 1.0
VALIDATION_TAIL = 7  # last days of the training window used for the gate
BOOT_BLOCKS = 2000
BOOT_BLOCK_DAYS = 7


def pv_hourly_kw(pv: np.ndarray) -> np.ndarray:
    """(365, 24) actual PV (kW) at each hour h = 1..24 (slot 6h-1)."""
    return pv[:, np.arange(1, 25) * 6 - 1] * 6.0


def official_hourly_at(hourly: np.ndarray, d: int, k: int, h: int) -> float:
    """Official forecast (kW) of issuance k for hour h of day d (h > issue)."""
    return float(hourly[d, k, h - ISSUE_HOURS[k] - 1])


def official_errors_at(hourly: np.ndarray, pv_h: np.ndarray, d: int, k: int,
                       h: int) -> float:
    return official_hourly_at(hourly, d, k, h) - float(pv_h[d, h - 1])


def build_self_forecasts(hour: int, base_k: int, next_hour: int,
                         pv: np.ndarray, pv_h: np.ndarray,
                         hourly: np.ndarray) -> dict:
    """Self-forecasts for one candidate hour over all days.

    Returns
    -------
    dict
        ``hourly_kw``   (365, 24) corrected hourly forecasts (kW, NaN for h<=hour)
        ``curve_kwh``   (365, 144) PCHIP ten-minute curves (kWh/10min, NaN < 6*hour)
        ``err_kwh``     (365, 144) self curve minus actual PV (kWh/10min)
        ``rho``         (365, 24) fitted shrunk regression coefficients
        ``fallback``    (365,) integer fallback code
        ``fallback_text`` dict code -> description
        ``operative``   (start, end) operative slot window used for screening
    """
    pv_max = float(pv.max() * 6.0)  # data-based kW ceiling for clipping
    hourly_kw = np.full((DAYS, 24), np.nan, dtype=float)
    curve_kwh = np.full((DAYS, T), np.nan, dtype=float)
    err_kwh = np.full((DAYS, T), np.nan, dtype=float)
    rho = np.full((DAYS, 24), np.nan, dtype=float)
    fallback = np.zeros(DAYS, dtype=int)

    targets = list(range(hour + 1, 25))
    op_targets = list(range(hour + 1, min(next_hour, 24) + 1))
    grid_minutes = 10.0 * (np.arange(T) + 1)

    for d in range(1, DAYS):
        first = max(0, d - WINDOW_DAYS)
        idx = np.arange(first, d)
        n = int(idx.size)

        if n < MIN_SAMPLES:
            fallback[d] = 1  # few samples -> official forecast
            _store_official(d, hour, base_k, pv, pv_h, hourly, hourly_kw,
                            curve_kwh, err_kwh, grid_minutes)
            rho[d, :] = 0.0
            continue

        e_cur = np.array([official_errors_at(hourly, pv_h, i, base_k, hour)
                          for i in idx])
        mu_cur = float(e_cur.mean())
        x = e_cur - mu_cur
        var_x = float(np.var(x))
        if var_x < MIN_CURRENT_VAR_KW2:
            fallback[d] = 2  # low current-error variance -> official forecast
            _store_official(d, hour, base_k, pv, pv_h, hourly, hourly_kw,
                            curve_kwh, err_kwh, grid_minutes)
            rho[d, :] = 0.0
            continue

        e_fut = np.full((n, 24), np.nan)
        for h in targets:
            e_fut[:, h - 1] = [official_errors_at(hourly, pv_h, i, base_k, h)
                               for i in idx]

        valid_counts = np.isfinite(e_fut).sum(axis=0)
        mu_h = np.divide(np.nansum(e_fut, axis=0), valid_counts,
                         out=np.zeros(24, dtype=float), where=valid_counts > 0)
        y = np.nan_to_num(e_fut - mu_h)
        denom = float(np.dot(x, x))
        rho_ols = np.nan_to_num(np.dot(x, y) / denom, nan=0.0)
        rho_hat = np.clip(rho_ols / (1.0 + RIDGE_FRACTION), 0.0, RHO_MAX)
        rho_hat[rho_ols <= 0.0] = 0.0  # sign anomaly -> no correction

        # ---- rolling validation gate (operative targets, last VALIDATION_TAIL days)
        if n >= VALIDATION_TAIL and len(op_targets):
            val_idx = idx[-VALIDATION_TAIL:]
            mae_off = mae_corr = 0.0
            for i in val_idx:
                e_cur_i = official_errors_at(hourly, pv_h, int(i), base_k, hour)
                for h in op_targets:
                    off = official_errors_at(hourly, pv_h, int(i), base_k, h)
                    corr = float(mu_h[h - 1] + rho_hat[h - 1] * (e_cur_i - mu_cur))
                    mae_off += abs(off)
                    mae_corr += abs(off - corr)
            if mae_corr > mae_off:
                fallback[d] = 3  # validation window no better than official
                rho_hat[:] = 0.0
        else:
            fallback[d] = 4  # validation gate unavailable (kept coefficients)

        e_cur_d = official_errors_at(hourly, pv_h, d, base_k, hour)
        corrected = np.full(24, np.nan)
        for h in targets:
            off = official_hourly_at(hourly, d, base_k, h)
            corr = float(mu_h[h - 1] + rho_hat[h - 1] * (e_cur_d - mu_cur))
            corrected[h - 1] = float(np.clip(off - corr, 0.0, pv_max))
            rho[d, h - 1] = float(rho_hat[h - 1])
        hourly_kw[d] = corrected

        # ---- PCHIP ten-minute conversion, anchored at the observed hour value
        anchor_x = 60.0 * np.arange(hour, 25)
        anchor_y = np.r_[float(pv_h[d, hour - 1]),
                         np.maximum(corrected[hour:], 0.0)]
        interp = PchipInterpolator(anchor_x, anchor_y, extrapolate=False)
        values = np.maximum(interp(grid_minutes[6 * hour:]), 0.0) / 6.0
        curve_kwh[d, 6 * hour:] = values
        err_kwh[d, 6 * hour:] = values - pv[d, 6 * hour:]

    fallback_text = {
        0: "fitted",
        1: "few_samples_use_official",
        2: "low_current_error_variance_use_official",
        3: "validation_window_not_better_use_official",
        4: "validation_gate_unavailable",
    }
    return {
        "hour": hour,
        "base_issuance": ISSUE_HOURS[base_k],
        "next_update_hour": next_hour,
        "hourly_kw": hourly_kw,
        "curve_kwh": curve_kwh,
        "err_kwh": err_kwh,
        "rho": rho,
        "fallback": fallback,
        "fallback_text": fallback_text,
        "operative": (6 * hour, 6 * min(next_hour, 24)),
    }


def _store_official(d, hour, base_k, pv, pv_h, hourly, hourly_kw, curve_kwh,
                    err_kwh, grid_minutes):
    """Fallback: the base official issuance used verbatim (PCHIP anchored)."""
    pv_max = float(pv.max() * 6.0)
    targets = list(range(hour + 1, 25))
    for h in targets:
        hourly_kw[d, h - 1] = float(np.clip(
            official_hourly_at(hourly, d, base_k, h), 0.0, pv_max))
    anchor_x = 60.0 * np.arange(hour, 25)
    anchor_y = np.r_[float(pv_h[d, hour - 1]),
                     np.maximum(hourly_kw[d, hour:], 0.0)]
    interp = PchipInterpolator(anchor_x, anchor_y, extrapolate=False)
    values = np.maximum(interp(grid_minutes[6 * hour:]), 0.0) / 6.0
    curve_kwh[d, 6 * hour:] = values
    err_kwh[d, 6 * hour:] = values - pv[d, 6 * hour:]


# --------------------------------------------------------------------------
# screening metrics (section 3.3 gates + monthly stability + bootstrap)
# --------------------------------------------------------------------------

def pooled_correlation(hour: int, next_hour: int, pv_h: np.ndarray,
                       hourly: np.ndarray, base_k: int) -> float:
    """Pooled correlation between the current error and future errors."""
    xv, yv = [], []
    for d in range(FORMAL_START, DAYS):
        e_cur = official_errors_at(hourly, pv_h, d, base_k, hour)
        for h in range(hour + 1, min(next_hour, 24) + 1):
            xv.append(e_cur)
            yv.append(official_errors_at(hourly, pv_h, d, base_k, h))
    x = np.asarray(xv)
    y = np.asarray(yv)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def improvement_metrics(spec: dict, pv: np.ndarray, fc: np.ndarray,
                        hourly: np.ndarray, pv_h: np.ndarray,
                        days: np.ndarray | None = None) -> dict:
    """RMSE/MAE/bias improvement of the self curve vs the official curve.

    Evaluated at the hourly anchors (slot 6h-1 for h in the operative
    window [hour+1, next_update]) of the formal period by default
    (``days`` overrides for bootstrap resampling).  The hourly-anchor
    convention matches the screening口径 of the second-sub-question plan
    section 2.3 (attachment-3-style hourly evaluation).
    """
    hour, base_k = spec["hour"], int(np.where(ISSUE_HOURS == spec["base_issuance"])[0][0])
    s0, s1 = spec["operative"]
    anchors = np.arange(s0 + 5, s1 + 1, 6)  # slots 6h-1 for h in (hour, next_update]
    if days is None:
        days = np.arange(FORMAL_START, DAYS)
    actual = pv[days][:, anchors]
    official = fc[days, base_k][:, anchors]
    selfv = spec["curve_kwh"][days][:, anchors]
    valid = ~np.isnan(selfv)
    actual, official, selfv = actual[valid], official[valid], selfv[valid]
    off_err = official - actual
    self_err = selfv - actual
    rmse_o = float(np.sqrt(np.mean(off_err ** 2)))
    rmse_s = float(np.sqrt(np.mean(self_err ** 2)))
    mae_o = float(np.mean(np.abs(off_err)))
    mae_s = float(np.mean(np.abs(self_err)))
    bias_o = float(np.mean(off_err))
    bias_s = float(np.mean(self_err))
    rmse_imp = 100.0 * (1.0 - rmse_s / rmse_o) if rmse_o > 1e-12 else np.nan
    mae_imp = 100.0 * (1.0 - mae_s / mae_o) if mae_o > 1e-12 else np.nan
    return {
        "rmse_improvement_pct": float(rmse_imp),
        "mae_improvement_pct": float(mae_imp),
        "rmse_official": rmse_o, "rmse_self": rmse_s,
        "mae_official": mae_o, "mae_self": mae_s,
        "rmse_delta": rmse_o - rmse_s,
        "mae_delta": mae_o - mae_s,
        "bias_official": bias_o, "bias_self": bias_s,
    }


def high_pv_tail_bias(spec: dict, pv: np.ndarray) -> dict:
    """Overestimation bias on high-PV slots (actual >= 75% of day max)."""
    hour = spec["hour"]
    s0, s1 = spec["operative"]
    rows = []
    for d in range(FORMAL_START, DAYS):
        day_pv = pv[d, s0:s1]
        thr = 0.75 * float(day_pv.max())
        sel = day_pv >= thr
        if not sel.any():
            continue
        err = spec["err_kwh"][d, s0:s1][sel]
        rows.append((float(err.mean()), float(err.sum())))
    arr = np.asarray(rows)
    return {
        "mean_error_kwh_high_pv": float(arr[:, 0].mean()),
        "sum_error_kwh_high_pv": float(arr[:, 1].sum()),
        "days": int(arr.shape[0]),
    }


def block_bootstrap(values: np.ndarray, blocks: int = BOOT_BLOCKS,
                    block_days: int = BOOT_BLOCK_DAYS, seed: int = 20250912,
                    stat="mean") -> tuple[float, float]:
    """Circular block bootstrap percentile CI for the statistic of ``values``."""
    rng = np.random.default_rng(seed)
    n = len(values)
    b = max(1, min(block_days, n))
    nblocks = int(np.ceil(n / b))
    draws = np.empty(blocks)
    for it in range(blocks):
        starts = rng.integers(0, n, size=nblocks)
        sample = np.concatenate([values[s:s + b] for s in starts])[:n]
        draws[it] = np.mean(sample) if stat == "mean" else np.median(sample)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def block_bootstrap_stat(fn, n_days: int, blocks: int = BOOT_BLOCKS,
                         block_days: int = BOOT_BLOCK_DAYS,
                         seed: int = 20250912) -> tuple[float, float]:
    """Circular block bootstrap CI for a pooled statistic ``fn(day_subset)``.

    ``fn`` takes an array of day indices (formal-period zero-based indices)
    and returns the statistic of the pooled resample.  Resampling whole
    blocks of days avoids the ratio-of-small-denominators noise that a
    mean-of-per-day-ratios statistic would produce.
    """
    rng = np.random.default_rng(seed)
    b = max(1, min(block_days, n_days))
    nblocks = int(np.ceil(n_days / b))
    draws = np.empty(blocks)
    for it in range(blocks):
        starts = rng.integers(0, n_days, size=nblocks)
        sample = np.concatenate(
            [np.arange(s, min(s + b, n_days)) for s in starts])[:n_days]
        draws[it] = fn(sample)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def monthly_improvement(spec: dict, pv: np.ndarray, fc: np.ndarray) -> pd.DataFrame:
    """Per-month pooled ratios and absolute error reductions.

    The stability gate uses the SIGN of the absolute deltas (rmse_delta,
    mae_delta), which stay well defined in low-PV months where the pooled
    ratio becomes noisy.
    """
    dates = pd.date_range("2025-01-01", periods=DAYS)
    hour = spec["hour"]
    base_k = int(np.where(ISSUE_HOURS == spec["base_issuance"])[0][0])
    s0, s1 = spec["operative"]
    anchors = np.arange(s0 + 5, s1 + 1, 6)
    rows = []
    for month in range(2, 13):
        days = np.array([d for d in range(FORMAL_START, DAYS)
                         if dates[d].month == month])
        actual = pv[days][:, anchors]
        official = fc[days, base_k][:, anchors]
        selfv = spec["curve_kwh"][days][:, anchors]
        off_err = official - actual
        self_err = selfv - actual
        rmse_o = float(np.sqrt(np.mean(off_err ** 2)))
        rmse_s = float(np.sqrt(np.mean(self_err ** 2)))
        mae_o = float(np.mean(np.abs(off_err)))
        mae_s = float(np.mean(np.abs(self_err)))
        rows.append({
            "candidate": f"{hour:02d}:00", "month": month,
            "rmse_improvement_pct": (100.0 * (1.0 - rmse_s / rmse_o)
                                     if rmse_o > 1e-12 else float("nan")),
            "mae_improvement_pct": (100.0 * (1.0 - mae_s / mae_o)
                                    if mae_o > 1e-12 else float("nan")),
            "rmse_delta": rmse_o - rmse_s,
            "mae_delta": mae_o - mae_s,
        })
    return pd.DataFrame(rows)


def high_pv_split(spec: dict, pv: np.ndarray, fc: np.ndarray) -> dict:
    """Pooled improvement on high-PV days (top tercile of mean operative PV)."""
    s0, s1 = spec["operative"]
    mean_pv = pv[FORMAL_START:, s0:s1].mean(axis=1)
    thr = float(np.quantile(mean_pv, 2.0 / 3.0))
    days = np.arange(FORMAL_START, DAYS)[mean_pv >= thr]
    met = improvement_metrics(spec, pv, fc, None, None, days=days)
    return {"high_pv_rmse_improvement_pct": met["rmse_improvement_pct"],
            "high_pv_mae_improvement_pct": met["mae_improvement_pct"],
            "high_pv_days": int(days.size)}


def build_all(root: Path = ROOT) -> dict:
    """Build all candidate self-forecasts and their screening metrics."""
    pv = load_pv_actuals(root)
    pv_h = pv_hourly_kw(pv)
    hourly = load_hourly_issuances(root)
    fc = build_issuance_curves(root)
    specs = {}
    for hour, base_k, next_hour in CANDIDATES:
        specs[hour] = build_self_forecasts(hour, base_k, next_hour, pv, pv_h,
                                           hourly)
        specs[hour]["correlation"] = pooled_correlation(
            hour, next_hour, pv_h, hourly, base_k)
        specs[hour]["metrics"] = improvement_metrics(
            specs[hour], pv, fc, hourly, pv_h)
        specs[hour]["tail"] = high_pv_tail_bias(specs[hour], pv)
        specs[hour]["monthly"] = monthly_improvement(specs[hour], pv, fc)
        specs[hour]["high_pv"] = high_pv_split(specs[hour], pv, fc)
    return specs, pv, pv_h, hourly, fc


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------

def write_outputs(out: Path, specs: dict, pv: np.ndarray, pv_h: np.ndarray,
                  hourly: np.ndarray, fc: np.ndarray) -> None:
    out.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2025-01-01", periods=DAYS)

    # ---- long-format parquet: self hourly + 10-min curves + errors ----
    long = []
    for hour, spec in specs.items():
        s0 = 6 * hour
        for d in range(1, DAYS):
            long.append(pd.DataFrame({
                "date": pd.Timestamp(dates[d]),
                "candidate_hour": f"{hour:02d}:00",
                "base_issuance": f"{spec['base_issuance']:02d}:00",
                "slot_index": np.arange(s0, T),
                "self_kwh": spec["curve_kwh"][d, s0:],
                "err_kwh": spec["err_kwh"][d, s0:],
                "official_kwh": fc[d, int(np.where(ISSUE_HOURS == spec["base_issuance"])[0][0]), s0:],
                "actual_kwh": pv[d, s0:],
                "fallback_code": np.full(T - s0, int(spec["fallback"][d])),
            }))
    frame = pd.concat(long, ignore_index=True)
    frame.to_parquet(out / "self_forecasts_10min.parquet", index=False)

    # ---- screening table (correlation / accuracy / operability gates) ----
    role = {7: "reject", 8: "weak_control", 9: "sensitivity", 10: "primary",
            11: "sensitivity_short_horizon", 13: "sensitivity", 14: "primary",
            15: "sensitivity", 16: "boundary_control",
            17: "reject_short_horizon"}
    screening = []
    for hour, spec in specs.items():
        m = spec["metrics"]
        s0, s1 = spec["operative"]
        screening.append({
            "candidate": f"{hour:02d}:00",
            "role": role[hour],
            "base_window": (f"{spec['base_issuance']:02d}:00-"
                            f"{spec['next_update_hour']:02d}:00"),
            "correlation": round(spec["correlation"], 3),
            "rmse_improvement_pct": round(m["rmse_improvement_pct"], 1),
            "mae_improvement_pct": round(m["mae_improvement_pct"], 1),
            "bias_official_kwh": round(m["bias_official"], 4),
            "bias_self_kwh": round(m["bias_self"], 4),
            "remaining_hours": spec["next_update_hour"] - hour,
            "operative_slots": f"{s0}-{s1}",
            "fitted_days": int((spec["fallback"] == 0).sum()),
            "fallback_days": int((spec["fallback"] != 0).sum()),
            "fallback_reasons": {int(k): int(v) for k, v in zip(
                *np.unique(spec["fallback"], return_counts=True))},
            "high_pv_tail_mean_error_kwh": round(spec["tail"]["mean_error_kwh_high_pv"], 4),
            "high_pv_rmse_improvement_pct": round(spec["high_pv"]["high_pv_rmse_improvement_pct"], 1),
            "high_pv_mae_improvement_pct": round(spec["high_pv"]["high_pv_mae_improvement_pct"], 1),
            "high_pv_days": spec["high_pv"]["high_pv_days"],
        })
    pd.DataFrame(screening).to_csv(out / "candidate_screening.csv", index=False,
                                   encoding="utf-8-sig")

    # ---- monthly robustness ----
    monthly = []
    for hour, spec in specs.items():
        monthly.append(spec["monthly"])
    pd.concat(monthly, ignore_index=True).to_csv(
        out / "monthly_robustness.csv", index=False, encoding="utf-8-sig")

    # ---- bootstrap CIs for the pooled accuracy improvement ----
    boot = []
    for hour, spec in specs.items():
        n_days = DAYS - FORMAL_START
        for metric in ["rmse_improvement_pct", "mae_improvement_pct"]:
            def fn(subset, spec=spec, metric=metric):
                return improvement_metrics(spec, pv, fc, None, None,
                                           days=subset)[metric]

            lo, hi = block_bootstrap_stat(fn, n_days)
            boot.append({"candidate": f"{hour:02d}:00", "metric": metric,
                         "pooled": round(fn(np.arange(FORMAL_START, DAYS)), 2),
                         "ci_low": round(lo, 2), "ci_high": round(hi, 2)})
    pd.DataFrame(boot).to_csv(out / "bootstrap_ci.csv", index=False,
                              encoding="utf-8-sig")

    # ---- anti-noise sensitivity: 30-minute median residual instead of the
    # instantaneous hourly current error (metric-level only) ----
    anti = []
    for hour, base_k, next_hour in CANDIDATES:
        base_idx = int(np.where(ISSUE_HOURS == ISSUE_HOURS[base_k])[0][0])
        s0, s1 = 6 * hour, 6 * min(next_hour, 24)
        for d in range(FORMAL_START, DAYS):
            off_err = (fc[d, base_idx, s0 - 3:s0] - pv[d, s0 - 3:s0]) * 6.0
            anti.append({"date": pd.Timestamp(dates[d]),
                         "candidate_hour": f"{hour:02d}:00",
                         "instant_error_kw": official_errors_at(
                             hourly, pv_h, d, base_k, hour),
                         "median30_error_kw": float(np.median(off_err))})
    anti_frame = pd.DataFrame(anti)
    anti_frame.to_csv(out / "anti_noise_current_error.csv", index=False,
                      encoding="utf-8-sig")

    summary = {
        "method": ("latest official issuance + same-day dynamic residual "
                   "correction (21-day rolling ridge, shrink 0.8, clip [0,1.5])"),
        "fallback_rules": ["few_samples(<14)->official",
                           "low_current_error_variance->official",
                           "sign_anomaly_or_clipped->no correction",
                           "validation_window_not_better->official"],
        "operative_windows": {f"{hour:02d}:00": list(map(int, spec["operative"]))
                              for hour, spec in specs.items()},
        "evaluation_grid": "hourly anchors (slot 6h-1) inside the operative window",
        "formal_period": "2025-02-01..2025-12-31 (334 days)",
        "causality": "day d uses only days before d; PCHIP anchored at the observed candidate-hour value",
        "units": "curves in kWh per 10 minutes; hourly in kW",
    }
    (out / "forecast_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    out = ROOT / "outputs/question3/second_subquestion/forecasts"
    specs, pv, pv_h, hourly, fc = build_all(ROOT)
    write_outputs(out, specs, pv, pv_h, hourly, fc)
    screening = pd.read_csv(out / "candidate_screening.csv")
    print(screening[["candidate", "correlation", "rmse_improvement_pct",
                     "mae_improvement_pct", "remaining_hours"]].to_string(index=False))
    print(f"outputs written to {out}")


if __name__ == "__main__":
    main()
