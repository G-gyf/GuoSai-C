"""Forecast error comparison: Q2 seven-day mean PV vs Q3 attachment-3 issuances.

Compares, against attachment-2 actual PV over the formal period
(2025-02-01..12-31, 334 days):

1. Q2 seven-day same-slot mean (``question2.forecasts``) — all 144 slots.
2. Q3 0:00-issuance PCHIP curve — all 144 slots.
3. Q3 6:00 / 12:00 / 18:00 issuance curves over their own remaining
   same-day horizons — the error seen after each forecast update.
4. Fair same-target comparison: for the same target window, the error of
   each successive issuance (0:00, 6:00, 12:00, 18:00), showing how much
   each update improves identical targets.
5. Hour-of-day RMSE profile for the 0:00 head-to-head (Q2 7-day mean vs
   Q3 0:00 issuance).

Error convention: forecast minus actual (kW).  Daytime mask: actual PV at
least 1 kW (shared across all methods, per 问题三_10分钟化实测记录).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.optimization.question2 import ROOT, T, load_inputs, forecasts
from src.data_pipeline.question3_forecasts import (
    build_issuance_curves,
    load_pv_actuals,
)

FORMAL_START = 31  # 2025-02-01
ISSUE_HOURS = [0, 6, 12, 18]


def metrics(err_kw: np.ndarray) -> dict:
    err = err_kw[np.isfinite(err_kw)]
    return {
        "n": int(err.size),
        "rmse_kw": float(np.sqrt(np.mean(err**2))) if err.size else float("nan"),
        "mae_kw": float(np.mean(np.abs(err))) if err.size else float("nan"),
        "bias_kw": float(np.mean(err)) if err.size else float("nan"),
        "mean_abs_actual_kw": None,
    }


def main() -> None:
    out = ROOT / "outputs/question3/analysis/forecast_errors"
    out.mkdir(parents=True, exist_ok=True)
    dates, load, pv, prices = load_inputs()
    actuals = load_pv_actuals(ROOT)  # kWh (365, 144)
    _, fv = forecasts(load, pv)      # Q2 seven-day same-slot mean, kWh
    fc = build_issuance_curves(ROOT)  # Q3 issuance PCHIP curves, kWh

    d0, d1 = FORMAL_START, 365
    daymask = np.zeros(actuals.shape[1], bool) if False else None
    daytime = (actuals[d0:d1] * 6.0) >= 1.0  # kW

    rows = []

    def add(name, err_all, err_day):
        m_all = metrics(err_all)
        m_day = metrics(err_day)
        rows.append({
            "forecast": name,
            "slots": "all",
            **{f"{k}_all": v for k, v in m_all.items()},
            **{f"{k}_day": v for k, v in m_day.items()},
        })

    # ---- 1 & 2: 0:00 head-to-head over the full day ----
    err_7d = (fv[d0:d1] - actuals[d0:d1]) * 6.0
    err_fc0 = (fc[d0:d1, 0] - actuals[d0:d1]) * 6.0
    add("Q2_7day_mean", err_7d, err_7d[daytime])
    add("Q3_issue_00:00", err_fc0, err_fc0[daytime])

    # ---- 3: each issuance over its own remaining horizon ----
    for k, hour in enumerate(ISSUE_HOURS):
        start = 6 * hour
        err_k = (fc[d0:d1, k, start:] - actuals[d0:d1, start:]) * 6.0
        add(f"Q3_issue_{hour:02d}:00_own_horizon", err_k, err_k[daytime[:, start:]])

    # ---- 4: fair same-target comparison per window ----
    windows = [(0, 36, [0]), (36, 72, [0, 1]), (72, 108, [0, 1, 2]), (108, 144, [0, 1, 2, 3])]
    for s, e, ks in windows:
        for k in ks:
            hour = ISSUE_HOURS[k]
            err = (fc[d0:d1, k, s:e] - actuals[d0:d1, s:e]) * 6.0
            add(f"same_target_{s//6:02d}-{e//6:02d}h_issue_{hour:02d}:00", err, err[daytime[:, s:e]])

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "forecast_error_comparison.csv", index=False, encoding="utf-8-sig")

    # ---- 5: hour-of-day RMSE profile, 0:00 head-to-head ----
    profile = []
    for h in range(24):
        sl = slice(6 * h, 6 * (h + 1))
        d7 = (fv[d0:d1, sl] - actuals[d0:d1, sl]) * 6.0
        f0 = (fc[d0:d1, 0, sl] - actuals[d0:d1, sl]) * 6.0
        dm = daytime[:, sl]
        profile.append({
            "hour": f"{h:02d}:00",
            "q2_7d_rmse_kw": metrics(d7[dm])["rmse_kw"],
            "q3_00_rmse_kw": metrics(f0[dm])["rmse_kw"],
            "q2_7d_bias_kw": metrics(d7[dm])["bias_kw"],
            "q3_00_bias_kw": metrics(f0[dm])["bias_kw"],
        })
    prof = pd.DataFrame(profile)
    prof.to_csv(out / "hour_of_day_profile.csv", index=False, encoding="utf-8-sig")

    # ---- printable summary ----
    pd.set_option("display.width", 200)
    sel = frame[["forecast", "n_all", "rmse_kw_all", "mae_kw_all", "bias_kw_all",
                 "n_day", "rmse_kw_day", "mae_kw_day", "bias_kw_day"]]
    print("=== formal period (2025-02-01..12-31) forecast error, kW ===")
    print(sel.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
    print()
    print("=== hour-of-day daytime RMSE (kW): Q2 7-day mean vs Q3 0:00 issuance ===")
    print(prof.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))


if __name__ == "__main__":
    main()
