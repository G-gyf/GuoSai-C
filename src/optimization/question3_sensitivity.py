"""Sensitivity sweep of the adjustment up-region quantile.

Runs the derived variant Q70 (main), plus Q50 / Q80 / Q90 as robustness
controls, for the M6 and M612 strategies from the same common 1 February
opening inventory produced by the shared warm-up.  The main quantile is
fixed a priori from the settlement marginals (F = 1 - R'/(5c) = 0.7 for
the up region); this sweep only REPORTS robustness and must never be used
to select the main quantile from backtest costs.

Run:  python -m src.optimization.question3_sensitivity
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.optimization.question2 import ROOT, load_inputs, load_forecast_weekly_persist
from src.optimization.question3 import Q3Settings, run_strategy
from src.data_pipeline.question3_forecasts import build_issuance_curves

SWEEP = [0.5, 0.7, 0.8, 0.9]
STRATEGIES = ["M6", "M612"]


def main() -> None:
    out = ROOT / "outputs/question3/sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    summaries = json.loads(
        (ROOT / "outputs/question3/current/question3_summary.json").read_text(encoding="utf-8")
    )
    m612 = next(s for s in summaries if s["settings"]["strategy"] == "M612")
    common_soc = float(m612["settings"]["common_initial_soc_kwh"])
    assert common_soc > 0

    dates, load, pv, prices = load_inputs()
    fl = load_forecast_weekly_persist(load)
    fc = build_issuance_curves(ROOT)

    rows = []
    for strategy in STRATEGIES:
        for q_up in SWEEP:
            settings = Q3Settings(strategy=strategy, adjustment_up_quantile=q_up)
            frame, diag, pair, summary = run_strategy(
                dates, load, pv, prices, fl, fc, settings,
                start_idx=31, initial_energy=common_soc,
            )
            m = summary["formal_period"]
            rows.append({
                "strategy": strategy,
                "q_up": q_up,
                "derived": q_up == 0.7,
                "total_cost": m["total_cost"],
                "retained_cost": m["retained_cost"],
                "down_cost": m["down_cost"],
                "up_cost": m["up_cost"],
                "emergency_cost": m["emergency_cost"],
                "emergency_kwh": m["emergency_kwh"],
                "up_kwh": m["up_kwh"],
                "down_kwh": m["down_kwh"],
                "initial_soc_kwh": m["initial_soc_kwh"],
                "final_soc_kwh": m["final_soc_kwh"],
            })
            print(f"{strategy} q_up={q_up:.1f} total={m['total_cost']:,.2f} "
                  f"up={m['up_cost']:,.2f} emergency={m['emergency_cost']:,.2f}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "quantile_sweep.csv", index=False, encoding="utf-8-sig")
    pivot = frame.pivot_table(index="strategy", columns="q_up", values="total_cost")
    pivot.to_csv(out / "quantile_sweep_total_pivot.csv", encoding="utf-8-sig")
    print(pivot.to_string(float_format=lambda x: f"{x:,.2f}"))
    (out / "quantile_sweep_summary.json").write_text(
        frame.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
