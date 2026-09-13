"""Run bounded Question 2 LDR sensitivity cases without overwriting main outputs."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.optimization.question2 import load_forecast_weekly_persist, load_inputs
from src.optimization.question2_ldr import LDRSettings, run_ldr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20250912)
    parser.add_argument("--residual-days", type=int, default=21)
    parser.add_argument("--maxiter", type=int, default=8)
    parser.add_argument("--popsize", type=int, default=5)
    parser.add_argument("--delta-bound", type=float, default=9600.0)
    parser.add_argument("--lambda-bound", type=float, default=2.0)
    args = parser.parse_args()

    settings = LDRSettings(
        name=args.name,
        residual_days=args.residual_days,
        search_seed=args.seed,
        search_maxiter=args.maxiter,
        search_popsize=args.popsize,
        delta_bound_kwh=args.delta_bound,
        lambda_bound=args.lambda_bound,
        load_forecast="weekly_persist",
    )
    dates, load, pv, prices = load_inputs()
    fl = load_forecast_weekly_persist(load)
    _, _, _, summary, _, _ = run_ldr(dates, load, pv, prices, settings, fl=fl)

    result = {
        "case": args.name,
        "settings": asdict(settings),
        "formal_period": summary["formal_period"],
        "calibrated_days": summary["calibrated_days"],
        "accepted_search_days": summary["accepted_search_days"],
        "search_evaluations": summary["search_evaluations"],
        "empirical_scenario_improvement": summary["empirical_scenario_improvement"],
        "seconds": summary["seconds"],
        "planner_seconds": summary["planner_seconds"],
        "calibration_seconds": summary["calibration_seconds"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
