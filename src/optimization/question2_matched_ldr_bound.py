"""One-off: matched perfect-foresight lower bound for the current LDR bundle.

Computes the 2/1-12/31 perfect-information LP lower bound with the current
LDR run's actual February opening inventory, and writes the result next to
the other perfect-foresight benchmarks without overwriting them.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.optimization.question2 import ROOT, load_inputs
from src.optimization.question2_perfect_foresight import solve_horizon


def main() -> None:
    out = ROOT / "outputs/question2/benchmark/perfect_foresight"
    summaries = json.loads(
        (ROOT / "outputs/question2/current/ldr/question2_summary.json").read_text(
            encoding="utf-8"
        )
    )
    ldr = next(s for s in summaries if s["settings"]["name"] == "ldr_quantile_a08")
    dates, load, pv, prices = load_inputs()
    bound = solve_horizon(
        dates[31:],
        load[31:],
        pv[31:],
        prices,
        float(ldr["initial_result_soc_kwh"]),
        "matched_ldr",
        out,
    )
    bound["causal_policy_cost_yuan"] = ldr["total_cost"]
    bound["gap_to_bound_yuan"] = ldr["total_cost"] - bound["cost_lower_bound_yuan"]
    bound["gap_as_fraction_of_actual"] = bound["gap_to_bound_yuan"] / ldr["total_cost"]
    bound["excess_over_lower_bound"] = (
        bound["gap_to_bound_yuan"] / bound["cost_lower_bound_yuan"]
    )
    (out / "matched_ldr_summary.json").write_text(
        json.dumps(bound, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(bound, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
