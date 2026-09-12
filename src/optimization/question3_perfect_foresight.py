"""Question 3 matched perfect-foresight lower bound for the M612 bundle.

With perfect information the final adjusted purchase equals the original
plan (no deviation fees) and no emergency purchase is ever needed, so the
question-2 perfect-information LP is the same physical relaxation for
question 3: it is a lower bound on every executable question-3 strategy
with the same opening inventory.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.optimization.question2 import ROOT, load_inputs
from src.optimization.question2_perfect_foresight import solve_horizon


def main() -> None:
    out = ROOT / "outputs/question3/benchmark"
    out.mkdir(parents=True, exist_ok=True)
    summaries = json.loads(
        (ROOT / "outputs/question3/current/question3_summary.json").read_text(encoding="utf-8")
    )
    m612 = next(s for s in summaries if s["settings"]["strategy"] == "M612")
    dates, load, pv, prices = load_inputs()
    bound = solve_horizon(
        dates[31:],
        load[31:],
        pv[31:],
        prices,
        float(m612["initial_result_soc_kwh"]),
        "matched_m612",
        out,
    )
    bound["causal_policy_cost_yuan"] = m612["total_cost"]
    bound["gap_to_bound_yuan"] = m612["total_cost"] - bound["cost_lower_bound_yuan"]
    bound["gap_as_fraction_of_actual"] = bound["gap_to_bound_yuan"] / m612["total_cost"]
    bound["excess_over_lower_bound"] = bound["gap_to_bound_yuan"] / bound["cost_lower_bound_yuan"]
    bound["note"] = (
        "perfect information implies qA == q0 (zero deviation fees) and zero "
        "emergency purchases, so the question-2 physical LP is a valid "
        "question-3 lower bound at the same opening inventory"
    )
    (out / "matched_m612_summary.json").write_text(
        json.dumps(bound, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(bound, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
