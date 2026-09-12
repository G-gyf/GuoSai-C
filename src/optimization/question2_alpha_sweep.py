"""Sweep the quantile level alpha of the day-ahead plan and score each by LB2.

Historical exploratory scan only. The formal alpha is fixed at 0.8 by the
user and the single-slot newsvendor motivation. These full-year retrospective
relaxed-bound scores neither select formal parameters nor prove optimality
in the storage-coupled problem.

Run:  python -m src.optimization.question2_alpha_sweep
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.optimization.question2 import ROOT, load_inputs
from src.optimization.question2_g_search import gen_schedule, lb2_of

ALPHAS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.833333, 0.85, 0.90, 0.95]


def main():
    out = ROOT / "outputs" / "question2" / "analysis" / "g_search"
    dates, load, pv, prices = load_inputs()
    rows = []
    for alpha in ALPHAS:
        t0 = time.perf_counter()
        g = gen_schedule(dates, load, pv, prices, method="quantile", alpha=alpha)
        cost, emergency = lb2_of(g, dates, load, pv, prices)
        rows.append({"alpha": alpha, "lb2": cost, "emergency": emergency,
                     "planned": cost - emergency, "sec": time.perf_counter() - t0})
        print(f"alpha={alpha:8.6f}  LB2={cost:>15,.2f}  emergency={emergency:>12,.2f}  planned={cost-emergency:>13,.2f}", flush=True)

    best = min(rows, key=lambda r: r["lb2"])
    (out / "alpha_sweep.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nBEST alpha={best['alpha']:.6f}  LB2={best['lb2']:,.2f}")


if __name__ == "__main__":
    main()
