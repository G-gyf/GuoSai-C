"""Extract every number needed for the weekly-persistence paper rewrite."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs/question2/current/ldr"
KEY = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]


def time_label(slot: int) -> str:
    return f"{slot // 6:02d}:{slot % 6 * 10:02d}"


def merged_emergency(day: pd.DataFrame):
    q = day.emergency_kwh.to_numpy()
    rows = []
    t = 0
    while t < 144:
        if q[t] <= 1e-6:
            t += 1
            continue
        begin = t
        while t < 144 and q[t] > 1e-6:
            t += 1
        rows.append((f"{time_label(begin)}-{time_label(t)}", float(q[begin:t].sum())))
    return rows


def main() -> None:
    summaries = json.loads((SRC / "question2_summary.json").read_text(encoding="utf-8"))
    ldr = next(s for s in summaries if s["settings"]["name"] == "ldr_quantile_a08")
    beta1 = next(s for s in summaries if s["settings"]["name"] == "risk_reserve")
    beta0 = next(s for s in summaries if s["settings"]["name"] == "risk_greedy")
    scenario = next(
        s
        for s in json.loads(
            (ROOT / "outputs/question2/archive/scenarios/question2_summary.json").read_text(
                encoding="utf-8"
            )
        )
        if s["settings"]["name"] == "scenario_value_no_NAC"
    )
    strategies = {"beta1": beta1, "beta0": beta0, "scenario": scenario, "ldr": ldr}
    print("== 8.2 strategy table ==")
    for name, s in strategies.items():
        print(
            name,
            f"plan={s['planned_cost']:.2f} emer={s['emergency_cost']:.2f} "
            f"total={s['total_cost']:.2f} emer_kwh={s['emergency_kwh']:.2f} "
            f"init={s['initial_result_soc_kwh']:.6f} final={s['final_soc_kwh']:.6f}",
        )
    for ref in ["beta1", "beta0", "scenario"]:
        delta = ldr["total_cost"] - strategies[ref]["total_cost"]
        print(
            f"delta vs {ref}: {delta:+.2f} yuan ({delta / strategies[ref]['total_cost'] * 100:+.4f}%)"
        )
    print()
    print("== 8.3 weighted emergency prices ==")
    for name, s in strategies.items():
        print(name, f"{s['emergency_cost'] / s['emergency_kwh']:.4f}")
    print()
    paired = pd.read_csv(SRC / "same_plan_controller_comparison.csv", parse_dates=["date"])
    formal_paired = paired[paired.date >= pd.Timestamp("2025-02-01")]
    print("== 8.4 paired totals ==")
    print(
        "ldr", f"{formal_paired.ldr_total_cost.sum():.2f}",
        "beta1", f"{formal_paired.beta1_total_cost.sum():.2f}",
        "beta0", f"{formal_paired.beta0_total_cost.sum():.2f}",
        "planned", f"{formal_paired.planned_cost.sum():.2f}",
    )
    print(
        "savings vs beta1:", f"{formal_paired.beta1_total_cost.sum() - formal_paired.ldr_total_cost.sum():.2f}",
        "vs beta0:", f"{formal_paired.beta0_total_cost.sum() - formal_paired.ldr_total_cost.sum():.2f}",
    )
    diff = formal_paired.ldr_total_cost - formal_paired.beta0_total_cost
    print(
        "vs beta0 daily: wins", int((diff < -1e-5).sum()),
        "losses", int((diff > 1e-5).sum()),
        "ties", int((diff.abs() <= 1e-5).sum()),
    )
    diff1 = formal_paired.ldr_total_cost - formal_paired.beta1_total_cost
    print(
        "vs beta1 daily: wins", int((diff1 < -1e-5).sum()),
        "losses", int((diff1 > 1e-5).sum()),
        "ties", int((diff1.abs() <= 1e-5).sum()),
    )
    print()
    daily = pd.read_csv(SRC / "question2_daily.csv", parse_dates=["date"])
    daily = daily[daily.date >= pd.Timestamp("2025-02-01")]
    print("== 8.1 emergency days ==")
    print("days with emergency:", int((daily.emergency_kwh > 1e-6).sum()))
    print("days without:", int((daily.emergency_kwh <= 1e-6).sum()))
    max_day = daily.loc[daily.emergency_cost.idxmax()]
    print("max daily emergency:", max_day.date.date(), f"{max_day.emergency_cost:.2f}")
    print("monthly total costs:")
    print(
        daily.assign(month=daily.date.dt.month)
        .groupby("month")[["planned_cost", "emergency_cost", "total_cost"]]
        .sum()
        .round(2)
        .to_string()
    )
    print()
    f = pd.read_csv(SRC / "question2_schedule.csv", parse_dates=["date"])
    params = pd.read_csv(SRC / "ldr_daily_parameters.csv", parse_dates=["date"])
    print("== section 9 key dates ==")
    for ds in KEY:
        day = f[f.date == ds]
        d = pd.Timestamp(ds)
        g = day.grid_kwh.to_numpy()
        print(f"--- {ds} ---")
        print("表1型:", {f"{h}:00-{h}:10": round(float(g[h * 6]), 6) for h in [10, 12, 14, 16, 18, 20]})
        print(f"全天购电量 {g.sum():.6f} 计划费 {day.planned_cost.sum():.2f} 实际总费 {day.total_cost.sum():.2f}")
        print(f"紧急电量 {day.emergency_kwh.sum():.6f} 紧急费 {day.emergency_cost.sum():.2f}")
        print(f"0:00 SOC {day.soc_start_kwh.iloc[0]:.6f}  24:00 SOC {day.soc_end_kwh.iloc[-1]:.6f}")
        print(f"充电总量 {day.charge_kwh.sum():.6f} 放电总量 {day.discharge_kwh.sum():.6f} 未利用 {day.unused_kwh.sum():.6f}")
        for block in range(6):
            blk = day.iloc[block * 24 : (block + 1) * 24]
            print(
                f"  {block*4}:00-{(block+1)*4}:00  充 {blk.charge_kwh.sum():.6f}  放 {blk.discharge_kwh.sum():.6f}"
            )
        merged = merged_emergency(day)
        print("紧急合并区间:", merged if merged else "无")
        p = params[params.date == d].iloc[0]
        print(
            "附录D:",
            [f"{p[c]:.4f}" for c in [
                "delta_00_06_kwh", "delta_06_12_kwh", "delta_12_18_kwh", "delta_18_24_kwh",
                "lambda_06_12", "lambda_12_18", "lambda_18_24",
            ]],
            "improvement", f"{p.scenario_improvement:.2f}",
        )
        print()
    print("== net load midday check (actuals) ==")
    for ds in KEY:
        day = f[f.date == ds]
        net = (day.load_kwh - day.pv_kwh).to_numpy()
        print(ds, "midday(10-15h) min net:", round(float(net[60:90].min()), 2))
    print()
    print("== 10.x validation from summary ==")
    print(json.dumps(ldr["validation"], ensure_ascii=False, indent=2))
    print("calibrated_days", ldr["calibrated_days"], "accepted", ldr["accepted_search_days"],
          "evaluations", ldr["search_evaluations"], "seconds", round(ldr["seconds"], 2))
    print("full_year", json.dumps(ldr["full_year"], ensure_ascii=False))


if __name__ == "__main__":
    main()
