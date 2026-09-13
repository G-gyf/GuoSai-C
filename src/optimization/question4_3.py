# -*- coding: utf-8 -*-
"""Question 4-3: fluctuating prices, question-3 pipeline (M612 main strategy).

Confirmed rule set (问题四口径, option B):

* decisions use the 0:00 weekly-persistence price forecast locked for the
  whole day, except that at every update instant (0:00, 6:00, 12:00 and the
  18:00 re-optimisation) the interval starting at that instant has an
  already-realized price which replaces the forecast in the decision row;
  scenario price paths are determinized at that first column.  Settlement
  always uses the realized attachment-4 price:
  p·min(q0,qA) + 0.5p(q0-qA)+ + 1.5p(qA-q0)+ + 5p·b;
* the same 21 historical days supply same-day paired price residuals for
  every calibration horizon (scenario prices clipped at zero);
* risk levels stay fixed a priori: Q80 plan / Q90 down / Q70 up;
* 2025-01-01 is a frozen initial-condition day (no plan, no battery action,
  SOC stays 6000 kWh, excluded from statistics); the shared M0 January
  warm-up under variable prices then yields the common 1 February inventory;
  M0 and M612 fork from it for the formal period.

Run:  python -m src.optimization.question4_3
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.optimization.question2 import ROOT, T, load_inputs, load_forecast_weekly_persist
from src.optimization.question3 import (
    MAIN_STRATEGY,
    Q3Settings,
    build_issuance_curves,
    run_strategy,
    write_main_outputs,
    write_strategy_outputs,
)
from src.data_pipeline.question4_prices import (
    determinize_scenario_first_column,
    load_question4_prices,
    price_scenario_paths,
)

KEY_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]


def write_q4_3_report(out: Path, summaries: list) -> None:
    by_name = {s["settings"]["strategy"]: s for s in summaries}
    m0 = by_name["M0"]
    m612 = by_name["M612"]
    v_total = m0["total_cost"] - m612["total_cost"]
    fm = m612["formal_period"]
    key = pd.read_csv(out / "question3_key_dates.csv", encoding="utf-8-sig")
    key_lines = ["| 日期 | 全天q0/kWh | 全天qA/kWh | 面值费/元 | 常规结算费/元 | 下调费/元 | 紧急费/元 | 实际总费/元 |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, row in key.iterrows():
        key_lines.append(
            f"| {row['date']} | {row['day_q0_kwh']:,.6f} | {row['day_qA_kwh']:,.6f} "
            f"| {row['face_cost']:,.2f} | {row['settlement_cost']:,.2f} "
            f"| {row['down_cost']:,.2f} | {row['emergency_cost']:,.2f} | {row['total_cost']:,.2f} |"
        )
    lines = [
        "# 问题四-3 波动电价下问题三重算（M612 主策略）",
        "",
        "## 结论摘要",
        "",
        "| 策略 | 常规结算费 | 下调违约费 | 上调新增费 | 紧急购电费 | 实际总费 | 紧急电量(kWh) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        m = s["formal_period"]
        lines.append(
            f"| {s['settings']['strategy']} | {m['retained_cost']:,.2f} | {m['down_cost']:,.2f} "
            f"| {m['up_cost']:,.2f} | {m['emergency_cost']:,.2f} | {m['total_cost']:,.2f} "
            f"| {m['emergency_kwh']:,.2f} |"
        )
    lines += [
        "",
        f"- 6:00＋12:00 滚动调整的总价值 C_M0 − C_M612 = {v_total:,.2f} 元（波动电价口径；"
        "V6/V12 的单独分解需运行 M6 消融，可在需要时补充）。",
        "",
        "## 主策略 M612 结构（波动电价口径）",
        "",
        "- 0:00：周持久化负载 + 附件3 0:00 预报（PCHIP）→ 80% 分位数 LP（决策价 = [已实现首区间价, 周持久化预报 p̂_d=p_{d-7} 后续列]）得到 q0 与 E^p,0；校准 δ0（1 维，0—6 时）。",
        "- 6:00：按分段净结算对 q0 重解 6:00—24:00（双面报童曲线 median(Q70,Q90,q0)，决策价首列（6:00—6:10）用已实现价、其余沿用 0:00 价格预报），锁定 6:00—12:00；校准 (δ6, λ6)（2 维）。",
        "- 12:00：重解并锁定 12:00—24:00（决策价首列用已实现价）；联合校准 (δ12, λ12, δ18, λ18)（4 维）。",
        "- 18:00：不调整购电、不使用 18:00 预报、不重新校准；仅代入实测 a18 更新保留阈值。（若重解，决策价首列用已实现价。）",
        "- 结算：全部按附件4 实际价逐区间结算（保留 p·min、下调 0.5p、上调 1.5p、紧急 5p）；决策用的价格预报全天锁定（各更新时点首列除外）。",
        "- 情景：同一 21 个历史日同日配对（负载/光伏/价格），情景价 max(0, p̂_d+ε^p_i) 进入各阶段 LDR 校准目标，且每个校准时域首列按该更新时点已实现价确定化；风险分位数 Q80/Q70/Q90 先验固定。",
        "- 共同起点：2025-01-01 冻结（无计划、电池不动作、SOC 恒 6000 kWh、不入统计）；1月2–31日由 M0（仅0:00）按波动价预热一次，各策略在2月1日以同一库存分叉。",
        "",
        "## 四个指定日期",
        "",
        *key_lines,
        "",
        "## 校验",
        "",
    ]
    for k, v in m612["validation"].items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## 解释限制",
        "",
        "0:00 计划未显式建模 6:00 可调整的期权价值（近视滚动 MPC）；计划层/调整层 LP 对价格采用"
        "确定性等价（预报价 p̂），实际账单按实际价结算；价格情景只进入 LDR 校准目标；"
        "每阶段直接搜索不宣称全局最优。",
    ]
    (out / "问题四-3实施与结果说明.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/question4/result4-3")
    parser.add_argument("--strategies", nargs="*", default=None, choices=["M0", "M612"])
    parser.add_argument("--search-seed", type=int, default=Q3Settings.search_seed)
    parser.add_argument("--search-maxiter", type=int, default=Q3Settings.search_maxiter)
    parser.add_argument("--search-popsize", type=int, default=Q3Settings.search_popsize)
    args = parser.parse_args()
    strategies = args.strategies or ["M0", "M612"]

    dates, load, pv, _ = load_inputs()
    bundle = load_question4_prices(ROOT)
    p_act = bundle["price_actual"]
    p_hat = bundle["price_forecast"]
    residual = bundle["price_residual"]
    fl = load_forecast_weekly_persist(load)
    fc = build_issuance_curves(ROOT)

    def scen_prices(d: int, h0: int, m: int) -> np.ndarray | None:
        if m <= 0:
            return None
        paths = price_scenario_paths(d, p_hat, residual, 21, h0)
        if paths is None:
            row = p_hat[d, h0:].copy()
            row[0] = p_act[d][h0]
            return row[None, :]
        if paths.shape[0] != m:
            raise ValueError(f"price scenario count {paths.shape[0]} != net scenario count {m}")
        # Correction 2: the interval starting at the update instant h0 has an
        # already-realized price, so every scenario path is determinized at
        # its first column (no scenario uncertainty there).
        return determinize_scenario_first_column(paths, p_act[d][h0])

    if args.days != 365:
        dates = dates[: args.days]
        load = load[: args.days]
        pv = pv[: args.days]
        fl = fl[: args.days]
        p_act = p_act[: args.days]
        p_hat = p_hat[: args.days]
        pilot_out = args.output / "pilot"
        pilot_out.mkdir(parents=True, exist_ok=True)
        for name in strategies:
            settings = Q3Settings(strategy=name, search_seed=args.search_seed,
                                  search_maxiter=args.search_maxiter,
                                  search_popsize=args.search_popsize)
            frame, diag, pair, summary = run_strategy(
                dates, load, pv, np.asarray(p_hat[0]), fl, fc, settings, limit=args.days,
                prices_plan=p_hat, prices_actual=p_act, scen_prices_fn=scen_prices,
            )
            (pilot_out / f"pilot_{name}_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    warmup_settings = Q3Settings(strategy="M0", search_seed=args.search_seed,
                                 search_maxiter=args.search_maxiter,
                                 search_popsize=args.search_popsize)
    warmup_frame, _, _, warmup_summary = run_strategy(
        dates, load, pv, np.asarray(p_hat[0]), fl, fc, warmup_settings, limit=31,
        prices_plan=p_hat, prices_actual=p_act, scen_prices_fn=scen_prices,
    )
    common_feb1_soc = float(warmup_frame.soc_end_kwh.iloc[-1])
    warmup_frame.to_csv(args.output / "warmup_january_schedule.csv",
                        index=False, encoding="utf-8-sig")
    print(f"common warm-up done: 1 February opening SOC = {common_feb1_soc:.6f} kWh", flush=True)

    summaries = []
    frames = {}
    for name in strategies:
        settings = Q3Settings(strategy=name, search_seed=args.search_seed,
                              search_maxiter=args.search_maxiter,
                              search_popsize=args.search_popsize)
        frame, diag, pair, summary = run_strategy(
            dates, load, pv, np.asarray(p_hat[0]), fl, fc, settings,
            start_idx=31, initial_energy=common_feb1_soc,
            prices_plan=p_hat, prices_actual=p_act, scen_prices_fn=scen_prices,
        )
        write_strategy_outputs(args.output / name, name, frame, diag, pair, summary,
                               np.asarray(p_hat[0]))
        summaries.append(summary)
        frames[name] = frame
        print(f"{name} total={summary['formal_period']['total_cost']:,.2f} "
              f"emergency={summary['formal_period']['emergency_cost']:,.2f}", flush=True)

    comparison = pd.DataFrame([
        {"strategy": s["settings"]["strategy"],
         **{k: v for k, v in s["formal_period"].items() if not isinstance(v, dict)}}
        for s in summaries
    ])
    comparison.to_csv(args.output / "question3_comparison.csv", index=False, encoding="utf-8-sig")
    (args.output / "question3_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    main_frame = frames[MAIN_STRATEGY]
    write_main_outputs(args.output, main_frame, p_act)
    for name in strategies:
        meta_path = args.output / name / "question3_metadata.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "question": "问题4-3（波动电价下重算问题3）",
                "price_source": "附件/附件4.xlsx（实际价逐区间结算）",
                "price_forecast": "weekly_persist p_d=p_{d-7}; 1/2-1/7 available-history mean; 1/1 frozen initial-condition day",
                "price_decision_rule": "at each update instant (0:00/6:00/12:00/18:00 re-optimisation) the first interval of the horizon has an already-realized price that replaces the forecast; scenario price first column determinized",
                "price_scenarios": "same 21 historical days, same-day paired residuals, clipped at 0, first column determinized",
                "settlement": "p*min(q0,qA) + 0.5p*(q0-qA)+ + 1.5p*(qA-q0)+ + 5p*b",
                "workbook_template": "附件/附件5/result4-3.xlsx",
                "source_sha256": {
                    "附件/附件4.xlsx": hashlib.sha256((ROOT / "附件/附件4.xlsx").read_bytes()).hexdigest(),
                    **metadata["source_sha256"],
                },
            }
        )
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_q4_3_report(args.output, summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
