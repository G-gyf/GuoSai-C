# -*- coding: utf-8 -*-
"""Question 4-2: fluctuating prices, question-2 pipeline (LDR main strategy).

Confirmed rule set (问题四口径, option B):

* decisions at 0:00 use the weekly-persistence price forecast
  p_hat[d] = p_actual[d-7] (warm-up 1/2-1/7 available-history mean), except
  the first interval (00:00-00:10) whose price is already realized at 0:00
  and replaces the forecast (scenario price column 0 is determinized to it);
  1/1 is a frozen initial-condition day (no plan, no battery action, SOC
  stays 6000 kWh, excluded from statistics); the realized attachment-4
  price settles the bill interval by interval (plan Σp·g, emergency 5p·b);
* the same 21 historical days that supply the net-load residuals also supply
  same-day paired price residuals; scenario prices are clipped at zero and
  enter the LDR calibration objective;
* risk level stays the fixed a-priori 80% quantile, load forecast stays
  weekly persistence, PV forecast stays the seven-day mean.

Run:  python -m src.optimization.question4_2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.optimization.question2 import (
    ROOT,
    T,
    ETA,
    EMIN,
    Settings,
    load_forecast_weekly_persist,
    load_inputs,
    run_case,
    write_outputs,
)
from src.optimization.question2_ldr import (
    LDRSettings,
    run_ldr,
)
from src.data_pipeline.question4_prices import (
    determinize_scenario_first_column,
    load_question4_prices,
    price_scenario_paths,
)

KEY_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]


def _summary_row(label: str, period: str, metrics: dict) -> dict:
    return {"strategy": label, "period": period, **metrics}


def write_q4_2_report(out: Path, summary: dict, baselines: list[dict]) -> None:
    formal = summary["formal_period"]
    full = summary["full_year"]
    by_name = {item["settings"]["name"]: item for item in baselines}
    beta1 = by_name["risk_reserve"]
    beta0 = by_name["risk_greedy"]
    delta_beta1 = formal["total_cost"] - beta1["total_cost"]
    delta_beta0 = formal["total_cost"] - beta0["total_cost"]
    key = pd.read_csv(out / "question2_key_dates.csv", encoding="utf-8-sig")
    key_lines = ["| 日期 | 计划购电量/kWh | 计划费/元 | 紧急电量/kWh | 紧急费/元 | 实际总费/元 |", "|---|---:|---:|---:|---:|---:|"]
    for _, row in key.iterrows():
        key_lines.append(
            f"| {row['date']} | {row['grid_kwh']:,.6f} | {row['planned_cost']:,.2f} "
            f"| {row['emergency_kwh']:,.6f} | {row['emergency_cost']:,.2f} | {row['total_cost']:,.2f} |"
        )
    verdict = (
        "LDR费用低于两个独立全年基线，作为问题4-2主方案。"
        if delta_beta1 < 0 and delta_beta0 < 0
        else "LDR未同时优于两个独立全年基线，如实披露。"
    )
    text = f"""# 问题四-2 波动电价下问题二重算（LDR 主方案）

## 结论摘要

按已确认的波动电价口径（方案 B）：0:00 制定计划时仅首区间（00:00—00:10）电价已实现、其余
区间未知，决策价 = [已实现首区间价, 周持久化预报 p̂_d=p_{{d-7}} 的后续区间]（1/2—1/7 可用历史
均值）；结算全部按附件4 实际价；同一 21 个历史日同日配对价格残差进入 LDR 情景校准，情景价
首列按已实现价确定化；风险分位数固定 0.8。2025-01-01 为冻结初始条件日（无计划、电池不动作、
SOC 恒 6000 kWh、不入统计），1月2日起连续运行至12月31日，2月1日至12月31日为正式期（334 天）。

## 正式期结果（2025-02-01—2025-12-31）

| 指标 | LDR | β=1基线 | β=0基线 |
|---|---:|---:|---:|
| 实际总费用（元） | {formal['total_cost']:,.2f} | {beta1['total_cost']:,.2f} | {beta0['total_cost']:,.2f} |
| 计划购电费用（元） | {formal['planned_cost']:,.2f} | {beta1['planned_cost']:,.2f} | {beta0['planned_cost']:,.2f} |
| 紧急购电费用（元） | {formal['emergency_cost']:,.2f} | {beta1['emergency_cost']:,.2f} | {beta0['emergency_cost']:,.2f} |
| 紧急购电量（kWh） | {formal['emergency_kwh']:,.2f} | {beta1['emergency_kwh']:,.2f} | {beta0['emergency_kwh']:,.2f} |
| 未利用供能（kWh） | {formal['unused_kwh']:,.2f} | {beta1['unused_kwh']:,.2f} | {beta0['unused_kwh']:,.2f} |
| 正式期期初SOC（kWh） | {formal['initial_soc_kwh']:,.2f} | {beta1['initial_result_soc_kwh']:,.2f} | {beta0['initial_result_soc_kwh']:,.2f} |
| 年末SOC（kWh） | {formal['final_soc_kwh']:,.2f} | {beta1['final_soc_kwh']:,.2f} | {beta0['final_soc_kwh']:,.2f} |

相对β=1，LDR费用变化为{delta_beta1:+,.2f}元；相对β=0，变化为{delta_beta0:+,.2f}元。{verdict}

## 全年连续账本（2025-01-02—2025-12-31；1月1日为冻结初始条件日，不计入统计）

- 实际总费用：{full['total_cost']:,.2f}元
- 计划购电费用：{full['planned_cost']:,.2f}元
- 紧急购电费用：{full['emergency_cost']:,.2f}元
- 紧急购电量：{full['emergency_kwh']:,.2f} kWh
- 年初SOC：{full['initial_soc_kwh']:,.2f} kWh；年末SOC：{full['final_soc_kwh']:,.2f} kWh

## 四个指定日期

{chr(10).join(key_lines)}

## 口径记录

- 决策价：0:00 时首区间（00:00—00:10）电价已实现，决策价 = [实际价首列, 周持久化预报 p̂_d=p_{{d-7}} 后续列]；预热期 1/2—1/7 为可用历史均价。2025-01-01 为冻结初始条件日（无计划、电池不动作、SOC 恒 6000 kWh、不入统计）。
- 结算价：附件4 实际价逐区间结算；计划费 Σp·g、紧急费 5p·b。
- 情景：21 历史日同日配对（负载/光伏/价格），情景价 max(0, p̂_d+ε^p_i)，且首列按 0:00 已实现电价确定化；风险曲线仍为负载/光伏情景的 80% 分位数（先验固定）。
- 日末库存价值代理 ν_d = min_t p̂_{{d,t}}/η，逐日计算。
- 附件4 缺 2025-01-01 00:00—00:10 电价，按确认口径用当日 00:10 标签价回填（仅预热账本）。

## 校验

{json.dumps(summary['validation'], ensure_ascii=False, indent=2)}

## 解释限制

与问题二一致的陈述：LDR 为“截断仿射保留阈值＋分段执行规则”，每日有限预算直接搜索不保证
七维经验目标全局最优；价格情景把价格—缺口耦合信息带入控制器，但计划层 LP 对价格采用
确定性等价（预报价），实际账单按实际价结算。
"""
    (out / "问题四-2实施与结果说明.md").write_text(text, encoding="utf-8")


def write_q4_2_outputs(
    out: Path,
    frame,
    daily,
    daily_all,
    summary,
    diagnostics,
    paired,
    baselines,
    template_price_ref: np.ndarray,
    bundle: dict,
) -> None:
    write_outputs(out, frame, daily, [summary, *baselines], template_price_ref)
    daily_all.to_csv(out / "question2_daily_with_warmup.csv", encoding="utf-8-sig")
    diagnostics.to_csv(out / "ldr_daily_parameters.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(out / "same_plan_controller_comparison.csv", index=False, encoding="utf-8-sig")
    rows = [
        _summary_row("LDR", "2025-01-02/2025-12-31", summary["full_year"]),
        _summary_row("LDR", "2025-02-01/2025-12-31", summary["formal_period"]),
    ]
    pd.DataFrame(rows).to_csv(out / "ldr_period_summary.csv", index=False, encoding="utf-8-sig")
    metadata_path = out / "question2_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "question": "问题4-2（波动电价下重算问题2）",
            "price_source": "附件/附件4.xlsx（实际价逐区间结算）",
            "price_forecast": "weekly_persist p_d=p_{d-7}; 1/2-1/7 available-history mean; 1/1 frozen initial-condition day",
            "price_decision_rule": "first interval (00:00-00:10) price is realized at 0:00 and replaces the forecast; scenario price column 0 determinized to the realized price",
            "price_scenarios": "same 21 historical days, same-day paired residuals, clipped at 0, first column determinized",
            "price_missing_fill": "2025-01-01 00:00-00:10 uses the 1/1 00:10 label price (warm-up ledger)",
            "settlement": "plan bill sum p_act*g; emergency 5*p_act*b",
            "risk_alpha": "0.8 fixed a priori (newsvendor ratio independent of price)",
            "workbook_template": "附件/附件5/result4-2.xlsx",
            "source_sha256": {
                "附件/附件4.xlsx": hashlib.sha256((ROOT / "附件/附件4.xlsx").read_bytes()).hexdigest(),
                **metadata["source_sha256"],
            },
        }
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_q4_2_report(out, summary, baselines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/question4/result4-2")
    parser.add_argument("--search-seed", type=int, default=LDRSettings.search_seed)
    parser.add_argument("--search-maxiter", type=int, default=LDRSettings.search_maxiter)
    parser.add_argument("--search-popsize", type=int, default=LDRSettings.search_popsize)
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()
    settings = LDRSettings(
        search_seed=args.search_seed,
        search_maxiter=args.search_maxiter,
        search_popsize=args.search_popsize,
        load_forecast="weekly_persist",
    )
    dates, load, pv, _ = load_inputs()
    bundle = load_question4_prices(ROOT)
    p_act = bundle["price_actual"]
    p_hat = bundle["price_forecast"]
    residual = bundle["price_residual"]

    fl = load_forecast_weekly_persist(load)
    if args.days != 365:
        dates = dates[: args.days]
        load = load[: args.days]
        pv = pv[: args.days]
        fl = fl[: args.days]
        p_act = p_act[: args.days]
        p_hat = p_hat[: args.days]

    def scenario_prices(d: int, m: int) -> np.ndarray | None:
        if m <= 0:
            return None
        paths = price_scenario_paths(d, p_hat, residual, settings.residual_days)
        if paths is None:
            row = p_hat[d].copy()
            row[0] = p_act[d][0]
            return row[None, :]
        if paths.shape[0] != m:
            raise ValueError(f"price scenario count {paths.shape[0]} != net scenario count {m}")
        # Correction 2: the 00:00-start price of the first interval is already
        # realized when the 0:00 plan is made, so every scenario path is
        # determinized at column 0 (no scenario uncertainty there).
        return determinize_scenario_first_column(paths, p_act[d][0])

    started = time.perf_counter()
    frame, daily, daily_all, summary, diagnostics, paired = run_ldr(
        dates, load, pv, np.asarray(p_hat[0]),
        settings, limit=args.days, fl=fl,
        prices_plan=p_hat, prices_actual=p_act,
        price_scenarios_fn=scenario_prices,
    )
    print(f"Q4-2 LDR run finished in {time.perf_counter() - started:.2f}s", flush=True)
    if args.days != 365:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "pilot_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return

    baselines: list[dict] = []
    if not args.skip_baselines:
        for setting in [Settings(), Settings(name="risk_greedy", beta=0.0)]:
            _, _, baseline_summary = run_case(
                dates, load, pv, np.asarray(p_hat[0]), setting, fl_override=fl,
                prices_plan=p_hat, prices_actual=p_act,
            )
            baseline_summary["settings"]["load_forecast"] = "weekly_persist"
            baseline_summary["settings"]["price_source"] = "attachment4_actual_settlement"
            baselines.append(baseline_summary)
    else:
        raise SystemExit("--skip-baselines without cached Q4 baselines is not supported")
    key_idx = dates.get_loc(pd.Timestamp(KEY_DATES[0]))
    write_q4_2_outputs(
        args.output, frame, daily, daily_all, summary, diagnostics, paired,
        baselines, p_act[key_idx], bundle,
    )
    print(json.dumps([summary, *baselines], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
