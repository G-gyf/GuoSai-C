# -*- coding: utf-8 -*-
"""Question-4 v2 reports: implementation notes, four-key-date paper tables
and the new-vs-old comparison (docs/问题四/问题四优化实施方案.md pipeline).

Run after question4_2_v2 / question4_3_v2 and build_workbook_q4_v2:
    python -m src.reporting.q4_v2_report
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.optimization.question2 import ROOT, T

V2 = ROOT / "outputs/question4/v2"
Q4_2 = V2 / "result4-2"
Q4_3 = V2 / "result4-3"
KEY_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
KEY_SLOTS = [60, 72, 84, 96, 108, 120]  # 10:00 12:00 14:00 16:00 18:00 20:00


def _read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _fmt(x, digits=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:,.{digits}f}"


def q4_2_key_dates(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date in KEY_DATES:
        day = frame[frame.date == pd.Timestamp(date)]
        row = {"date": date, **{
            f"{s // 6:02d}:00-{s // 6:02d}:10": float(day[day.slot == s].grid_kwh.iloc[0])
            for s in KEY_SLOTS
        }}
        row["全天购电量"] = float(day.grid_kwh.sum())
        row["全天计划购电费"] = float(day.planned_cost.sum())
        row["紧急电量"] = float(day.emergency_kwh.sum())
        row["紧急费"] = float(day.emergency_cost.sum())
        row["实际总费"] = float(day.total_cost.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def q4_3_key_dates(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date in KEY_DATES:
        day = frame[frame.date == pd.Timestamp(date)]
        row = {"date": date}
        for s in KEY_SLOTS:
            r = day[day.slot == s].iloc[0]
            row[f"{s // 6:02d}:00 q0"] = float(r.q0_kwh)
            row[f"{s // 6:02d}:00 qA"] = float(r.qA_kwh)
        row["全天q0"] = float(day.q0_kwh.sum())
        row["全天qA"] = float(day.qA_kwh.sum())
        row["面值费"] = float((day.price * day.q0_kwh).sum())
        row["常规结算费"] = float((day.retained_cost + day.down_cost + day.up_cost).sum())
        row["下调费"] = float(day.down_cost.sum())
        row["上调费"] = float(day.up_cost.sum())
        row["紧急费"] = float(day.emergency_cost.sum())
        row["实际总费"] = float(day.total_cost.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _emergency_rows(frame: pd.DataFrame) -> pd.DataFrame:
    from src.optimization.q4_v2_common import emergency_rows
    rows = [r for r in emergency_rows(frame)
            if r[0] in KEY_DATES]
    return pd.DataFrame(rows, columns=["date", "interval", "emergency_kwh"])


def _battery_blocks(frame: pd.DataFrame, date: str) -> pd.DataFrame:
    day = frame[frame.date == pd.Timestamp(date)]
    blocks = []
    for k in range(6):
        b = day.iloc[k * 24:(k + 1) * 24]
        blocks.append({"date": date, "block": f"{k * 4}:00-{(k + 1) * 4}:00",
                       "charge_kwh": float(b.charge_kwh.sum()),
                       "discharge_kwh": float(b.discharge_kwh.sum()),
                       "soc0_kwh": float(day.soc_start_kwh.iloc[0]) if k == 0
                       else (float(day.soc_end_kwh.iloc[-1]) if k == 1 else None)})
    return pd.DataFrame(blocks)


def write_q4_2_report():
    summary = _read_json(Q4_2 / "question2_summary.json")
    baselines = _read_json(Q4_2 / "baselines/baseline_summaries.json")
    sens = _read_json(Q4_2 / "sensitivity/sensitivity_summaries.json")
    bound = _read_json(Q4_2 / "perfect_foresight_bound.json")
    preeval = _read_json(V2 / "pre_evaluation/pre_evaluation.json")
    frame = pd.read_csv(Q4_2 / "question2_schedule.csv", parse_dates=["date"])
    key = q4_2_key_dates(frame)
    formal = summary["formal_period"]

    lines = [
        "# 问题四-2 波动电价下问题二重算（优化方案 v2：价格加权Q80 + 正则化LDR）",
        "",
        "## 结论摘要",
        "",
        f"- 电价预测主模型：**{preeval['chosen_price_model']}**（1/8–1/31 预评价窗选定，2/1 前冻结；"
        f"正则强度 γ={preeval['chosen_gamma']}）；另一模型作为敏感性对照。",
        f"- 主方案正式期（2/1–12/31，334 天）实际总费 **{formal['total_cost']:,.2f} 元**"
        f"（计划费 {formal['planned_cost']:,.2f} ＋ 紧急费 {formal['emergency_cost']:,.2f}），"
        f"紧急购电量 {formal['emergency_kwh']:,.2f} kWh。",
        f"- 2/1 期初 SOC {formal['initial_soc_kwh']:,.2f} kWh（1/1 零计划真实运行日 + 1/2–1/31 WP 预热连续传递），"
        f"年末 {formal['final_soc_kwh']:,.2f} kWh。",
        f"- 发布账本—执行账本桥接：正式期执行计划费 {summary['exec_plan_cost']:,.2f} 元 = "
        f"发布行费用 {summary['issue_cost']:,.2f} ＋ carry-in {summary['carry_in']:,.2f} "
        f"− carry-out {summary['carry_out']:,.2f}（桥接误差 {summary['bridge_error']:.2e} 元）。",
        "",
        "## 策略对照（正式期，同一 2/1 期初 SOC 分叉）",
        "",
        "| 方案 | 计划费 | 紧急费 | 实际总费 | 紧急电量(kWh) |",
        "|---|---:|---:|---:|---:|",
        f"| 价格加权Q80＋正则化LDR（主） | {formal['planned_cost']:,.2f} | {formal['emergency_cost']:,.2f} | {formal['total_cost']:,.2f} | {formal['emergency_kwh']:,.2f} |",
        f"| 普通Q80＋β=1 | {baselines['beta1']['planned_cost']:,.2f} | {baselines['beta1']['emergency_cost']:,.2f} | {baselines['beta1']['total_cost']:,.2f} | {baselines['beta1']['emergency_kwh']:,.2f} |",
        f"| 普通Q80＋β=0 | {baselines['beta0']['planned_cost']:,.2f} | {baselines['beta0']['emergency_cost']:,.2f} | {baselines['beta0']['total_cost']:,.2f} | {baselines['beta0']['emergency_kwh']:,.2f} |",
        f"| 普通Q80＋原LDR（21日窗、同日内终端价值、样本内验收） | {baselines['orig']['planned_cost']:,.2f} | {baselines['orig']['emergency_cost']:,.2f} | {baselines['orig']['total_cost']:,.2f} | {baselines['orig']['emergency_kwh']:,.2f} |",
        f"| 完美预见下界（按主方案 2/1 SOC 匹配） | — | — | {bound['total_cost']:,.2f} | — |",
        "",
        f"主方案相对 β=1 节费 {baselines['beta1']['total_cost'] - formal['total_cost']:,.2f} 元、"
        f"相对 β=0 节费 {baselines['beta0']['total_cost'] - formal['total_cost']:,.2f} 元、"
        f"相对原 LDR 节费 {baselines['orig']['total_cost'] - formal['total_cost']:,.2f} 元；"
        f"与完美预见下界相差 {formal['total_cost'] - bound['total_cost']:,.2f} 元"
        f"（{100 * (formal['total_cost'] - bound['total_cost']) / bound['total_cost']:.2f}%）。",
        "",
        "## 敏感性（主方案单维变动）",
        "",
        "| 变体 | 实际总费 | 紧急费 |",
        "|---|---:|---:|",
    ]
    for name, m in sens.items():
        lines.append(f"| {name} | {m['total_cost']:,.2f} | {m['emergency_cost']:,.2f} |")
    lines += [
        "",
        "## 四个指定日期",
        "",
    ]
    for _, row in key.iterrows():
        lines.append(
            f"- {row['date']}：10:00 {_fmt(row['10:00-10:10'])}、12:00 {_fmt(row['12:00-12:10'])}、"
            f"14:00 {_fmt(row['14:00-14:10'])}、16:00 {_fmt(row['16:00-16:10'])}、"
            f"18:00 {_fmt(row['18:00-18:10'])}、20:00 {_fmt(row['20:00-20:10'])} kWh；"
            f"全天购电 {_fmt(row['全天购电量'])} kWh、计划费 {_fmt(row['全天计划购电费'])} 元、"
            f"紧急 {_fmt(row['紧急电量'])} kWh/{_fmt(row['紧急费'])} 元、"
            f"总费 {_fmt(row['实际总费'])} 元。")
    (Q4_2 / "问题四-2实施与结果说明.md").write_text("\n".join(lines), encoding="utf-8")
    key.to_csv(Q4_2 / "question2_key_dates.csv", index=False, encoding="utf-8-sig")
    return lines


def write_q4_3_report():
    run = _read_json(Q4_3 / "run_summary.json")
    sens = run.get("sensitivity", {})
    frame = pd.read_csv(Q4_3 / "M612/question3_schedule.csv", parse_dates=["date"])
    key = q4_3_key_dates(frame)
    m0, m612 = run["M0"], run["M612"]
    lines = [
        "# 问题四-3 波动电价下问题三重算（优化方案 v2：M0/M6/M612/M61218）",
        "",
        "## 结论摘要",
        "",
        "| 策略 | 更新时间 | 常规结算费 | 下调费 | 上调费 | 紧急费 | 实际总费 | 紧急电量(kWh) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("M0", "M6", "M612", "M61218"):
        m = run[name]
        lines.append(
            f"| {name} | {'0:00' if name == 'M0' else ('0:00,6:00' if name == 'M6' else ('0:00,6:00,12:00' if name == 'M612' else '0:00,6:00,12:00,18:00'))} "
            f"| {m['retained_cost'] + m['down_cost'] + m['up_cost']:,.2f} | {m['down_cost']:,.2f} "
            f"| {m['up_cost']:,.2f} | {m['emergency_cost']:,.2f} | {m['total_cost']:,.2f} "
            f"| {m['emergency_kwh']:,.2f} |")
    lines += [
        "",
        f"- 滚动调整总价值 C_M0 − C_M612 = {m0['total_cost'] - m612['total_cost']:,.2f} 元；"
        f"V6 = C_M0 − C_M6 = {m0['total_cost'] - run['M6']['total_cost']:,.2f} 元；"
        f"V12 = C_M6 − C_M612 = {run['M6']['total_cost'] - m612['total_cost']:,.2f} 元；"
        f"V18 = C_M612 − C_M61218 = {m612['total_cost'] - run['M61218']['total_cost']:,.2f} 元。",
        f"- 主策略 M612 正式期常规结算 {m612['retained_cost'] + m612['down_cost'] + m612['up_cost']:,.2f} 元"
        f"（保留 {m612['retained_cost']:,.2f}＋下调 {m612['down_cost']:,.2f}＋上调 {m612['up_cost']:,.2f}），"
        f"紧急 {m612['emergency_cost']:,.2f} 元。",
        f"- 2/1 共同期初 SOC {m612['initial_soc_kwh']:,.2f} kWh，年末 {m612['final_soc_kwh']:,.2f} kWh。",
        "",
        "## 敏感性（M612 单维变动）",
        "",
        "| 变体 | 实际总费 | 紧急费 |",
        "|---|---:|---:|",
    ]
    for name, m in sens.items():
        lines.append(f"| {name} | {m['total_cost']:,.2f} | {m['emergency_cost']:,.2f} |")
    lines += [
        "",
        "## 四个指定日期",
        "",
    ]
    for _, row in key.iterrows():
        lines.append(
            f"- {row['date']}：q0/qA：10:00 {_fmt(row['10:00 q0'])}/{_fmt(row['10:00 qA'])}、"
            f"12:00 {_fmt(row['12:00 q0'])}/{_fmt(row['12:00 qA'])}、"
            f"16:00 {_fmt(row['16:00 q0'])}/{_fmt(row['16:00 qA'])}、"
            f"18:00 {_fmt(row['18:00 q0'])}/{_fmt(row['18:00 qA'])} kWh；"
            f"全天 q0 {_fmt(row['全天q0'])} / qA {_fmt(row['全天qA'])} kWh、"
            f"面值费 {_fmt(row['面值费'])} 元、常规结算 {_fmt(row['常规结算费'])} 元、"
            f"下调 {_fmt(row['下调费'])} 元、上调 {_fmt(row['上调费'])} 元、"
            f"紧急 {_fmt(row['紧急费'])} 元、总费 {_fmt(row['实际总费'])} 元。")
    (Q4_3 / "问题四-3实施与结果说明.md").write_text("\n".join(lines), encoding="utf-8")
    key.to_csv(Q4_3 / "question3_key_dates.csv", index=False, encoding="utf-8-sig")
    return lines


def write_old_vs_new():
    """Compare v2 (optimized plan) against the previous option-B results."""
    old2 = json.loads((ROOT / "outputs/question4/result4-2/question2_summary.json")
                      .read_text(encoding="utf-8"))
    old3 = json.loads((ROOT / "outputs/question4/result4-3/question3_summary.json")
                      .read_text(encoding="utf-8"))
    new2 = _read_json(Q4_2 / "question2_summary.json")
    new3 = _read_json(Q4_3 / "run_summary.json")
    lines = [
        "# 问题四 新旧口径结果对比（方案B → 优化方案 v2）",
        "",
        "> 口径差异：v2 采用 1/1 零计划真实运行日（方案 B 为冻结日）、发布/执行双账本"
        "（0:00 计划覆盖 [00:10_d, 00:10_{d+1})）、价格加权分位数、42 日衰减情景、"
        "正则化 LDR（验证日验收）与次日终端价值。两种口径的正式期费用不可直接视为"
        "同一口径的『改进』，仅供方案结构对比。",
        "",
        "| 指标 | 方案B（旧） | 优化方案 v2 |",
        "|---|---:|---:|",
    ]
    o2f = [s for s in old2 if isinstance(s, dict) and s.get("settings", {}).get("name") == "ldr_quantile_a08"]
    if o2f:
        o2 = o2f[0]["formal_period"]
        n2 = new2["formal_period"]
        lines += [
            f"| Q4-2 实际总费 | {o2['total_cost']:,.2f} | {n2['total_cost']:,.2f} |",
            f"| Q4-2 计划费 | {o2['planned_cost']:,.2f} | {n2['planned_cost']:,.2f} |",
            f"| Q4-2 紧急费 | {o2['emergency_cost']:,.2f} | {n2['emergency_cost']:,.2f} |",
            f"| Q4-2 紧急电量 | {o2['emergency_kwh']:,.2f} | {n2['emergency_kwh']:,.2f} |",
            f"| Q4-2 2/1 期初 SOC | {o2['initial_soc_kwh']:,.2f} | {n2['initial_soc_kwh']:,.2f} |",
        ]
    o3m = [s for s in old3 if s["settings"]["strategy"] == "M612"][0]["formal_period"]
    n3m = new3["M612"]
    lines += [
        f"| Q4-3 M612 实际总费 | {o3m['total_cost']:,.2f} | {n3m['total_cost']:,.2f} |",
        f"| Q4-3 M612 常规结算费 | {o3m['retained_cost'] + o3m['down_cost'] + o3m['up_cost']:,.2f} "
        f"| {n3m['retained_cost'] + n3m['down_cost'] + n3m['up_cost']:,.2f} |",
        f"| Q4-3 M612 紧急费 | {o3m['emergency_cost']:,.2f} | {n3m['emergency_cost']:,.2f} |",
        f"| Q4-3 M0 实际总费 | {[s for s in old3 if s['settings']['strategy'] == 'M0'][0]['formal_period']['total_cost']:,.2f} "
        f"| {new3['M0']['total_cost']:,.2f} |",
    ]
    (V2 / "问题四新旧口径对比.md").write_text("\n".join(lines), encoding="utf-8")
    return lines


def main():
    write_q4_2_report()
    write_q4_3_report()
    write_old_vs_new()
    print("reports written")


if __name__ == "__main__":
    main()
