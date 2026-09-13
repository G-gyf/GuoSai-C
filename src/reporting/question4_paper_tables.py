# -*- coding: utf-8 -*-
"""Assemble the question-4 four-key-date tables (题目表1/表2/表3 format) from
the Q4-2 and Q4-3 result bundles.

Run:  python -m src.reporting.question4_paper_tables
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
KEY_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
HOURS = [10, 12, 14, 16, 18, 20]


def load_q42(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sched = pd.read_csv(root / "outputs/question4/result4-2/question2_schedule.csv",
                        parse_dates=["date"], encoding="utf-8-sig")
    batt = pd.read_csv(root / "outputs/question4/result4-2/question2_key_battery.csv",
                       encoding="utf-8-sig")
    emer = pd.read_csv(root / "outputs/question4/result4-2/question2_emergency.csv",
                       encoding="utf-8-sig")
    return sched, batt, emer


def load_q43(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sched = pd.read_csv(root / "outputs/question4/result4-3/M612/question3_schedule.csv",
                        parse_dates=["date"], encoding="utf-8-sig")
    batt = pd.read_csv(root / "outputs/question4/result4-3/question3_key_battery.csv",
                       encoding="utf-8-sig")
    emer = pd.read_csv(root / "outputs/question4/result4-3/question3_emergency.csv",
                       encoding="utf-8-sig")
    daily = pd.read_csv(root / "outputs/question4/result4-3/M612/question3_daily.csv",
                        parse_dates=["date"], encoding="utf-8-sig")
    return sched, batt, emer, daily


def battery_table(batt: pd.DataFrame, value_cols: list[str]) -> str:
    lines = ["| 日期 | 时段 | 充电量 | 放电量 | 时段 | 充电量 | 放电量 |",
             "|---|---|---:|---:|---|---:|---:|"]
    rows = {}
    for _, r in batt.iterrows():
        rows.setdefault(r["date"], {})[r["block"]] = r
    for date in KEY_DATES:
        blocks = rows[date]
        a, b, c = blocks["0:00-4:00"], blocks["4:00-8:00"], blocks["8:00-12:00"]
        d, e, f = blocks["12:00-16:00"], blocks["16:00-20:00"], blocks["20:00-24:00"]
        lines.append(
            f"| {date} | 0:00-4:00 | {a['charge_kwh']:,.6f} | {a['discharge_kwh']:,.6f} "
            f"| 4:00-8:00 | {b['charge_kwh']:,.6f} | {b['discharge_kwh']:,.6f} |")
        lines.append(
            f"|  | 8:00-12:00 | {c['charge_kwh']:,.6f} | {c['discharge_kwh']:,.6f} "
            f"| 12:00-16:00 | {d['charge_kwh']:,.6f} | {d['discharge_kwh']:,.6f} |")
        lines.append(
            f"|  | 16:00-20:00 | {e['charge_kwh']:,.6f} | {e['discharge_kwh']:,.6f} "
            f"| 20:00-24:00 | {f['charge_kwh']:,.6f} | {f['discharge_kwh']:,.6f} |")
    return "\n".join(lines)


def main():
    out = ROOT / "outputs/question4"
    s2, b2, e2 = load_q42(ROOT)
    s3, b3, e3, d3 = load_q43(ROOT)
    lines = ["# 问题四 四个指定日期的结果表（题目表1/表2/表3 格式）", ""]

    # ---------------- Q4-2 ----------------
    lines += ["## 一、问题4-2（波动电价下的问题二，LDR 主方案）", ""]
    lines += ["### 表1 指定区间与全天计划购电（kWh / 元）", ""]
    lines += ["| 日期 | 时刻 | 购电量 | 时刻 | 购电量 | 全天购电量 | 全天计划购电费 | 实际总费 |",
              "|---|---|---:|---|---:|---:|---:|---:|"]
    for date in KEY_DATES:
        day = s2[s2.date == pd.Timestamp(date)]
        vals = {h: float(day[day.slot == h * 6].grid_kwh.iloc[0]) for h in HOURS}
        lines.append(
            f"| {date} | 10:00-10:10 | {vals[10]:,.6f} | 12:00-12:10 | {vals[12]:,.6f} "
            f"| {day.grid_kwh.sum():,.6f} | {day.planned_cost.sum():,.2f} | {day.total_cost.sum():,.2f} |")
        lines.append(
            f"|  | 14:00-14:10 | {vals[14]:,.6f} | 16:00-16:10 | {vals[16]:,.6f} | | | |")
        lines.append(
            f"|  | 18:00-18:10 | {vals[18]:,.6f} | 20:00-20:10 | {vals[20]:,.6f} | | | |")
    lines += ["", "### 表2 指定日期储能充放电与边界库存（kWh）", "", battery_table(b2, []), ""]
    lines += ["### 表3 指定日期紧急购电区间（kWh）", "", "| 日期 | 紧急购电时间段 | 紧急购电量 |", "|---|---|---:|"]
    for date in KEY_DATES:
        sub = e2[e2.date == date]
        if (sub.emergency_kwh > 1e-6).any():
            for _, r in sub[sub.emergency_kwh > 1e-6].iterrows():
                lines.append(f"| {date} | {r['interval']} | {r['emergency_kwh']:,.6f} |")
        else:
            lines.append(f"| {date} | 无 | 0.000000 |")
    lines += [""]

    # ---------------- Q4-3 ----------------
    lines += ["## 二、问题4-3（波动电价下的问题三，M612 主策略）", ""]
    lines += ["### 表1 指定区间原计划 q0 与最终生效 qA（kWh）", ""]
    lines += ["| 日期 | 时刻 | q0 | qA | 时刻 | q0 | qA |", "|---|---|---:|---:|---|---:|---:|"]
    for date in KEY_DATES:
        day = s3[s3.date == pd.Timestamp(date)]
        vals = {h: (float(day[day.slot == h * 6].q0_kwh.iloc[0]),
                    float(day[day.slot == h * 6].qA_kwh.iloc[0])) for h in HOURS}
        lines.append(
            f"| {date} | 10:00-10:10 | {vals[10][0]:,.6f} | {vals[10][1]:,.6f} "
            f"| 12:00-12:10 | {vals[12][0]:,.6f} | {vals[12][1]:,.6f} |")
        lines.append(
            f"|  | 14:00-14:10 | {vals[14][0]:,.6f} | {vals[14][1]:,.6f} "
            f"| 16:00-16:10 | {vals[16][0]:,.6f} | {vals[16][1]:,.6f} |")
        lines.append(
            f"|  | 18:00-18:10 | {vals[18][0]:,.6f} | {vals[18][1]:,.6f} "
            f"| 20:00-20:10 | {vals[20][0]:,.6f} | {vals[20][1]:,.6f} |")
    lines += ["", "### 指定日期全天汇总（kWh / 元）", ""]
    lines += ["| 日期 | 全天q0 | 全天qA | 面值费 | 常规结算费 | 下调费 | 上调费 | 紧急费 | 实际总费 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for date in KEY_DATES:
        day = s3[s3.date == pd.Timestamp(date)]
        q0, qA = day.q0_kwh.sum(), day.qA_kwh.sum()
        face = float((day.price * day.q0_kwh).sum())
        retained = float((day.price * pd.concat([day.q0_kwh, day.qA_kwh], axis=1).min(axis=1)).sum())
        down = float((0.5 * day.price * (day.q0_kwh - day.qA_kwh).clip(lower=0)).sum())
        up = float((1.5 * day.price * (day.qA_kwh - day.q0_kwh).clip(lower=0)).sum())
        lines.append(
            f"| {date} | {q0:,.6f} | {qA:,.6f} | {face:,.2f} | {retained + down + up:,.2f} "
            f"| {down:,.2f} | {up:,.2f} | {day.emergency_cost.sum():,.2f} | {day.total_cost.sum():,.2f} |")
    lines += ["", "### 表2 指定日期储能充放电与边界库存（kWh）", "", battery_table(b3, []), ""]
    lines += ["### 表3 指定日期紧急购电区间（kWh）", "", "| 日期 | 紧急购电时间段 | 紧急购电量 |", "|---|---|---:|"]
    for date in KEY_DATES:
        sub = e3[e3.date == date]
        if (sub.emergency_kwh > 1e-6).any():
            for _, r in sub[sub.emergency_kwh > 1e-6].iterrows():
                lines.append(f"| {date} | {r['interval']} | {r['emergency_kwh']:,.6f} |")
        else:
            lines.append(f"| {date} | 无 | 0.000000 |")

    text = "\n".join(lines) + "\n"
    (out / "问题四_四日结果表.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
