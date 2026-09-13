# -*- coding: utf-8 -*-
"""Assemble the question-3 (第一小问, M612 主策略) four-key-date result tables,
strictly following the official 题目表1/表2/表3 layouts.

表1  微网在指定时间段的购电量及全天的购电量和购电费
     —— 购电量 = 最终生效购电量 qA（0:00 计划 q0 经 6:00/12:00 调整后的结算量）；
        全天购电费 = 当日实际总费（按 qA 分段净结算 + 紧急购电）。
表2  储能设备在指定时间段的充放电量及 0:00 和 24:00 的储电量
表3  微网在指定日期的紧急购电量（四个日期并排）

Run:  python -m src.reporting.question3_paper_tables
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
KEY_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
HOURS_LEFT = [10, 12, 14]     # 表1 左列组
HOURS_RIGHT = [16, 18, 20]    # 表1 右列组
BLOCKS_LEFT = ["0:00-4:00", "4:00-8:00", "8:00-12:00"]     # 表2 左列组
BLOCKS_RIGHT = ["12:00-16:00", "16:00-20:00", "20:00-24:00"]  # 表2 右列组
M612 = ROOT / "outputs/question3/current/M612"
CUR = ROOT / "outputs/question3/current"


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sched = pd.read_csv(M612 / "question3_schedule.csv",
                        parse_dates=["date"], encoding="utf-8-sig")
    batt = pd.read_csv(CUR / "question3_key_battery.csv", encoding="utf-8-sig")
    emer = pd.read_csv(CUR / "question3_emergency.csv", encoding="utf-8-sig")
    daily = pd.read_csv(M612 / "question3_daily.csv",
                        parse_dates=["date"], encoding="utf-8-sig")
    return sched, batt, emer, daily


def table1_grid(date: str, day: pd.DataFrame) -> list[str]:
    """题目表1 格式的一张（单个日期）：
    第1行 10:00-10:10 / 12:00-12:10 / 14:00-14:10，
    第2行 16:00-16:10 / 18:00-18:10 / 20:00-20:10，
    第3行 全天购电量 / 全天购电费（严格按题面行列顺序）。"""
    qa = {h: float(day[day.slot == h * 6].qA_kwh.iloc[0]) for h in HOURS_LEFT + HOURS_RIGHT}
    total_qa = float(day.qA_kwh.sum())
    total_cost = float(day.total_cost.sum())
    lines = [
        f"**{date}**",
        "",
        "| 时间段 | 购电量 | 时间段 | 购电量 | 时间段 | 购电量 |",
        "|---|---:|---|---:|---|---:|",
        f"| 10:00-10:10 | {qa[10]:,.3f} | 12:00-12:10 | {qa[12]:,.3f} "
        f"| 14:00-14:10 | {qa[14]:,.3f} |",
        f"| 16:00-16:10 | {qa[16]:,.3f} | 18:00-18:10 | {qa[18]:,.3f} "
        f"| 20:00-20:10 | {qa[20]:,.3f} |",
        f"| 全天购电量 | {total_qa:,.3f} | 全天购电费 | {total_cost:,.2f} |  |  |",
        "",
    ]
    return lines


def table2_grid(date: str, blocks: dict[str, pd.Series], s0: float, s24: float) -> list[str]:
    """题目表2 格式的一张（单个日期）：
    第1行 0:00-4:00 / 4:00-8:00，第2行 8:00-12:00 / 12:00-16:00，
    第3行 16:00-20:00 / 20:00-24:00，第4行 0:00储电量 / 24:00储电量（严格按题面行列顺序）。"""
    def cell(b: str, col: str) -> str:
        return f"{float(blocks[b][col]):,.2f}"

    lines = [
        f"**{date}**",
        "",
        "| 时间段 | 充电量 | 放电量 | 时间段 | 充电量 | 放电量 |",
        "|---|---:|---:|---|---:|---:|",
        f"| 0:00-4:00 | {cell('0:00-4:00', 'charge_kwh')} | {cell('0:00-4:00', 'discharge_kwh')} "
        f"| 4:00-8:00 | {cell('4:00-8:00', 'charge_kwh')} | {cell('4:00-8:00', 'discharge_kwh')} |",
        f"| 8:00-12:00 | {cell('8:00-12:00', 'charge_kwh')} | {cell('8:00-12:00', 'discharge_kwh')} "
        f"| 12:00-16:00 | {cell('12:00-16:00', 'charge_kwh')} | {cell('12:00-16:00', 'discharge_kwh')} |",
        f"| 16:00-20:00 | {cell('16:00-20:00', 'charge_kwh')} | {cell('16:00-20:00', 'discharge_kwh')} "
        f"| 20:00-24:00 | {cell('20:00-24:00', 'charge_kwh')} | {cell('20:00-24:00', 'discharge_kwh')} |",
        f"| 0:00 储电量 | {s0:,.2f} |  | 24:00 储电量 | {s24:,.2f} |  |",
        "",
    ]
    return lines


def table3_grid(emer: pd.DataFrame) -> list[str]:
    """题目表3 格式：四个日期并排（2025.3.20 等），各含 时间段|购电量 两列。"""
    cols: list[list[str]] = []
    for date in KEY_DATES:
        sub = emer[emer.date == date]
        sub = sub[(sub.emergency_kwh > 1e-6) & (sub.interval.astype(str) != "无")]
        rows: list[str] = []
        if len(sub):
            for _, r in sub.iterrows():
                rows.append(f"{r['interval']} | {float(r['emergency_kwh']):,.2f}")
        else:
            rows.append("无 | 0.00")
        cols.append(rows)
    n = max(len(c) for c in cols)
    lines = [
        "| 2025.3.20 |  | 2025.6.21 |  | 2025.9.23 |  | 2025.12.21 |  |",
        "|---|---|---|---|---|---|---|---|",
        "| 时间段 | 购电量 | 时间段 | 购电量 | 时间段 | 购电量 | 时间段 | 购电量 |",
    ]
    for i in range(n):
        cells: list[str] = []
        for c in cols:
            if i < len(c):
                cells.append(c[i])
            else:
                cells.append(" | ")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def main():
    sched, batt, emer, daily = load()
    blocks_by_date = {}
    for _, r in batt.iterrows():
        blocks_by_date.setdefault(r["date"], {})[r["block"]] = r
    soc = {
        str(r["date"].date()): (float(r["soc_start_kwh"]), float(r["soc_end_kwh"]))
        for _, r in daily[daily.date.isin(pd.to_datetime(KEY_DATES))].iterrows()
    }
    lines = [
        "# 问题三第一小问（M612 主策略）四个指定日期的结果表（严格按题目表1/表2/表3 格式）",
        "",
        "指定日期：2025-03-20、2025-06-21、2025-09-23、2025-12-21。",
        "表1 的购电量为**最终生效购电量 qA**（0:00 原计划经 6:00/12:00 调整后的结算量）；"
        "全天购电费为该日实际总购电费（按 qA 分段净结算＋紧急购电）。"
        "原计划 q0 与调整量见 result3.xlsx 的“计划购电量”“调整购电量”工作表。",
        "",
        "## 表1 微网在指定时间段的购电量及全天的购电量和购电费（kWh / 元）",
        "",
    ]
    for date in KEY_DATES:
        lines += table1_grid(date, sched[sched.date == pd.Timestamp(date)])
    lines += ["## 表2 储能设备在指定时间段的充放电量及 0:00 和 24:00 的储电量（kWh）", ""]
    for date in KEY_DATES:
        s0, s24 = soc[date]
        lines += table2_grid(date, blocks_by_date[date], s0, s24)
    lines += ["## 表3 微网在指定日期的紧急购电量（kWh）", ""]
    lines += table3_grid(emer)

    text = "\n".join(lines)
    out = ROOT / "outputs/question3/current/paper/问题三第一小问_四日结果表.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(out)
    print(text)


if __name__ == "__main__":
    main()
