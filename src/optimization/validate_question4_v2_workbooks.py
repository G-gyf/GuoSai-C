# -*- coding: utf-8 -*-
"""Independent validation of the question-4 v2 workbooks (§10.1 items 7-11).

Checks (all against attachment-4 prices recomputed from source):
  * every workbook row's 144 purchase columns sum to its 全天购电量;
  * result4-2 row cost == sum p_template * g_issue of the same row;
  * result4-3 face cost / adjustment-table settlement cost == the row
    formulas on q0 / (q0, qA);
  * execution ledger == issue ledger + carry-in - carry-out (formal period);
  * 12/31 last column is causally generated (matches the issue ledger and is
    not mechanically zero unless the plan itself is zero);
  * battery blocks and emergency rows match the execution schedule.

Run:  python -m src.optimization.validate_question4_v2_workbooks
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from src.optimization.question2 import ROOT, T

Q4_2_DIR = ROOT / "outputs/question4/v2/result4-2"
Q4_3_DIR = ROOT / "outputs/question4/v2/result4-3"


def template_prices(root: Path = ROOT) -> np.ndarray:
    raw = pd.read_excel(root / "附件/附件4.xlsx", sheet_name=0)
    return raw.iloc[:, 1:145].to_numpy(float)


def _serial_to_date(v) -> pd.Timestamp | None:
    if v is None:
        return None
    return pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(v))


def validate_q4_2(result_dir: Path, p_tpl: np.ndarray) -> dict:
    wb = openpyxl.load_workbook(result_dir / "result4-2.xlsx", data_only=True)
    p_sheet = wb["计划购电量"]
    issue = pd.read_csv(result_dir / "question2_issue.csv", parse_dates=["date"])
    frame = pd.read_csv(result_dir / "question2_schedule.csv", parse_dates=["date"])
    p_act = pd.read_excel(ROOT / "附件/附件4.xlsx", sheet_name=0).iloc[:, 1:145].to_numpy(float)

    max_err = 0.0
    issue_cost_sum = 0.0
    row = 2
    for _, ir in issue.iterrows():
        date = pd.Timestamp(ir["date"])
        if not (pd.Timestamp("2025-02-01") <= date <= pd.Timestamp("2025-12-31")):
            continue
        grid = ir[[f"g{j}" for j in range(T)]].to_numpy(float)
        exported = np.asarray([p_sheet.cell(row, 2 + j).value for j in range(T)], float)
        max_err = max(max_err, float(np.max(np.abs(exported - grid))))
        assert abs(p_sheet.cell(row, 146).value - grid.sum()) < 1e-4, \
            f"row total mismatch on {date.date()}"
        d_idx = (date - pd.Timestamp("2025-01-01")).days
        expected_cost = float(np.dot(p_tpl[d_idx], grid))
        assert abs(p_sheet.cell(row, 147).value - expected_cost) < 1e-4
        issue_cost_sum += expected_cost
        row += 1

    # carry bridge on the execution ledger
    exec_plan = float(frame.planned_cost.sum())
    carry_in = carry_out = 0.0
    for _, ir in issue.iterrows():
        date = pd.Timestamp(ir["date"])
        d_idx = (date - pd.Timestamp("2025-01-01")).days
        last = float(ir["g143"])
        if date == pd.Timestamp("2025-01-31"):
            carry_in = float(p_tpl[d_idx][143] * last)
        if date == pd.Timestamp("2025-12-31"):
            carry_out = float(p_tpl[d_idx][143] * last)
    bridge_error = abs(issue_cost_sum + carry_in - carry_out - exec_plan)

    # 12/31 last column causally generated (from the issue ledger, not zero-fill)
    last_issue = issue[issue.date == pd.Timestamp("2025-12-31")].iloc[0]
    g143 = float(last_issue["g143"])
    exported_143 = float(p_sheet.cell(row - 1, 2 + 143).value)
    assert abs(exported_143 - g143) < 1e-9

    # battery blocks + emergency rows vs execution ledger
    b_sheet = wb["充放电量"]
    e_sheet = wb["紧急购电量"]
    formal = frame[frame.date >= pd.Timestamp("2025-02-01")]
    block_err = 0.0
    r = 2
    for date, day in formal.groupby("date"):
        for k in range(6):
            block = day.iloc[k * 24:(k + 1) * 24]
            assert abs(b_sheet.cell(r, 3).value - block.charge_kwh.sum()) < 1e-4
            assert abs(b_sheet.cell(r, 4).value - block.discharge_kwh.sum()) < 1e-4
            r += 1
    emergency_ok = True
    r = 2
    for date, day in formal.groupby("date"):
        q = day.emergency_kwh.to_numpy()
        t = 0
        while t < T:
            if q[t] <= 1e-6:
                t += 1
                continue
            begin = t
            while t < T and q[t] > 1e-6:
                t += 1
            cell_q = e_sheet.cell(r, 3).value
            if abs(cell_q - q[begin:t].sum()) > 1e-4:
                emergency_ok = False
            r += 1
        if not (q > 1e-6).any():
            r += 1
    return {
        "workbook": "result4-2.xlsx",
        "max_workbook_schedule_error_kwh": max_err,
        "bridge_error": bridge_error,
        "issue_cost_sum": issue_cost_sum,
        "exec_plan_cost": exec_plan,
        "carry_in": carry_in,
        "carry_out": carry_out,
        "dec31_last_column_kwh": g143,
        "emergency_rows_match": emergency_ok,
        "passed": max_err < 1e-6 and bridge_error < 1e-4 and emergency_ok,
    }


def validate_q4_3(result_dir: Path, p_tpl: np.ndarray) -> dict:
    wb = openpyxl.load_workbook(result_dir / "result4-3.xlsx", data_only=True)
    p0 = wb["计划购电量"]
    pA = wb["调整购电量"]
    q0 = pd.read_csv(result_dir / "M612/question3_issue_q0.csv", parse_dates=["date"])
    qA = pd.read_csv(result_dir / "M612/question3_issue_qA.csv", parse_dates=["date"])
    frame = pd.read_csv(result_dir / "M612/question3_schedule.csv", parse_dates=["date"])

    max_err = 0.0
    issue_settle = 0.0
    row = 2
    for _, (r0, rA_) in enumerate(zip(q0.iterrows(), qA.iterrows())):
        date = pd.Timestamp(r0[1]["date"])
        if not (pd.Timestamp("2025-02-01") <= date <= pd.Timestamp("2025-12-31")):
            continue
        g0 = r0[1][[f"g{j}" for j in range(T)]].to_numpy(float)
        gA = rA_[1][[f"g{j}" for j in range(T)]].to_numpy(float)
        exp0 = np.asarray([p0.cell(row, 2 + j).value for j in range(T)], float)
        expA = np.asarray([pA.cell(row, 2 + j).value for j in range(T)], float)
        max_err = max(max_err, float(np.max(np.abs(exp0 - g0))),
                      float(np.max(np.abs(expA - gA))))
        assert abs(p0.cell(row, 146).value - g0.sum()) < 1e-4
        assert abs(pA.cell(row, 146).value - gA.sum()) < 1e-4
        d_idx = (date - pd.Timestamp("2025-01-01")).days
        p = p_tpl[d_idx]
        face = float(np.dot(p, g0))
        settle_row = float(np.dot(p, np.minimum(g0, gA)
                                  + 0.5 * np.maximum(g0 - gA, 0.0)
                                  + 1.5 * np.maximum(gA - g0, 0.0)))
        assert abs(p0.cell(row, 147).value - face) < 1e-4
        assert abs(pA.cell(row, 147).value - settle_row) < 1e-4
        issue_settle += settle_row
        row += 1

    # execution-ledger settlement vs issue settlement with carry bridge
    formal = frame[frame.date >= pd.Timestamp("2025-02-01")]
    exec_settle = float((formal.retained_cost + formal.down_cost + formal.up_cost).sum())
    carry_in = carry_out = 0.0
    for _, (r0, rA_) in enumerate(zip(q0.iterrows(), qA.iterrows())):
        date = pd.Timestamp(r0[1]["date"])
        d_idx = (date - pd.Timestamp("2025-01-01")).days
        g0l = float(r0[1]["g143"])
        gAl = float(rA_[1]["g143"])
        p = p_tpl[d_idx][143]
        cost = p * (min(g0l, gAl) + 0.5 * max(g0l - gAl, 0.0) + 1.5 * max(gAl - g0l, 0.0))
        if date == pd.Timestamp("2025-01-31"):
            carry_in = cost
        if date == pd.Timestamp("2025-12-31"):
            carry_out = cost
    bridge_error = abs(issue_settle + carry_in - carry_out - exec_settle)

    last_q0 = q0[q0.date == pd.Timestamp("2025-12-31")].iloc[0]
    last_qA = qA[qA.date == pd.Timestamp("2025-12-31")].iloc[0]
    g0_143 = float(last_q0["g143"])
    gA_143 = float(last_qA["g143"])
    assert abs(p0.cell(row - 1, 2 + 143).value - g0_143) < 1e-9
    assert abs(pA.cell(row - 1, 2 + 143).value - gA_143) < 1e-9

    return {
        "workbook": "result4-3.xlsx",
        "max_workbook_schedule_error_kwh": max_err,
        "bridge_error": bridge_error,
        "issue_settle": issue_settle,
        "exec_settle": exec_settle,
        "carry_in": carry_in,
        "carry_out": carry_out,
        "dec31_last_column_q0_kwh": g0_143,
        "dec31_last_column_qA_kwh": gA_143,
        "passed": max_err < 1e-6 and bridge_error < 1e-4,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q4-2-dir", type=Path, default=Q4_2_DIR)
    parser.add_argument("--q4-3-dir", type=Path, default=Q4_3_DIR)
    args = parser.parse_args()
    p_tpl = template_prices(ROOT)
    r2 = validate_q4_2(args.q4_2_dir, p_tpl)
    r3 = validate_q4_3(args.q4_3_dir, p_tpl)
    for r in (r2, r3):
        print(json.dumps(r, ensure_ascii=False, indent=2, default=float))
    out2 = args.q4_2_dir / "question4_2_workbook_validation.json"
    out3 = args.q4_3_dir / "question4_3_workbook_validation.json"
    out2.write_text(json.dumps(r2, ensure_ascii=False, indent=2, default=float))
    out3.write_text(json.dumps(r3, ensure_ascii=False, indent=2, default=float))
    if not (r2["passed"] and r3["passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
