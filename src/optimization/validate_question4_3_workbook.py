"""Read-only, independent reconciliation of result4-3.xlsx against the Q4-3
M612 dispatch ledger, with the fluctuating-price four-component bill
recomputed from attachment 4 actuals per day."""
import argparse
import json
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from src.data_pipeline.question4_prices import load_price_actual

ROOT = Path(__file__).resolve().parents[2]


def slot_of(label):
    h, m = map(int, label.split(":"))
    assert m % 10 == 0
    return h * 6 + m // 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/question4/result4-3")
    out = parser.parse_args().output_dir
    f = pd.read_csv(out / "M612" / "question3_schedule.csv", parse_dates=["date"])
    w = openpyxl.load_workbook(out / "result4-3.xlsx", data_only=True)
    template = openpyxl.load_workbook(ROOT / "附件/附件5/result4-3.xlsx", data_only=False)
    assert w.sheetnames == template.sheetnames, (w.sheetnames, template.sheetnames)
    for s in w:
        assert [c.value for c in s[1]] == [c.value for c in template[s.title][1]]
        assert not any(c.data_type == "e" for row in s for c in row)

    plan, adjusted, battery, emergency = w.worksheets
    assert (plan.max_row, plan.max_column) == (335, 147)
    assert (adjusted.max_row, adjusted.max_column) == (335, 147)
    assert (battery.max_row, battery.max_column) == (2005, 6)

    _, prices = load_price_actual(ROOT)  # (365, 144) physical interval-start order
    year0 = pd.Timestamp("2025-01-01")
    max_error = 0.0
    face_bill = 0.0
    settlement_bill = 0.0
    q0_lookup = {d: day.q0_kwh.to_numpy() for d, day in f.groupby("date")}
    qa_lookup = {d: day.qA_kwh.to_numpy() for d, day in f.groupby("date")}
    dates = list(pd.date_range("2025-02-01", "2025-12-31"))

    for j, date in enumerate(dates):
        day = f[f.date == date]
        row_p = list(plan.values)[j + 1]
        row_a = list(adjusted.values)[j + 1]
        assert pd.Timestamp(row_p[0]) == date
        assert pd.Timestamp(row_a[0]) == date

        g0 = day.q0_kwh.to_numpy()
        ga = day.qA_kwh.to_numpy()
        exported_p = np.asarray(row_p[1:145], float)
        exported_a = np.asarray(row_a[1:145], float)
        next_date = date + pd.Timedelta(days=1)
        expected_next0_p = float(q0_lookup[next_date][0]) if next_date in q0_lookup else 0.0
        expected_next0_a = float(qa_lookup[next_date][0]) if next_date in qa_lookup else 0.0
        for exported, g, expected_next0 in [
            (exported_p, g0, expected_next0_p), (exported_a, ga, expected_next0_a)
        ]:
            err1 = float(np.max(np.abs(exported[0:143] - g[1:144])))
            err2 = abs(exported[143] - expected_next0)
            assert err2 < 1e-5
            max_error = max(max_error, err1, err2)
        assert abs(g0.sum() - row_p[145]) < 1e-5
        assert abs(ga.sum() - row_a[145]) < 1e-5
        p_act = prices[(date - year0).days]
        face = float(np.dot(g0, p_act))
        assert abs(face - row_p[146]) < 1e-5
        face_bill += row_p[146]
        settlement = float(day.retained_cost.sum() + day.down_cost.sum() + day.up_cost.sum())
        assert abs(settlement - row_a[146]) < 1e-5
        settlement_bill += row_a[146]

        for k in range(6):
            br = [battery.cell(2 + j * 6 + k, c).value for c in range(1, 7)]
            block = day.iloc[k * 24:(k + 1) * 24]
            assert abs(br[2] - block.charge_kwh.sum()) < 1e-5
            assert abs(br[3] - block.discharge_kwh.sum()) < 1e-5
            if k == 0:
                assert pd.Timestamp(br[0]) == date
                assert abs(br[5] - day.soc_start_kwh.iloc[0]) < 1e-5
            if k == 1:
                assert abs(br[5] - day.soc_end_kwh.iloc[-1]) < 1e-5

    reconstructed = np.zeros(len(f))
    seen = set()
    for date, interval, q in list(emergency.values)[1:]:
        date = pd.Timestamp(date)
        seen.add(date)
        if interval == "无":
            assert q == 0
            continue
        start, end = interval.split("-")
        i, j = slot_of(start), slot_of(end)
        assert 0 <= i < j <= 144
        offset = dates.index(date) * 144
        actual = f.emergency_kwh.to_numpy()[offset + i:offset + j]
        assert (actual > 1e-6).all()
        assert abs(actual.sum() - q) < 1e-5
        assert not reconstructed[offset + i:offset + j].any()
        reconstructed[offset + i:offset + j] = actual
    assert len(seen) == 334
    np.testing.assert_allclose(reconstructed, f.emergency_kwh, atol=1e-6, rtol=0)
    emergency_bill = float(
        np.dot(reconstructed, 5 * prices[[(d - year0).days for d in dates]].ravel())
    )

    retained = float((f.price * np.minimum(f.q0_kwh, f.qA_kwh)).sum())
    down = float((0.5 * f.price * np.maximum(f.q0_kwh - f.qA_kwh, 0)).sum())
    up = float((1.5 * f.price * np.maximum(f.qA_kwh - f.q0_kwh, 0)).sum())
    assert abs(retained + down + up - settlement_bill) < 1e-4
    assert abs(emergency_bill - f.emergency_cost.sum()) < 1e-4
    assert abs(settlement_bill + emergency_bill - f.total_cost.sum()) < 1e-4
    assert max_error < 1e-6

    result = {
        "passed": True,
        "headers_preserved": True,
        "plan_days": 334,
        "plan_intervals": 48096,
        "battery_blocks": 2004,
        "emergency_rows": emergency.max_row - 1,
        "max_workbook_schedule_error_kwh": max_error,
        "independent_face_bill": face_bill,
        "independent_settlement_bill": settlement_bill,
        "independent_emergency_bill": emergency_bill,
        "independent_total_bill": settlement_bill + emergency_bill,
        "four_component_recheck": {"retained": retained, "down": down, "up": up,
                                   "emergency": emergency_bill},
    }
    (out / "question4_3_workbook_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
