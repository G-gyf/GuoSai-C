"""Read-only, independent reconciliation of exported workbook to actual dispatch."""
import json
import argparse
from pathlib import Path
import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/question2')
    out = parser.parse_args().output_dir
    f = pd.read_csv(out/'question2_schedule.csv',parse_dates=['date'])
    w = openpyxl.load_workbook(out/'result2.xlsx',data_only=True)
    template = openpyxl.load_workbook(ROOT/'附件/附件5/result2.xlsx',data_only=False)
    assert w.sheetnames == template.sheetnames
    for s in w:
        assert [c.value for c in s[1]] == [c.value for c in template[s.title][1]]
        assert not any(c.data_type == 'e' for row in s for c in row)
    p,b,e = w.worksheets
    assert (p.max_row,p.max_column) == (335,147)
    assert (b.max_row,b.max_column) == (2005,6)
    price_raw = pd.read_excel(ROOT/'附件/附件1.xlsx').iloc[:,1].to_numpy(float)
    prices = np.r_[price_raw[-1],price_raw[:-1]]
    max_error = 0.
    planned_bill = 0.
    purchase_rows = list(p.values)
    grid_lookup = {d: day.grid_kwh.to_numpy() for d, day in f.groupby('date')}
    for j,(date,day) in enumerate(f.groupby('date')):
        row = purchase_rows[j+1]
        assert pd.Timestamp(row[0]) == date
        exported = np.asarray(row[1:145],float)
        g = day.grid_kwh.to_numpy()
        err1 = float(np.max(np.abs(exported[0:143]-g[1:144])))
        next_date = date + pd.Timedelta(days=1)
        expected_next0 = float(grid_lookup[next_date][0]) if next_date in grid_lookup else 0.0
        err2 = abs(exported[143]-expected_next0)
        assert err2 < 1e-5
        max_error = max(max_error, err1, err2)
        assert abs(g.sum()-row[145]) < 1e-5
        assert abs(np.dot(g,prices)-row[146]) < 1e-5
        planned_bill += row[146]
        for k in range(6):
            br = [b.cell(2+j*6+k,c).value for c in range(1,7)]
            block=day.iloc[k*24:(k+1)*24]
            assert abs(br[2]-block.charge_kwh.sum()) < 1e-5
            assert abs(br[3]-block.discharge_kwh.sum()) < 1e-5
            if k == 0:
                assert pd.Timestamp(br[0]) == date
                assert abs(br[5]-day.soc_start_kwh.iloc[0]) < 1e-5
            if k == 1:
                assert abs(br[5]-day.soc_end_kwh.iloc[-1]) < 1e-5
    reconstructed = np.zeros(len(f))
    dates = list(pd.date_range('2025-02-01','2025-12-31'))
    seen = set()
    for date,interval,q in list(e.values)[1:]:
        date=pd.Timestamp(date)
        seen.add(date)
        if interval == '无':
            assert q == 0
            continue
        start,end=interval.split('-')
        def slot(s):
            h,m=map(int,s.split(':'))
            assert m%10==0
            return h*6+m//10
        i,j=slot(start),slot(end)
        assert 0<=i<j<=144
        offset=dates.index(date)*144
        actual=f.emergency_kwh.to_numpy()[offset+i:offset+j]
        assert (actual>1e-6).all()
        assert abs(actual.sum()-q)<1e-5
        assert not reconstructed[offset+i:offset+j].any()
        reconstructed[offset+i:offset+j]=actual
    assert len(seen)==334
    np.testing.assert_allclose(reconstructed,f.emergency_kwh,atol=1e-6,rtol=0)
    emergency_bill=float(np.dot(reconstructed,5*np.tile(prices,334)))
    assert abs(planned_bill-f.planned_cost.sum())<1e-4
    assert abs(emergency_bill-f.emergency_cost.sum())<1e-4
    assert max_error<1e-6
    result={'passed':True,'headers_preserved':True,'plan_days':334,'plan_intervals':48096,
            'battery_blocks':2004,'emergency_rows':e.max_row-1,
            'max_workbook_schedule_error_kwh':max_error,
            'independent_planned_bill':planned_bill,'independent_emergency_bill':emergency_bill,
            'independent_total_bill':planned_bill+emergency_bill}
    (out/'question2_workbook_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
