"""Causal Q2: weekly-persistence load forecast, seven-day mean PV forecast,
21 residual days, risk LP and real-time storage.

Only attachments 1 (tariffs and the warm-up load curve), 2 (actuals), and
the official output template are inputs. Actual observations are revealed to
the controller one slot at a time. The planning objective is a proxy, never
reported as the realized bill. The adopted Q2 convention is
``load_forecast_weekly_persist`` for load (L_{d-7}, attachment 1 for d < 7)
and the seven-day same-slot mean for PV; the mean-based load forecast
remains available via ``forecasts`` for comparison studies only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, lil_matrix

from src.optimization.question1 import load_question1_inputs
from src.data_pipeline.build_timeline import validate_source_headers

ROOT = Path(__file__).resolve().parents[2]
T, ETA, EMIN, EMAX, S = 144, .9, 1200., 10800., 5000 / 6
KEY_DATES = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']


@dataclass(frozen=True)
class Settings:
    name: str = 'risk_reserve'
    risk: bool = True
    alpha: float = .8
    beta: float = 1.
    forecast_days: int = 7
    residual_days: int = 21


def load_inputs(root=ROOT):
    raw = pd.read_excel(root / '附件/附件2.xlsx', sheet_name=None)
    lf, vf = raw['小区负载'], raw['光伏发电实际功率']
    validate_source_headers(lf.columns[1:])
    validate_source_headers(vf.columns[1:])
    dates = pd.DatetimeIndex(pd.to_datetime(lf.iloc[:, 0]))
    assert dates.equals(pd.DatetimeIndex(pd.to_datetime(vf.iloc[:, 0])))
    assert dates.equals(pd.date_range('2025-01-01', '2025-12-31'))
    load, pv = lf.iloc[:, 1:].to_numpy(float) / 6, vf.iloc[:, 1:].to_numpy(float) / 6
    assert load.shape == pv.shape == (365, T)
    assert np.isfinite(load).all() and np.isfinite(pv).all()
    assert (load >= 0).all() and (pv >= 0).all()
    prices = load_question1_inputs(root / '附件/附件1.xlsx').price_yuan_per_kwh.to_numpy()
    assert (prices > 0).all()
    return dates, load, pv, prices


def forecasts(load, pv):
    fl, fv = np.full_like(load, np.nan), np.full_like(pv, np.nan)
    for d in range(1, len(load)):
        fl[d] = load[max(0, d-7):d].mean(axis=0)
        fv[d] = pv[max(0, d-7):d].mean(axis=0)
    return fl, fv


def load_forecast_weekly_persist(load, representative):
    """Weekly persistence load forecast L_{d-7}; attachment 1 for d < 7.

    Mirrors the b1 scheme of the load forecast comparison (问题二预测.docx):
    the day-0 row stays NaN and is never used (cold-start zero plan).  All
    values are in kWh per 10-minute interval.
    """
    fl = np.full_like(load, np.nan)
    for d in range(1, len(load)):
        fl[d] = load[d - 7] if d >= 7 else representative
    return fl


def planning_net(d, load, pv, fl, fv, setting):
    """No row d or later of actuals is accessed here."""
    net = fl[d] - fv[d]
    first = max(7, d-setting.residual_days)
    count = max(0, d-first)
    if count == 0 or not setting.risk:
        return net.copy(), count
    sl = np.maximum(0, fl[d] + load[first:d] - fl[first:d])
    sv = np.maximum(0, fv[d] + pv[first:d] - fv[first:d])
    q = np.quantile(sl-sv, setting.alpha, axis=0, method='linear')
    return np.maximum(net, q), count


def equality_matrix(n):
    # Variables g, C, D, U, E, all measured in kWh.
    a = lil_matrix((2*n, 5*n))
    for t in range(n):
        a[t, t], a[t, n+t], a[t, 2*n+t], a[t, 3*n+t] = 1, -1, 1, -1
        a[n+t, 4*n+t], a[n+t, n+t], a[n+t, 2*n+t] = 1, -ETA, 1/ETA
        if t:
            a[n+t, 4*n+t-1] = -1
    return a.tocsr()


def solve_plan(net, prices, initial, terminal_value):
    n = len(net)
    a = equality_matrix(n)
    b = np.r_[net, initial, np.zeros(n-1)]
    bounds = [(0, None)]*n + [(0, S)]*(2*n) + [(0, None)]*n + [(EMIN, EMAX)]*n
    obj = np.zeros(5*n)
    obj[:n], obj[-1] = prices, -terminal_value
    first = linprog(obj, A_eq=a, b_eq=b, bounds=bounds, method='highs')
    if not first.success:
        raise RuntimeError(first.message)
    throughput = np.zeros(5*n)
    throughput[n:3*n] = 1
    second = linprog(throughput, A_eq=a, b_eq=b, bounds=bounds,
                     A_ub=csr_matrix(obj.reshape(1, -1)), b_ub=[first.fun+1e-6], method='highs')
    if not second.success:
        raise RuntimeError(second.message)
    x = second.x.copy()
    x[np.abs(x) < 1e-8] = 0
    assert np.max(np.abs(a@x-b)) < 1e-5
    assert np.min(x) >= -1e-6
    assert np.max(np.minimum(x[n:2*n], x[2*n:3*n])) < 1e-5
    return x.reshape(5, n), float(first.fun)


def execute_slot(load, pv, grid, energy, reserve):
    r = load-pv-grid
    charge = max(0., min(max(-r, 0.), S, (EMAX-energy)/ETA))
    discharge = max(0., min(max(r, 0.), S, ETA*max(energy-reserve, 0.)))
    emergency = max(r-discharge, 0.)
    unused = max(-r-charge, 0.)
    end = energy+ETA*charge-discharge/ETA
    return charge, discharge, emergency, unused, end


def run_case(dates, load, pv, prices, setting, fl_override=None):
    start = time.perf_counter()
    fl, fv = forecasts(load, pv)
    if fl_override is not None:
        fl = fl_override
    # Lowest tariff is the fixed valley replacement-cost approximation.
    nu = float(prices.min()/ETA)
    energy, records = 6000., []
    for d, date in enumerate(dates):
        if d:
            net, count = planning_net(d, load, pv, fl, fv, setting)
            plan, obj = solve_plan(net, prices, energy, nu)
        else:
            net, count, obj = np.zeros(T), 0, 0.
            plan = np.zeros((5, T))
            plan[4] = EMIN
        reserve = EMIN + setting.beta*(plan[4]-EMIN)
        for t in range(T):
            initial = energy
            c, dis, buy, unused, energy = execute_slot(load[d,t], pv[d,t], plan[0,t], initial, reserve[t])
            records.append((date, t, date+pd.Timedelta(minutes=10*t),
                            load[d,t], pv[d,t], prices[t], fl[d,t], fv[d,t], net[t],
                            count, plan[0,t], plan[1,t], plan[2,t], plan[4,t], reserve[t],
                            initial, c, dis, buy, unused, energy,
                            prices[t]*plan[0,t], 5*prices[t]*buy))
        if (d+1) % 90 == 0:
            print(f'{setting.name}: {d+1}/{len(dates)} days', flush=True)
    columns = ['date','slot','interval_start','load_kwh','pv_kwh','price','forecast_load_kwh',
               'forecast_pv_kwh','planning_net_kwh','residual_count','grid_kwh','plan_charge_kwh',
               'plan_discharge_kwh','plan_soc_kwh','reserve_kwh','soc_start_kwh','charge_kwh',
               'discharge_kwh','emergency_kwh','unused_kwh','soc_end_kwh','planned_cost','emergency_cost']
    frame = pd.DataFrame.from_records(records, columns=columns)
    frame['total_cost'] = frame.planned_cost+frame.emergency_cost
    valid = validate_schedule(frame)
    formal = frame[frame.date >= '2025-02-01']
    daily = formal.groupby('date').agg(
        grid_kwh=('grid_kwh','sum'), charge_kwh=('charge_kwh','sum'), discharge_kwh=('discharge_kwh','sum'),
        emergency_kwh=('emergency_kwh','sum'), unused_kwh=('unused_kwh','sum'),
        planned_cost=('planned_cost','sum'), emergency_cost=('emergency_cost','sum'), total_cost=('total_cost','sum'),
        soc_start_kwh=('soc_start_kwh','first'), soc_end_kwh=('soc_end_kwh','last'))
    summary = {'settings':asdict(setting), 'terminal_value':nu, 'seconds':time.perf_counter()-start,
               'result_days':len(daily), 'result_intervals':len(formal),
               **{k:float(formal[k].sum()) for k in ['grid_kwh','charge_kwh','discharge_kwh',
                    'emergency_kwh','unused_kwh','planned_cost','emergency_cost','total_cost']},
               'initial_result_soc_kwh':float(formal.soc_start_kwh.iloc[0]),
               'final_soc_kwh':float(formal.soc_end_kwh.iloc[-1]),
               'emergency_intervals':int((formal.emergency_kwh>1e-6).sum()), 'validation':valid}
    return frame, daily, summary


def validate_schedule(f):
    balance = f.grid_kwh+f.emergency_kwh+f.pv_kwh+f.discharge_kwh-f.load_kwh-f.charge_kwh-f.unused_kwh
    state = f.soc_start_kwh+ETA*f.charge_kwh-f.discharge_kwh/ETA-f.soc_end_kwh
    checks = {
        'max_balance_error_kwh':float(np.abs(balance).max()),
        'max_state_error_kwh':float(np.abs(state).max()),
        'max_continuity_error_kwh':float(np.abs(f.soc_start_kwh.to_numpy()[1:]-f.soc_end_kwh.to_numpy()[:-1]).max()),
        'min_soc_kwh':float(min(f.soc_start_kwh.min(),f.soc_end_kwh.min())),
        'max_soc_kwh':float(max(f.soc_start_kwh.max(),f.soc_end_kwh.max())),
        'max_charge_kwh':float(f.charge_kwh.max()), 'max_discharge_kwh':float(f.discharge_kwh.max()),
        'simultaneous_charge_discharge':int(((f.charge_kwh>1e-6)&(f.discharge_kwh>1e-6)).sum()),
        'emergency_while_charging':int(((f.charge_kwh>1e-6)&(f.emergency_kwh>1e-6)).sum()),
    }
    assert max(checks[k] for k in ['max_balance_error_kwh','max_state_error_kwh','max_continuity_error_kwh']) < 1e-5
    assert checks['min_soc_kwh'] >= EMIN-1e-5 and checks['max_soc_kwh'] <= EMAX+1e-5
    assert max(checks['max_charge_kwh'],checks['max_discharge_kwh']) <= S+1e-5
    assert checks['simultaneous_charge_discharge'] == checks['emergency_while_charging'] == 0
    assert (f[['grid_kwh','charge_kwh','discharge_kwh','emergency_kwh','unused_kwh']] >= -1e-7).all().all()
    checks['passed'] = True
    return checks


def time_label(slot):
    return f'{slot//6:02d}:{slot%6*10:02d}'


def summarize_emergency(frame):
    rows = []
    for date, day in frame.groupby('date'):
        q = day.emergency_kwh.to_numpy()
        t = 0
        while t < T:
            if q[t] <= 1e-6:
                t += 1
                continue
            begin = t
            while t < T and q[t] > 1e-6:
                t += 1
            rows.append([str(date.date()), f'{time_label(begin)}-{time_label(t)}', float(q[begin:t].sum())])
        if not (q>1e-6).any():
            rows.append([str(date.date()), '无', 0.])
    return rows


def write_outputs(out, frame, daily, summaries, prices):
    out.mkdir(parents=True, exist_ok=True)
    formal = frame[frame.date >= '2025-02-01'].copy()
    frame.to_csv(out/'question2_schedule_with_warmup.csv', index=False, encoding='utf-8-sig')
    formal.to_csv(out/'question2_schedule.csv', index=False, encoding='utf-8-sig')
    daily.to_csv(out/'question2_daily.csv', encoding='utf-8-sig')
    (out/'question2_summary.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    comparison = pd.DataFrame([{k:v for k,v in s.items() if not isinstance(v,dict)} | {'case':s['settings']['name']} for s in summaries])
    comparison.to_csv(out/'question2_comparison.csv', index=False, encoding='utf-8-sig')
    # The template's final purchase column "0:00-0:10+1" is the NEXT calendar
    # day's 00:00-00:10 interval (its load/PV is the next-day 00:10 observation).
    # The preceding 143 columns are this date's 00:10-24:00 intervals, so this
    # date's own 00:00-00:10 lives in the PREVIOUS row's final column.  The last
    # result date (12-31) has no 2026-01-01 data, so its final column is 0.
    grid_by_date = {d: day.grid_kwh.to_numpy() for d, day in frame.groupby('date')}
    purchases, batteries, selected, keyblocks = [], [], [], []
    for date, day in formal.groupby('date'):
        g = day.grid_kwh.to_numpy()
        next_date = date + pd.Timedelta(days=1)
        next_first = float(grid_by_date[next_date][0]) if next_date in grid_by_date else 0.0
        purchases.append([str(date.date()), *np.r_[g[1:], next_first].tolist(), float(g.sum()), float(day.planned_cost.sum())])
        for block in range(6):
            f = day.iloc[block*24:(block+1)*24]
            row = [str(date.date()) if block == 0 else None, f'{block*4}:00-{(block+1)*4}:00',
                   float(f.charge_kwh.sum()),float(f.discharge_kwh.sum()),
                   '0:00' if block==0 else ('24:00' if block==1 else None),
                   float(day.soc_start_kwh.iloc[0]) if block==0 else (float(day.soc_end_kwh.iloc[-1]) if block==1 else None)]
            batteries.append(row)
            if str(date.date()) in KEY_DATES:
                keyblocks.append([str(date.date()),*row[1:]])
        if str(date.date()) in KEY_DATES:
            selected.append({'date':str(date.date()), **{f'{h}:00-{h}:10':float(g[h*6]) for h in [10,12,14,16,18,20]},
                             **daily.loc[date].to_dict()})
    emergency = summarize_emergency(formal)
    payload = {'purchases':purchases,'batteries':batteries,'emergency':emergency,
               'prices_template_order':np.r_[prices[1:],prices[0]].tolist()}
    (out/'question2_workbook_payload.json').write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    pd.DataFrame(selected).to_csv(out/'question2_key_dates.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(keyblocks,columns=['date','block','charge_kwh','discharge_kwh','time','soc_kwh']).to_csv(out/'question2_key_battery.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(emergency,columns=['date','interval','emergency_kwh']).to_csv(out/'question2_emergency.csv',index=False,encoding='utf-8-sig')
    metadata = {'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['附件/附件1.xlsx','附件/附件2.xlsx','附件/附件5/result2.xlsx']},
                'forecast_window':7,'residual_window':21,'quantile_method':'linear',
                'information_cutoff':'Forecast and residuals strictly before the issue date; actuals revealed slot by slot.',
                'template_last_purchase_column':'Header 0:00-0:10+1 is the NEXT day 00:00-00:10 (next-day 00:10 observation); this row own 00:00-00:10 is in the previous row; 12-31 last column is 0.',
                'reported_bill':'planned purchase bill plus 5x emergency bill; no terminal credit deducted'}
    (out/'question2_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/question2/archive/baseline')
    args = parser.parse_args()
    dates, load, pv, prices = load_inputs()
    main_frame, main_daily, main_summary = run_case(dates,load,pv,prices,Settings())
    summaries = [main_summary]
    for setting in [Settings(name='mean_greedy',risk=False,beta=0),Settings(name='risk_greedy',beta=0)]:
        _, _, summary = run_case(dates,load,pv,prices,setting)
        summaries.append(summary)
    write_outputs(args.output,main_frame,main_daily,summaries,prices)
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
