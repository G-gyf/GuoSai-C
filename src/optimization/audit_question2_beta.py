"""Fixed-plan beta ablations; these replays do not regenerate later plans."""
import json
import numpy as np
import pandas as pd
from src.optimization.question2 import ROOT, EMIN, ETA, S, execute_slot, validate_schedule


def replay(f, reset_daily=False):
    energy = float(f.soc_start_kwh.iloc[0])
    out = f.copy()
    records = []
    for row in f.itertuples():
        if reset_daily and row.slot == 0:
            energy = float(row.soc_start_kwh)
        start = energy
        c, d, b, u, energy = execute_slot(row.load_kwh,row.pv_kwh,row.grid_kwh,energy,EMIN)
        records.append((start,c,d,b,u,energy))
    out[['soc_start_kwh','charge_kwh','discharge_kwh','emergency_kwh','unused_kwh','soc_end_kwh']] = records
    out['emergency_cost'] = 5*out.price*out.emergency_kwh
    out['total_cost'] = out.planned_cost+out.emergency_cost
    if not reset_daily:
        validate_schedule(out)
    return out


def totals(f):
    return {k:float(f[k].sum()) for k in ['planned_cost','emergency_cost','total_cost','emergency_kwh','charge_kwh','discharge_kwh','unused_kwh']}


def main():
    folder=ROOT/'outputs/question2/analysis/beta_audit'
    folder.mkdir(exist_ok=True)
    results=[]
    folder_map = {'question2': 'archive/baseline', 'question2_scenarios': 'archive/scenarios'}
    for name, rel in folder_map.items():
        f=pd.read_csv(ROOT/'outputs'/rel/'question2_schedule.csv')
        frozen=replay(f)
        paired=replay(f,True)
        r=np.maximum(f.load_kwh-f.pv_kwh-f.grid_kwh,0)
        available=np.minimum(np.minimum(r,S),ETA*np.maximum(f.soc_start_kwh-EMIN,0))
        extra=np.maximum(available-f.discharge_kwh,0)
        result={'strategy':name,'period':'2025-02-01..2025-12-31',
                'original':totals(f),'fixed_plan_continuous_beta0':totals(frozen),
                'same_day_same_opening_beta0':totals(paired),
                'original_minus_fixed_plan_continuous_bill':float(f.total_cost.sum()-frozen.total_cost.sum()),
                'original_minus_daily_paired_bill':float(f.total_cost.sum()-paired.total_cost.sum()),
                'locally_withheld_while_emergency_slots':int(((extra>1e-6)&(f.emergency_kwh>1e-6)).sum()),
                'locally_available_extra_discharge_kwh':float(extra.sum()),
                'original_end_soc':float(f.soc_end_kwh.iloc[-1]),
                'fixed_plan_beta0_end_soc':float(frozen.soc_end_kwh.iloc[-1])}
        results.append(result)
        paired_day=paired.groupby('date')[['emergency_cost','soc_end_kwh']].agg({'emergency_cost':'sum','soc_end_kwh':'last'})
        paired_day['original_emergency_cost']=f.groupby('date').emergency_cost.sum()
        paired_day['original_minus_beta0_emergency_cost']=paired_day.original_emergency_cost-paired_day.emergency_cost
        result['paired_beta0_cheaper_days']=int((paired_day.original_minus_beta0_emergency_cost>1e-5).sum())
        result['paired_original_cheaper_days']=int((paired_day.original_minus_beta0_emergency_cost < -1e-5).sum())
        paired_day.to_csv(folder/f'{name}_paired_days.csv')
    (folder/'beta_audit.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
