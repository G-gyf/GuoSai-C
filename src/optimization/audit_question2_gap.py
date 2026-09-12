"""Recheck fixed-plan counterfactuals; never changes an operational plan."""
from __future__ import annotations
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from src.optimization.question2 import ROOT, ETA, EMIN, EMAX, S, equality_matrix, load_inputs, validate_schedule


def dispatch(frame, physical_direction):
    n = len(frame)
    r = (frame.load_kwh-frame.pv_kwh-frame.grid_kwh).to_numpy()
    p = frame.price.to_numpy()
    initial = float(frame.soc_start_kwh.iloc[0])
    a = equality_matrix(n)
    rhs = np.r_[r, initial, np.zeros(n-1)]
    upper_c = np.minimum(S, np.maximum(-r, 0)) if physical_direction else np.full(n, S)
    upper_d = np.minimum(S, np.maximum(r, 0)) if physical_direction else np.full(n, S)
    bounds = [(0, None)]*n + [(0, float(x)) for x in upper_c] + [(0, float(x)) for x in upper_d] + [(0, None)]*n + [(EMIN, EMAX)]*n
    result = linprog(np.r_[5*p, np.zeros(4*n)], A_eq=a, b_eq=rhs, bounds=bounds, method='highs')
    if not result.success:
        raise RuntimeError(result.message)
    b, c, d, u, e = result.x.reshape(5, n)
    equality_error = float(np.max(np.abs(a@result.x-rhs)))
    lower = np.array([x[0] for x in bounds])
    upper = np.array([np.inf if x[1] is None else x[1] for x in bounds])
    finite = np.isfinite(upper)
    dual = rhs@result.eqlin.marginals + lower@result.lower.marginals + upper[finite]@result.upper.marginals[finite]
    assert equality_error < 1e-5 and abs(result.fun-dual) < .01
    if physical_direction:
        assert np.max(np.minimum(c,d)) < 1e-5
        assert np.max(np.minimum(c,b)) < 1e-5
    return {'total_cost': float(p@frame.grid_kwh.to_numpy()+result.fun),
            'emergency_cost': float(result.fun), 'end_soc': float(e[-1]),
            'equality_error': equality_error, 'duality_gap': float(abs(result.fun-dual)),
            'emergency_charging_slots': int(np.sum((b>1e-5)&(c>1e-5)))}


def main():
    out = ROOT/'outputs/question2/analysis/gap_audit'
    out.mkdir(parents=True, exist_ok=True)
    dates, load, pv, prices = load_inputs()
    benchmarks = {x['case']: x['cost_lower_bound_yuan'] for x in json.loads((ROOT/'outputs/question2/benchmark/perfect_foresight/perfect_foresight_summary.json').read_text(encoding='utf-8'))}
    rows = []
    for folder, formal, benchmark in [('question2', False, 'full_year'), ('question2', True, 'matched_risk_reserve'), ('question2_scenarios', True, 'matched_scenario')]:
        rel = 'archive/baseline' if folder == 'question2' else 'archive/scenarios'
        path = ROOT/'outputs'/rel/('question2_schedule.csv' if formal else 'question2_schedule_with_warmup.csv')
        f = pd.read_csv(path)
        mask = dates >= '2025-02-01' if formal else np.ones(len(dates), dtype=bool)
        np.testing.assert_allclose(f.load_kwh, load[mask].ravel(), atol=1e-8)
        np.testing.assert_allclose(f.pv_kwh, pv[mask].ravel(), atol=1e-8)
        np.testing.assert_allclose(f.price, np.tile(prices, int(mask.sum())), atol=1e-12)
        validation = validate_schedule(f)
        actual = float((f.price*(f.grid_kwh+5*f.emergency_kwh)).sum())
        np.testing.assert_allclose(actual, f.total_cost.sum(), atol=.01)
        relaxed = dispatch(f, False)
        direction = dispatch(f, True)
        lb = benchmarks[benchmark]
        assert lb <= relaxed['total_cost']+.01 <= direction['total_cost']+.02 <= actual+.03
        row = {'strategy': folder, 'period': 'Feb-Dec' if formal else 'Jan-Dec',
               'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'initial_soc': float(f.soc_start_kwh.iloc[0]), 'actual': actual,
               'perfect_information_lower_bound': lb, 'fixed_g_relaxed': relaxed,
               'fixed_g_direction': direction, 'actual_minus_fixed_g_direction': actual-direction['total_cost'],
               'fixed_g_direction_minus_information_bound': direction['total_cost']-lb,
               'identity_error': (actual-lb)-((actual-direction['total_cost'])+(direction['total_cost']-lb)),
               'actual_validation': validation}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        (out/'gap_audit.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
