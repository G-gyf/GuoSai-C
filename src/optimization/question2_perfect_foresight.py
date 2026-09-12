"""Joint-horizon perfect-information cost lower bounds for Question 2."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from src.optimization.question2 import (
    ROOT, T, ETA, EMIN, EMAX, S, equality_matrix, load_inputs,
    execute_slot, validate_schedule,
)


def solve_horizon(dates,load,pv,daily_prices,initial,label,out):
    start=time.perf_counter()
    l,v=load.ravel(),pv.ravel(); p=np.tile(daily_prices,len(dates)); n=len(l)
    a=equality_matrix(n); rhs=np.r_[l-v,initial,np.zeros(n-1)]
    obj=np.r_[p,np.zeros(4*n)]
    bounds=[(0,None)]*n+[(0,S)]*(2*n)+[(0,None)]*n+[(EMIN,EMAX)]*n
    result=linprog(obj,A_eq=a,b_eq=rhs,bounds=bounds,method='highs')
    if not result.success: raise RuntimeError(result.message)
    grid=np.maximum(result.x[:n],0)
    # A greedy physical dispatch with the same optimized grid purchases is
    # feasible: its SOC is never below that of an arbitrary feasible LP
    # dispatch (it never wastes discharges and accepts all usable surplus).
    # This removes degeneracy without a second, large LP or integer variables.
    energy=float(initial); rows=[]; numerical_makeup=0.
    for t in range(n):
        c,d,b,u,end=execute_slot(l[t],v[t],grid[t],energy,EMIN)
        assert b<1e-4, f'Perfect plan cannot supply slot {t}: missing {b}'
        # Attribute numerical residual only to the known day-ahead plan.
        numerical_makeup+=b; grid[t]+=b
        rows.append((energy,c,d,u,end)); energy=end
    x=np.asarray(rows)
    f=pd.DataFrame({'date':np.repeat(dates,T),'slot':np.tile(np.arange(T),len(dates)),
                    'interval_start':pd.date_range(dates[0],periods=n,freq='10min'),
                    'load_kwh':l,'pv_kwh':v,'price':p,'grid_kwh':grid,
                    'soc_start_kwh':x[:,0],'charge_kwh':x[:,1],'discharge_kwh':x[:,2],
                    'unused_kwh':x[:,3],'soc_end_kwh':x[:,4],
                    'emergency_kwh':np.zeros(n),'planned_cost':p*grid,'emergency_cost':np.zeros(n)})
    f['total_cost']=f.planned_cost
    checks=validate_schedule(f)
    bill=float(p@grid)
    # Independent LP dual objective, including finite variable bounds.
    lo=np.array([b[0] for b in bounds],float)
    hi=np.array([np.inf if b[1] is None else b[1] for b in bounds])
    finite=np.isfinite(hi)
    dual=float(rhs@result.eqlin.marginals+lo@result.lower.marginals+hi[finite]@result.upper.marginals[finite])
    assert abs(bill-result.fun)<.01
    assert abs(result.fun-dual)<.01
    daily=f.groupby('date').agg(**{k:(k,'sum') for k in ['grid_kwh','charge_kwh','discharge_kwh','unused_kwh','total_cost']},
                  soc_start_kwh=('soc_start_kwh','first'),soc_end_kwh=('soc_end_kwh','last'))
    f.to_csv(out/f'{label}_schedule.csv',index=False,encoding='utf-8-sig')
    daily.to_csv(out/f'{label}_daily.csv',encoding='utf-8-sig')
    summary={'case':label,'start_date':str(dates[0].date()),'end_date':str(dates[-1].date()),'days':len(dates),
              'intervals':n,'initial_soc_kwh':initial,'final_soc_kwh':energy,'cost_lower_bound_yuan':float(result.fun),
              'feasible_schedule_bill_yuan':bill,'dual_objective_yuan':dual,'duality_gap_yuan':abs(result.fun-dual),
              'numerical_plan_makeup_kwh':numerical_makeup,'grid_kwh':float(grid.sum()),
              'charge_kwh':float(f.charge_kwh.sum()),'discharge_kwh':float(f.discharge_kwh.sum()),
              'unused_kwh':float(f.unused_kwh.sum()),'emergency_kwh':0.,
              'seconds':time.perf_counter()-start,'validation':checks}
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


def main():
    out=ROOT/'outputs/question2_perfect_foresight'; out.mkdir(parents=True,exist_ok=True)
    dates,l,v,p=load_inputs()
    scenarios=json.loads((ROOT/'outputs/question2_scenarios/question2_summary.json').read_text(encoding='utf-8'))
    annual=solve_horizon(dates,l,v,p,6000.,'full_year',out)
    results=[annual]
    # Match each causal policy's ACTUAL February opening inventory. A suffix
    # of the full-year optimum is not used as the suffix-only lower bound.
    for name,s in [('matched_scenario',scenarios[0]),('matched_risk_reserve',scenarios[1])]:
        bound=solve_horizon(dates[31:],l[31:],v[31:],p,s['initial_result_soc_kwh'],name,out)
        bound['causal_policy_cost_yuan']=s['total_cost']
        bound['gap_to_bound_yuan']=s['total_cost']-bound['cost_lower_bound_yuan']
        bound['gap_as_fraction_of_actual']=bound['gap_to_bound_yuan']/s['total_cost']
        bound['excess_over_lower_bound']=bound['gap_to_bound_yuan']/bound['cost_lower_bound_yuan']
        results.append(bound)
    (out/'perfect_foresight_summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    rows='\n'.join(f'| {r["case"]} | {r["start_date"]}—{r["end_date"]} | {r["initial_soc_kwh"]:,.6f} | {r["final_soc_kwh"]:,.6f} | {r["cost_lower_bound_yuan"]:,.2f} |' for r in results)
    comparisons='\n'.join(f'| {r["case"]} | {r["causal_policy_cost_yuan"]:,.2f} | {r["cost_lower_bound_yuan"]:,.2f} | {r["gap_to_bound_yuan"]:,.2f} | {r["gap_as_fraction_of_actual"]:.2%} | {r["excess_over_lower_bound"]:.2%} |' for r in results[1:])
    report=f'''# 问题二完美预见费用下限

将附件2全年实际负载与光伏视为事先完全已知，按附件1固定周期电价，对全时域进行一次统一线性规划。2025年1—12月购电费用下限为 **{annual['cost_lower_bound_yuan']:,.2f}元**，购电量{annual['grid_kwh']:,.6f} kWh。该下限不是七天预测模型，也不是365个单日最优值简单相加。

## 口径

保留10分钟粒度、充放电各90%效率、1200—10800 kWh库存、5000 kW充放电功率、无售电与允许免费未利用供能。全年从1月1日6000 kWh起步，库存跨天连续。不设每日闭环，不强制年末回到6000，不加入人为日末库存价值项；年末仅受实际库存上下界约束。原预测方案的首日冷启动规则不适用于这个完美信息反事实，因为首日也已知供需。

在完全预见且计划购电无限额的题设下，紧急购电可提前转为同一时段普通计划购电，价格更低，所以最优解不需要紧急购电，紧急费用为0。

目标为min sum(c*g)，约束为g+D=L−V+C+U，E下一时段=E当前+0.9C−D/0.9，配合设备边界及非负性。一次性优化52560个时段，约262800个连续变量。末日之后的数据与电量价值均不引入。

## 下限与可比结果

| 情形 | 统计区间 | 初始库存 kWh | 末库存 kWh | 费用下限 元 |
|---|---|---:|---:|---:|
{rows}

full_year是完整自然年下限。matched_scenario和matched_risk_reserve分别将2月1日起点设为两种实际策略的预热后库存，独立求解2—12月下限。不能拿全年下限直接减2—12月实际费用，也不能把全年最优解的2—12月切片直接称为独立正式期下限。

| 可比策略 | 实际费用 元 | 同期同初始库存下限 元 | 差额 元 | 差额占实际费用 | 实际费用高于下限 |
|---|---:|---:|---:|---:|---:|
{comparisons}

差额同时包含预测不确定性代价、逐日规划边界和控制近似等因素，不能解释为仅更换优化求解器就一定能消除的费用。该下限允许年末仅保留1200 kWh，而实际策略的末库存略高；这使下限保持乐观，也是比较口径的一部分。

## 最优性与可行性核验

全部LP正常求得最优解，利用等式约束及变量边界的对偶乘子独立复算对偶目标。保存的可行账单与求解器最优目标差异均低于0.01元，对偶差也低于0.01元。原始LP若存在退化充放电，通过固定其购电计划、按当前富余充电和缺口放电构造物理可行轨迹。该轨迹库存不会低于原LP轨迹，供电可行且购电费用不变，保证不同时充放电、不无效放电弃置。

逐时核验供需平衡、储能递推、跨日连续、容量功率边界，均通过。数值容差量级的供电残差计入事先计划购电，记录在numerical_plan_makeup_kwh，不作为真实紧急事件。详细最优值、对偶值与误差见perfect_foresight_summary.json。

各情形逐时明细保存于相应_schedule.csv，每日购电、充放电、费用与日界库存保存于_daily.csv。文件仅作为完美预见基准，不替代可实际部署的result2.xlsx。

复现：python -m src.optimization.question2_perfect_foresight。
'''
    (out/'完美预见下限测算说明.md').write_text(report,encoding='utf-8')


if __name__=='__main__': main()
