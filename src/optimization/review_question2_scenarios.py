"""Post-run comparison and fixed-plan numerical convergence audit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.optimization.question2 import ROOT,KEY_DATES,forecasts,load_inputs,execute_slot
from src.optimization.question2_scenarios import scenarios,value_reserves


def main():
    out=ROOT/'outputs/question2_scenarios'
    summaries=json.loads((out/'question2_summary.json').read_text(encoding='utf-8'))
    main=summaries[0]
    f=pd.read_csv(out/'question2_schedule.csv',parse_dates=['date'])
    dates,l,v,p=load_inputs(); fl,fv=forecasts(l,v); nu=main['terminal_value']
    checks=[]
    for date in KEY_DATES:
        d=dates.get_loc(pd.Timestamp(date)); day=f[f.date==date]
        g=day.grid_kwh.to_numpy(); net=scenarios(d,l,v,fl,fv)
        for step in [25.,12.5,5.]:
            reserve=value_reserves(net,g,p,nu,step)
            energy=float(day.soc_start_kwh.iloc[0]); bill=0.
            for t in range(144):
                c,dis,b,u,energy=execute_slot(l[d,t],v[d,t],g[t],energy,reserve[t])
                bill+=5*p[t]*b
            checks.append({'date':date,'grid_step_kwh':step,'emergency_cost':bill,'final_soc_kwh':energy})
    conv=pd.DataFrame(checks)
    conv.to_csv(out/'value_grid_convergence.csv',index=False)
    diag=pd.read_csv(out/'scenario_solver_diagnostics.csv')
    # The fixed-direction LPs all solved to optimality. The stored relative
    # gap is an upper/lower-bound certificate for the unrestricted-direction
    # problem, not a solver termination gap or a time-limit status.
    diag.loc[diag['method']=='fixed_direction_LP_with_global_relaxation_bound','solver_status']=0
    diag.to_csv(out/'scenario_solver_diagnostics.csv',index=False)
    formal=diag[diag.date>='2025-02-01']
    gap=float(formal.relative_gap.max())
    # Financial objective includes terminal value; compare like with like.
    realized=f.groupby('date').agg(cost=('total_cost','sum'),end=('soc_end_kwh','last'))
    comparison=[]
    for _,r in formal.iterrows():
        rr=realized.loc[pd.Timestamp(r.date)]
        comparison.append({'date':r.date,'planning_objective':r.direction_feasible_objective,
                           'realized_cost_less_same_terminal_proxy':rr['cost']-nu*rr['end']})
    pd.DataFrame(comparison).to_csv(out/'planning_vs_execution.csv',index=False)
    pairs=pd.read_csv(out/'same_day_controller_comparison.csv')
    pairs=pairs[pairs.date>='2025-02-01']
    metrics={'max_direction_approximation_bound':gap,'days_bound_above_0_1_percent':int((formal.relative_gap>.001).sum()),
             'days_with_integer_direction_repair':int((formal.binary_cells>0).sum()),
             'mean_day_solve_seconds':float(formal.seconds.mean()),'max_day_solve_seconds':float(formal.seconds.max()),
             'sum_direction_penalty_vs_lp':float((formal.direction_feasible_objective-formal.relaxed_objective).sum()),
             'paired_greedy_emergency_cost':float(pairs.same_plan_greedy_emergency_cost.sum()),
             'max_grid25_vs5_emergency_cost_difference_yuan':float((conv[conv.grid_step_kwh==25].set_index('date').emergency_cost-conv[conv.grid_step_kwh==5].set_index('date').emergency_cost).abs().max())}
    (out/'scenario_review_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    labels=['原情景法与近似价值执行','此前风险分位数与库存保留','七天均值基础方案','此前风险分位数与即时补缺口']
    rows='\n'.join(f'| {label} | {s["planned_cost"]:,.2f} | {s["emergency_cost"]:,.2f} | {s["total_cost"]:,.2f} | {s["emergency_kwh"]:,.2f} | {s["final_soc_kwh"]:,.2f} |' for label,s in zip(labels,summaries))
    delta=summaries[1]['total_cost']-main['total_cost']
    key=pd.read_csv(out/'question2_key_dates.csv')
    keyrows='\n'.join(f'| {r.date} | {r.grid_kwh:,.6f} | {r.planned_cost:,.2f} | {r.emergency_kwh:,.6f} | {r.emergency_cost:,.2f} | {r.total_cost:,.2f} |' for r in key.itertuples())
    text=f'''# 问题二原情景法近似解检验

本次恢复“日前两阶段情景规划与日内近似价值函数控制”，不施加储能追索的非预见性约束。其余口径保留：七天同一时段均值、最近21个有效残差日、同日负载与光伏误差配对、首日专门初始化、跨日实际库存继承、常规计划全额计费与五倍紧急电价。未修改此前正式输出，新增文件位于outputs/question2_scenarios。

## 实际回测结果

正式结果期为2025年2—12月，334天48096个时段，1月预热从6000 kWh开始。情景法实际总费用为{main['total_cost']:,.2f}元；此前正式方案减去本次费用为{delta:,.2f}元，占此前费用{delta/summaries[1]['total_cost']:.2%}。差值为正表示本次节费，为负表示本次更贵。

| 方案 | 计划费 元 | 紧急费 元 | 实际总费 元 | 紧急电量 kWh | 年末库存 kWh |
|---|---:|---:|---:|---:|---:|
{rows}

本次2月1日库存为{main['initial_result_soc_kwh']:,.6f} kWh。各独立全年策略从同一1月1日库存开始，但预热后2月1日库存可能不同。正式账单不扣除日末库存价值，不用不同库存制造退款。不能把情景内理想追索目标当作上述真实执行费用。

该比较检验的是整套方案：日前从逐时分位数改成联合历史情景，日内从规划库存阈值改成平均未来价值阈值。因此费用差异不能全部归因于“没有非预见性约束”；此前简化主模型本身也未建立非预见性约束。

## 日前规划与方向修复

所有情景共用g，各情景分别具有b、C、D、U、E。目标为sum(c*g)+mean(sum(5*c*b)−ν*E末)，ν=min(c)/0.9={nu:.9f}。每个情景满足10分钟平衡、库存递推、1200—10800 kWh库存及5000/6 kWh充放电上限。

与分位数方案不同，本次直接使用21条净负荷情景，不再先压缩为80%分位数。α不适用。实时保留量由情景未来价值导出，不再用β乘规划库存，β也不适用；这不是通过全年结果重新调参。

连续松弛可能给出紧急购电充电，故不能在保留现有运行口径的同时直接接受。全年正式对照采用两次连续LP：先求原模型松弛最优g0，再按N与g0的大小确定各情景时段的供需方向。缺口分支设置C=0、g+D≤N；富余分支设置D=0、C−g≤−N。随后在这些固定分支内重新优化共同g及全部追索变量。约束自动确保分支符号不改变；第二次得到的g可以在分支允许范围内变化，不是简单固定第一次的g。

这样不用整数变量，也没有非预见性约束。各情景的储能追索仍可依赖该历史情景完整未来，故日前能力评估依旧偏理想；第二次LP只修复物理方向，不修复信息结构。

第一次松弛最优目标L是严格方向约束问题的下界，第二次可行目标U是上界，所以L≤严格方向模型最优值≤U。正式期最大(U−L)/|U|为{gap:.6%}；这是一项保守的近似误差证据，不代表实际随机运行费用与全局最优的差距。两次LP本身均正常求得最优解。平均每日日前求解{metrics['mean_day_solve_seconds']:.3f}秒，最大{metrics['max_day_solve_seconds']:.3f}秒。边界是在当日实际初始库存给定条件下成立，不能把全年逐日边界简单解释为跨日全局最优性证明。

早期另做35天的整数方向修复试算，耗时明显高于固定方向LP，因此仅保留在pilot目录作方法检验，不混入本次全年结果。全年主对照中的binary_cells全部为0；代码保留可选MILP求解函数，但默认全年路径不调用该分支。

## 原方案近似价值控制

g在0:00锁定后，每条历史情景在固定g下反向计算剩余紧急费用函数，终端为−νE。富余时仅用富余电量充电；缺口时仅在缺口范围内放电。逐情景价值函数再取均值，当前选择使“当前紧急费用＋平均未来费用”最小的动作。实际控制仅读取当前供需和实际库存，未读取未来实际数据。

内部库存采用25 kWh网格上的凸分段线性插值，实际库存连续保存，不取整。阈值从mean(H下一时段(E))+5*c*0.9*E的最小点得到。该函数仍是完全信息历史情景价值的平均，属于近似未来价值，并非严格随机Bellman函数。

四个指定日期以相同g、相同当天初始库存分别用25、12.5、5 kWh网格执行，25与5 kWh的单日紧急费用最大差异为{metrics['max_grid25_vs5_emergency_cost_difference_yuan']:.6f}元，详见value_grid_convergence.csv。该检查只检验固定计划的控制离散误差，不等于整个全年重新优化后的误差界。

另保存same_day_controller_comparison.csv：逐日固定本次g与相同起始库存，比较即时补缺口执行器。该对照每日重新对齐库存，不是独立全年策略，不能将其累计费用直接与独立全年策略排名。

## 指定日期

| 日期 | 计划购电 kWh | 计划费 元 | 紧急购电 kWh | 紧急费 元 | 实际总费 元 |
|---|---:|---:|---:|---:|---:|
{keyrows}

## 验证与文件

逐时核验供需平衡、库存递推与跨日连续、充放电功率边界、无同时充放电、无紧急购电充电，全部通过。新增测试覆盖情景因果性、单时段80%分位数退化结果、情景物理方向以及未来高价下的保留行为。每日日前状态、方向近似的上下界差与连续松弛目标保存于scenario_solver_diagnostics.csv。代码中relative_gap在全年固定方向LP路径表示(U−L)/|U|，不是LP求解器数值误差。

question2_schedule.csv为正式实际执行明细；question2_schedule_with_warmup.csv包含预热；question2_comparison.csv为与既有方案的比较；question2_key_dates.csv及question2_key_battery.csv为题定日期与分块结果。planning_vs_execution.csv用同一终端库存代理口径并列规划目标与实际结果，其差异包含预测误差和控制近似，不能全部归因于非预见性。

如导出result2.xlsx，采用与正式结果一致的官方模板映射：最后“0:00-0:10+1”列是次日00:00—00:10（其负荷/光伏取次日00:10观测），其他143列是当天00:10—24:00，当天自身00:00—00:10落在上一行末列，12-31末列填0，保留原表头。

复现：python -m src.optimization.question2_scenarios。程序逐日保存本地checkpoint.pkl以便断点续算；更改参数或代码后应使用新的--output目录，避免混用旧检查点。数值复核与报告：python -m src.optimization.review_question2_scenarios。该模块仅做事后审查，不反向修改已生成的计划。
'''
    (out/'原情景法检验与对比.md').write_text(text,encoding='utf-8')
    meta=json.loads((out/'question2_metadata.json').read_text(encoding='utf-8'))
    meta.update(method='No-NAC scenario planning with direction repair and causal mean-value controller',
                quantile_method='not used in this experiment',soc_grid_step_kwh=25,
                planner='relaxed_LP_then_fixed_direction_LP',nonanticipativity=False)
    (out/'question2_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__': main()
