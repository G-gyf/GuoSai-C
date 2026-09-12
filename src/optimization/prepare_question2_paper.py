"""Read existing LDR results, audit them, and produce paper tables and figures."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.optimization.question2 import ROOT, ETA, EMIN, EMAX, S, load_inputs, forecasts, load_forecast_weekly_persist, planning_net, Settings, solve_plan, validate_schedule
from src.optimization.question2_ldr import PARAMETER_NAMES, ScenarioObjective
from src.optimization.question2_g_search import scenario_net_matrix

OUT=ROOT/'outputs/question2/current/paper'
SRC=ROOT/'outputs/question2/current/ldr'
KEY=['2025-03-20','2025-06-21','2025-09-23','2025-12-21']


def main():
    OUT.mkdir(exist_ok=True)
    (OUT/'figures').mkdir(exist_ok=True)
    (OUT/'tables').mkdir(exist_ok=True)
    f=pd.read_csv(SRC/'question2_schedule_with_warmup.csv',parse_dates=['date'])
    formal=f[f.date>='2025-02-01'].copy()
    params=pd.read_csv(SRC/'ldr_daily_parameters.csv',parse_dates=['date'])
    dates,l,v,p=load_inputs()
    np.testing.assert_allclose(f.load_kwh,l.ravel(),atol=1e-8,rtol=0)
    np.testing.assert_allclose(f.pv_kwh,v.ravel(),atol=1e-8,rtol=0)
    np.testing.assert_allclose(f.price,np.tile(p,365),atol=1e-12,rtol=0)
    summaries=json.loads((SRC/'question2_summary.json').read_text(encoding='utf-8'))
    main_settings=next(s for s in summaries if s['settings']['name']=='ldr_quantile_a08')['settings']
    fl,fv=forecasts(l,v)
    if main_settings.get('load_forecast','mean7d')=='weekly_persist':
        representative=pd.read_excel(ROOT/'附件/附件1.xlsx',sheet_name=0).iloc[:,2].to_numpy(float)/6.0
        fl=load_forecast_weekly_persist(l,representative)
    max_rule_error=max_score_error=max_grid_error=0.
    score_checked=0
    for d, date in enumerate(dates):
        day=f[f.date==date]
        theta=params.loc[d,PARAMETER_NAMES].to_numpy(float)
        delta=theta[:4]; lambdas=np.r_[0.,theta[4:]]
        e=float(day.soc_start_kwh.iloc[0]); errsum=0.; signal=0.
        if d:
            net,count=planning_net(d,l,v,fl,fv,Settings(alpha=.8))
            np.testing.assert_allclose(day.planning_net_kwh,net,atol=1e-8,rtol=0)
            plan,_=solve_plan(net,p,e,float(p.min()/ETA))
            max_grid_error=max(max_grid_error,float(np.max(np.abs(plan[0]-day.grid_kwh))))
        else:
            plan=np.zeros((5,144)); plan[4]=EMIN
        for t,row in enumerate(day.itertuples()):
            k=t//36
            if t%36==0:
                signal=errsum/t if t else 0.
            reserve=float(np.clip(row.plan_soc_kwh+delta[k]+lambdas[k]*signal,EMIN,EMAX))
            r=l[d,t]-v[d,t]-row.grid_kwh
            c=min(max(-r,0),S,max((EMAX-e)/ETA,0))
            dis=min(max(r,0),S,ETA*max(e-reserve,0))
            b=max(r-dis,0); u=max(-r-c,0)
            e=e+ETA*c-dis/ETA
            expected=np.array([reserve,signal,c,dis,b,u,e])
            saved=np.array([row.reserve_kwh,row.stage_error_mean_kwh,row.charge_kwh,row.discharge_kwh,row.emergency_kwh,row.unused_kwh,row.soc_end_kwh])
            max_rule_error=max(max_rule_error,float(abs(expected-saved).max()))
            errsum+=l[d,t]-v[d,t]-(fl[d,t]-fv[d,t]) if d else l[d,t]-v[d,t]
        if d>=28:
            scenarios,count=scenario_net_matrix(d,l,v,fl,fv)
            objective=ScenarioObjective(scenarios,fl[d]-fv[d],day.grid_kwh.to_numpy(),day.plan_soc_kwh.to_numpy(),p,float(day.soc_start_kwh.iloc[0]),float(p.min()/ETA))
            selected,zero=float(objective(theta)),float(objective(np.zeros(7)))
            max_score_error=max(max_score_error,abs(selected-params.selected_scenario_score.iloc[d]),abs(zero-params.zero_scenario_score.iloc[d]))
            assert selected<=zero+1e-6
            score_checked+=1
        if (d+1)%90==0:
            print(f'Paper audit: {d+1}/365 days checked',flush=True)
    assert max_rule_error<1e-5 and max_score_error<1e-5 and max_grid_error<1e-4
    audit={'actual_physical':validate_schedule(f),'raw_inputs_match':True,'recomputed_plan_days':364,
           'recomputed_rule_intervals':len(f),'recomputed_score_days':score_checked,
           'max_grid_error_kwh':max_grid_error,'max_rule_error_kwh':max_rule_error,'max_score_error_yuan':max_score_error,
           'independent_total_bill':float(np.sum(formal.price*(formal.grid_kwh+5*formal.emergency_kwh))),
           'source_hashes':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [SRC/'question2_schedule_with_warmup.csv',SRC/'ldr_daily_parameters.csv',ROOT/'src/optimization/question2_ldr.py']},'passed':True}
    (OUT/'paper_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    summaries=json.loads((SRC/'question2_summary.json').read_text(encoding='utf-8'))
    scenario=json.loads((ROOT/'outputs/question2/archive/scenarios/question2_summary.json').read_text(encoding='utf-8'))[0]
    selected=[summaries[1],summaries[2],scenario,summaries[0]]
    labels=['分位数＋β=1','分位数＋β=0','情景价值控制','分位数＋LDR']
    comparison=pd.DataFrame([{'方案':label,**{k:s[k] for k in ['planned_cost','emergency_cost','total_cost','emergency_kwh','initial_result_soc_kwh','final_soc_kwh']}} for label,s in zip(labels,selected)])
    comparison['emergency_average_price']=comparison.emergency_cost/comparison.emergency_kwh
    comparison.to_csv(OUT/'tables/strategy_comparison.csv',index=False,encoding='utf-8-sig')
    daily=formal.groupby('date').agg(**{k:(k,'sum') for k in ['planned_cost','emergency_cost','total_cost','emergency_kwh','load_kwh','pv_kwh','unused_kwh']})
    monthly=daily.groupby(daily.index.month).sum()
    monthly.index.name='month'
    monthly.to_csv(OUT/'tables/monthly_summary.csv',encoding='utf-8-sig')
    paired=pd.read_csv(SRC/'same_plan_controller_comparison.csv',parse_dates=['date'])
    paired=paired[paired.date>='2025-02-01']
    paired['saving_vs_beta1']=paired.beta1_total_cost-paired.ldr_total_cost
    paired['saving_vs_beta0']=paired.beta0_total_cost-paired.ldr_total_cost
    mp=paired.groupby(paired.date.dt.month)[['saving_vs_beta1','saving_vs_beta0']].sum()
    mp.to_csv(OUT/'tables/monthly_paired_savings.csv',encoding='utf-8-sig')
    extra={'no_emergency_days':int((daily.emergency_kwh<=1e-6).sum()),'emergency_days':int((daily.emergency_kwh>1e-6).sum()),
           'max_daily_emergency_date':str(daily.emergency_cost.idxmax().date()),'max_daily_emergency_cost':float(daily.emergency_cost.max()),
           'paired_savings_vs_beta1':float(paired.saving_vs_beta1.sum()),'paired_savings_vs_beta0':float(paired.saving_vs_beta0.sum()),
           'paired_wins_beta0':int((paired.saving_vs_beta0>1e-5).sum()),'paired_losses_beta0':int((paired.saving_vs_beta0< -1e-5).sum()),
           'soc_min_share':float(np.mean(formal.soc_end_kwh<=EMIN+1e-6)),'soc_max_share':float(np.mean(formal.soc_end_kwh>=EMAX-1e-6)),
           'de_messages':params.loc[params.residual_count>=21,'solver_message'].value_counts().to_dict()}
    (OUT/'paper_metrics.json').write_text(json.dumps(extra,ensure_ascii=False,indent=2),encoding='utf-8')
    for filename in ['question2_key_dates.csv','question2_key_battery.csv']:
        pd.read_csv(SRC/filename).to_csv(OUT/'tables'/filename,index=False,encoding='utf-8-sig')
    emergency=pd.read_csv(SRC/'question2_emergency.csv')
    emergency[emergency.iloc[:,0].isin(KEY)].to_csv(OUT/'tables/key_emergency_intervals.csv',index=False,encoding='utf-8-sig')
    # Static, publication-oriented charts: blue/orange roots, neutral context.
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    blue='#2864A0'; orange='#CF7A25'; gray='#646B73'
    fig,axs=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    for ax,col,title in [(axs[0],'total_cost','实际总购电费用'),(axs[1],'emergency_cost','紧急购电费用')]:
        vals=comparison[col].to_numpy()/1e4
        bars=ax.bar(np.arange(4),vals,color=['#CAD6E2']*3+[blue],edgecolor=gray,linewidth=.7)
        ax.set_xticks(np.arange(4),['分位数\nβ=1','分位数\nβ=0','情景价值\n控制','分位数\nLDR'])
        ax.set_ylim(0,max(vals)*1.17);ax.set_ylabel('费用 / 万元');ax.set_title(title)
        ax.bar_label(bars,labels=[f'{z:,.2f}' for z in vals],padding=4,fontsize=9)
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('图1  各策略正式期费用比较\n2025年2—12月；各策略独立连续运行，期初期末库存见表',fontsize=12)
    fig.savefig(OUT/'figures/figure1_cost_comparison.png',dpi=300);plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(10.5,7),layout='constrained')
    x=monthly.index.to_numpy()
    axs[0].bar(x,monthly.planned_cost/1e4,color=blue,label='计划费')
    axs[0].bar(x,monthly.emergency_cost/1e4,bottom=monthly.planned_cost/1e4,color=orange,hatch='//',label='紧急费')
    axs[0].set_ylabel('LDR费用 / 万元');axs[0].legend(ncol=2,frameon=False);axs[0].set_title('（a）LDR月度账单构成')
    axs[1].plot(x,mp.saving_vs_beta1/1e4,'o-',color=blue,label='相对β=1')
    axs[1].plot(x,mp.saving_vs_beta0/1e4,'s--',color=orange,label='相对β=0')
    axs[1].axhline(0,color=gray,lw=.8);axs[1].set_ylabel('配对节费 / 万元');axs[1].legend(ncol=2,frameon=False)
    axs[1].set_title('（b）同计划、同日初库存配对节费（正值表示LDR更低）')
    for ax in axs:
        ax.set_xticks(x,[f'{k}月' for k in x]);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('图2  正式期月度费用与控制规则配对比较\n2025年2—12月；配对试验逐日对齐库存，不作为独立全年账单',fontsize=12)
    fig.savefig(OUT/'figures/figure2_monthly_and_paired.png',dpi=300);plt.close(fig)
    fig,axs=plt.subplots(3,4,figsize=(15,9),layout='constrained')
    for j,ds in enumerate(KEY):
        d=f[f.date==ds];x=np.arange(144)/6
        axs[0,j].plot(x,d.load_kwh-d.pv_kwh,color=gray,lw=1,label='实际净负荷')
        axs[0,j].step(x,d.grid_kwh,where='post',color=blue,lw=1,label='计划购电')
        axs[0,j].axhline(0,color='#AAAAAA',lw=.6);axs[0,j].set_title(ds)
        axs[1,j].plot(np.arange(145)/6,np.r_[d.soc_start_kwh.iloc[0],d.soc_end_kwh],color=blue,lw=1.3,label='实际库存')
        axs[1,j].step(x,d.reserve_kwh,where='post',color=orange,ls='--',lw=1,label='保留阈值')
        for bound in [EMIN,EMAX]:axs[1,j].axhline(bound,color=gray,ls=':',lw=.7)
        axs[1,j].set_ylim(0,12000)
        axs[2,j].bar(x,d.emergency_kwh,width=1/6,color=orange)
        axs[2,j].set_xlabel('时刻 / h');axs[2,j].set_ylim(0,250)
        for i in range(3):
            axs[i,j].set_xlim(0,24);axs[i,j].set_xticks([0,6,12,18,24]);axs[i,j].grid(alpha=.15)
    axs[0,0].set_ylabel('每10分钟电量 / kWh');axs[1,0].set_ylabel('内部库存 / kWh');axs[2,0].set_ylabel('紧急购电 / kWh')
    axs[0,0].legend(fontsize=8,frameon=False);axs[1,0].legend(fontsize=8,frameon=False)
    fig.suptitle('图3  四个指定日期的计划购电、储能与紧急响应\n供需为区间电量；库存为边界时点状态；上下虚线为1200与10800 kWh',fontsize=12)
    fig.savefig(OUT/'figures/figure3_key_dates.png',dpi=300);plt.close(fig)
    print(json.dumps({'audit':audit,'metrics':extra},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
