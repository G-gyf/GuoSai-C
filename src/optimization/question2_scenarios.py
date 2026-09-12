"""No-NAC scenario planning and causal approximate-value execution.

Annual default: relaxed LP, then fixed-direction feasible LP; report their
lower/upper bound gap. An optional MILP implementation supports pilot audits.
Neither scenario-specific recourse nor its average future value is claimed
to solve the full causal stochastic control problem exactly.
"""
from __future__ import annotations
import argparse
import json
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix, csr_matrix, hstack, vstack
from src.optimization.question2 import (
    ROOT,T,ETA,EMIN,EMAX,S,Settings,forecasts,load_forecast_weekly_persist,load_inputs,execute_slot,
    validate_schedule,summarize_emergency,write_outputs,
)


def scenarios(d,load,pv,fl,fv):
    first=max(7,d-21)
    if first>=d:
        return (fl[d]-fv[d])[None,:]
    return np.maximum(0,fl[d]+load[first:d]-fl[first:d])-np.maximum(0,fv[d]+pv[first:d]-fv[first:d])


def stochastic_plan(net,prices,initial,nu,time_limit=5.,gap=1e-3,branch_only=False):
    start=time.perf_counter()
    m,n=net.shape
    nv=n+5*m*n
    # Each scenario uses b,C,D,U,E. g is shared by ALL scenarios.
    a=lil_matrix((2*m*n,nv)); rhs=np.zeros(2*m*n)
    objective=np.zeros(nv); objective[:n]=prices
    lower=np.zeros(nv); upper=np.full(nv,np.inf)
    upper[:n]=np.maximum(0,net.max(axis=0))+S
    for w in range(m):
        off=n+5*n*w
        for t in range(n):
            row=2*n*w+t
            a[row,t]=1
            for k,value in [(0,1),(1,-1),(2,1),(3,-1)]: a[row,off+k*n+t]=value
            rhs[row]=net[w,t]
            row+=n
            a[row,off+4*n+t]=1; a[row,off+n+t]=-ETA; a[row,off+2*n+t]=1/ETA
            if t: a[row,off+4*n+t-1]=-1
            else: rhs[row]=initial
        upper[off:off+n]=np.maximum(net[w],0)+S
        upper[off+n:off+3*n]=S
        lower[off+4*n:off+5*n]=EMIN; upper[off+4*n:off+5*n]=EMAX
        objective[off:off+n]=5*prices/m; objective[off+5*n-1]=-nu/m
    a=a.tocsr()
    lp=linprog(objective,A_eq=a,b_eq=rhs,bounds=list(zip(lower,upper)),method='highs')
    if not lp.success: raise RuntimeError(lp.message)
    relaxed=float(lp.fun)
    # Fix each scenario's sign to that induced by the relaxed shared plan.
    # This is a feasible LP restriction, with the unrestricted LP providing a
    # rigorous lower bound. If the gap is small no integer solve is necessary.
    mode=(net>=lp.x[:n])
    branch_bounds=list(zip(lower,upper))
    bc=lil_matrix((m*n,nv)); bu=np.zeros(m*n)
    for w in range(m):
        off=n+5*n*w
        for t in range(n):
            row=w*n+t
            if mode[w,t]:
                branch_bounds[off+n+t]=(0,0)
                bc[row,off+2*n+t]=1; bc[row,t]=1; bu[row]=net[w,t]
            else:
                branch_bounds[off+2*n+t]=(0,0)
                bc[row,off+n+t]=1; bc[row,t]=-1; bu[row]=-net[w,t]
    restricted=linprog(objective,A_eq=a,b_eq=rhs,A_ub=bc.tocsr(),b_ub=bu,
                       bounds=branch_bounds,method='highs')
    if not restricted.success: raise RuntimeError(restricted.message)
    branch_gap=max(0,(restricted.fun-relaxed)/max(abs(restricted.fun),1e-10))
    if branch_only or branch_gap<=gap:
        xx=restricted.x; tr=xx[n:].reshape(m,5,n)
        r=net-xx[:n]
        violation=float(max(np.max(tr[:,1]-np.maximum(-r,0)),np.max(tr[:,2]-np.maximum(r,0)),0))
        assert violation<1e-5
        assert np.max(np.abs(a@xx-rhs))<1e-4
        assert np.min(xx-lower)>-1e-4 and np.max(xx-upper)<1e-4
        return xx[:n],tr,{'seconds':time.perf_counter()-start,'scenario_count':m,'binary_cells':0,
            'separation_rounds':0,'solver_status':0,'relative_gap':branch_gap,
            'relaxed_objective':relaxed,'direction_feasible_objective':float(restricted.fun),
            'scenario_expected_emergency_cost':float(np.sum(tr[:,0]*5*prices)/m),
            'scenario_mean_final_soc':float(tr[:,4,-1].mean()),'max_direction_violation_kwh':violation,
            'method':'fixed_direction_LP_with_global_relaxation_bound'}
    x=lp.x.copy(); active=set(); rounds=0; solver_gap=0.; status=0
    def clean(v):
        q=v[:nv].copy(); traj=q[n:].reshape(m,5,n)
        delta=ETA*traj[:,1]-traj[:,2]/ETA
        traj[:,1]=np.maximum(delta,0)/ETA; traj[:,2]=ETA*np.maximum(-delta,0)
        r=net-q[:n]
        shortage=r+traj[:,1]-traj[:,2]
        traj[:,0]=np.maximum(shortage,0); traj[:,3]=np.maximum(-shortage,0)
        return q,traj
    while True:
        x,traj=clean(x)
        r=net-x[:n]
        bad=(traj[:,1]>np.maximum(-r,0)+1e-5)|(traj[:,2]>np.maximum(r,0)+1e-5)
        cells=set(map(tuple,np.argwhere(bad)))
        if not cells: break
        new=cells-active
        if not new: raise RuntimeError('Direction violations remain in enforced cells')
        active.update(new); rounds+=1
        # Avoid many small re-solves when degeneracy moves a violation to an
        # adjacent slot. After two selective rounds enforce every cell.
        if rounds>=3:
            active.update((w,t) for w in range(m) for t in range(n))
        indexed=sorted(active); nz=len(indexed)
        aa=hstack([a,csr_matrix((a.shape[0],nz))],format='csr')
        cuts=lil_matrix((4*nz,nv+nz)); ub=[]
        for j,(w,t) in enumerate(indexed):
            off=n+5*n*w; z=nv+j; N=net[w,t]
            cp,dp=off+n+t,off+2*n+t
            cuts[4*j,cp]=1; cuts[4*j,z]=S; ub.append(S)
            cuts[4*j+1,dp]=1; cuts[4*j+1,z]=-S; ub.append(0)
            bigp=max(0,upper[t]-N)
            cuts[4*j+2,t]=1; cuts[4*j+2,dp]=1; cuts[4*j+2,z]=bigp; ub.append(N+bigp)
            bign=max(0,N)
            cuts[4*j+3,cp]=1; cuts[4*j+3,t]=-1; cuts[4*j+3,z]=-bign; ub.append(-N)
        # Shared g implies shortage indicators are monotone in scenario net
        # demand at each time. These valid inequalities strengthen the MILP;
        # they are NOT nonanticipativity constraints on storage decisions.
        chains=[]
        for t in range(n):
            ids=sorted([j for j,(w,tt) in enumerate(indexed) if tt==t],key=lambda j:net[indexed[j]])
            chains.extend(zip(ids[:-1],ids[1:]))
        order=lil_matrix((len(chains),nv+nz))
        for k,(j1,j2) in enumerate(chains): order[k,nv+j1]=1; order[k,nv+j2]=-1
        result=milp(np.r_[objective,np.zeros(nz)],integrality=np.r_[np.zeros(nv),np.ones(nz)],
                    bounds=Bounds(np.r_[lower,np.zeros(nz)],np.r_[upper,np.ones(nz)]),
                    constraints=[LinearConstraint(aa,rhs,rhs),LinearConstraint(cuts.tocsr(),-np.inf,np.array(ub)),
                                 LinearConstraint(order.tocsr(),-np.inf,np.zeros(len(chains)))],
                    options={'time_limit':time_limit,'mip_rel_gap':gap})
        if result.x is None: raise RuntimeError('No feasible MILP incumbent: '+result.message)
        x=result.x; status=int(result.status); solver_gap=float(result.mip_gap)
        if rounds>30: raise RuntimeError('Separation exceeded 30 rounds')
    assert np.max(np.abs(a@x-rhs))<1e-4
    assert np.min(x-lower)>-1e-4
    assert np.max(x-upper)<1e-4
    info={'seconds':time.perf_counter()-start,'scenario_count':m,'binary_cells':len(active),
          'separation_rounds':rounds,'solver_status':status,'relative_gap':solver_gap,
          'relaxed_objective':relaxed,'direction_feasible_objective':float(objective@x),
          'scenario_expected_emergency_cost':float(np.sum(traj[:,0]*5*prices)/m),
          'scenario_mean_final_soc':float(traj[:,4,-1].mean()),
          'max_direction_violation_kwh':float(max(np.max(traj[:,1]-np.maximum(-r,0)),np.max(traj[:,2]-np.maximum(r,0)),0))}
    return x[:n],traj,info


def value_reserves(net,grid,prices,nu,step=25.):
    """Convex piecewise-linear interpolation on continuous SOC (never round SOC).

    Each path value has perfect information ONLY within the historical
    scenario calculation. Average next-step values drive the causal policy.
    """
    m,n=net.shape
    states=np.linspace(EMIN,EMAX,int(np.ceil((EMAX-EMIN)/step))+1)
    values=np.broadcast_to(-nu*states,(m,len(states))).copy()
    reserves=np.empty(n)
    for t in range(n-1,-1,-1):
        k=5*prices[t]*ETA
        mean=values.mean(axis=0)+k*states
        reserves[t]=states[np.flatnonzero(mean<=mean.min()+1e-8)[-1]]
        nxt=np.empty_like(values)
        for w in range(m):
            r=net[w,t]-grid[t]
            if r<=0:
                y=np.minimum(EMAX,states+ETA*min(S,-r))
                nxt[w]=np.interp(y,states,values[w])
            else:
                f=values[w]+k*states
                opt=states[np.flatnonzero(f<=f.min()+1e-8)[-1]]
                y=np.clip(opt,np.maximum(EMIN,states-min(S,r)/ETA),states)
                nxt[w]=5*prices[t]*(r-ETA*(states-y))+np.interp(y,states,values[w])
        # Grid interpolation of convex value functions should remain convex.
        assert np.diff(nxt,n=2,axis=1).min()>-1e-6
        values=nxt
    return reserves


def run(dates,load,pv,prices,limit,step=25.,time_limit=30.,checkpoint=None,fl=None,load_forecast="mean7d"):
    base_fl,fv=forecasts(load,pv); energy=6000.; nu=float(prices.min()/ETA)
    if fl is None:
        fl=base_fl
    records=[]; diagnostics=[]; paired=[]
    begin=time.perf_counter()
    first_day=0
    if checkpoint and checkpoint.exists():
        with checkpoint.open('rb') as handle:
            first_day,energy,records,diagnostics,paired=pickle.load(handle)
        print(f'Resume at day {first_day+1}',flush=True)
    for d in range(first_day,limit):
        date=dates[d]
        if d:
            net=scenarios(d,load,pv,fl,fv)
            grid,traj,diag=stochastic_plan(net,prices,energy,nu,time_limit,branch_only=True)
            reserves=value_reserves(net,grid,prices,nu,step)
            means=traj.mean(axis=0)
        else:
            net=np.zeros((1,T)); grid=np.zeros(T); reserves=np.full(T,EMIN)
            means=np.zeros((5,T)); means[4]=EMIN
            diag={'seconds':0.,'scenario_count':0,'binary_cells':0,'separation_rounds':0,
                  'solver_status':0,'relative_gap':0.,'relaxed_objective':0.,'direction_feasible_objective':0.}
        diag['date']=str(date.date()); diagnostics.append(diag)
        # Paired controller audit: same day's g AND same actual starting SOC.
        # The greedy path is reset daily; it is NOT an independent annual policy.
        greedy_energy=energy; greedy_cost=0.
        for t in range(T):
            old=energy
            c,dis,b,u,energy=execute_slot(load[d,t],pv[d,t],grid[t],old,reserves[t])
            gc,gd,gb,gu,greedy_energy=execute_slot(load[d,t],pv[d,t],grid[t],greedy_energy,EMIN)
            greedy_cost+=5*prices[t]*gb
            records.append((date,t,date+pd.Timedelta(minutes=t*10),load[d,t],pv[d,t],prices[t],
                fl[d,t],fv[d,t],net[:,t].mean(),max(0,min(21,d-7)),grid[t],means[1,t],means[2,t],
                means[4,t],reserves[t],old,c,dis,b,u,energy,prices[t]*grid[t],5*prices[t]*b))
        paired.append({'date':str(date.date()),'same_plan_greedy_emergency_cost':greedy_cost,
                       'same_plan_greedy_final_soc':greedy_energy,'value_policy_final_soc':energy})
        if checkpoint:
            with checkpoint.with_suffix('.tmp').open('wb') as handle:
                pickle.dump((d+1,energy,records,diagnostics,paired),handle)
            for attempt in range(10):
                try:
                    checkpoint.with_suffix('.tmp').replace(checkpoint)
                    break
                except PermissionError:
                    if attempt==9: raise
                    time.sleep(.1)
        print(f'{date.date()} scenarios={diag["scenario_count"]} binary={diag["binary_cells"]} '
              f'solve={diag["seconds"]:.2f}s gap={diag["relative_gap"]:.6g}',flush=True)
    cols=['date','slot','interval_start','load_kwh','pv_kwh','price','forecast_load_kwh','forecast_pv_kwh',
          'planning_net_kwh','residual_count','grid_kwh','plan_charge_kwh','plan_discharge_kwh','plan_soc_kwh',
          'reserve_kwh','soc_start_kwh','charge_kwh','discharge_kwh','emergency_kwh','unused_kwh','soc_end_kwh',
          'planned_cost','emergency_cost']
    f=pd.DataFrame(records,columns=cols); f['total_cost']=f.planned_cost+f.emergency_cost
    checks=validate_schedule(f)
    formal=f[f.date>='2025-02-01']
    daily=formal.groupby('date').agg(**{k:(k,'sum') for k in ['grid_kwh','charge_kwh','discharge_kwh',
         'emergency_kwh','unused_kwh','planned_cost','emergency_cost','total_cost']},
         soc_start_kwh=('soc_start_kwh','first'),soc_end_kwh=('soc_end_kwh','last'))
    summary={'settings':{'name':'scenario_value_no_NAC','forecast_days':7,'residual_days':21,'soc_grid_step':step,
                        'load_forecast':load_forecast,
                        'planner':'relaxed_LP_then_fixed_direction_LP_with_bound',
                        'nonanticipativity':False,'direction_enforced':True},'terminal_value':nu,
             'planner_seconds':float(sum(z['seconds'] for z in diagnostics)),
             'current_run_segment_seconds':time.perf_counter()-begin,
             'result_days':len(daily),'result_intervals':len(formal),'validation':checks}
    if len(formal):
        summary.update({k:float(formal[k].sum()) for k in ['grid_kwh','charge_kwh','discharge_kwh','emergency_kwh',
                 'unused_kwh','planned_cost','emergency_cost','total_cost']})
        summary.update(initial_result_soc_kwh=float(formal.soc_start_kwh.iloc[0]),final_soc_kwh=energy,
                       emergency_intervals=int((formal.emergency_kwh>1e-6).sum()))
    return f,daily,summary,pd.DataFrame(diagnostics),pd.DataFrame(paired)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--days',type=int,default=365)
    ap.add_argument('--step',type=float,default=25); ap.add_argument('--time-limit',type=float,default=5)
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/question2/archive/scenarios')
    ap.add_argument('--load-forecast',choices=['weekly_persist','mean7d'],default='weekly_persist',
                    help='day-ahead LOAD forecast method (PV forecast stays the 7-day mean)')
    args=ap.parse_args(); dates,load,pv,prices=load_inputs()
    args.output.mkdir(parents=True,exist_ok=True)
    fl=None
    if args.load_forecast=='weekly_persist':
        representative=pd.read_excel(ROOT/'附件/附件1.xlsx',sheet_name=0).iloc[:,2].to_numpy(float)/6.0
        assert representative.shape==(T,)
        fl=load_forecast_weekly_persist(load,representative)
    f,daily,s,diag,pair=run(dates,load,pv,prices,args.days,args.step,args.time_limit,
                            args.output/'checkpoint.pkl',fl=fl,load_forecast=args.load_forecast)
    diag.to_csv(args.output/'scenario_solver_diagnostics.csv',index=False)
    pair.to_csv(args.output/'same_day_controller_comparison.csv',index=False)
    if args.days==365:
        old=json.loads((ROOT/'outputs/question2/archive/baseline/question2_summary.json').read_text(encoding='utf-8'))
        write_outputs(args.output,f,daily,[s,*old],prices)
    else:
        f.to_csv(args.output/'pilot_schedule.csv',index=False)
        (args.output/'pilot_summary.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
    print(json.dumps(s,indent=2),flush=True)


if __name__=='__main__': main()
