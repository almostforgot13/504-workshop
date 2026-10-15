"""Perfect-information COST LOWER BOUND, formulated independently.

The model uses actual customer load limits and deficit replacement accounting.
It relaxes future stopping-inventory constraints and imposes no final zero-power
constraint. Thus its feasible set contains the executed trajectories. A finite
solver dual bound is an optimistic lower bound, not a deployable controller.
"""
from pathlib import Path
import json,sys,time,warnings
import numpy as np
import pandas as pd
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from independent_check import check_frame,prices_for,FIELDS
OUT=HERE.parent;INPUT=OUT/'输入';DEST=OUT/'结果/离线成本下界';DEST.mkdir(parents=True,exist_ok=True)

def main():
    p=json.loads((INPUT/'config/devices.json').read_text(encoding='utf-8'))
    cfg=json.loads((INPUT/'config/experiment.json').read_text(encoding='utf-8'));costs=cfg['costs']
    source=pd.read_csv(INPUT/'data/processed/A_15min_virtual_2017_2019.csv');source['time']=pd.to_datetime(source.time,utc=True)
    begin=pd.Timestamp('2019-01-14T00:00:00Z');actual=source.loc[(source.time>=begin)&(source.time<begin+pd.Timedelta(days=1))].reset_index(drop=True)
    n=len(actual);assert n==96;dt=.25;b,h,el,fc,g=(p[k] for k in ('battery','hydrogen_tank','electrolyzer','fuel_cell','grid'))
    prices=prices_for(actual.time,costs);N=(actual.load_kw-actual.pv_kw-actual.wind_kw).to_numpy()
    lower=[];upper=[];objective=[];integer=[];rows=[];lhs=[];rhs=[]
    def var(lo=0.,hi=np.inf,c=0.,kind=0):
        i=len(lower);lower.append(lo);upper.append(hi);objective.append(c);integer.append(kind);return i
    def constraint(terms,lo=-np.inf,hi=np.inf):rows.append(terms);lhs.append(lo);rhs.append(hi)
    emin=b['soc_min']*b['energy_capacity_kwh'];emax=b['soc_max']*b['energy_capacity_kwh']
    Ei=b['soc_initial']*b['energy_capacity_kwh'];Hi=h['initial_kg'];steps=[]
    for t in range(n):
        ids={}
        specs=[('G',0,g['import_power_max_kw'],prices[t]),('C',0,b['charge_power_max_kw'],costs['battery_throughput_per_kwh']),
            ('D',0,b['discharge_power_max_kw'],costs['battery_throughput_per_kwh']),('EL',0,el['power_max_kw'],costs['hydrogen_device_throughput_per_kwh']),
            ('FC',0,fc['power_max_kw'],costs['hydrogen_device_throughput_per_kwh']),('S',0,np.inf,costs['curtailment_per_kwh']),
            ('U',0,float(actual.load_kw.iloc[t]),costs['unserved_per_kwh']),('E',emin,emax,0),('H',h['min_kg'],h['max_kg'],0)]
        for name,lo,hi,c in specs:ids[name]=var(lo,hi,c*dt)
        for name in ['z','onEL','onFC']:ids[name]=var(0,1,0,1)
        ids['startEL']=var(0,1,costs['electrolyzer_start']);ids['startFC']=var(0,1,costs['fuel_cell_start'])
        prev=steps[-1] if steps else None
        constraint({ids['G']:1,ids['D']:1,ids['FC']:1,ids['U']:1,ids['C']:-1,ids['EL']:-1,ids['S']:-1},N[t],N[t])
        er={ids['E']:1,ids['C']:-dt*b['charge_efficiency'],ids['D']:dt/b['discharge_efficiency']}
        hr={ids['H']:1,ids['EL']:-dt/el['electricity_kwh_per_kg_h2'],ids['FC']:dt/fc['electric_output_kwh_per_kg_h2']}
        if prev:er[prev['E']]=-1;hr[prev['H']]=-1
        constraint(er,0 if prev else Ei,0 if prev else Ei);constraint(hr,0 if prev else Hi,0 if prev else Hi)
        constraint({ids['C']:1,ids['z']:-b['charge_power_max_kw']},hi=0)
        constraint({ids['D']:1,ids['z']:b['discharge_power_max_kw']},hi=b['discharge_power_max_kw'])
        for name,on,equip,start in [('EL','onEL',el,'startEL'),('FC','onFC',fc,'startFC')]:
            constraint({ids[name]:1,ids[on]:-equip['power_max_kw']},hi=0)
            constraint({ids[name]:1,ids[on]:-equip['power_min_on_kw']},lo=0)
            turn={ids[start]:1,ids[on]:-1}
            if prev:turn[prev[on]]=1
            constraint(turn,lo=0)
        constraint({ids['onEL']:1,ids['onFC']:1},hi=1)
        for name,terms,rate in [('battery',{ids['D']:1,ids['C']:-1},b['net_power_ramp_kw_per_h']*dt),
                              ('EL',{ids['EL']:1},el['power_ramp_kw_per_h']*dt),('FC',{ids['FC']:1},fc['power_ramp_kw_per_h']*dt)]:
            if prev:
                if name=='battery':terms[prev['D']]=-1;terms[prev['C']]=1
                else:terms[prev[name]]=-1
            constraint(terms,-rate,rate)
        constraint({ids['C']:1,ids['EL']:1},hi=g['import_power_max_kw']+(b['net_power_ramp_kw_per_h']+el['power_ramp_kw_per_h'])*dt)
        steps.append(ids)
    e_def=var(c=costs['terminal_replacement_price_per_kwh']/b['charge_efficiency'])
    h_def=var(c=costs['terminal_replacement_price_per_kwh']*el['electricity_kwh_per_kg_h2'])
    constraint({steps[-1]['E']:1,e_def:1},lo=Ei);constraint({steps[-1]['H']:1,h_def:1},lo=Hi)
    rr=[];cc=[];vv=[]
    for i,terms in enumerate(rows):
        for j,v in terms.items():rr.append(i);cc.append(j);vv.append(v)
    A=coo_matrix((vv,(rr,cc)),shape=(len(rows),len(lower))).tocsr()
    membership=[]
    for method in ['C_tcn_blend_q50','D_raw_reserve']:
        candidate_path=OUT/'结果/压力测试/normal'/method/'dispatch.csv'
        if not candidate_path.exists():continue
        candidate=pd.read_csv(candidate_path);v=np.zeros(len(lower));oldEL=oldFC=0.
        for t,ids in enumerate(steps):
            r=candidate.iloc[t]
            for key,column in [('G',FIELDS[0]),('C',FIELDS[1]),('D',FIELDS[2]),('EL',FIELDS[3]),('FC',FIELDS[4]),('S',FIELDS[5]),('U',FIELDS[6]),
                ('E','battery_energy_end_kwh'),('H','hydrogen_end_kg')]:v[ids[key]]=float(r[column])
            v[ids['z']]=int(r.battery_charge_kw>1e-7);v[ids['onEL']]=int(r.electrolyzer_kw>1e-7);v[ids['onFC']]=int(r.fuel_cell_kw>1e-7)
            v[ids['startEL']]=int(oldEL<=1e-7 and r.electrolyzer_kw>1e-7);v[ids['startFC']]=int(oldFC<=1e-7 and r.fuel_cell_kw>1e-7)
            oldEL,oldFC=float(r.electrolyzer_kw),float(r.fuel_cell_kw)
        v[e_def]=max(0,Ei-float(candidate.battery_energy_end_kwh.iloc[-1]));v[h_def]=max(0,Hi-float(candidate.hydrogen_end_kg.iloc[-1]))
        av=A@v;violation=max(0.,float(np.max(np.array(lower)-v)),float(np.max(v-np.array(upper))),float(np.max(np.array(lhs)-av)),float(np.max(av-np.array(rhs))))
        membership.append({'strategy':method,'reference_constraints_satisfied':violation<=1e-5,'max_violation':violation,'same_objective_cost_cny':float(np.dot(objective,v))})
    if membership and not all(r['reference_constraints_satisfied'] for r in membership):raise RuntimeError('Reference is not a valid relaxation for compared trajectories')
    started=time.perf_counter()
    print('Solving independent actual-net cost relaxation, 96 steps, 300s ceiling.',flush=True)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Unrecognized options detected:.*threads.*')
        result=milp(np.array(objective),integrality=np.array(integer),bounds=Bounds(lower,upper),
            constraints=LinearConstraint(A,lhs,rhs),options={'time_limit':300.,'mip_rel_gap':1e-4,'threads':1})
    info={'status':int(result.status),'message':str(result.message),'wall_seconds':time.perf_counter()-started,
        'objective_cny':float(result.fun) if result.fun is not None else None,
        'dual_bound_cny':float(result.mip_dual_bound) if getattr(result,'mip_dual_bound',None) is not None and np.isfinite(result.mip_dual_bound) else None,
        'mip_gap':float(result.mip_gap) if getattr(result,'mip_gap',None) is not None and np.isfinite(result.mip_gap) else None,
        'reference_type':'Perfect-information relaxation. Removed future stopping-inventory requirements and final-power-zero requirements. Same actual load, ratings, ramps, mutual exclusion, tariffs, startup charges and deficit replacement accounting.',
        'not_online_controller':True,'not_claimed_globally_optimal_for_original_controller':True,'steps':n}
    info['executed_trajectory_membership']=membership
    if result.x is not None:
        x=result.x;prevE=Ei;prevH=Hi;oldEL=oldFC=0.;records=[]
        for t,ids in enumerate(steps):
            G,C,D,EL,FC,S,U=[max(0.,float(x[ids[k]])) for k in ['G','C','D','EL','FC','S','U']]
            flow=dict(zip(FIELDS,[G,C,D,EL,FC,S,U]))
            cost=dt*(prices[t]*G+costs['unserved_per_kwh']*U+costs['curtailment_per_kwh']*S+costs['battery_throughput_per_kwh']*(C+D)+costs['hydrogen_device_throughput_per_kwh']*(EL+FC))
            cost+=costs['electrolyzer_start']*int(oldEL<=1e-7 and EL>1e-7)+costs['fuel_cell_start']*int(oldFC<=1e-7 and FC>1e-7)
            records.append({'timestamp_utc':str(actual.time.iloc[t]),'strategy':'offline_cost_relaxation',
                **{k:float(actual[k].iloc[t]) for k in ['load_kw','pv_kw','wind_kw']},**flow,
                'battery_energy_start_kwh':prevE,'battery_energy_end_kwh':float(x[ids['E']]),
                'hydrogen_start_kg':prevH,'hydrogen_end_kg':float(x[ids['H']]),'step_cost':cost,'price_per_kwh':prices[t]})
            prevE=float(x[ids['E']]);prevH=float(x[ids['H']]);oldEL,oldFC=EL,FC
        frame=pd.DataFrame(records);frame.to_csv(DEST/'参考轨迹.csv',index=False)
        relaxed=check_frame(frame,p,cfg,source,stop_required=False);strict=check_frame(frame,p,cfg,source,stop_required=True)
        info.update(relaxed_physics_audit_passed=relaxed['passed'],original_stop_audit_passed=strict['passed'],
            original_stop_violations=strict['violation_counts'],unserved_kwh=float(frame.unserved_kw.sum()*.25),
            audited_adjusted_cost_cny=relaxed['adjusted_cost_cny'],objective_accounting_difference=abs(relaxed['adjusted_cost_cny']-float(result.fun)))
        (DEST/'松弛模型独立审计.json').write_text(json.dumps(relaxed,ensure_ascii=False,indent=2),encoding='utf-8')
        if not relaxed['passed'] or info['objective_accounting_difference']>1e-4:raise RuntimeError('Offline reference audit failed')
    (DEST/'求解证据.json').write_text(json.dumps(info,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
