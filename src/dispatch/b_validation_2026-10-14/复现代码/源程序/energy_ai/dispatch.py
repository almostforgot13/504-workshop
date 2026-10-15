"""Physical MILP with first-hour endpoint recourse and soft energy reserves.

Endpoint recourse is a local feasibility model, not a simultaneous 96-step
chance constraint. Sustained reserve is explicitly a conservative heuristic.
All executed actions share measured-interval recourse and an independent audit.
"""
import json
import math
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from .data import ROOT, config, read_data, read_forecasts, dump_json
from .device_model import DeviceState, FLOW_FIELDS, apply_dispatch, initial_state, load_parameters, ramp_down_energy_kwsteps

NAMES=('grid','charge','discharge','electrolyzer','fuel_cell','curtailment','unserved','energy','hydrogen','charge_on','electrolyzer_on','fuel_cell_on')
ACTION_NAMES=NAMES[:7]
FLOW_MAP=dict(zip(ACTION_NAMES,FLOW_FIELDS))

def parameters():return load_parameters(ROOT/'config/devices.json')

def tariff(stamps,costs):
    t=pd.DatetimeIndex(stamps).tz_convert(costs['tariff_timezone'])
    hour=t.hour.to_numpy()
    return np.where(hour<7,costs['valley_price_per_kwh'],np.where(((hour>=10)&(hour<13))|((hour>=18)&(hour<21)),costs['peak_price_per_kwh'],costs['normal_price_per_kwh']))

def safe_limits(state,p):
    b,h,el,fc=(p[k] for k in ('battery','hydrogen_tank','electrolyzer','fuel_cell'));dt=p['meta']['time_step_h']
    specs={
        'charge':(b['charge_power_max_kw'],b['net_power_ramp_kw_per_h']*dt,0,(b['soc_max']*b['energy_capacity_kwh']-state.battery_energy_kwh)/(dt*b['charge_efficiency'])),
        'discharge':(b['discharge_power_max_kw'],b['net_power_ramp_kw_per_h']*dt,0,(state.battery_energy_kwh-b['soc_min']*b['energy_capacity_kwh'])*b['discharge_efficiency']/dt),
        'electrolyzer':(el['power_max_kw'],el['power_ramp_kw_per_h']*dt,el['power_min_on_kw'],(h['max_kg']-state.hydrogen_kg)*el['electricity_kwh_per_kg_h2']/dt),
        'fuel_cell':(fc['power_max_kw'],fc['power_ramp_kw_per_h']*dt,fc['power_min_on_kw'],(state.hydrogen_kg-h['min_kg'])*fc['electric_output_kwh_per_kg_h2']/dt)}
    out={}
    for name,(rated,ramp,minimum,available) in specs.items():
        lo,hi=0.,rated
        for _ in range(35):
            v=(lo+hi)/2
            if v+ramp_down_energy_kwsteps(v,ramp,minimum)<=available+1e-7:lo=v
            else:hi=v
        out[name]=lo
    return out

def assert_stop_safe(state,p,tol=1e-5):
    """Inventory covers mandatory future ramp-down, including charge headroom."""
    b,h,el,fc=(p[k] for k in ('battery','hydrogen_tank','electrolyzer','fuel_cell'));dt=.25
    bn=state.battery_net_power_kw
    required=ramp_down_energy_kwsteps(abs(bn),b['net_power_ramp_kw_per_h']*dt)*dt
    if bn>=0:
        if state.battery_energy_kwh-required/b['discharge_efficiency']<b['soc_min']*b['energy_capacity_kwh']-tol:raise ValueError('Insufficient battery stopping inventory')
    else:
        if state.battery_energy_kwh+required*b['charge_efficiency']>b['soc_max']*b['energy_capacity_kwh']+tol:raise ValueError('Insufficient battery stopping headroom')
    ef=ramp_down_energy_kwsteps(state.electrolyzer_power_kw,el['power_ramp_kw_per_h']*dt,el['power_min_on_kw'])*dt/el['electricity_kwh_per_kg_h2']
    ff=ramp_down_energy_kwsteps(state.fuel_cell_power_kw,fc['power_ramp_kw_per_h']*dt,fc['power_min_on_kw'])*dt/fc['electric_output_kwh_per_kg_h2']
    if state.hydrogen_kg+ef>h['max_kg']+tol or state.hydrogen_kg-ff<h['min_kg']-tol:raise ValueError('Insufficient hydrogen stopping inventory/headroom')

class Model:
    def __init__(self):
        self.lo=[];self.hi=[];self.cost=[];self.integer=[];self.rows=[];self.rl=[];self.rh=[]
    def var(self,lo=0,hi=np.inf,cost=0,integer=0):
        i=len(self.lo);self.lo.append(lo);self.hi.append(hi);self.cost.append(cost);self.integer.append(integer);return i
    def row(self,terms,lo=-np.inf,hi=np.inf):
        self.rows.append(terms);self.rl.append(lo);self.rh.append(hi)
    def solve(self,time_limit,gap):
        ri=[];ci=[];values=[]
        for r,terms in enumerate(self.rows):
            for c,v in terms.items():
                if v:ri.append(r);ci.append(c);values.append(v)
        A=coo_matrix((values,(ri,ci)),shape=(len(self.rows),len(self.lo))).tocsr()
        start=time.perf_counter()
        # SciPy forwards this native HiGHS option. Fixing one thread avoids
        # differing implicit core counts when seasonal jobs run concurrently.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message="Unrecognized options detected:.*threads.*",category=RuntimeWarning)
            result=milp(np.asarray(self.cost),integrality=np.asarray(self.integer),bounds=Bounds(self.lo,self.hi),
                constraints=LinearConstraint(A,self.rl,self.rh),options={'time_limit':time_limit,'mip_rel_gap':gap,'threads':1})
        elapsed=time.perf_counter()-start
        if result.status not in (0,1) or result.x is None:raise RuntimeError(f'MILP has no feasible incumbent: status={result.status}')
        x=result.x;lhs=A@x
        tol=1e-5
        if (np.any(x<np.asarray(self.lo)-tol) or np.any(x>np.asarray(self.hi)+tol) or
            np.any(lhs<np.asarray(self.rl)-tol) or np.any(lhs>np.asarray(self.rh)+tol) or
            np.any(np.abs(x[np.asarray(self.integer)==1]-np.rint(x[np.asarray(self.integer)==1]))>tol)):
            raise RuntimeError('Solver incumbent failed full constraint audit')
        g=getattr(result,'mip_gap',None)
        return x,{'solver_status':'optimal' if result.status==0 else 'time_limit_feasible',
            'solver_seconds':elapsed,'mip_gap':float(g) if g is not None and np.isfinite(g) else None,
            'plan_objective_cny':float(result.fun)}

def solve_plan(net,p,state,prices,costs,*,lower=None,upper=None,time_limit=.25,gap=.01,
               target=None,customer_load=None,terminal=True,risk_scale=1.0):
    net=np.asarray(net,dtype=float);prices=np.asarray(prices,dtype=float);n=len(net)
    if not 1<=n<=96 or prices.shape!=(n,) or not np.isfinite(net).all():raise ValueError('Bad planning horizon/prices')
    if not np.isfinite(prices).all() or (prices<0).any() or time_limit<=0:raise ValueError('Bad prices/time limit')
    reserve=lower is not None or upper is not None
    if reserve:
        lower,upper=np.asarray(lower),np.asarray(upper)
        if lower.shape!=net.shape or upper.shape!=net.shape or not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower>net).any() or (upper<net).any():raise ValueError('Bad reserve interval')
    b,h,el,fc,g=(p[k] for k in ('battery','hydrogen_tank','electrolyzer','fuel_cell','grid'));dt=.25
    emin=b['soc_min']*b['energy_capacity_kwh'];emax=b['soc_max']*b['energy_capacity_kwh']
    rb=b['net_power_ramp_kw_per_h']*dt;re=el['power_ramp_kw_per_h']*dt;rf=fc['power_ramp_kw_per_h']*dt
    m=Model();limits=safe_limits(state,p)

    def step(value,price,previous,weight=1.,is_first=False):
        caps={'grid':g['import_power_max_kw'],'charge':b['charge_power_max_kw'],'discharge':b['discharge_power_max_kw'],
            'electrolyzer':el['power_max_kw'],'fuel_cell':fc['power_max_kw'],'unserved':max(0.,value) if customer_load is None else customer_load,
            'energy':emax,'hydrogen':h['max_kg']}
        weights={'grid':price,'curtailment':costs['curtailment_per_kwh'],'unserved':costs['unserved_per_kwh'],
            'charge':costs['battery_throughput_per_kwh'],'discharge':costs['battery_throughput_per_kwh'],
            'electrolyzer':costs['hydrogen_device_throughput_per_kwh'],'fuel_cell':costs['hydrogen_device_throughput_per_kwh']}
        ix={name:m.var(lo=emin if name=='energy' else h['min_kg'] if name=='hydrogen' else 0,
            hi=1 if name.endswith('_on') else min(caps.get(name,np.inf),limits.get(name,np.inf)) if is_first else caps.get(name,np.inf),
            cost=weights.get(name,0)*dt*weight,integer=int(name.endswith('_on'))) for name in NAMES}
        row=m.row
        row({ix['grid']:1,ix['discharge']:1,ix['fuel_cell']:1,ix['unserved']:1,ix['charge']:-1,ix['electrolyzer']:-1,ix['curtailment']:-1},value,value)
        row({ix['charge']:1,ix['charge_on']:-b['charge_power_max_kw']},hi=0)
        row({ix['discharge']:1,ix['charge_on']:b['discharge_power_max_kw']},hi=b['discharge_power_max_kw'])
        for name,equipment in [('electrolyzer',el),('fuel_cell',fc)]:
            row({ix[name]:1,ix[name+'_on']:-equipment['power_max_kw']},hi=0)
            row({ix[name]:1,ix[name+'_on']:-equipment['power_min_on_kw']},lo=0)
        row({ix['electrolyzer_on']:1,ix['fuel_cell_on']:1},hi=1)
        row({ix['charge']:1,ix['electrolyzer']:1},hi=g['import_power_max_kw']+rb+re)
        energy={ix['energy']:1,ix['charge']:-dt*b['charge_efficiency'],ix['discharge']:dt/b['discharge_efficiency']}
        hydrogen={ix['hydrogen']:1,ix['electrolyzer']:-dt/el['electricity_kwh_per_kg_h2'],ix['fuel_cell']:dt/fc['electric_output_kwh_per_kg_h2']}
        ramp={ix['discharge']:1,ix['charge']:-1}
        if previous is None:
            row(energy,state.battery_energy_kwh,state.battery_energy_kwh);row(hydrogen,state.hydrogen_kg,state.hydrogen_kg)
            row(ramp,state.battery_net_power_kw-rb,state.battery_net_power_kw+rb)
        else:
            energy[previous['energy']]=-1;hydrogen[previous['hydrogen']]=-1
            row(energy,0,0);row(hydrogen,0,0)
            ramp[previous['discharge']]=-1;ramp[previous['charge']]=1;row(ramp,-rb,rb)
        for name,rate,old in [('electrolyzer',re,state.electrolyzer_power_kw),('fuel_cell',rf,state.fuel_cell_power_kw)]:
            terms={ix[name]:1}
            if previous is not None:terms[previous[name]]=-1;old=0
            row(terms,old-rate,old+rate)
            start=m.var(hi=1,cost=costs[name+'_start']*weight)
            terms={start:1,ix[name+'_on']:-1}
            if previous is not None:terms[previous[name+'_on']]=1;old_on=0
            else:old_on=int(old>1e-7)
            row(terms,lo=-old_on)
        return ix

    base=[]
    for t,value in enumerate(net):base.append(step(value,prices[t],base[-1] if base else None,is_first=t==0))
    if terminal:
        for name in ('charge','discharge','electrolyzer','fuel_cell'):m.hi[base[-1][name]]=0
        initial=initial_state(p);replacement=costs['terminal_replacement_price_per_kwh']
        for name,goal,unit_cost in [('energy',initial.battery_energy_kwh,replacement/b['charge_efficiency']),
            ('hydrogen',initial.hydrogen_kg,replacement*el['electricity_kwh_per_kg_h2'])]:
            deficit=m.var(cost=unit_cost);m.row({base[-1][name]:1,deficit:1},lo=goal)
    reserve_indices=[];reserve_slacks={}
    if reserve:
        # Each alternative starts from the same nominal previous state. These
        # independent endpoints are not a jointly realizable scenario tree.
        for t in range(min(4,n)):
            for side,values in [('lower',lower),('upper',upper)]:
                value=net[t]+risk_scale*(values[t]-net[t])
                reserve_indices.append((side,t,step(value,prices[t],base[t-1] if t else None,
                    weight=0.2,is_first=t==0)))
        upside=np.maximum(upper-net,0)*risk_scale;downside=np.maximum(net-lower,0)*risk_scale
        # Sum of marginal widths is a reserve heuristic, not an energy quantile.
        targets={
            'battery_up_kwh':min(emax-emin,float(upside[:4].sum()*dt/b['discharge_efficiency'])),
            'battery_down_kwh':min(emax-emin,float(downside[:4].sum()*dt*b['charge_efficiency'])),
            'hydrogen_up_kg':min(h['max_kg']-h['min_kg'],float(upside[4:min(24,n)].sum()*dt/fc['electric_output_kwh_per_kg_h2'])),
            'hydrogen_down_kg':min(h['max_kg']-h['min_kg'],float(downside[4:min(24,n)].sum()*dt/el['electricity_kwh_per_kg_h2']))}
        for label,state_name,sign,goal,unit in [
            ('battery_up_kwh','energy',1,emin+targets['battery_up_kwh'],1),
            ('battery_down_kwh','energy',-1,-emax+targets['battery_down_kwh'],1),
            ('hydrogen_up_kg','hydrogen',1,h['min_kg']+targets['hydrogen_up_kg'],fc['electric_output_kwh_per_kg_h2']),
            ('hydrogen_down_kg','hydrogen',-1,-h['max_kg']+targets['hydrogen_down_kg'],el['electricity_kwh_per_kg_h2'])]:
            slack=m.var(cost=costs['reserve_shortfall_per_kw_interval']*unit)
            m.row({base[0][state_name]:sign,slack:1},lo=goal);reserve_slacks[label]=slack
    if target is not None:
        for name in ('charge','discharge','electrolyzer','fuel_cell'):
            deviation=m.var(cost=10.0*dt)
            m.row({base[0][name]:1,deviation:-1},hi=target[name])
            m.row({base[0][name]:-1,deviation:-1},hi=-target[name])
    x,info=m.solve(time_limit,gap)
    action={name:max(0.,float(x[base[0][name]])) for name in ACTION_NAMES}
    info.update({'reserve_enabled':reserve,'reserve_targets':targets if reserve else {},
        'reserve_shortfalls':{label:float(x[ix]) for label,ix in reserve_slacks.items()},
        'endpoint_unserved_kw':sum(float(x[ix['unserved']]) for _,_,ix in reserve_indices)})
    return action,info

def settle(action,actual,p):
    net=actual['load_kw']-actual['pv_kw']-actual['wind_kw']+action['charge']+action['electrolyzer']-action['discharge']-action['fuel_cell']
    grid=min(p['grid']['import_power_max_kw'],max(0.,net));unserved=max(0.,net-grid)
    if unserved>actual['load_kw']+1e-6:raise ValueError('Device consumption cannot be labeled as customer shortfall')
    values={'grid':grid,'charge':action['charge'],'discharge':action['discharge'],'electrolyzer':action['electrolyzer'],
        'fuel_cell':action['fuel_cell'],'curtailment':max(0.,-net),'unserved':unserved}
    return {FLOW_MAP[k]:v for k,v in values.items()}

def brake_action(state,p):
    b,el,fc=(p[k] for k in ('battery','electrolyzer','fuel_cell'));dt=.25
    net=math.copysign(max(0.,abs(state.battery_net_power_kw)-b['net_power_ramp_kw_per_h']*dt),state.battery_net_power_kw)
    action={name:0. for name in ACTION_NAMES};action['discharge']=max(0.,net);action['charge']=max(0.,-net)
    for name,equipment,old in [('electrolyzer',el,state.electrolyzer_power_kw),('fuel_cell',fc,state.fuel_cell_power_kw)]:
        v=max(0.,old-equipment['power_ramp_kw_per_h']*dt)
        action[name]=max(equipment['power_min_on_kw'],v) if v>1e-7 else 0.
    return action

def rule_action(net,state,p):
    action=brake_action(state,p);limits=safe_limits(state,p);b,el,fc=(p[k] for k in ('battery','electrolyzer','fuel_cell'))
    if state.battery_net_power_kw*net>=0:
        if net>=0:action['discharge']=max(action['discharge'],min(net,limits['discharge'],state.battery_net_power_kw+b['net_power_ramp_kw_per_h']*.25))
        else:action['charge']=max(action['charge'],min(-net,limits['charge'],-state.battery_net_power_kw+b['net_power_ramp_kw_per_h']*.25))
    residual=net+action['charge']-action['discharge']
    if residual>=0 and action['electrolyzer']<=1e-7:
        v=min(residual,limits['fuel_cell'],state.fuel_cell_power_kw+fc['power_ramp_kw_per_h']*.25)
        action['fuel_cell']=max(action['fuel_cell'],v if v>=fc['power_min_on_kw'] else 0)
    elif residual<0 and action['fuel_cell']<=1e-7:
        v=min(-residual,limits['electrolyzer'],state.electrolyzer_power_kw+el['power_ramp_kw_per_h']*.25)
        action['electrolyzer']=max(action['electrolyzer'],v if v>=el['power_min_on_kw'] else 0)
    if action['charge']+action['electrolyzer']>p['grid']['import_power_max_kw']+(b['net_power_ramp_kw_per_h']+el['power_ramp_kw_per_h'])*.25:
        return brake_action(state,p)
    return action

def step_cost(flow,prior,price,costs):
    return (.25*(price*flow['grid_import_kw']+costs['unserved_per_kwh']*flow['unserved_kw']+
        costs['curtailment_per_kwh']*flow['curtailment_kw']+costs['battery_throughput_per_kwh']*(flow['battery_charge_kw']+flow['battery_discharge_kw'])+
        costs['hydrogen_device_throughput_per_kwh']*(flow['electrolyzer_kw']+flow['fuel_cell_kw']))+
        costs['electrolyzer_start']*int(prior.electrolyzer_power_kw<=1e-7 and flow['electrolyzer_kw']>1e-7)+costs['fuel_cell_start']*int(prior.fuel_cell_power_kw<=1e-7 and flow['fuel_cell_kw']>1e-7))

def replacement_cost(state,p,costs):
    target=initial_state(p);price=costs['terminal_replacement_price_per_kwh']
    return price*(max(0.,target.battery_energy_kwh-state.battery_energy_kwh)/p['battery']['charge_efficiency']+
        max(0.,target.hydrogen_kg-state.hydrogen_kg)*p['electrolyzer']['electricity_kwh_per_kg_h2'])

def run_strategy(d,forecast,strategy,start,steps,*,time_limit=None,p=None,cfg=None,renewable_scale=1.,force_failure=False,mode='default'):
    cfg=config() if cfg is None else cfg;p=parameters() if p is None else p;costs=cfg['costs']
    budget=cfg['solver_time_limit_s'] if time_limit is None else time_limit
    if strategy not in cfg['methods'] or steps<1 or not 0<=renewable_scale<=2:raise ValueError('Invalid strategy/steps/scenario')
    start=pd.Timestamp(start);actuals=d.loc[(d.time>=start)&(d.time<start+pd.Timedelta(minutes=15*steps))].copy()
    if len(actuals)!=steps:raise ValueError('Missing execution actuals')
    actuals['pv_kw']*=renewable_scale;actuals['wind_kw']*=renewable_scale
    if strategy!='A_rule':
        f=forecast.copy();f['issue_time']=pd.to_datetime(f.issue_time,utc=True)
        windows={issue:frame for issue,frame in f.groupby('issue_time',sort=False)}
    else:windows={}
    state=initial_state(p);results=[];log=[]
    prices_all=tariff(pd.date_range(start,periods=steps+95,freq='15min'),costs)
    for t,(_,r) in enumerate(actuals.iterrows()):
        stamp=r.time;actual={k:float(r[k]) for k in ('load_kw','pv_kw','wind_kw')};prior=state;corrected=False;fallback=False
        info={'solver_status':'rule','solver_seconds':0.,'mip_gap':None,'reserve_enabled':False,'reserve_targets':{},'reserve_shortfalls':{},'endpoint_unserved_kw':0.}
        lower=upper=None;window=None
        if strategy=='A_rule':planned=rule_action(actual['load_kw']-actual['pv_kw']-actual['wind_kw'],state,p)
        else:
            if stamp not in windows:raise ValueError(f'Missing forecast at {stamp}')
            window=windows[stamp];n=min(96,steps-t);net=window.q50_kw.to_numpy()[:n]
            if strategy in ('D_raw_reserve','E_cqr_reserve') and mode!='economy':
                fields=('lower_cal_kw','upper_cal_kw') if strategy=='E_cqr_reserve' else ('q10_kw','q90_kw')
                if not set(fields).issubset(window):raise ValueError('Missing calibrated bounds')
                lower=window[fields[0]].to_numpy()[:n];upper=window[fields[1]].to_numpy()[:n]
            try:
                attempt_start=time.perf_counter()
                if force_failure:raise RuntimeError('Injected solver failure')
                planned,info=solve_plan(net,p,state,prices_all[t:t+n],costs,lower=lower,upper=upper,time_limit=budget,gap=cfg['solver_mip_gap'],risk_scale=cfg['risk_scale'])
            except RuntimeError as exc:
                planned=brake_action(state,p);fallback=True;info['solver_status']='fallback';info['failure_reason']=str(exc)
                info['solver_seconds']=time.perf_counter()-attempt_start
        try:
            recourse_start=time.perf_counter()
            # Same measured-interval optimization for all methods, including
            # rule and timeout fallback. It cannot use the next interval.
            if force_failure:raise RuntimeError('Injected recourse solver failure')
            executed,recourse=solve_plan([actual['load_kw']-actual['pv_kw']-actual['wind_kw']],p,state,[prices_all[t]],costs,
                time_limit=budget,gap=cfg['solver_mip_gap'],target=planned,customer_load=actual['load_kw'],terminal=False)
            corrected=any(abs(executed[k]-planned[k])>1e-5 for k in ('charge','discharge','electrolyzer','fuel_cell'))
            flow=settle(executed,actual,p)
            candidate,residual=apply_dispatch(prior,**actual,flow=flow,p=p);assert_stop_safe(candidate,p)
        except (RuntimeError,ValueError) as exc:
            fallback=True;recourse={'solver_status':'fallback','solver_seconds':time.perf_counter()-recourse_start}
            info['recourse_failure_reason']=str(exc)
            flow=settle(brake_action(state,p),actual,p)
            candidate,residual=apply_dispatch(prior,**actual,flow=flow,p=p);assert_stop_safe(candidate,p)
        state=candidate
        cost=step_cost(flow,prior,prices_all[t],costs)
        row={'timestamp_utc':str(stamp),'strategy':strategy,**actual,**flow,
            'battery_energy_start_kwh':prior.battery_energy_kwh,'battery_energy_end_kwh':state.battery_energy_kwh,
            'hydrogen_start_kg':prior.hydrogen_kg,'hydrogen_end_kg':state.hydrogen_kg,
            'solver_status':info['solver_status'],'step_cost':cost,'price_per_kwh':float(prices_all[t]),
            'fallback':fallback,'actual_correction':corrected,'balance_residual_kw':residual,
            'solver_seconds':info['solver_seconds'],'recourse_seconds':recourse['solver_seconds'],
            'mip_gap':info['mip_gap'],'recourse_status':recourse['solver_status']}
        results.append(row)
        entry={'issue_time':str(stamp),**info,'recourse_status':recourse['solver_status']}
        if window is not None:
            entry.update({'q10_first_kw':float(window.q10_kw.iloc[0]),'q50_first_kw':float(window.q50_kw.iloc[0]),'q90_first_kw':float(window.q90_kw.iloc[0]),
                'lower_used_kw':float(lower[0]) if lower is not None else None,'upper_used_kw':float(upper[0]) if upper is not None else None})
        log.append(entry)
        if (t+1)%96==0:print(f'{strategy} {start.date()}: {t+1}/{steps}',flush=True)
    frame=pd.DataFrame(results);summary=summarize(frame,p,cfg);summary.update({'start':str(start),'renewable_scale':renewable_scale,'mode':mode,'solver_budget_s':budget})
    return frame,log,summary

def summarize(frame,p,cfg):
    last=frame.iloc[-1]
    state=DeviceState(float(last.battery_energy_end_kwh),float(last.hydrogen_end_kg))
    cost=float(frame.step_cost.sum());replacement=replacement_cost(state,p,cfg['costs'])
    gap=frame.mip_gap.dropna()
    return {'strategy':str(frame.strategy.iloc[0]),'steps':len(frame),'operating_cost_cny':cost,'replacement_accounting_cny':replacement,
        'adjusted_cost_cny':cost+replacement,'terminal_battery_kwh':state.battery_energy_kwh,'terminal_hydrogen_kg':state.hydrogen_kg,
        'battery_target_kwh':initial_state(p).battery_energy_kwh,'hydrogen_target_kg':initial_state(p).hydrogen_kg,
        'grid_import_kwh':float(frame.grid_import_kw.sum()*.25),'curtailment_kwh':float(frame.curtailment_kw.sum()*.25),
        'unserved_kwh':float(frame.unserved_kw.sum()*.25),'battery_throughput_kwh':float((frame.battery_charge_kw+frame.battery_discharge_kw).sum()*.25),
        'electrolyzer_starts':int(((frame.electrolyzer_kw>1e-7)&(frame.electrolyzer_kw.shift(1,fill_value=0)<=1e-7)).sum()),
        'fuel_cell_starts':int(((frame.fuel_cell_kw>1e-7)&(frame.fuel_cell_kw.shift(1,fill_value=0)<=1e-7)).sum()),
        'max_balance_residual_kw':float(frame.balance_residual_kw.abs().max()),'constraint_violations':0,
        'fallback_steps':int(frame.fallback.sum()),'actual_corrections':int(frame.actual_correction.sum()),
        'time_limit_feasible_steps':int((frame.solver_status=='time_limit_feasible').sum()),
        'max_mip_gap':float(gap.max()) if len(gap) else None,'mean_plan_seconds':float(frame.solver_seconds.mean()),
            'costs_are_scenario_assumptions':True,'terminal_policy':cfg['terminal_policy'],'settlement_policy':cfg['settlement_policy']}

def audit(frame,p,cfg,expected_actual=None,renewable_scale=1.):
    """Recompute physics, temporal continuity, prices and costs from saved CSV."""
    state=initial_state(p);cost_sum=0.;max_residual=0.
    stamps=pd.to_datetime(frame.timestamp_utc,utc=True)
    if stamps.duplicated().any() or not (stamps.diff().iloc[1:]==pd.Timedelta(minutes=15)).all():raise ValueError('Bad result timestamps')
    prices=tariff(stamps,cfg['costs'])
    if expected_actual is not None:
        source=expected_actual.set_index('time').loc[pd.DatetimeIndex(stamps),['load_kw','pv_kw','wind_kw']].copy()
        source[['pv_kw','wind_kw']]*=renewable_scale
        if not np.allclose(source.to_numpy(),frame[['load_kw','pv_kw','wind_kw']].to_numpy(),atol=1e-5,rtol=0):raise ValueError('Saved actuals differ from the declared input scenario')
    for t,(_,r) in enumerate(frame.iterrows()):
        if abs(state.battery_energy_kwh-r.battery_energy_start_kwh)>1e-5 or abs(state.hydrogen_kg-r.hydrogen_start_kg)>1e-5:raise ValueError('State discontinuity')
        flow={k:float(r[k]) for k in FLOW_FIELDS};actual={k:float(r[k]) for k in ('load_kw','pv_kw','wind_kw')}
        new,residual=apply_dispatch(state,**actual,flow=flow,p=p);assert_stop_safe(new,p)
        if abs(new.battery_energy_kwh-r.battery_energy_end_kwh)>1e-5 or abs(new.hydrogen_kg-r.hydrogen_end_kg)>1e-5:raise ValueError('Inventory accounting mismatch')
        cost=step_cost(flow,state,prices[t],cfg['costs'])
        if abs(cost-r.step_cost)>1e-5 or abs(prices[t]-r.price_per_kwh)>1e-8:raise ValueError('Cost/tariff mismatch')
        state=new;cost_sum+=cost;max_residual=max(max_residual,abs(residual))
    return {'passed':True,'steps':len(frame),'source_actuals_checked':expected_actual is not None,'independently_recomputed_cost_cny':cost_sum,
        'replacement_accounting_cny':replacement_cost(state,p,cfg['costs']),'max_balance_residual_kw':max_residual}
