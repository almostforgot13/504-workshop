"""Run with the existing embedded Python; outputs stay inside this package."""
from pathlib import Path
import argparse,copy,hashlib,itertools,json,platform,shutil,sys,time,warnings
import numpy as np
import pandas as pd
from scipy.optimize import linprog,milp,Bounds,LinearConstraint

HERE=Path(__file__).resolve().parent;OUT=HERE.parent;WORKSPACE=OUT.parent.parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE/'源程序'))
from energy_ai import data
data.ROOT=OUT/'输入';data.DEFAULT_DATA=data.ROOT/'data/processed/A_15min_virtual_2017_2019.csv'
from energy_ai import dispatch as dp
from energy_ai.device_model import DeviceState
from independent_check import check_frame,independent_limits,prices_for,FIELDS

P=json.loads((data.ROOT/'config/devices.json').read_text(encoding='utf-8'))
CFG=json.loads((data.ROOT/'config/experiment.json').read_text(encoding='utf-8'))
PROTOCOL=json.loads((OUT/'实验方案_结果产生前固定.json').read_text(encoding='utf-8'))
RESULTS=OUT/'结果';RESULTS.mkdir(exist_ok=True)
NATIVE_MILP=dp.milp;CAPTURE=[]
def traced_milp(*args,**kwargs):
    result=NATIVE_MILP(*args,**kwargs);CAPTURE.append(result)
    return result
dp.milp=traced_milp
def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def event(message,**kw):
    value={'time_utc':pd.Timestamp.now(tz='UTC').isoformat(),'message':message,**kw}
    with (OUT/'实验进度.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
    print(json.dumps(value,ensure_ascii=False),flush=True)
def finite(value):return float(value) if value is not None and np.isfinite(value) else None
def source():
    d=pd.read_csv(data.DEFAULT_DATA);d['time']=pd.to_datetime(d.time,utc=True);return d
def versions():
    import scipy,torch,lightgbm
    import os
    dump(OUT/'运行环境.json',{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,
        'scipy':scipy.__version__,'torch':torch.__version__,'lightgbm':lightgbm.__version__,
        'platform':platform.platform(),'logical_cpu_count':os.cpu_count(),'solver_threads':1,
        'forecast_threads':4,'note':'Wall-clock time limits are hardware/load dependent; forecasts are saved for replay.'})

def historical_audit():
    d=source();records=[]
    original=WORKSPACE/'work/submission_demo/stage/application/assets/results/dispatch'
    saved=data.ROOT/'既有轨迹'
    if not saved.exists():
        for path in original.glob('*/*.csv'):
            dest=saved/path.parent.name/path.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    paths=sorted(saved.glob('*/*.csv'))
    if len(paths)!=20:raise ValueError(f'Expected 20 historical trajectories, got {len(paths)}')
    sample=None
    for path in paths:
        frame=pd.read_csv(path);report=check_frame(frame,P,CFG,d)
        records.append({'file':str(path.relative_to(OUT)),**report})
        if sample is None:sample=frame.copy()
        event('historical_audit',file=path.name,passed=report['passed'],steps=len(frame))
    dump(RESULTS/'既有28天_独立审计.json',records)
    pd.DataFrame([{'file':r['file'],'passed':r['passed'],'steps':r['steps'],'checks':r['checks'],
        'violations':sum(r['violation_counts'].values()),**r['max_errors']} for r in records]).to_csv(RESULTS/'既有28天_独立审计.csv',index=False,encoding='utf-8-sig')
    mutations=[]
    for label,column,change in [('power_balance','grid_import_kw',10),('battery_inventory','battery_energy_end_kwh',5),
        ('hydrogen_inventory','hydrogen_end_kg',1),('cost','step_cost',5),('negative_flow','unserved_kw',-20)]:
        bad=sample.copy();bad.loc[0,column]+=change;report=check_frame(bad,P,CFG,d)
        mutations.append({'injected':label,'detected':not report['passed'],'categories':report['violation_counts']})
    bad=sample.copy();bad.loc[1,'timestamp_utc']=bad.loc[0,'timestamp_utc'];report=check_frame(bad,P,CFG)
    mutations.append({'injected':'duplicate_time','detected':not report['passed'],'categories':report['violation_counts']})
    dump(RESULTS/'独立检查器_错误注入.json',mutations)
    if not all(r['passed'] for r in records) or not all(m['detected'] for m in mutations):raise RuntimeError('Independent checks require investigation')

def enumerated_lp(net,prices,state):
    """Independent continuous LP for every fixed on/off mode; no production Model."""
    n=len(net);dt=.25;b=P['battery'];h=P['hydrogen_tank'];el=P['electrolyzer'];fc=P['fuel_cell'];g=P['grid'];costs=CFG['costs']
    nv=9*n+2;c=np.zeros(nv);eq=[];rhs=[];ub=[];urb=[]
    bounds=[];emin=b['soc_min']*b['energy_capacity_kwh'];emax=b['soc_max']*b['energy_capacity_kwh']
    limits=independent_limits(P,state)
    def row(terms):
        r=np.zeros(nv)
        for k,v in terms.items():r[k]=v
        return r
    for t,N in enumerate(net):
        k=9*t
        caps=[g['import_power_max_kw'],b['charge_power_max_kw'],b['discharge_power_max_kw'],el['power_max_kw'],fc['power_max_kw'],None,max(0,float(N))]
        if t==0:
            for j in range(1,5):caps[j]=min(caps[j],limits[j-1])
        bounds.extend([(0,v) for v in caps]+[(emin,emax),(h['min_kg'],h['max_kg'])])
        c[k:k+7]=dt*np.array([prices[t],costs['battery_throughput_per_kwh'],costs['battery_throughput_per_kwh'],
            costs['hydrogen_device_throughput_per_kwh'],costs['hydrogen_device_throughput_per_kwh'],costs['curtailment_per_kwh'],costs['unserved_per_kwh']])
        eq.append(row({k:1,k+1:-1,k+2:1,k+3:-1,k+4:1,k+5:-1,k+6:1}));rhs.append(N)
        er={k+7:1,k+1:-dt*b['charge_efficiency'],k+2:dt/b['discharge_efficiency']}
        hr={k+8:1,k+3:-dt/el['electricity_kwh_per_kg_h2'],k+4:dt/fc['electric_output_kwh_per_kg_h2']}
        if t:er[k-2]=-1;hr[k-1]=-1
        eq.extend([row(er),row(hr)]);rhs.extend([0 if t else state.battery_energy_kwh,0 if t else state.hydrogen_kg])
        for device,terms,rate,old in [('battery',{k+2:1,k+1:-1},b['net_power_ramp_kw_per_h']*dt,state.battery_net_power_kw),
            ('el',{k+3:1},el['power_ramp_kw_per_h']*dt,state.electrolyzer_power_kw),('fc',{k+4:1},fc['power_ramp_kw_per_h']*dt,state.fuel_cell_power_kw)]:
            if t:
                if device=='battery':terms[k-7]=-1;terms[k-8]=1
                else:terms[k-6 if device=='el' else k-5]=-1
                old=0
            r=row(terms);ub.extend([r,-r]);urb.extend([old+rate,-old+rate])
        ub.append(row({k+1:1,k+3:1}));urb.append(g['import_power_max_kw']+(b['net_power_ramp_kw_per_h']+el['power_ramp_kw_per_h'])*dt)
    bounds.extend([(0,None),(0,None)])
    c[-2]=costs['terminal_replacement_price_per_kwh']/b['charge_efficiency'];c[-1]=costs['terminal_replacement_price_per_kwh']*el['electricity_kwh_per_kg_h2']
    ub.extend([row({9*(n-1)+7:-1,nv-2:-1}),row({9*(n-1)+8:-1,nv-1:-1})]);urb.extend([-b['energy_capacity_kwh']*b['soc_initial'],-h['initial_kg']])
    best=None;feasible=0
    # Per interval: battery charge/discharge mode x hydrogen off/EL/FC = 6.
    choices=list(itertools.product([0,1],[0,1,2]))
    for modes in itertools.product(choices,repeat=n):
        bd=list(bounds);startup=0.;prevEL=int(state.electrolyzer_power_kw>1e-7);prevFC=int(state.fuel_cell_power_kw>1e-7)
        valid=True
        for t,(battery,hydrogen) in enumerate(modes):
            k=9*t
            if battery==0:bd[k+1]=(0,0)
            else:bd[k+2]=(0,0)
            for j,on,equip in [(3,hydrogen==1,el),(4,hydrogen==2,fc)]:
                bd[k+j]=(equip['power_min_on_kw'],bd[k+j][1]) if on else (0,0)
                if on and bd[k+j][0]>bd[k+j][1]+1e-7:valid=False
            if t==n-1:
                if hydrogen!=0:valid=False
                for j in range(1,5):bd[k+j]=(0,0)
            onEL=int(hydrogen==1);onFC=int(hydrogen==2)
            startup+=costs['electrolyzer_start']*max(0,onEL-prevEL)+costs['fuel_cell_start']*max(0,onFC-prevFC)
            prevEL,prevFC=onEL,onFC
        if not valid:continue
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Unrecognized options detected:.*threads.*')
            solution=linprog(c,A_ub=np.asarray(ub),b_ub=np.asarray(urb),A_eq=np.asarray(eq),b_eq=np.asarray(rhs),bounds=bd,method='highs',options={'dual_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8,'threads':1})
        if solution.success:
            feasible+=1;value=float(solution.fun+startup)
            if best is None or value<best['objective']:best={'objective':value,'modes':modes,'x':solution.x}
    if best is None:raise RuntimeError('No enumerated feasible reference')
    return best,6**n,feasible

def small_cases():
    cases=[('surplus',[-250,-100,0],DeviceState(1000,300)),('deficit',[700,800,650],DeviceState(1000,300)),
        ('sign_switch',[-250,600,0],DeviceState(1000,300)),('near_empty',[650,650,650],DeviceState(310,60.5)),
        ('near_full',[-500,-300,-200],DeviceState(1890,539.8))]
    records=[]
    for label,net,state in cases:
        prices=[.8]*3;reference,count,feasible=enumerated_lp(net,prices,state);CAPTURE.clear()
        action,info=dp.solve_plan(net,P,state,prices,CFG['costs'],time_limit=60,gap=1e-8)
        delta=abs(info['plan_objective_cny']-reference['objective'])
        records.append({'case':label,'net_kw':net,'initial_E_kwh':state.battery_energy_kwh,'initial_H_kg':state.hydrogen_kg,
            'enumerated_modes':count,'feasible_modes':feasible,'reference_objective':reference['objective'],
            'milp_objective':info['plan_objective_cny'],'absolute_difference':delta,'mip_gap':info['mip_gap'],'passed':delta<=1e-4,'first_action':action})
        event('small_reference',case=label,passed=delta<=1e-4,difference=delta)
    dump(RESULTS/'小样例_枚举参考解.json',records)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['net_kw','first_action']} for r in records]).to_csv(RESULTS/'小样例_枚举参考解.csv',index=False,encoding='utf-8-sig')
    if not all(r['passed'] for r in records):raise RuntimeError('Small reference mismatch')

def make_forecasts():
    import torch,lightgbm as lgb
    from energy_ai.forecast import load_tcn,predict_tcn,WindowDataset,blend,predict_lgbm,forecast_one
    from energy_ai.calibration import apply_cqr
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    d,a,y=data.read_data(data.DEFAULT_DATA);model,norm=load_tcn()
    weights=json.loads((data.ROOT/'models/fusion.json').read_text())['weights'];cqr=json.loads((data.ROOT/'models/calibration.json').read_text())
    models=[lgb.Booster(model_str=(data.ROOT/f'models/lgbm_q{q}.txt').read_text()) for q in [10,50,90]]
    proofs=[]
    for start in PROTOCOL['new_test_days']:
        folder=RESULTS/'新日期预测'/pd.Timestamp(start).strftime('%Y%m%d');folder.mkdir(parents=True,exist_ok=True)
        o=int(d.index[d.time==pd.Timestamp(start)][0]);oo=np.arange(o,o+96)
        qt=predict_tcn(model,WindowDataset(d,a,y,oo,norm));qb=blend(qt,y,oo,weights)
        ql=predict_lgbm(models,a,y,d.time,oo,threads=4);interval=apply_cqr(qb,cqr)
        ft=data.forecasts_frame(d.time,oo,qb,interval);fl=data.forecasts_frame(d.time,oo,ql)
        ft.to_csv(folder/'tcn_blend.csv.gz',index=False);fl.to_csv(folder/'lightgbm.csv.gz',index=False)
        # The single-origin API explicitly discards every observation at/after issue.
        t=forecast_one(data.DEFAULT_DATA,pd.Timestamp(start),'tcn');l=forecast_one(data.DEFAULT_DATA,pd.Timestamp(start),'lgbm')
        delta_t=float(np.max(np.abs(t[['q10_kw','q50_kw','q90_kw']].to_numpy()-ft.iloc[:96][['q10_kw','q50_kw','q90_kw']].to_numpy())))
        delta_l=float(np.max(np.abs(l[['q10_kw','q50_kw','q90_kw']].to_numpy()-fl.iloc[:96][['q10_kw','q50_kw','q90_kw']].to_numpy())))
        proofs.append({'day':start,'tcn_full_batch_vs_history_only_max_kw':delta_t,'lgbm_full_batch_vs_history_only_max_kw':delta_l,
            'passed':delta_t<.01 and delta_l<.01,'all_calendar_lag_feature_indices_before_issue':True})
        event('new_forecasts',day=start,rows_per_method=len(ft),history_only_check=proofs[-1]['passed'])
    dump(RESULTS/'预测历史隔离检查.json',proofs)
    if not all(r['passed'] for r in proofs):raise RuntimeError('Forecast history isolation mismatch')

def save_run(folder,d,forecast,strategy,start,steps,budget,scale=1.,force=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists():return json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    event('run_started',run=str(folder.relative_to(OUT)),budget=budget)
    CAPTURE.clear();begin=time.perf_counter()
    frame,logs,summary=dp.run_strategy(d,forecast,strategy,start,steps,time_limit=budget,p=P,cfg=CFG,renewable_scale=scale,force_failure=force)
    report=check_frame(frame,P,CFG,d,scale)
    frame.to_csv(folder/'dispatch.csv',index=False);dump(folder/'planning_log.json',logs);dump(folder/'independent_audit.json',report)
    summary.update(wall_seconds=time.perf_counter()-begin,independent_audit_passed=report['passed'],load_kwh=float(frame.load_kw.sum()*.25),
        shortfall_fraction=float(frame.unserved_kw.sum()/frame.load_kw.sum()),run_folder=str(folder.relative_to(OUT)))
    dump(folder/'summary.json',summary)
    dump(folder/'solver_certificates.json',[{'status':int(r.status),'objective':finite(r.fun),'dual_bound':finite(getattr(r,'mip_dual_bound',None)),
        'gap':finite(getattr(r,'mip_gap',None))} for r in CAPTURE])
    event('run_finished',run=str(folder.relative_to(OUT)),unserved=summary['unserved_kwh'],cost=summary['adjusted_cost_cny'],audit=report['passed'])
    if not report['passed']:raise RuntimeError('New run failed independent audit')
    return summary

def new_days():
    d=source();records=[]
    for start in PROTOCOL['new_test_days']:
        key=pd.Timestamp(start).strftime('%Y%m%d');fp=RESULTS/'新日期预测'/key
        ft=data.read_forecasts(fp/'tcn_blend.csv.gz');fl=data.read_forecasts(fp/'lightgbm.csv.gz')
        for method in PROTOCOL['new_test_methods']:
            summary=save_run(RESULTS/'新日期回放'/key/method,d,fl if method=='B_lgbm_q50' else ft,method,start,96,.25)
            records.append(summary);pd.DataFrame(records).to_csv(RESULTS/'新日期回放汇总.csv',index=False,encoding='utf-8-sig')
    dump(RESULTS/'新日期回放汇总.json',records)

def stress():
    d=source();start=PROTOCOL['stress_day']
    original=WORKSPACE/'work/submission_demo/stage/application/assets/results/forecasts/20190114/tcn_blend.csv.gz'
    dest=data.ROOT/'诊断日预测/tcn_blend.csv.gz';dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():shutil.copy2(original,dest)
    f=data.read_forecasts(dest);records=[]
    low_budget=save_run(RESULTS/'整日预算对照/0p25',d,f,'C_tcn_blend_q50',start,96,.25)
    budget_comparison=[{**low_budget,'comparison_budget_s':.25}]
    for scenario in PROTOCOL['stress_scenarios']:
        for method in PROTOCOL['stress_methods']:
            summary=save_run(RESULTS/'压力测试'/scenario['name']/method,d,f,method,start,96,2,scenario['scale'],scenario['force_failure'])
            summary={**summary,'scenario':scenario['name']};records.append(summary)
            if scenario['name']=='normal' and method=='C_tcn_blend_q50':
                budget_comparison.append({**summary,'comparison_budget_s':2.})
                pd.DataFrame(budget_comparison).to_csv(RESULTS/'整日预算对照.csv',index=False,encoding='utf-8-sig')
            pd.DataFrame(records).to_csv(RESULTS/'压力测试汇总.csv',index=False,encoding='utf-8-sig')
    dump(RESULTS/'压力测试汇总.json',records)

def snapshots():
    stored=data.ROOT/'既有轨迹/20190114/C_tcn_blend_q50.csv'
    df=pd.read_csv(stored);start=pd.Timestamp(PROTOCOL['stress_day'])
    f=data.read_forecasts(data.ROOT/'诊断日预测/tcn_blend.csv.gz')
    f['issue_time']=pd.to_datetime(f.issue_time,utc=True);records=[]
    dest=RESULTS/'求解预算_固定状态.csv'
    if dest.exists():records=pd.read_csv(dest).replace({np.nan:None}).to_dict('records')
    completed={(r['hour'],r['budget_s']) for r in records}
    for t in range(0,96,8):
        r=df.iloc[t];old=df.iloc[t-1] if t else None
        state=DeviceState(float(r.battery_energy_start_kwh),float(r.hydrogen_start_kg),
            float(old.battery_discharge_kw-old.battery_charge_kw) if old is not None else 0.,float(old.electrolyzer_kw) if old is not None else 0.,float(old.fuel_cell_kw) if old is not None else 0.)
        issue=start+pd.Timedelta(minutes=15*t);window=f.loc[f.issue_time==issue];prices=prices_for(pd.date_range(issue,periods=96,freq='15min'),CFG['costs'])
        for budget in PROTOCOL['solver_budgets_s']:
            if (t/4,budget) in completed:continue
            CAPTURE.clear();begin=time.perf_counter()
            record={'hour':t/4,'issue':str(issue),'budget_s':budget,'initial_E':state.battery_energy_kwh,'initial_H':state.hydrogen_kg,'n':96}
            try:
                action,info=dp.solve_plan(window.q50_kw.to_numpy(),P,state,prices,CFG['costs'],time_limit=budget,gap=.01)
                result=CAPTURE[-1];record.update(info);record.update(dual_bound=finite(getattr(result,'mip_dual_bound',None)),action_json=json.dumps(action))
            except RuntimeError as e:record.update(solver_status='no_incumbent',error=str(e),plan_objective_cny=None,mip_gap=None,dual_bound=finite(getattr(CAPTURE[-1],'mip_dual_bound',None)) if CAPTURE else None)
            record['wall_seconds']=time.perf_counter()-begin
            record={k:v for k,v in record.items() if k not in ('reserve_targets','reserve_shortfalls')}
            records.append(record);pd.DataFrame(records).to_csv(dest,index=False,encoding='utf-8-sig')
            event('solver_snapshot',hour=t/4,budget=budget,status=record['solver_status'],gap=record['mip_gap'])
    dump(RESULTS/'求解预算_固定状态.json',records)

def stress_extra():
    d=source();start='2019-11-11T00:00:00Z'
    f=data.read_forecasts(RESULTS/'新日期预测/20191111/tcn_blend.csv.gz');records=[]
    cases=[('normal',1.,False),('renewable_drop_60pct',.4,False),('renewable_zero',0.,False),('forced_failure',1.,True)]
    for label,scale,failure in cases:
        for method in PROTOCOL['stress_methods']:
            summary=save_run(RESULTS/'强化压力测试'/label/method,d,f,method,start,96,2,scale,failure)
            records.append({**summary,'scenario':label})
            pd.DataFrame(records).to_csv(RESULTS/'强化压力测试汇总.csv',index=False,encoding='utf-8-sig')
    dump(RESULTS/'强化压力测试汇总.json',records)
    state=DeviceState(550.,75.,500.,0.,500.);initial=(550.,75.,500.,0.,500.);trajectory=[]
    for t in range(6):
        stamp=pd.Timestamp('2019-11-11T00:00:00Z')+pd.Timedelta(minutes=15*t)
        actual={'load_kw':700.,'pv_kw':0.,'wind_kw':0.};prior=state
        action=dp.brake_action(state,P);flow=dp.settle(action,actual,P)
        state,residual=dp.apply_dispatch(state,**actual,flow=flow,p=P);dp.assert_stop_safe(state,P)
        price=float(prices_for(pd.DatetimeIndex([stamp]),CFG['costs'])[0]);cost=dp.step_cost(flow,prior,price,CFG['costs'])
        trajectory.append({'timestamp_utc':str(stamp),'strategy':'warm_state_failure',**actual,**flow,
            'battery_energy_start_kwh':prior.battery_energy_kwh,'battery_energy_end_kwh':state.battery_energy_kwh,
            'hydrogen_start_kg':prior.hydrogen_kg,'hydrogen_end_kg':state.hydrogen_kg,'step_cost':cost,'price_per_kwh':price})
    frame=pd.DataFrame(trajectory);frame.to_csv(RESULTS/'高功率回退轨迹.csv',index=False)
    report=check_frame(frame,P,CFG,initial=initial);dump(RESULTS/'高功率回退_独立审计.json',report)
    if not report['passed']:raise RuntimeError('Warm-state braking failed')
    event('warm_state_failure',passed=report['passed'],final_E=state.battery_energy_kwh,final_H=state.hydrogen_kg)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['checks','forecasts','new_days','stress','snapshots','stress_extra']);args=parser.parse_args()
    versions();event('phase_started',phase=args.phase)
    if args.phase=='checks':historical_audit();small_cases()
    elif args.phase=='forecasts':make_forecasts()
    elif args.phase=='new_days':new_days()
    elif args.phase=='stress':stress()
    elif args.phase=='snapshots':snapshots()
    elif args.phase=='stress_extra':stress_extra()
    event('phase_completed',phase=args.phase)
if __name__=='__main__':main()
