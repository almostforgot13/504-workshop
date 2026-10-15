"""CSV checks written independently: no energy_ai functions are imported."""
from collections import Counter
import math
import numpy as np
import pandas as pd

FIELDS = ['grid_import_kw','battery_charge_kw','battery_discharge_kw','electrolyzer_kw',
          'fuel_cell_kw','curtailment_kw','unserved_kw']

def stopping_sum(power, decrement, minimum=0.):
    if decrement <= 0:
        return float('inf') if power > 1e-7 else 0.
    total=0.
    while power > decrement+1e-7:
        power=max(minimum,power-decrement)
        total+=power
    return total

def independent_limits(p,state):
    dt=p['meta']['time_step_h'];b=p['battery'];h=p['hydrogen_tank'];el=p['electrolyzer'];fc=p['fuel_cell']
    E,H=state.battery_energy_kwh,state.hydrogen_kg
    specs=[(b['charge_power_max_kw'],b['net_power_ramp_kw_per_h']*dt,0.,(b['soc_max']*b['energy_capacity_kwh']-E)/(dt*b['charge_efficiency'])),
           (b['discharge_power_max_kw'],b['net_power_ramp_kw_per_h']*dt,0.,(E-b['soc_min']*b['energy_capacity_kwh'])*b['discharge_efficiency']/dt),
           (el['power_max_kw'],el['power_ramp_kw_per_h']*dt,el['power_min_on_kw'],(h['max_kg']-H)*el['electricity_kwh_per_kg_h2']/dt),
           (fc['power_max_kw'],fc['power_ramp_kw_per_h']*dt,fc['power_min_on_kw'],(H-h['min_kg'])*fc['electric_output_kwh_per_kg_h2']/dt)]
    result=[]
    for rated,ramp,minimum,available in specs:
        low,high=0.,rated
        for _ in range(35):
            mid=(low+high)/2
            if mid+stopping_sum(mid,ramp,minimum)<=available+1e-7:low=mid
            else:high=mid
        result.append(low)
    return result

def prices_for(stamps,costs):
    hours=pd.DatetimeIndex(stamps).tz_convert(costs['tariff_timezone']).hour
    return np.where(hours<7,costs['valley_price_per_kwh'],
        np.where(((hours>=10)&(hours<13))|((hours>=18)&(hours<21)),costs['peak_price_per_kwh'],costs['normal_price_per_kwh']))

def check_frame(frame,p,cfg,source=None,scale=1.,initial=None,stop_required=True,tol=1e-5):
    b,h,el,fc,g=(p[k] for k in ('battery','hydrogen_tank','electrolyzer','fuel_cell','grid'))
    dt=p['meta']['time_step_h'];costs=cfg['costs'];counts=Counter();examples=[];checks=0
    maxima=Counter();E=b['energy_capacity_kwh']*b['soc_initial'];H=h['initial_kg'];bn=e0=f0=0.
    if initial is not None:E,H,bn,e0,f0=initial
    def validate(ok,kind,row):
        nonlocal checks
        checks+=1
        if not ok:
            counts[kind]+=1
            if len(examples)<12:examples.append({'row':int(row),'kind':kind})
    stamps=pd.to_datetime(frame.timestamp_utc,utc=True)
    validate(not stamps.duplicated().any(),'duplicate_timestamp',0)
    validate((stamps.diff().iloc[1:]==pd.Timedelta(minutes=15)).all(),'time_continuity',0)
    prices=prices_for(stamps,costs);total_cost=0.
    expected=None
    if source is not None:
        expected=source.set_index('time').loc[pd.DatetimeIndex(stamps),['load_kw','pv_kw','wind_kw']].copy()
        expected[['pv_kw','wind_kw']]*=scale
    for i,(_,r) in enumerate(frame.iterrows()):
        values=np.array([r[k] for k in FIELDS]+[r.load_kw,r.pv_kw,r.wind_kw],float)
        validate(np.isfinite(values).all(),'finite',i);validate((values>=-tol).all(),'nonnegative',i)
        if not np.isfinite(values).all():continue
        G,C,D,EL,FC,S,U=[float(r[k]) for k in FIELDS]
        if expected is not None:validate(np.allclose(expected.iloc[i].to_numpy(),[r.load_kw,r.pv_kw,r.wind_kw],atol=tol,rtol=0),'declared_actuals',i)
        validate(abs(r.battery_energy_start_kwh-E)<=tol,'battery_continuity',i)
        validate(abs(r.hydrogen_start_kg-H)<=tol,'hydrogen_continuity',i)
        balance=r.pv_kw+r.wind_kw+G+D+FC-(r.load_kw-U+C+EL+S)
        maxima['balance_kw']=max(maxima['balance_kw'],abs(balance))
        validate(abs(balance)<=tol,'power_balance',i)
        validate(not(C>tol and D>tol),'battery_mutual_exclusion',i)
        validate(not(EL>tol and FC>tol),'hydrogen_mutual_exclusion',i)
        for label,value,cap in [('grid',G,g['import_power_max_kw']),('charge',C,b['charge_power_max_kw']),
            ('discharge',D,b['discharge_power_max_kw']),('electrolyzer',EL,el['power_max_kw']),('fuel_cell',FC,fc['power_max_kw'])]:
            validate(value<=cap+tol,label+'_rating',i)
        validate(EL<=tol or EL>=el['power_min_on_kw']-tol,'electrolyzer_min_on',i)
        validate(FC<=tol or FC>=fc['power_min_on_kw']-tol,'fuel_cell_min_on',i)
        for label,value,old,rate in [('battery',D-C,bn,b['net_power_ramp_kw_per_h']),('electrolyzer',EL,e0,el['power_ramp_kw_per_h']),('fuel_cell',FC,f0,fc['power_ramp_kw_per_h'])]:
            validate(abs(value-old)<=rate*dt+tol,label+'_ramp',i)
        En=E+dt*(C*b['charge_efficiency']-D/b['discharge_efficiency'])
        Hn=H+dt*(EL/el['electricity_kwh_per_kg_h2']-FC/fc['electric_output_kwh_per_kg_h2'])
        errorE=abs(En-r.battery_energy_end_kwh);errorH=abs(Hn-r.hydrogen_end_kg)
        maxima['battery_accounting_kwh']=max(maxima['battery_accounting_kwh'],errorE)
        maxima['hydrogen_accounting_kg']=max(maxima['hydrogen_accounting_kg'],errorH)
        validate(errorE<=tol,'battery_accounting',i);validate(errorH<=tol,'hydrogen_accounting',i)
        emin=b['soc_min']*b['energy_capacity_kwh'];emax=b['soc_max']*b['energy_capacity_kwh']
        validate(emin-tol<=En<=emax+tol,'battery_bounds',i)
        validate(h['min_kg']-tol<=Hn<=h['max_kg']+tol,'hydrogen_bounds',i)
        validate(U<=r.load_kw+tol,'customer_shortfall_bound',i)
        validate(not(G>tol and S>tol),'grid_spill_exclusion',i)
        if stop_required:
            future=stopping_sum(abs(D-C),b['net_power_ramp_kw_per_h']*dt)*dt
            need=future/b['discharge_efficiency'] if D>=C else 0.
            room=future*b['charge_efficiency'] if C>D else 0.
            validate(En-need>=emin-tol and En+room<=emax+tol,'battery_stop_inventory',i)
            production=stopping_sum(EL,el['power_ramp_kw_per_h']*dt,el['power_min_on_kw'])*dt/el['electricity_kwh_per_kg_h2']
            use=stopping_sum(FC,fc['power_ramp_kw_per_h']*dt,fc['power_min_on_kw'])*dt/fc['electric_output_kwh_per_kg_h2']
            validate(Hn+production<=h['max_kg']+tol and Hn-use>=h['min_kg']-tol,'hydrogen_stop_inventory',i)
        cost=dt*(prices[i]*G+costs['unserved_per_kwh']*U+costs['curtailment_per_kwh']*S+
            costs['battery_throughput_per_kwh']*(C+D)+costs['hydrogen_device_throughput_per_kwh']*(EL+FC))
        cost+=costs['electrolyzer_start']*int(e0<=1e-7 and EL>1e-7)+costs['fuel_cell_start']*int(f0<=1e-7 and FC>1e-7)
        if 'step_cost' in frame:validate(abs(cost-r.step_cost)<=tol,'cost_accounting',i)
        if 'price_per_kwh' in frame:validate(abs(prices[i]-r.price_per_kwh)<=1e-8,'price',i)
        total_cost+=cost;E,H,bn,e0,f0=En,Hn,D-C,EL,FC
    replacement=costs['terminal_replacement_price_per_kwh']*(max(0,b['energy_capacity_kwh']*b['soc_initial']-E)/b['charge_efficiency']+
        max(0,h['initial_kg']-H)*el['electricity_kwh_per_kg_h2'])
    return {'passed':not counts,'steps':len(frame),'checks':checks,'violation_counts':dict(counts),'examples':examples,
        'max_errors':dict(maxima),'operating_cost_cny':total_cost,'replacement_cny':replacement,
        'adjusted_cost_cny':total_cost+replacement,'terminal_battery_kwh':E,'terminal_hydrogen_kg':H,
        'tolerance':tol,'stop_inventory_checked':stop_required,'actual_source_checked':source is not None}
