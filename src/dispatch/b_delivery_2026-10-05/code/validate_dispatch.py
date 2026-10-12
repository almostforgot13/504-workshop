"""Recheck every four-strategy CSV row, including source, state and braking room."""
import argparse
import csv
from datetime import timedelta
import json
from pathlib import Path
from device_model import apply_dispatch, initial_state, load_parameters, ramp_down_energy_kwsteps
from rolling_dispatch import FLOW_NAMES, RESULT_FIELDS, read_actual, utc_stamp, write_csv
from reserve import available_buffer_kwh
from dispatch import STRATEGIES


def verify(directory, actual_path, p):
    reports=[]
    common_stamps=None
    for strategy in STRATEGIES:
        with (directory/f'{strategy}_dispatch_results.csv').open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.DictReader(stream)
            if tuple(reader.fieldnames or ())!=RESULT_FIELDS: raise ValueError('Wrong result columns')
            rows=list(reader)
        if not rows: raise ValueError('Empty dispatch result')
        stamps=[utc_stamp(r['timestamp_utc']) for r in rows]
        if any(v!=stamps[0]+timedelta(minutes=15*t) for t,v in enumerate(stamps)): raise ValueError('Discontinuous times')
        if common_stamps is None: common_stamps=stamps
        if stamps!=common_stamps: raise ValueError('Strategies do not share the same period')
        measured=read_actual(actual_path,stamps[0],len(rows))
        with (directory/f'{strategy}_step_log.csv').open(encoding='utf-8',newline='') as stream: logs=list(csv.DictReader(stream))
        if len(logs)!=len(rows): raise ValueError('Log rows missing')
        state=initial_state(p)
        checks=[]
        errors=[]
        max_residual=0
        b,h,el,fc=(p[k] for k in ('battery','hydrogen_tank','electrolyzer','fuel_cell'))
        dt=p['meta']['time_step_h'];rb=b['net_power_ramp_kw_per_h']*dt;re=el['power_ramp_kw_per_h']*dt;rf=fc['power_ramp_kw_per_h']*dt
        for i,(row,actual,log) in enumerate(zip(rows,measured,logs)):
            problems=[]
            try:
                if row['strategy']!=strategy or row['step_cost']!='': raise ValueError('Wrong strategy/nonblank unapproved cost')
                if log['timestamp_utc']!=row['timestamp_utc'] or log['solver_status']!=row['solver_status']: raise ValueError('Log differs from result')
                for name in actual:
                    if abs(float(row[name])-actual[name])>1e-5: raise ValueError('Result uses a different source value')
                if abs(float(row['battery_energy_start_kwh'])-state.battery_energy_kwh)>1e-5 or abs(float(row['hydrogen_start_kg'])-state.hydrogen_kg)>1e-5: raise ValueError('Start-state discontinuity')
                flow={k:float(row[k]) for k in FLOW_NAMES}
                state,residual=apply_dispatch(state,**actual,flow=flow,p=p)
                max_residual=max(max_residual,abs(residual))
                if abs(float(row['battery_energy_end_kwh'])-state.battery_energy_kwh)>1e-5 or abs(float(row['hydrogen_end_kg'])-state.hydrogen_kg)>1e-5: raise ValueError('End state differs from physical recurrence')
                charge,dis,electro,fuel=(flow[k] for k in ('battery_charge_kw','battery_discharge_kw','electrolyzer_kw','fuel_cell_kw'))
                if state.battery_energy_kwh+ramp_down_energy_kwsteps(charge,rb)*dt*b['charge_efficiency']>b['soc_max']*b['energy_capacity_kwh']+1e-5: raise ValueError('Insufficient charge braking room')
                if state.battery_energy_kwh-ramp_down_energy_kwsteps(dis,rb)*dt/b['discharge_efficiency']<b['soc_min']*b['energy_capacity_kwh']-1e-5: raise ValueError('Insufficient discharge braking room')
                if state.hydrogen_kg+ramp_down_energy_kwsteps(electro,re,el['power_min_on_kw'])*dt/el['electricity_kwh_per_kg_h2']>h['max_kg']+1e-5: raise ValueError('Insufficient electrolyzer braking room')
                if state.hydrogen_kg-ramp_down_energy_kwsteps(fuel,rf,fc['power_min_on_kw'])*dt/fc['electric_output_kwh_per_kg_h2']<h['min_kg']-1e-5: raise ValueError('Insufficient fuel-cell braking room')
                mandatory=max(0,charge-rb)+(max(el['power_min_on_kw'],electro-re) if electro>re+1e-7 else 0)
                if mandatory>p['grid']['import_power_max_kw']+1e-5: raise ValueError('Next-step zero-renewable mandatory load exceeds grid limit')
                expected=max(0,float(log['reserve_requested_kwh'])-available_buffer_kwh(state,p))
                if abs(expected-float(log['reserve_actual_shortfall_kwh']))>1e-5: raise ValueError('Reserve shortfall log mismatch')
            except ValueError as exc:
                problems.append(str(exc));errors.append(f'row {i+1}: {exc}')
            checks.append(dict(timestamp_utc=row['timestamp_utc'],strategy=strategy,passed=not problems,
                reason='; '.join(problems),battery_soc=state.battery_energy_kwh/b['energy_capacity_kwh'],
                hydrogen_state=state.hydrogen_kg/h['capacity_kg']))
        summary=json.loads((directory/f'{strategy}_metrics.json').read_text())
        for key,col in (('grid_import_kwh','grid_import_kw'),('curtailment_kwh','curtailment_kw'),('unserved_kwh','unserved_kw')):
            if abs(summary[key]-sum(float(r[col])*dt for r in rows))>1e-5: errors.append(f'Summary mismatch {key}')
        if summary['total_cost'] is not None or summary['steps']!=len(rows): errors.append('Cost or step count mismatch')
        for key,value in (('terminal_battery_kwh',state.battery_energy_kwh),('terminal_hydrogen_kg',state.hydrogen_kg)):
            if abs(summary[key]-value)>1e-5: errors.append(f'Terminal inventory mismatch {key}')
        if summary['fallback_steps']!=sum(log['fallback']=='True' for log in logs): errors.append('Fallback count mismatch')
        with (directory/f'{strategy}_fallback_log.csv').open(encoding='utf-8',newline='') as stream: events=list(csv.DictReader(stream))
        if sum(event['fallback']=='True' for event in events)!=summary['fallback_steps']: errors.append('Missing fallback event')
        write_csv(directory/f'{strategy}_constraint_check.csv',checks,checks[0].keys())
        reports.append(dict(strategy=strategy,passed=not errors,rows_checked=len(rows),constraint_violations=len(errors),max_balance_residual_kw=max_residual,errors=errors))
    report=dict(passed=all(r['passed'] for r in reports),strategies=reports,total_rows_checked=sum(r['rows_checked'] for r in reports))
    (directory/'verification_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,required=True)
    parser.add_argument('--actual',type=Path,required=True)
    parser.add_argument('--parameters',type=Path,default=Path(__file__).with_name('candidate_parameters_15min.json'))
    args=parser.parse_args()
    report=verify(args.results_dir,args.actual,load_parameters(args.parameters))
    print(json.dumps(report,indent=2))
    if not report['passed']: raise SystemExit(1)
