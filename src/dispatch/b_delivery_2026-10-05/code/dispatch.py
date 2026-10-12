"""Run all B strategies with common physical constraints and explicit fallbacks."""
from __future__ import annotations
import argparse
import csv
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import time

from device_model import apply_dispatch, initial_state, load_parameters
from rolling_dispatch import (FLOW_NAMES, RESULT_FIELDS, actual_flow, iter_forecasts,
                              read_actual, solve_window, utc_stamp, write_csv)
from rule_dispatch import fixed_rule
from reserve import reserve_profile, available_buffer_kwh

HERE = Path(__file__).resolve().parent
STRATEGIES = ('rule', 'persistence_milp', 'q50_milp', 'quantile_reserve_milp')


def failure_kind(error):
    message=str(error)
    if 'MILP status 1' in message: return 'timeout_no_incumbent'
    if 'MILP status 2' in message: return 'infeasible'
    if 'MILP status 3' in message: return 'unbounded'
    if 'violates' in message: return 'invalid_incumbent'
    return 'solver_failure'


def last_observed_net(path, start):
    wanted = start - timedelta(minutes=15)
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if utc_stamp(row['time']) == wanted:
                return float(row['load_kw']) - float(row['pv_kw']) - float(row['wind_kw'])
    raise ValueError('Missing last completed observation before first issue')


def simulate(strategy, windows, actuals, previous_net, p, config, start, *, solve=solve_window):
    if strategy not in STRATEGIES:
        raise ValueError('Unknown strategy')
    state = initial_state(p)
    results, logs, failures = [], [], []
    iterator = iter(windows) if windows is not None else None
    for t, actual in enumerate(actuals):
        stamp = start + timedelta(minutes=15*t)
        issue = stamp.isoformat(sep=' ')
        window = None
        if strategy in ('q50_milp', 'quantile_reserve_milp'):
            try:
                window = next(iterator)
            except StopIteration as exc:
                raise ValueError(f'Forecast missing at {issue}') from exc
        elif strategy == 'persistence_milp':
            window = [dict(q10_kw=previous_net, q50_kw=previous_net, q90_kw=previous_net) for _ in range(96)]
        reserve = reserve_profile(window, config['reserve']) if strategy == 'quantile_reserve_milp' else [0.0]*96
        prior = state
        solver_info = dict(status='rule', mip_gap=None, objective_demo_units=None, reserve_shortfall_kwh=0.0)
        fallback, corrected = False, False
        reason = ''
        clock_start = time.perf_counter()
        def safe_rule():
            try: return fixed_rule(prior, **actual, p=p)
            except ValueError as exc:
                raise ValueError(f'{strategy} no physically feasible rule action at {issue}; state={prior}: {exc}') from exc
        if strategy == 'rule':
            flow = safe_rule()
        else:
            try:
                planned, solver_info = solve([x['q50_kw'] for x in window], p, prior,
                    time_limit=config['time_limit_seconds'], reserve_kwh=reserve,
                    weights=config['objective_weights'], mip_gap=config['mip_relative_gap'])
            except RuntimeError as exc:
                reason = str(exc)
                fallback = True
                solver_info['status'] = failure_kind(exc)
                flow = safe_rule()
            else:
                try:
                    flow = actual_flow(planned, actual, p)
                    apply_dispatch(prior, **actual, flow=flow, p=p)
                except ValueError:
                    corrected = True
                    try:
                        action, _ = solve([actual['load_kw']-actual['pv_kw']-actual['wind_kw']], p, prior,
                            customer_load_kw=actual['load_kw'], target=planned,
                            time_limit=config['time_limit_seconds'], weights=config['objective_weights'],
                            mip_gap=config['mip_relative_gap'])
                        flow = dict(zip(FLOW_NAMES, (action[k] for k in
                            ('grid','charge','discharge','electrolyzer','fuel_cell','curtailment','unserved'))))
                        apply_dispatch(prior, **actual, flow=flow, p=p)
                    except (RuntimeError, ValueError) as exc:
                        fallback = True
                        reason = 'Measured correction failed: ' + str(exc)
                        flow = safe_rule()
        # No physical exception is suppressed. A genuinely infeasible external
        # state aborts with a timestamped failure report rather than a fake row.
        try:
            state, residual = apply_dispatch(prior, **actual, flow=flow, p=p)
        except ValueError as exc:
            raise ValueError(f'{strategy} physically infeasible at {issue}: {exc}') from exc
        status = solver_info['status'] + ('_fallback' if fallback else '_corrected' if corrected else '')
        if strategy == 'rule':
            status = 'rule_unserved' if flow['unserved_kw'] > 1e-7 else 'rule_curtailment' if flow['curtailment_kw'] > 1e-7 else 'rule_ok'
        results.append(dict(timestamp_utc=issue, strategy=strategy, **actual, **flow,
            battery_energy_start_kwh=prior.battery_energy_kwh, battery_energy_end_kwh=state.battery_energy_kwh,
            hydrogen_start_kg=prior.hydrogen_kg, hydrogen_end_kg=state.hydrogen_kg,
            solver_status=status, step_cost=''))
        requested = reserve[0]
        executed_shortfall = max(0.0, requested - available_buffer_kwh(state, p))
        logs.append(dict(timestamp_utc=issue, strategy=strategy, solver_status=status,
            solve_seconds=time.perf_counter()-clock_start, mip_gap=solver_info['mip_gap'],
            objective_demo_units=solver_info['objective_demo_units'],
            q50_first_kw=None if window is None else window[0]['q50_kw'],
            previous_observed_net_kw=previous_net,
            actual_net_kw=actual['load_kw']-actual['pv_kw']-actual['wind_kw'],
            reserve_requested_kwh=requested, reserve_available_kwh=available_buffer_kwh(state,p),
            reserve_plan_shortfall_kwh=solver_info['reserve_shortfall_kwh'],
            reserve_actual_shortfall_kwh=executed_shortfall,
            fallback=fallback, actual_correction=corrected, balance_residual_kw=residual))
        if fallback or corrected or solver_info['status'] == 'time_limit_feasible' or executed_shortfall > 1e-6:
            failures.append(dict(timestamp_utc=issue, strategy=strategy, event=status,
                fallback=fallback, actual_correction=corrected, reason=reason,
                mip_gap=solver_info['mip_gap'], reserve_shortfall_kwh=executed_shortfall))
        previous_net = actual['load_kw']-actual['pv_kw']-actual['wind_kw']
        if (t+1) % 24 == 0:
            print(f'{strategy}: {t+1}/{len(actuals)} physical steps passed', flush=True)
    if iterator is not None:
        # Exhaust the final generator continuation so incomplete-window/count
        # errors cannot be bypassed by zip's shortest-iterator behaviour.
        sentinel = object()
        if next(iterator, sentinel) is not sentinel:
            raise ValueError('Forecast contains more selected windows than execution steps')
    dt = p['meta']['time_step_h']
    summary = dict(strategy=strategy, steps=len(results), duration_h=len(results)*dt,
        time_step_h=dt, planning_horizon_steps=96, total_cost=None,
        cost_note='Undefined team cost model; blank step_cost is not zero cost',
        grid_import_kwh=sum(x['grid_import_kw']*dt for x in results),
        curtailment_kwh=sum(x['curtailment_kw']*dt for x in results),
        unserved_kwh=sum(x['unserved_kw']*dt for x in results),
        unserved_steps=sum(x['unserved_kw']>1e-7 for x in results),
        unserved_hours=sum(x['unserved_kw']>1e-7 for x in results)*dt,
        initial_battery_kwh=initial_state(p).battery_energy_kwh,
        initial_hydrogen_kg=initial_state(p).hydrogen_kg,
        terminal_battery_kwh=state.battery_energy_kwh, terminal_hydrogen_kg=state.hydrogen_kg,
        terminal_policy=config['terminal_policy'],
        fallback_steps=sum(x['fallback'] for x in logs),
        correction_steps=sum(x['actual_correction'] for x in logs),
        time_limit_feasible_steps=sum(x['solver_status'].startswith('time_limit_feasible') for x in logs),
        max_mip_gap=max((x['mip_gap'] for x in logs if x['mip_gap'] is not None),default=None),
        reserve_shortfall_steps=sum(x['reserve_actual_shortfall_kwh']>1e-6 for x in logs),
        max_balance_residual_kw=max(abs(x['balance_residual_kw']) for x in logs))
    return results, logs, failures, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual', type=Path, required=True)
    parser.add_argument('--forecast', type=Path, required=True)
    parser.add_argument('--start', required=True)
    parser.add_argument('--steps', type=int, default=192)
    parser.add_argument('--strategy', choices=(*STRATEGIES,'all'), default='all')
    parser.add_argument('--parameters', type=Path, default=HERE/'candidate_parameters_15min.json')
    parser.add_argument('--config', type=Path, default=HERE/'experiment_config.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error('steps must be positive')
    p = load_parameters(args.parameters)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    start = utc_stamp(args.start)
    actuals = read_actual(args.actual,start,args.steps)
    previous = last_observed_net(args.actual,start)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output-dir must be empty; use a new run directory to avoid stale results')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    selected = STRATEGIES if args.strategy=='all' else (args.strategy,)
    summary_table = []
    provenance = dict(status='running',completed_strategies=[],start_utc=start.isoformat(), steps=args.steps,
        actual=str(args.actual), forecast=str(args.forecast), configuration=config,
        parameters_sha256=hashlib.sha256(args.parameters.read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest())
    (args.output_dir/'run_manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    for strategy in selected:
        windows = iter_forecasts(args.forecast,start,args.steps) if strategy in ('q50_milp','quantile_reserve_milp') else None
        try:
            rows,logs,failures,summary=simulate(strategy,windows,actuals,previous,p,config,start)
        except Exception as exc:
            (args.output_dir/f'{strategy}_infeasible.json').write_text(
                json.dumps(dict(strategy=strategy,error=str(exc)),indent=2),encoding='utf-8')
            provenance.update(status='failed',error=str(exc))
            (args.output_dir/'run_manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
            raise
        write_csv(args.output_dir/f'{strategy}_dispatch_results.csv',rows,RESULT_FIELDS)
        write_csv(args.output_dir/f'{strategy}_step_log.csv',logs,logs[0].keys())
        write_csv(args.output_dir/f'{strategy}_fallback_log.csv',failures,
            ('timestamp_utc','strategy','event','fallback','actual_correction','reason','mip_gap','reserve_shortfall_kwh'))
        (args.output_dir/f'{strategy}_metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        summary_table.append(summary)
        provenance['completed_strategies'].append(strategy)
        (args.output_dir/'run_manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    write_csv(args.output_dir/'comparison_metrics.csv',summary_table,summary_table[0].keys())
    provenance['status']='complete'
    (args.output_dir/'run_manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
