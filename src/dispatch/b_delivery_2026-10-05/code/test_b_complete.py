import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import unittest
from dispatch import simulate
from device_model import load_parameters, initial_state, DeviceState, apply_dispatch
from rolling_dispatch import FLOW_NAMES
from rule_dispatch import fixed_rule
from rolling_dispatch import solve_window
from reserve import reserve_profile

HERE=Path(__file__).parent
P=load_parameters(HERE/'candidate_parameters_15min.json')
CONFIG=json.loads((HERE/'experiment_config.json').read_text())
START=datetime(2019,6,15,tzinfo=timezone.utc)


def idle(net,p,state,**kwargs):
    return dict(grid=0,charge=0,discharge=0,electrolyzer=0,fuel_cell=0,curtailment=0,unserved=0), dict(status='optimal',mip_gap=0,objective_demo_units=0,reserve_shortfall_kwh=0)


class CompleteBTests(unittest.TestCase):
    def test_reserve_monotonic_and_capped(self):
        windows=[dict(q10_kw=0,q50_kw=x/2,q90_kw=x) for x in (0,100,1000)]
        self.assertEqual(reserve_profile(windows,CONFIG['reserve']),[0,50,300])

    def test_crossed_quantiles_rejected(self):
        with self.assertRaises(ValueError):
            reserve_profile([dict(q10_kw=2,q50_kw=1,q90_kw=3)],CONFIG['reserve'])

    def test_persistence_only_uses_completed_observations(self):
        observed=[]
        def capture(net,p,state,**kwargs):
            observed.append(net)
            return idle(net,p,state,**kwargs)
        actual=[dict(load_kw=10,pv_kw=0,wind_kw=0),dict(load_kw=20,pv_kw=0,wind_kw=0)]
        simulate('persistence_milp',None,actual,5,P,CONFIG,START,solve=capture)
        self.assertEqual(observed,[[5]*96,[10]*96])

    def test_future_actual_change_does_not_change_first_plan(self):
        first=[]
        def capture(net,p,state,**kwargs):
            first.append((list(net),state))
            return idle(net,p,state,**kwargs)
        windows=[[dict(q10_kw=1,q50_kw=2,q90_kw=3)]*96]*2
        actual=[dict(load_kw=10,pv_kw=0,wind_kw=0),dict(load_kw=20,pv_kw=0,wind_kw=0)]
        simulate('q50_milp',iter(windows),actual,5,P,CONFIG,START,solve=capture)
        original=first[0]
        first.clear()
        altered=copy.deepcopy(actual);altered[1]['load_kw']=2000
        simulate('q50_milp',iter(windows),altered,5,P,CONFIG,START,solve=capture)
        self.assertEqual(original,first[0])

    def test_no_incumbent_falls_back_without_dropping_step(self):
        def fail(*args,**kwargs):
            raise RuntimeError('MILP status 1: no incumbent')
        actual=[dict(load_kw=100,pv_kw=0,wind_kw=0)]
        rows,log,failure,metrics=simulate('persistence_milp',None,actual,5,P,CONFIG,START,solve=fail)
        self.assertEqual(len(rows),1)
        self.assertTrue(log[0]['fallback'])
        self.assertIn('no incumbent',failure[0]['reason'])
        self.assertEqual(metrics['fallback_steps'],1)

    def test_failed_measured_correction_has_explicit_rule_fallback(self):
        calls=0
        def bad_then_fail(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==2: raise RuntimeError('correction timeout')
            action,info=idle(*args,**kwargs)
            action.update(charge=500,electrolyzer=300)
            return action,info
        actual=[dict(load_kw=0,pv_kw=0,wind_kw=0)]
        rows,log,failure,metrics=simulate('persistence_milp',None,actual,5,P,CONFIG,START,solve=bad_then_fail)
        self.assertEqual(rows[0]['unserved_kw'],0)
        self.assertTrue(log[0]['actual_correction'])
        self.assertTrue(log[0]['fallback'])
        self.assertIn('correction timeout',failure[0]['reason'])

    def test_impossible_reserve_uses_explicit_soft_shortfall(self):
        action,info=solve_window([0],P,initial_state(P),reserve_kwh=[10000],time_limit=5)
        self.assertGreater(info['reserve_shortfall_kwh'],0)
        self.assertEqual(action['unserved'],0)

    def test_final_forecast_generator_validation_is_exhausted(self):
        def windows():
            yield [dict(q10_kw=1,q50_kw=2,q90_kw=3)]*96
            raise ValueError('incomplete final window')
        with self.assertRaisesRegex(ValueError,'incomplete final window'):
            simulate('q50_milp',windows(),[dict(load_kw=10,pv_kw=0,wind_kw=0)],5,P,CONFIG,START,solve=idle)

    def test_tight_inventory_still_allows_next_rule_fallback(self):
        b=P['battery']
        state=DeviceState(b['soc_min']*b['energy_capacity_kwh']+375*.25/b['discharge_efficiency'],300,250,0,0)
        action,_=solve_window([1000],P,state,customer_load_kw=1000,time_limit=5)
        flow=dict(zip(FLOW_NAMES,(action[k] for k in ('grid','charge','discharge','electrolyzer','fuel_cell','curtailment','unserved'))))
        next_state,_=apply_dispatch(state,1000,0,0,flow,P)
        fallback=fixed_rule(next_state,1000,0,0,P)
        apply_dispatch(next_state,1000,0,0,fallback,P)


if __name__=='__main__': unittest.main()
