"""An analytic lower bound on unserved ENERGY, independent of the optimizer."""
from pathlib import Path
import csv,json
OUT=Path(__file__).resolve().parent.parent;R=OUT/'结果'
p=json.loads((OUT/'输入/config/devices.json').read_text(encoding='utf-8'));b=p['battery'];h=p['hydrogen_tank'];fc=p['fuel_cell']
B0=(b['soc_initial']-b['soc_min'])*b['energy_capacity_kwh']*b['discharge_efficiency']
H0=(h['initial_kg']-h['min_kg'])*fc['electric_output_kwh_per_kg_h2']
assert b['charge_efficiency']*b['discharge_efficiency']<=1
assert fc['electric_output_kwh_per_kg_h2']<=p['electrolyzer']['electricity_kwh_per_kg_h2']
records=[]
for path in sorted(R.rglob('dispatch.csv')):
    with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    L=sum(float(x['load_kw'])*.25 for x in rows)
    renewable=sum((float(x['pv_kw'])+float(x['wind_kw']))*.25 for x in rows)
    grid_limit=p['grid']['import_power_max_kw']*len(rows)*.25
    bound=max(0.,L-renewable-grid_limit-B0-H0)
    observed=sum(float(x['unserved_kw'])*.25 for x in rows)
    records.append({'file':str(path.relative_to(R)),'steps':len(rows),'load_kwh':L,'renewable_kwh':renewable,
        'max_grid_kwh':grid_limit,'initial_usable_battery_bus_kwh':B0,'initial_usable_hydrogen_electric_kwh':H0,
        'unserved_energy_lower_bound_kwh':bound,'observed_unserved_kwh':observed,
        'observed_minus_lower_bound_kwh':observed-bound,'attains_energy_lower_bound_within_1e_5':abs(observed-bound)<=1e-5})
with (R/'能量守恒_缺供下界.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
(R/'能量守恒_缺供下界.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps([r for r in records if r['file'].startswith('强化压力测试')],ensure_ascii=False,indent=2))
