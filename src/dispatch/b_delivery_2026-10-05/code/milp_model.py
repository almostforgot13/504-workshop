"""Small-sample, 15-minute battery/hydrogen MILP (SciPy HiGHS).

The objective weights are demonstration assumptions, not agreed tariffs.
Run from this directory with --input and --output. A 24 h day has 96 rows.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from device_model import FLOW_FIELDS, apply_dispatch, initial_state, load_parameters


VARS = ("grid", "charge", "discharge", "electrolyzer", "fuel_cell", "curtailment",
        "unserved", "energy", "hydrogen", "battery_charge_on", "electrolyzer_on", "fuel_cell_on")


def read_day(path: Path, day: str) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = [r for r in csv.DictReader(stream) if r["time"].startswith(day)]
    if len(rows) != 96:
        raise ValueError(f"Expected 96 rows for {day}; found {len(rows)}")
    from datetime import datetime, timedelta, timezone
    start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    for t, row in enumerate(rows):
        stamp = datetime.fromisoformat(row["time"].replace("Z", "+00:00"))
        if stamp != start + timedelta(minutes=15 * t):
            raise ValueError(f"Non-contiguous UTC timestamp at step {t}: {row['time']}")
        for name in ("load_kw", "pv_kw", "wind_kw"):
            row[name] = float(row[name])
            if not np.isfinite(row[name]) or row[name] < 0:
                raise ValueError(f"Invalid {name} at step {t}")
    return rows


def solve_day(rows: list[dict], p: dict, *, time_limit: float = 120.0) -> dict:
    """A clairvoyant training-day reference using the common terminal-safe MILP."""
    from rolling_dispatch import solve_window, RESULT_FIELDS
    state = initial_state(p)
    _, info = solve_window([r["load_kw"]-r["pv_kw"]-r["wind_kw"] for r in rows],
                            p, state, time_limit=time_limit, return_plan=True)
    plan = info.pop("plan")
    dispatch = []
    maximum = 0.0
    for source, action in zip(rows, plan):
        flow = dict(zip(FLOW_FIELDS, (action[k] for k in
                        ("grid","charge","discharge","electrolyzer","fuel_cell","curtailment","unserved"))))
        next_state, residual = apply_dispatch(state,source["load_kw"],source["pv_kw"],source["wind_kw"],flow,p)
        maximum=max(maximum,abs(residual))
        row=dict(timestamp_utc=source["time"],strategy="clairvoyant_training_day_reference",
                 load_kw=source["load_kw"],pv_kw=source["pv_kw"],wind_kw=source["wind_kw"],**flow,
                 battery_energy_start_kwh=state.battery_energy_kwh,battery_energy_end_kwh=next_state.battery_energy_kwh,
                 hydrogen_start_kg=state.hydrogen_kg,hydrogen_end_kg=next_state.hydrogen_kg,
                 solver_status=info["status"],step_cost="")
        dispatch.append({key:row[key] for key in RESULT_FIELDS})
        state=next_state
    dt=p["meta"]["time_step_h"]
    report=dict(solver_status=0 if info["status"]=="optimal" else 1,
                solver_message=info["status"],objective_demo_units=info["objective_demo_units"],
                mip_gap=info["mip_gap"],rows=len(rows),time_step_h=dt,
                physical_check="passed",max_balance_residual_kw=maximum,
                grid_import_kwh=sum(r["grid_import_kw"]*dt for r in dispatch),
                curtailment_kwh=sum(r["curtailment_kw"]*dt for r in dispatch),
                unserved_kwh=sum(r["unserved_kw"]*dt for r in dispatch),
                terminal_battery_kwh=state.battery_energy_kwh,terminal_hydrogen_kg=state.hydrogen_kg,
                terminal_device_powers_zero=all(abs(plan[-1][k])<1e-5 for k in ('charge','discharge','electrolyzer','fuel_cell')),total_cost=None)
    return {"report":report,"dispatch":dispatch}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, default=Path(__file__).with_name("candidate_parameters_15min.json"))
    parser.add_argument("--day", default="2017-01-01")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=120)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():
        parser.error('Use a new output filename to avoid stale sample results')
    p = load_parameters(args.parameters)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = solve_day(read_day(args.input, args.day), p, time_limit=args.time_limit)
    except (RuntimeError, ValueError) as exc:
        args.output.with_suffix('.json').write_text(json.dumps({'solver_status':'failed','error':str(exc)},indent=2),encoding='utf-8')
        raise SystemExit(2) from exc
    args.output.with_suffix(".json").write_text(json.dumps(result["report"], indent=2), encoding="utf-8")
    if result["dispatch"]:
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=result["dispatch"][0].keys())
            writer.writeheader(); writer.writerows(result["dispatch"])
    print(json.dumps(result["report"], indent=2))
    if result["report"]["solver_status"] != 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
