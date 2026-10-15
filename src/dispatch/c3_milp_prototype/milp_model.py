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
    n = len(rows)
    dt = p["meta"]["time_step_h"]
    if abs(dt - 0.25) > 1e-12:
        raise ValueError("This example expects 15-minute parameters")
    b, el, fc, h, g = (p[k] for k in ("battery", "electrolyzer", "fuel_cell", "hydrogen_tank", "grid"))
    initial = initial_state(p)
    m = len(VARS)
    idx = lambda t, name: t * m + VARS.index(name)
    lower = np.zeros(n * m)
    upper = np.full(n * m, np.inf)
    integer = np.zeros(n * m)
    objective = np.zeros(n * m)
    for t, row in enumerate(rows):
        caps = {"grid": g["import_power_max_kw"], "charge": b["charge_power_max_kw"],
                "discharge": b["discharge_power_max_kw"], "electrolyzer": el["power_max_kw"],
                "fuel_cell": fc["power_max_kw"], "unserved": row["load_kw"],
                "energy": b["soc_max"] * b["energy_capacity_kwh"],
                "hydrogen": h["max_kg"]}
        for name, cap in caps.items():
            upper[idx(t, name)] = cap
        lower[idx(t, "energy")] = b["soc_min"] * b["energy_capacity_kwh"]
        lower[idx(t, "hydrogen")] = h["min_kg"]
        for name in ("battery_charge_on", "electrolyzer_on", "fuel_cell_on"):
            upper[idx(t, name)] = 1
            integer[idx(t, name)] = 1
        # Illustrative dimensionless weights per kWh; approval needed for cost comparison.
        for name, weight in (("grid", 1), ("curtailment", 0.1), ("unserved", 100)):
            objective[idx(t, name)] = dt * weight

    data, rr, cc, lows, highs = [], [], [], [], []
    def row(terms, lo=-np.inf, hi=np.inf):
        r = len(lows)
        for col, val in terms.items():
            if val:
                rr.append(r); cc.append(col); data.append(val)
        lows.append(lo); highs.append(hi)

    for t, source in enumerate(rows):
        def at(name): return idx(t, name)
        row({at("grid"): 1, at("discharge"): 1, at("fuel_cell"): 1,
             at("unserved"): 1, at("charge"): -1, at("electrolyzer"): -1,
             at("curtailment"): -1},
            source["load_kw"] - source["pv_kw"] - source["wind_kw"],
            source["load_kw"] - source["pv_kw"] - source["wind_kw"])
        # Mode gates and minimum stable on-power.
        row({at("charge"): 1, at("battery_charge_on"): -b["charge_power_max_kw"]}, hi=0)
        row({at("discharge"): 1, at("battery_charge_on"): b["discharge_power_max_kw"]},
            hi=b["discharge_power_max_kw"])
        for name, mode, power in (("electrolyzer", "electrolyzer_on", el),
                                  ("fuel_cell", "fuel_cell_on", fc)):
            row({at(name): 1, at(mode): -power["power_max_kw"]}, hi=0)
            row({at(name): 1, at(mode): -power["power_min_on_kw"]}, lo=0)
        row({at("electrolyzer_on"): 1, at("fuel_cell_on"): 1}, hi=1)
        # State recurrences and signed battery-net ramp.
        er = {at("energy"): 1, at("charge"): -dt * b["charge_efficiency"],
              at("discharge"): dt / b["discharge_efficiency"]}
        hr = {at("hydrogen"): 1, at("electrolyzer"): -dt / el["electricity_kwh_per_kg_h2"],
              at("fuel_cell"): dt / fc["electric_output_kwh_per_kg_h2"]}
        br = {at("discharge"): 1, at("charge"): -1}
        for name, terms, previous in (("energy", er, initial.battery_energy_kwh),
                                      ("hydrogen", hr, initial.hydrogen_kg)):
            if t:
                terms[idx(t - 1, name)] = -1
                row(terms, 0, 0)
            else:
                row(terms, previous, previous)
        if t:
            br[idx(t - 1, "discharge")] = -1
            br[idx(t - 1, "charge")] = 1
            center = 0
        else:
            center = initial.battery_net_power_kw
        ramp = b["net_power_ramp_kw_per_h"] * dt
        row(br, center - ramp, center + ramp)
        for name, limit, previous in (("electrolyzer", el["power_ramp_kw_per_h"] * dt,
                                       initial.electrolyzer_power_kw),
                                      ("fuel_cell", fc["power_ramp_kw_per_h"] * dt,
                                       initial.fuel_cell_power_kw)):
            terms = {at(name): 1}
            if t:
                terms[idx(t - 1, name)] = -1
                previous = 0
            row(terms, previous - limit, previous + limit)

    A = coo_matrix((data, (rr, cc)), shape=(len(lows), n * m)).tocsr()
    result = milp(objective, integrality=integer, bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(A, lows, highs),
                  options={"time_limit": time_limit, "mip_rel_gap": 0.001})
    report = {"solver_status": int(result.status), "solver_message": str(result.message),
              "objective_demo_units": None if result.fun is None else float(result.fun),
              "mip_gap": None if getattr(result, "mip_gap", None) is None else float(result.mip_gap),
              "rows": n, "time_step_h": dt,
              "objective_weights_per_kwh": {"grid": 1, "curtailment": 0.1, "unserved": 100}}
    if result.x is None or result.status != 0:
        return {"report": report, "dispatch": []}
    state = initial
    dispatch = []
    max_residual = 0.0
    for t, source in enumerate(rows):
        x = result.x
        flow = dict(zip(FLOW_FIELDS, (max(0, x[idx(t, key)]) for key in
                       ("grid", "charge", "discharge", "electrolyzer", "fuel_cell",
                        "curtailment", "unserved"))))
        next_state, residual = apply_dispatch(state, source["load_kw"], source["pv_kw"],
                                              source["wind_kw"], flow, p)
        max_residual = max(max_residual, abs(residual))
        dispatch.append({"time": source["time"], **flow,
                         "battery_energy_start_kwh": state.battery_energy_kwh,
                         "battery_energy_end_kwh": next_state.battery_energy_kwh,
                         "hydrogen_start_kg": state.hydrogen_kg,
                         "hydrogen_end_kg": next_state.hydrogen_kg})
        state = next_state
    report.update({"physical_check": "passed", "max_balance_residual_kw": max_residual,
                   "grid_import_kwh": sum(r["grid_import_kw"] * dt for r in dispatch),
                   "curtailment_kwh": sum(r["curtailment_kw"] * dt for r in dispatch),
                   "unserved_kwh": sum(r["unserved_kw"] * dt for r in dispatch),
                   "terminal_battery_kwh": state.battery_energy_kwh,
                   "terminal_hydrogen_kg": state.hydrogen_kg})
    return {"report": report, "dispatch": dispatch}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, default=Path(__file__).with_name("candidate_parameters_15min.json"))
    parser.add_argument("--day", default="2017-01-01")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=120)
    args = parser.parse_args()
    p = load_parameters(args.parameters)
    result = solve_day(read_day(args.input, args.day), p, time_limit=args.time_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
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
