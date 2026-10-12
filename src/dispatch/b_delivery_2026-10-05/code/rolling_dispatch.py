"""Prediction-driven 15-minute receding-horizon dispatch prototype.

Each UTC issue supplies 96 q10/q50/q90 net-load values. Plan with q50, execute
only step one, settle bus flows against that step's measured load/PV/wind, and
carry actual battery/hydrogen states and previous powers into the next solve.
The objective weights are demonstrative, not approved costs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import gzip
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from device_model import DeviceState, apply_dispatch, initial_state, load_parameters, ramp_down_energy_kwsteps


NAMES = ("grid", "charge", "discharge", "electrolyzer", "fuel_cell", "curtailment",
         "unserved", "energy", "hydrogen", "charge_on", "electrolyzer_on", "fuel_cell_on")
FLOW_NAMES = ("grid_import_kw", "battery_charge_kw", "battery_discharge_kw",
              "electrolyzer_kw", "fuel_cell_kw", "curtailment_kw", "unserved_kw")
RESULT_FIELDS = ("timestamp_utc", "strategy", "load_kw", "pv_kw", "wind_kw",
                 *FLOW_NAMES, "battery_energy_start_kwh", "battery_energy_end_kwh",
                 "hydrogen_start_kg", "hydrogen_end_kg", "solver_status", "step_cost")


def utc_stamp(value: str) -> datetime:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
        raise ValueError(f"Expected UTC timestamp: {value}")
    if stamp.minute % 15 or stamp.second or stamp.microsecond:
        raise ValueError(f"Timestamp is not on a 15-minute boundary: {value}")
    return stamp.astimezone(timezone.utc)


def iter_forecasts(path: Path, start: datetime, steps: int):
    """Yield complete 96-row windows in order; memory stays constant for 2019."""
    end = start + timedelta(minutes=15 * steps)
    expected_issue = start
    window = []
    count = 0
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = ["issue_time", "target_time", "horizon_step", "horizon_hours",
                  "q10_kw", "q50_kw", "q90_kw"]
        if reader.fieldnames != fields:
            raise ValueError("Forecast columns/order do not match FORECAST_15MIN.md")
        for row in reader:
            issue = utc_stamp(row["issue_time"])
            if issue < start:
                continue
            if issue >= end:
                break
            if issue != expected_issue:
                if len(window) != 96 or issue != expected_issue + timedelta(minutes=15):
                    raise ValueError(f"Missing or incomplete forecast issue at {expected_issue}")
                yield window
                count += 1
                expected_issue = issue
                window = []
            horizon = int(row["horizon_step"])
            if horizon != len(window) + 1 or not 1 <= horizon <= 96:
                raise ValueError(f"Missing/duplicate/unsorted horizon at {issue}: {horizon}")
            if utc_stamp(row["target_time"]) != issue + timedelta(minutes=15 * (horizon - 1)):
                raise ValueError(f"Target time mismatch at {issue}, horizon {horizon}")
            if abs(float(row["horizon_hours"]) - horizon * 0.25) > 1e-6:
                raise ValueError(f"Horizon hours mismatch at {issue}, horizon {horizon}")
            values = [float(row[k]) for k in ("q10_kw", "q50_kw", "q90_kw")]
            if not all(math.isfinite(v) for v in values) or not values[0] <= values[1] <= values[2]:
                raise ValueError(f"Invalid quantiles at {issue}, horizon {horizon}")
            window.append(dict(zip(("q10_kw", "q50_kw", "q90_kw"), values)))
    if len(window) != 96:
        raise ValueError(f"Expected all 96 horizons for final issue {expected_issue}")
    yield window
    count += 1
    if count != steps:
        raise ValueError(f"Expected {steps} forecast issues, got {count}")


def read_forecasts(path: Path, start: datetime, steps: int) -> list[list[dict]]:
    """Materialize a short fixture; use iter_forecasts for long rolling runs."""
    return list(iter_forecasts(path, start, steps))


def read_actual(path: Path, start: datetime, steps: int) -> list[dict]:
    expected = {start + timedelta(minutes=15 * t): t for t in range(steps)}
    actual = [None] * steps
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        needed = {"time", "load_kw", "pv_kw", "wind_kw"}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError(f"Actual data is missing columns: {sorted(needed - set(reader.fieldnames or []))}")
        for row in reader:
            stamp = utc_stamp(row["time"])
            if stamp not in expected:
                continue
            t = expected[stamp]
            if actual[t] is not None:
                raise ValueError(f"Duplicate actual timestamp: {stamp}")
            values = {k: float(row[k]) for k in ("load_kw", "pv_kw", "wind_kw")}
            if not all(math.isfinite(v) and v >= 0 for v in values.values()):
                raise ValueError(f"Invalid actual power at {stamp}")
            actual[t] = values
    if any(row is None for row in actual):
        raise ValueError("Actual data is missing one or more execution timestamps")
    return actual


def _safe_max(rated: float, permitted) -> float:
    if permitted(rated):
        return rated
    low, high = 0.0, rated
    for _ in range(60):
        mid = (low + high) / 2
        if permitted(mid):
            low = mid
        else:
            high = mid
    return low


def safe_first_limits(state: DeviceState, p: dict) -> dict[str, float]:
    """Reserve enough inventory to ramp each device to zero after this step."""
    dt = p["meta"]["time_step_h"]
    b, h, el, fc = (p[k] for k in ("battery", "hydrogen_tank", "electrolyzer", "fuel_cell"))
    rb = b["net_power_ramp_kw_per_h"] * dt
    re = el["power_ramp_kw_per_h"] * dt
    rf = fc["power_ramp_kw_per_h"] * dt
    return {
        "charge": _safe_max(b["charge_power_max_kw"], lambda v:
            (v + ramp_down_energy_kwsteps(v, rb)) * dt * b["charge_efficiency"]
            <= b["soc_max"] * b["energy_capacity_kwh"] - state.battery_energy_kwh - 1e-4),
        "discharge": _safe_max(b["discharge_power_max_kw"], lambda v:
            (v + ramp_down_energy_kwsteps(v, rb)) * dt / b["discharge_efficiency"]
            <= state.battery_energy_kwh - b["soc_min"] * b["energy_capacity_kwh"] - 1e-4),
        "electrolyzer": _safe_max(el["power_max_kw"], lambda v:
            (v + ramp_down_energy_kwsteps(v, re, el["power_min_on_kw"])) * dt
            / el["electricity_kwh_per_kg_h2"] <= h["max_kg"] - state.hydrogen_kg - 1e-4),
        "fuel_cell": _safe_max(fc["power_max_kw"], lambda v:
            (v + ramp_down_energy_kwsteps(v, rf, fc["power_min_on_kw"])) * dt
            / fc["electric_output_kwh_per_kg_h2"] <= state.hydrogen_kg - h["min_kg"] - 1e-4),
    }


def solve_window(net_kw: list[float], p: dict, state: DeviceState,
                 *, customer_load_kw: float | None = None, target: dict | None = None,
                 time_limit: float = 15.0, reserve_kwh=None, weights=None, mip_gap=0.001,
                 return_plan=False):
    """MILP for 96 forecast net loads, or one actual interval for recourse."""
    n = len(net_kw)
    dt = p["meta"]["time_step_h"]
    if abs(dt - 0.25) > 1e-12 or n not in (1, 96):
        raise ValueError("Expected one actual step or 96 forecast steps at 0.25 h")
    b, h, el, fc, grid = (p[k] for k in ("battery", "hydrogen_tank", "electrolyzer", "fuel_cell", "grid"))
    m = len(NAMES)
    extra = 4 if target is not None else 0
    reserve_kwh = [0.0] * n if reserve_kwh is None else reserve_kwh
    if len(reserve_kwh) != n or any(not math.isfinite(v) or v < 0 for v in reserve_kwh):
        raise ValueError('Invalid reserve vector')
    weights = weights or {"grid": 1, "curtailment": 0.1, "unserved": 100,
                          "throughput": 0, "reserve_shortfall": 20}
    if any(not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError('Invalid objective weights')
    size = n * m + extra + n
    at = lambda t, name: t * m + NAMES.index(name)
    lower = np.zeros(size)
    upper = np.full(size, np.inf)
    integrality = np.zeros(size)
    objective = np.zeros(size)
    for t, net in enumerate(net_kw):
        if not math.isfinite(net):
            raise ValueError("Nonfinite net-load prediction")
        caps = {"grid": grid["import_power_max_kw"], "charge": b["charge_power_max_kw"],
                "discharge": b["discharge_power_max_kw"], "electrolyzer": el["power_max_kw"],
                "fuel_cell": fc["power_max_kw"],
                "unserved": max(0.0, net) if customer_load_kw is None else customer_load_kw,
                "energy": b["soc_max"] * b["energy_capacity_kwh"], "hydrogen": h["max_kg"]}
        for name, cap in caps.items():
            upper[at(t, name)] = cap
        lower[at(t, "energy")] = b["soc_min"] * b["energy_capacity_kwh"]
        lower[at(t, "hydrogen")] = h["min_kg"]
        for name in ("charge_on", "electrolyzer_on", "fuel_cell_on"):
            upper[at(t, name)] = 1
            integrality[at(t, name)] = 1
        for name, weight in (("grid", weights['grid']), ("curtailment", weights['curtailment']),
                             ("unserved", 10000 if target is not None else weights['unserved'])):
            objective[at(t, name)] = weight * dt
        for name in ('charge', 'discharge', 'electrolyzer', 'fuel_cell'):
            objective[at(t, name)] = weights.get('throughput', 0) * dt
        reserve_slack = n * m + extra + t
        upper[reserve_slack] = reserve_kwh[t]
        objective[reserve_slack] = weights.get('reserve_shortfall', 20)
    if target is not None:
        for j in range(4):
            objective[n * m + j] = 10 * dt

    values, r_idx, c_idx, lows, highs = [], [], [], [], []
    def row(terms, lo=-np.inf, hi=np.inf):
        r = len(lows)
        for col, val in terms.items():
            if val:
                r_idx.append(r); c_idx.append(col); values.append(val)
        lows.append(lo); highs.append(hi)

    first_caps = safe_first_limits(state, p)
    for name, cap in first_caps.items():
        upper[at(0, name)] = min(upper[at(0, name)], cap)
    rb = b["net_power_ramp_kw_per_h"] * dt
    re = el["power_ramp_kw_per_h"] * dt
    rf = fc["power_ramp_kw_per_h"] * dt
    for t, net in enumerate(net_kw):
        a = lambda name: at(t, name)
        row({a("grid"): 1, a("discharge"): 1, a("fuel_cell"): 1,
             a("unserved"): 1, a("charge"): -1, a("electrolyzer"): -1,
             a("curtailment"): -1}, net, net)
        row({a("charge"): 1, a("charge_on"): -b["charge_power_max_kw"]}, hi=0)
        row({a("discharge"): 1, a("charge_on"): b["discharge_power_max_kw"]},
            hi=b["discharge_power_max_kw"])
        for name, mode, equipment in (("electrolyzer", "electrolyzer_on", el),
                                      ("fuel_cell", "fuel_cell_on", fc)):
            row({a(name): 1, a(mode): -equipment["power_max_kw"]}, hi=0)
            row({a(name): 1, a(mode): -equipment["power_min_on_kw"]}, lo=0)
        row({a("electrolyzer_on"): 1, a("fuel_cell_on"): 1}, hi=1)
        # If wind/PV and customer load vanish next step, mandatory charging is
        # still coverable by the 500 kW import limit after one ramp-down step.
        row({a("charge"): 1, a("electrolyzer"): 1},
            hi=grid["import_power_max_kw"] + rb + re)
        energy = {a("energy"): 1, a("charge"): -dt * b["charge_efficiency"],
                  a("discharge"): dt / b["discharge_efficiency"]}
        hydrogen = {a("hydrogen"): 1, a("electrolyzer"): -dt / el["electricity_kwh_per_kg_h2"],
                    a("fuel_cell"): dt / fc["electric_output_kwh_per_kg_h2"]}
        if t:
            energy[at(t - 1, "energy")] = -1
            hydrogen[at(t - 1, "hydrogen")] = -1
            row(energy, 0, 0); row(hydrogen, 0, 0)
        else:
            row(energy, state.battery_energy_kwh, state.battery_energy_kwh)
            row(hydrogen, state.hydrogen_kg, state.hydrogen_kg)
        row({a('energy'): b['discharge_efficiency'],
             a('hydrogen'): fc['electric_output_kwh_per_kg_h2'],
             n * m + extra + t: 1},
            lo=reserve_kwh[t] + b['discharge_efficiency'] * b['soc_min'] * b['energy_capacity_kwh']
               + fc['electric_output_kwh_per_kg_h2'] * h['min_kg'])
        battery_ramp = {a("discharge"): 1, a("charge"): -1}
        for name, limit, previous in (("electrolyzer", re, state.electrolyzer_power_kw),
                                      ("fuel_cell", rf, state.fuel_cell_power_kw)):
            terms = {a(name): 1}
            if t:
                terms[at(t - 1, name)] = -1
                previous = 0
            row(terms, previous - limit, previous + limit)
        if t:
            battery_ramp[at(t - 1, "discharge")] = -1
            battery_ramp[at(t - 1, "charge")] = 1
            center = 0
        else:
            center = state.battery_net_power_kw
        row(battery_ramp, center - rb, center + rb)
    if n == 96:
        # A conservative finite-horizon safeguard, pending team agreement on
        # economic terminal inventory values and a validated terminal set.
        for name in ("charge", "discharge", "electrolyzer", "fuel_cell"):
            upper[at(n - 1, name)] = 0
    if target is not None:
        for j, name in enumerate(("charge", "discharge", "electrolyzer", "fuel_cell")):
            deviation = n * m + j
            goal = target[name]
            row({at(0, name): 1, deviation: -1}, hi=goal)
            row({at(0, name): -1, deviation: -1}, hi=-goal)
    A = coo_matrix((values, (r_idx, c_idx)), shape=(len(lows), size)).tocsr()
    result = milp(objective, integrality=integrality, bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(A, lows, highs),
                  options={"time_limit": time_limit, "mip_rel_gap": mip_gap})
    if result.status not in (0, 1) or result.x is None:
        raise RuntimeError(f"MILP status {result.status}: {result.message}; incumbent={result.x is not None}")
    # A time-limited feasible incumbent is usable for integration checks, but
    # must be reported as such and never described as an optimal solution.
    x = result.x
    lhs = A @ x
    if (np.any(x < lower - 1e-5) or np.any(x > upper + 1e-5)
            or np.any(lhs < np.asarray(lows) - 1e-5)
            or np.any(lhs > np.asarray(highs) + 1e-5)
            or np.any(abs(x[integrality == 1] - np.rint(x[integrality == 1])) > 1e-5)):
        raise RuntimeError("Solver incumbent violates a model bound or constraint")
    action = {name: max(0.0, float(result.x[at(0, name)])) for name in
              ("grid", "charge", "discharge", "electrolyzer", "fuel_cell", "curtailment", "unserved")}
    info = {"status": "optimal" if result.status == 0 else "time_limit_feasible",
                    "objective_demo_units": float(result.fun),
                    "mip_gap": float(result.mip_gap),
                    "reserve_shortfall_kwh": float(result.x[n * m + extra])}
    if return_plan:
        info['plan']=[{name:max(0.0,float(result.x[at(t,name)])) for name in
                     ('grid','charge','discharge','electrolyzer','fuel_cell','curtailment','unserved')}
                     for t in range(n)]
    return action, info


def actual_flow(action: dict, actual: dict, p: dict) -> dict:
    """Keep planned device setpoints; settle import, spill and customer shortfall."""
    residual = (actual["load_kw"] - actual["pv_kw"] - actual["wind_kw"]
                + action["charge"] + action["electrolyzer"]
                - action["discharge"] - action["fuel_cell"])
    grid = min(p["grid"]["import_power_max_kw"], max(0.0, residual))
    unserved = max(0.0, residual - grid)
    if unserved > actual["load_kw"] + 1e-7:
        raise ValueError("Planned device load would be mislabeled as customer shortfall")
    return dict(zip(FLOW_NAMES, (grid, action["charge"], action["discharge"],
                                 action["electrolyzer"], action["fuel_cell"],
                                 max(0.0, -residual), unserved)))


def run(forecasts: list[list[dict]], actuals: list[dict], p: dict, start: datetime,
        *, time_limit: float = 15.0):
    state = initial_state(p)
    results, log = [], []
    for t, (window, actual) in enumerate(zip(forecasts, actuals)):
        stamp = start + timedelta(minutes=15 * t)
        try:
            planned, solver = solve_window([r["q50_kw"] for r in window], p, state,
                                           time_limit=time_limit)
        except RuntimeError as exc:
            raise RuntimeError(f"Planning failed at {stamp.isoformat()}: {exc}") from exc
        prior = state
        corrected = False
        try:
            flow = actual_flow(planned, actual, p)
            state, residual = apply_dispatch(prior, **actual, flow=flow, p=p)
        except ValueError:
            # Actual net load can differ sharply from the forecast. Correct
            # only the first action with measured power, preserving the plan
            # where physically possible; never count device demand as customers.
            corrected = True
            executed, _ = solve_window([actual["load_kw"] - actual["pv_kw"] - actual["wind_kw"]],
                                       p, prior, customer_load_kw=actual["load_kw"],
                                       target=planned, time_limit=time_limit)
            flow = dict(zip(FLOW_NAMES, (executed[k] for k in
                       ("grid", "charge", "discharge", "electrolyzer", "fuel_cell",
                        "curtailment", "unserved"))))
            state, residual = apply_dispatch(prior, **actual, flow=flow, p=p)
        results.append({"timestamp_utc": stamp.isoformat(sep=" "),
                        "strategy": "q50_rolling_milp_15min_demo",
                        **actual, **flow,
                        "battery_energy_start_kwh": prior.battery_energy_kwh,
                        "battery_energy_end_kwh": state.battery_energy_kwh,
                        "hydrogen_start_kg": prior.hydrogen_kg,
                        "hydrogen_end_kg": state.hydrogen_kg,
                        "solver_status": solver["status"] + ("_corrected" if corrected else ""),
                        "step_cost": ""})
        log.append({"issue_time": stamp.isoformat(sep=" "),
                    "q10_first_kw": window[0]["q10_kw"],
                    "q50_first_kw": window[0]["q50_kw"],
                    "q90_first_kw": window[0]["q90_kw"],
                    "actual_net_kw": actual["load_kw"] - actual["pv_kw"] - actual["wind_kw"],
                    "plan_objective_demo_units": solver["objective_demo_units"],
                    "plan_solver_status": solver["status"],
                    "plan_mip_gap": solver["mip_gap"], "actual_correction": corrected,
                    "balance_residual_kw": residual})
        if (t + 1) % 24 == 0:
            print(f"Validated {t + 1} executed steps through {stamp.isoformat()}", flush=True)
    if len(results) != len(actuals):
        raise ValueError(f"Forecast ended after {len(results)} of {len(actuals)} execution steps")
    dt = p["meta"]["time_step_h"]
    report = {"steps": len(results), "time_step_h": dt, "planning_horizon_steps": 96,
              "forecast_used": "q50 net load", "actual_corrections": sum(x["actual_correction"] for x in log),
              "time_limit_feasible_steps": sum(x["plan_solver_status"] == "time_limit_feasible" for x in log),
              "max_mip_gap": max(x["plan_mip_gap"] for x in log),
              "max_balance_residual_kw": max(abs(x["balance_residual_kw"]) for x in log),
              "grid_import_kwh": sum(x["grid_import_kw"] * dt for x in results),
              "curtailment_kwh": sum(x["curtailment_kw"] * dt for x in results),
              "unserved_kwh": sum(x["unserved_kw"] * dt for x in results),
              "terminal_battery_kwh": state.battery_energy_kwh,
              "terminal_hydrogen_kg": state.hydrogen_kg,
              "objective_is_approved_cost": False}
    return results, log, report


def write_csv(path: Path, rows: list[dict], fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--parameters", type=Path,
                        default=Path(__file__).with_name("candidate_parameters_15min.json"))
    parser.add_argument("--start", required=True, help="UTC issue time, e.g. 2019-06-15T00:00:00Z")
    parser.add_argument("--steps", type=int, default=192)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=15.0)
    args = parser.parse_args()
    if args.steps < 1 or args.time_limit <= 0:
        parser.error("steps and time-limit must be positive")
    start = utc_stamp(args.start)
    p = load_parameters(args.parameters)
    windows = iter_forecasts(args.forecast, start, args.steps)
    actual = read_actual(args.actual, start, args.steps)
    rows, log, report = run(windows, actual, p, start, time_limit=args.time_limit)
    write_csv(args.output, rows, RESULT_FIELDS)
    write_csv(args.output.with_name(args.output.stem + "_step_log.csv"), log, log[0].keys())
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
