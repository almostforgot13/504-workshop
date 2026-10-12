"""15-minute fixed-rule dispatch with ramp-aware stopping and explicit shortfall.

No forecast is used. All four comparison strategies should reuse the same input
timestamps and equipment configuration. Cost is intentionally not implemented.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import csv
import json
import math
from pathlib import Path
from typing import Any

from device_model import (
    DeviceState, TOL, apply_dispatch, finite_number, initial_state,
    load_parameters, ramp_down_energy_kwsteps,
)


RESULT_FIELDS = (
    "timestamp_utc", "strategy", "load_kw", "pv_kw", "wind_kw",
    "grid_import_kw", "battery_charge_kw", "battery_discharge_kw",
    "electrolyzer_kw", "fuel_cell_kw", "curtailment_kw", "unserved_kw",
    "battery_energy_start_kwh", "battery_energy_end_kwh",
    "hydrogen_start_kg", "hydrogen_end_kg", "solver_status", "step_cost",
)
FLOW_FIELDS = (
    "grid_import_kw", "battery_charge_kw", "battery_discharge_kw",
    "electrolyzer_kw", "fuel_cell_kw", "curtailment_kw", "unserved_kw",
)


class InfeasibleDispatchError(ValueError):
    """The fixed rule cannot meet a hard physical limit for the current state."""

    def __init__(self, reason: str, details: dict[str, Any]):
        super().__init__(reason)
        self.details = details


def _forced_electro_next(power_kw: float, p: dict[str, Any]) -> float:
    """Smallest legal electrolyzer load at the following 15-minute step."""
    el = p["electrolyzer"]
    ramp_step = el["power_ramp_kw_per_h"] * p["meta"]["time_step_h"]
    if power_kw <= ramp_step + TOL:
        return 0.0
    return max(el["power_min_on_kw"], power_kw - ramp_step)


def _forced_battery_charge_next(battery_net_kw: float, p: dict[str, Any]) -> float:
    """Charge load that cannot be removed in one further ramp-limited step."""
    ramp_step = p["battery"]["net_power_ramp_kw_per_h"] * p["meta"]["time_step_h"]
    return max(0.0, -battery_net_kw - ramp_step)


def _largest_safe_power(rated_kw: float, feasible) -> float:
    """Bisection for an inventory-safe current setpoint and future ramp-down."""
    if rated_kw <= TOL:
        return 0.0
    if feasible(rated_kw):
        return rated_kw
    low, high = 0.0, rated_kw
    for _ in range(65):
        middle = (low + high) / 2
        if feasible(middle):
            low = middle
        else:
            high = middle
    return low


def _battery_interval(state: DeviceState, p: dict[str, Any]) -> tuple[float, float]:
    b = p["battery"]
    dt = p["meta"]["time_step_h"]
    ramp_step = b["net_power_ramp_kw_per_h"] * dt
    energy_min = b["soc_min"] * b["energy_capacity_kwh"]
    energy_max = b["soc_max"] * b["energy_capacity_kwh"]

    def safe_discharge(power: float) -> bool:
        current_use = power * dt / b["discharge_efficiency"]
        braking_use = ramp_down_energy_kwsteps(power, ramp_step) * dt / b["discharge_efficiency"]
        return current_use + braking_use <= state.battery_energy_kwh - energy_min

    def safe_charge(power: float) -> bool:
        current_gain = power * dt * b["charge_efficiency"]
        braking_gain = ramp_down_energy_kwsteps(power, ramp_step) * dt * b["charge_efficiency"]
        return current_gain + braking_gain <= energy_max - state.battery_energy_kwh

    safe_discharge_max = _largest_safe_power(b["discharge_power_max_kw"], safe_discharge)
    safe_charge_max = _largest_safe_power(b["charge_power_max_kw"], safe_charge)
    low = max(-safe_charge_max, state.battery_net_power_kw - ramp_step)
    high = min(safe_discharge_max, state.battery_net_power_kw + ramp_step)
    if low > high + TOL:
        raise ValueError("Battery state has no ramp- and SOC-feasible action")
    return low, high


def _on_interval(previous_kw: float, safe_max_kw: float, min_on_kw: float, ramp_step_kw: float, name: str) -> tuple[float, float]:
    """Return mandatory minimum and allowed maximum, respecting off/on gap."""
    if previous_kw <= ramp_step_kw + TOL:
        maximum = min(safe_max_kw, previous_kw + ramp_step_kw)
        maximum = maximum if maximum >= min_on_kw - TOL else 0.0
        return 0.0, maximum
    minimum = max(min_on_kw, previous_kw - ramp_step_kw)
    maximum = min(safe_max_kw, previous_kw + ramp_step_kw)
    if minimum > maximum + TOL:
        raise ValueError(f"{name} cannot ramp down before reaching inventory bound")
    return minimum, maximum


def _hydrogen_intervals(state: DeviceState, p: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    h, el, fc = p["hydrogen_tank"], p["electrolyzer"], p["fuel_cell"]
    dt = p["meta"]["time_step_h"]
    el_ramp_step = el["power_ramp_kw_per_h"] * dt
    fc_ramp_step = fc["power_ramp_kw_per_h"] * dt

    def safe_electro(power: float) -> bool:
        future = ramp_down_energy_kwsteps(power, el_ramp_step, el["power_min_on_kw"])
        production_kg = (power + future) * dt / el["electricity_kwh_per_kg_h2"]
        return production_kg <= h["max_kg"] - state.hydrogen_kg

    def safe_fuel(power: float) -> bool:
        future = ramp_down_energy_kwsteps(power, fc_ramp_step, fc["power_min_on_kw"])
        use_kg = (power + future) * dt / fc["electric_output_kwh_per_kg_h2"]
        return use_kg <= state.hydrogen_kg - h["min_kg"]

    safe_el = _largest_safe_power(el["power_max_kw"], safe_electro)
    safe_fc = _largest_safe_power(fc["power_max_kw"], safe_fuel)
    e_range = _on_interval(state.electrolyzer_power_kw, safe_el, el["power_min_on_kw"], el_ramp_step, "electrolyzer")
    f_range = _on_interval(state.fuel_cell_power_kw, safe_fc, fc["power_min_on_kw"], fc_ramp_step, "fuel cell")
    if e_range[0] > TOL and f_range[0] > TOL:
        raise ValueError("Electrolyzer and fuel cell are both required to remain on")
    return e_range, f_range


def fixed_rule(state: DeviceState, load_kw: float, pv_kw: float, wind_kw: float, p: dict[str, Any]) -> dict[str, float]:
    """Respect mandatory ramp-down, then use battery, hydrogen, grid, slack."""
    for name, value in (("load_kw", load_kw), ("pv_kw", pv_kw), ("wind_kw", wind_kw)):
        finite_number(value, name)
    b_min, b_max = _battery_interval(state, p)
    (e_min, e_max), (f_min, f_max) = _hydrogen_intervals(state, p)
    b_net = min(max(0.0, b_min), b_max)  # closest feasible signed battery power to zero
    electro, fuel = e_min, f_min
    residual = load_kw - pv_kw - wind_kw + electro - fuel - b_net

    flow = {name: 0.0 for name in FLOW_FIELDS}
    if residual > TOL:
        increase = min(residual, max(0.0, b_max - b_net))
        b_net += increase
        residual -= increase
        if electro <= TOL:
            fuel_room = max(0.0, f_max - fuel)
            candidate = min(residual, fuel_room)
            if fuel > TOL or candidate >= p["fuel_cell"]["power_min_on_kw"] - TOL:
                fuel += candidate
                residual -= candidate
        flow["grid_import_kw"] = min(max(0.0, residual), p["grid"]["import_power_max_kw"])
        residual -= flow["grid_import_kw"]
        flow["unserved_kw"] = max(0.0, residual)
    elif residual < -TOL:
        # Preserve enough *next-step* grid headroom to ramp mandatory flexible
        # loads down if renewable output and customer load both suddenly vanish.
        # This is a physical feasibility guard, not a capacity retune.
        grid_limit = p["grid"]["import_power_max_kw"]
        battery_ramp_step = p["battery"]["net_power_ramp_kw_per_h"] * p["meta"]["time_step_h"]
        electro_ramp_step = p["electrolyzer"]["power_ramp_kw_per_h"] * p["meta"]["time_step_h"]
        mandatory_electro_next = _forced_electro_next(e_min, p)
        headroom_after_electro = grid_limit - mandatory_electro_next
        if headroom_after_electro < -TOL:
            raise InfeasibleDispatchError("Mandatory electrolyzer ramp-down exceeds grid import limit", {
                "mandatory_electrolyzer_next_kw": mandatory_electro_next,
                "grid_import_limit_kw": grid_limit,
            })
        viable_b_min = max(b_min, -battery_ramp_step - max(0.0, headroom_after_electro))
        if viable_b_min > b_net + TOL:
            raise InfeasibleDispatchError("Battery cannot reduce charge to preserve next-step grid headroom", {
                "battery_net_power_kw": b_net,
                "battery_viable_min_kw": viable_b_min,
                "grid_import_limit_kw": grid_limit,
            })
        decrease = min(-residual, max(0.0, b_net - viable_b_min))
        b_net -= decrease
        residual += decrease
        if fuel <= TOL:
            remaining_grid_headroom = grid_limit - _forced_battery_charge_next(b_net, p)
            if remaining_grid_headroom < p["electrolyzer"]["power_min_on_kw"] - TOL:
                viable_e_max = min(e_max, electro_ramp_step)
            else:
                viable_e_max = min(e_max, electro_ramp_step + remaining_grid_headroom)
            if viable_e_max < electro - TOL:
                raise InfeasibleDispatchError("Mandatory electrolyzer load leaves no feasible next-step grid headroom", {
                    "electrolyzer_min_kw": electro,
                    "electrolyzer_viable_max_kw": viable_e_max,
                    "grid_import_limit_kw": grid_limit,
                })
            electro_room = max(0.0, viable_e_max - electro)
            candidate = min(-residual, electro_room)
            if electro > TOL or candidate >= p["electrolyzer"]["power_min_on_kw"] - TOL:
                electro += candidate
                residual += candidate
        flow["curtailment_kw"] = max(0.0, -residual)
    flow["battery_charge_kw"] = max(0.0, -b_net)
    flow["battery_discharge_kw"] = max(0.0, b_net)
    flow["electrolyzer_kw"] = electro
    flow["fuel_cell_kw"] = fuel
    if flow["unserved_kw"] > load_kw + TOL:
        raise InfeasibleDispatchError("Mandatory device demand exceeds supply even after full grid import", {
            "customer_load_kw": load_kw,
            "unserved_slack_required_kw": flow["unserved_kw"],
            "grid_import_limit_kw": p["grid"]["import_power_max_kw"],
            "mandatory_battery_charge_kw": flow["battery_charge_kw"],
            "mandatory_electrolyzer_kw": flow["electrolyzer_kw"],
        })
    return flow


def read_15min_week(path: Path, week_start: datetime) -> tuple[list[dict[str, Any]], int]:
    """Validate the *entire* source CSV and select a complete UTC week."""
    end = week_start + timedelta(days=7)
    selected: list[dict[str, Any]] = []
    previous: datetime | None = None
    seen: set[datetime] = set()
    flagged = 0
    required = ("time", "split", "load_kw", "pv_kw", "wind_kw")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for name in required:
            if name not in (reader.fieldnames or []):
                raise ValueError(f"Input CSV missing required column {name}")
        for line, row in enumerate(reader, 2):
            if any(row.get(name) in (None, "") for name in required):
                raise ValueError(f"Missing required input on CSV line {line}")
            try:
                timestamp = datetime.fromisoformat(row["time"])
            except ValueError as exc:
                raise ValueError(f"Invalid timestamp on CSV line {line}") from exc
            if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
                raise ValueError(f"Timestamp on line {line} must have a UTC offset")
            timestamp = timestamp.astimezone(timezone.utc)
            if timestamp.minute % 15 or timestamp.second or timestamp.microsecond:
                raise ValueError(f"Timestamp on line {line} is not aligned to a 15-minute boundary")
            if timestamp in seen:
                raise ValueError(f"Duplicate timestamp on CSV line {line}")
            if previous is not None and timestamp - previous != timedelta(minutes=15):
                raise ValueError(f"CSV timestamps are not strictly continuous at line {line}")
            seen.add(timestamp)
            previous = timestamp
            values = {}
            for name in ("load_kw", "pv_kw", "wind_kw"):
                try:
                    number = float(row[name])
                except ValueError as exc:
                    raise ValueError(f"Invalid {name} on CSV line {line}") from exc
                values[name] = finite_number(number, f"line {line} {name}")
            if "net_load_kw" in row:
                if row["net_load_kw"] in (None, ""):
                    raise ValueError(f"Missing net_load_kw on CSV line {line}")
                try:
                    supplied_net = float(row["net_load_kw"])
                except ValueError as exc:
                    raise ValueError(f"Invalid net_load_kw on CSV line {line}") from exc
                finite_number(supplied_net, f"line {line} net_load_kw", allow_negative=True)
                if abs(supplied_net - (values["load_kw"] - values["pv_kw"] - values["wind_kw"])) > 1e-3:
                    raise ValueError(f"net_load_kw identity fails on CSV line {line}")
            if week_start <= timestamp < end:
                selected.append({"timestamp_utc": timestamp.isoformat(), "split": row["split"], **values})
                if row.get("quality_flag", "source_profile") != "source_profile":
                    flagged += 1
    if len(selected) != 672:
        raise ValueError(f"Selected week has {len(selected)} rows; expected 672")
    if datetime.fromisoformat(selected[0]["timestamp_utc"]) != week_start or datetime.fromisoformat(selected[-1]["timestamp_utc"]) != end - timedelta(minutes=15):
        raise ValueError("Selected week does not cover the requested start/end")
    if len({row["split"] for row in selected}) != 1:
        raise ValueError("Selected week crosses train/validation/test boundaries")
    return selected, flagged


def run_week(rows: list[dict[str, Any]], p: dict[str, Any], flagged_steps: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state = initial_state(p)
    dt = p["meta"]["time_step_h"]
    results = []
    for source in rows:
        before = state
        try:
            flow = fixed_rule(before, source["load_kw"], source["pv_kw"], source["wind_kw"], p)
        except InfeasibleDispatchError as exc:
            exc.details["timestamp_utc"] = source["timestamp_utc"]
            exc.details["battery_energy_start_kwh"] = before.battery_energy_kwh
            exc.details["hydrogen_start_kg"] = before.hydrogen_kg
            raise
        state, residual = apply_dispatch(before, source["load_kw"], source["pv_kw"], source["wind_kw"], flow, p)
        status = "rule_unserved" if flow["unserved_kw"] > TOL else "rule_curtailment" if flow["curtailment_kw"] > TOL else "rule_ok"
        results.append({
            "timestamp_utc": source["timestamp_utc"],
            "strategy": "fixed_rule_battery_hydrogen_grid_15min",
            "load_kw": source["load_kw"],
            "pv_kw": source["pv_kw"],
            "wind_kw": source["wind_kw"],
            **flow,
            "battery_energy_start_kwh": before.battery_energy_kwh,
            "battery_energy_end_kwh": state.battery_energy_kwh,
            "hydrogen_start_kg": before.hydrogen_kg,
            "hydrogen_end_kg": state.hydrogen_kg,
            "solver_status": status,
            "step_cost": "",  # Cost definition is not agreed; do not report a numeric zero.
        })
        if abs(residual) > 1e-5:
            raise AssertionError("Device model returned nonzero power-balance residual")
    if len(results) != 672:
        raise ValueError("A seven-day 15-minute week must produce exactly 672 rows")
    energy_cols = (
        "load_kw", "pv_kw", "wind_kw", "grid_import_kw", "battery_charge_kw",
        "battery_discharge_kw", "electrolyzer_kw", "fuel_cell_kw",
        "curtailment_kw", "unserved_kw",
    )
    energy = {name.replace("_kw", "_kwh"): sum(row[name] for row in results) * dt for name in energy_cols}
    unserved_steps = sum(row["unserved_kw"] > TOL for row in results)
    curtailment_steps = sum(row["curtailment_kw"] > TOL for row in results)
    grid_steps = sum(row["grid_import_kw"] > TOL for row in results)
    summary = {
        "start_utc": results[0]["timestamp_utc"],
        "end_exclusive_utc": (datetime.fromisoformat(results[-1]["timestamp_utc"]) + timedelta(minutes=15)).isoformat(),
        "source_split": rows[0]["split"],
        "steps": len(results),
        "step_h": dt,
        "duration_h": len(results) * dt,
        "flagged_input_steps": flagged_steps,
        "energy_kwh": energy,
        "unserved_steps": unserved_steps,
        "unserved_duration_h": unserved_steps * dt,
        "curtailment_steps": curtailment_steps,
        "curtailment_duration_h": curtailment_steps * dt,
        "grid_import_steps": grid_steps,
        "grid_import_duration_h": grid_steps * dt,
        "max_unserved_kw": max(row["unserved_kw"] for row in results),
        "max_curtailment_kw": max(row["curtailment_kw"] for row in results),
        "battery_final_energy_kwh": state.battery_energy_kwh,
        "hydrogen_final_kg": state.hydrogen_kg,
        "solver_status_counts": {status: sum(row["solver_status"] == status for row in results) for status in sorted({r["solver_status"] for r in results})},
        "cost_status": "not_implemented_no_agreed_price_or_degradation_cost",
        "total_cost": None,
        "forecast_used": False,
        "note": "Unserved and curtailment are explicit; all energy totals equal sum(interval average kW * 0.25 h).",
    }
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run seven days of ramp-aware 15-minute fixed-rule dispatch")
    parser.add_argument("--input", required=True, type=Path, help="A's original 15-minute virtual CSV")
    parser.add_argument("--params", type=Path, default=Path(__file__).with_name("candidate_parameters_15min.json"))
    parser.add_argument("--week-start", default="2017-01-01", help="UTC date YYYY-MM-DD")
    parser.add_argument("--output", required=True, type=Path, help="672-row results CSV")
    parser.add_argument("--summary", required=True, type=Path, help="Metrics JSON")
    parser.add_argument("--failure-report", type=Path, help="Where to save an infeasible-state report if dispatch cannot continue")
    args = parser.parse_args()
    try:
        start = datetime.strptime(args.week_start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError("--week-start must be YYYY-MM-DD") from exc
    p = load_parameters(args.params)
    if not math.isclose(p["meta"]["time_step_h"], 0.25, rel_tol=0, abs_tol=1e-12):
        raise ValueError("This 15-minute input runner requires time_step_h=0.25")
    source, flagged = read_15min_week(args.input, start)
    try:
        results, summary = run_week(source, p, flagged)
    except InfeasibleDispatchError as exc:
        failure_path = args.failure_report or args.output.with_suffix(".infeasible.json")
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        report = {"status": "infeasible_hard_constraints", "reason": str(exc), "details": exc.details}
        failure_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(2) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(results)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
