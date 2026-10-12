"""Validated 15-minute battery and hydrogen inventory accounting.

Bus power: kW. Battery energy: kWh. Hydrogen inventory: kg. Each flow is a
nonnegative *average power over the interval*. Unserved load is an explicit
balance slack, never generated power.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


TOL = 1e-7
FLOW_FIELDS = (
    "grid_import_kw", "battery_charge_kw", "battery_discharge_kw",
    "electrolyzer_kw", "fuel_cell_kw", "curtailment_kw", "unserved_kw",
)


@dataclass(frozen=True)
class DeviceState:
    battery_energy_kwh: float
    hydrogen_kg: float
    battery_net_power_kw: float = 0.0  # discharge - charge
    electrolyzer_power_kw: float = 0.0
    fuel_cell_power_kw: float = 0.0


def finite_number(value: object, label: str, *, allow_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if not allow_negative and number < -TOL:
        raise ValueError(f"{label} must be nonnegative")
    return number


def load_parameters(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        p = json.load(stream)
    for section in ("meta", "battery", "electrolyzer", "hydrogen_tank", "fuel_cell", "grid"):
        if section not in p or not isinstance(p[section], dict):
            raise ValueError(f"Missing parameter section: {section}")
    dt = finite_number(p["meta"]["time_step_h"], "meta.time_step_h")
    if dt <= 0:
        raise ValueError("time_step_h must be positive")
    b, h, el, fc, grid = (p[k] for k in ("battery", "hydrogen_tank", "electrolyzer", "fuel_cell", "grid"))
    for name, value in (
        ("battery.energy_capacity_kwh", b["energy_capacity_kwh"]),
        ("battery.charge_power_max_kw", b["charge_power_max_kw"]),
        ("battery.discharge_power_max_kw", b["discharge_power_max_kw"]),
        ("electrolyzer.power_max_kw", el["power_max_kw"]),
        ("electrolyzer.electricity_kwh_per_kg_h2", el["electricity_kwh_per_kg_h2"]),
        ("fuel_cell.power_max_kw", fc["power_max_kw"]),
        ("fuel_cell.electric_output_kwh_per_kg_h2", fc["electric_output_kwh_per_kg_h2"]),
        ("hydrogen_tank.capacity_kg", h["capacity_kg"]),
    ):
        if finite_number(value, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("soc_min", "soc_initial", "soc_max", "charge_efficiency", "discharge_efficiency", "net_power_ramp_kw_per_h"):
        finite_number(b[name], f"battery.{name}")
    if not (0 <= b["soc_min"] <= b["soc_initial"] <= b["soc_max"] <= 1):
        raise ValueError("Battery SOC bounds/initial value are inconsistent")
    if not (0 < b["charge_efficiency"] <= 1 and 0 < b["discharge_efficiency"] <= 1):
        raise ValueError("Battery efficiencies must lie in (0, 1]")
    for name in ("initial_kg", "min_kg", "max_kg"):
        finite_number(h[name], f"hydrogen_tank.{name}")
    if not (0 <= h["min_kg"] <= h["initial_kg"] <= h["max_kg"] <= h["capacity_kg"]):
        raise ValueError("Hydrogen inventory bounds/initial value are inconsistent")
    for section_name, equipment in (("electrolyzer", el), ("fuel_cell", fc)):
        for name in ("power_min_on_kw", "power_ramp_kw_per_h"):
            finite_number(equipment[name], f"{section_name}.{name}")
        if equipment["power_min_on_kw"] > equipment["power_max_kw"]:
            raise ValueError(f"{section_name} min on-power exceeds maximum")
        if equipment["power_min_on_kw"] > equipment["power_ramp_kw_per_h"] * dt + TOL:
            raise ValueError(f"{section_name} cannot ramp from minimum on-power to off in one step")
    for name in ("import_power_max_kw", "export_power_max_kw"):
        finite_number(grid[name], f"grid.{name}")
    if grid["export_power_max_kw"] != 0:
        raise ValueError("This baseline does not implement grid export")
    return p


def initial_state(p: dict[str, Any]) -> DeviceState:
    b, h = p["battery"], p["hydrogen_tank"]
    return DeviceState(b["energy_capacity_kwh"] * b["soc_initial"], h["initial_kg"])


def ramp_down_energy_kwsteps(power_kw: float, ramp_kw_per_step: float, min_on_kw: float = 0.0) -> float:
    """Sum of future mandatory kW setpoints when reducing toward zero.

The caller multiplies by dt and an efficiency factor to convert to energy or
mass. For example, 500 kW with a 125 kW/step ramp requires future setpoints
375, 250 and 125 kW before zero.
"""
    if ramp_kw_per_step <= 0:
        if power_kw > TOL:
            raise ValueError("Positive power cannot stop with zero ramp rate")
        return 0.0
    if power_kw > TOL and min_on_kw > ramp_kw_per_step + TOL:
        raise ValueError("Minimum on-power cannot ramp down to zero in finite steps")
    total = 0.0
    current = power_kw
    while current > ramp_kw_per_step + TOL:
        next_power = max(min_on_kw, current - ramp_kw_per_step)
        total += next_power
        current = next_power
    return total


def apply_dispatch(
    state: DeviceState,
    load_kw: float,
    pv_kw: float,
    wind_kw: float,
    flow: dict[str, float],
    p: dict[str, Any],
) -> tuple[DeviceState, float]:
    """Apply one interval and reject nonfinite, physically illegal actions."""
    dt = p["meta"]["time_step_h"]
    b, h, el, fc, grid = (p[k] for k in ("battery", "hydrogen_tank", "electrolyzer", "fuel_cell", "grid"))
    for name, value in (("load_kw", load_kw), ("pv_kw", pv_kw), ("wind_kw", wind_kw)):
        finite_number(value, name)
    for name in state.__dataclass_fields__:
        finite_number(getattr(state, name), f"state.{name}", allow_negative=(name == "battery_net_power_kw"))
    if set(flow) != set(FLOW_FIELDS):
        raise ValueError(f"Flow fields must be exactly: {FLOW_FIELDS}")
    for name in FLOW_FIELDS:
        finite_number(flow[name], name)
    charge, discharge = flow["battery_charge_kw"], flow["battery_discharge_kw"]
    electro, fuel = flow["electrolyzer_kw"], flow["fuel_cell_kw"]
    if charge > TOL and discharge > TOL:
        raise ValueError("Battery charge/discharge are mutually exclusive")
    if electro > TOL and fuel > TOL:
        raise ValueError("Electrolyzer/fuel cell are mutually exclusive")
    for name, value, upper in (
        ("battery_charge_kw", charge, b["charge_power_max_kw"]),
        ("battery_discharge_kw", discharge, b["discharge_power_max_kw"]),
        ("electrolyzer_kw", electro, el["power_max_kw"]),
        ("fuel_cell_kw", fuel, fc["power_max_kw"]),
        ("grid_import_kw", flow["grid_import_kw"], grid["import_power_max_kw"]),
    ):
        if value > upper + TOL:
            raise ValueError(f"{name} exceeds its rating")
    for name, value, minimum in (
        ("electrolyzer", electro, el["power_min_on_kw"]),
        ("fuel_cell", fuel, fc["power_min_on_kw"]),
    ):
        if TOL < value < minimum - TOL:
            raise ValueError(f"{name} below its minimum on-power")
    battery_net = discharge - charge
    for name, current, previous, rate in (
        ("battery", battery_net, state.battery_net_power_kw, b["net_power_ramp_kw_per_h"]),
        ("electrolyzer", electro, state.electrolyzer_power_kw, el["power_ramp_kw_per_h"]),
        ("fuel_cell", fuel, state.fuel_cell_power_kw, fc["power_ramp_kw_per_h"]),
    ):
        if abs(current - previous) > rate * dt + TOL:
            raise ValueError(f"{name} ramp limit exceeded: {current} vs {previous} kW")
    next_energy = state.battery_energy_kwh + charge * b["charge_efficiency"] * dt - discharge * dt / b["discharge_efficiency"]
    next_hydrogen = state.hydrogen_kg + electro * dt / el["electricity_kwh_per_kg_h2"] - fuel * dt / fc["electric_output_kwh_per_kg_h2"]
    if not (b["soc_min"] * b["energy_capacity_kwh"] - TOL <= next_energy <= b["soc_max"] * b["energy_capacity_kwh"] + TOL):
        raise ValueError("Battery SOC limit exceeded")
    if not (h["min_kg"] - TOL <= next_hydrogen <= h["max_kg"] + TOL):
        raise ValueError("Hydrogen inventory limit exceeded")
    if flow["unserved_kw"] > load_kw + TOL:
        raise ValueError("Unserved customer load cannot exceed total customer load")
    supply = pv_kw + wind_kw + discharge + fuel + flow["grid_import_kw"] + flow["unserved_kw"]
    use = load_kw + charge + electro + flow["curtailment_kw"]
    balance_residual = supply - use
    if abs(balance_residual) > 1e-5:
        raise ValueError(f"Power balance residual {balance_residual:.6g} kW")
    return DeviceState(next_energy, next_hydrogen, battery_net, electro, fuel), balance_residual
