"""Independently validate the fixed 18-column, 672-step result interface."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import io
import json
import math
import os
from pathlib import Path
import unittest

from device_model import initial_state, load_parameters
from rule_dispatch import RESULT_FIELDS, read_15min_week


NUMERIC_COLUMNS = tuple(name for name in RESULT_FIELDS if name.endswith(("_kw", "_kwh", "_kg")) and name != "step_cost")
NONNEGATIVE = tuple(name for name in NUMERIC_COLUMNS if name not in ("battery_energy_start_kwh", "battery_energy_end_kwh", "hydrogen_start_kg", "hydrogen_end_kg"))
FLOW_COLUMNS = (
    "grid_import_kw", "battery_charge_kw", "battery_discharge_kw",
    "electrolyzer_kw", "fuel_cell_kw", "curtailment_kw", "unserved_kw",
)


def close(actual: float, expected: float, tol: float = 1e-5) -> bool:
    return math.isclose(actual, expected, rel_tol=0, abs_tol=tol)


def validate(result_path: Path, param_path: Path, summary_path: Path | None, source_path: Path | None) -> dict:
    p = load_parameters(param_path)
    dt = p["meta"]["time_step_h"]
    b, h, el, fc, grid = (p[k] for k in ("battery", "hydrogen_tank", "electrolyzer", "fuel_cell", "grid"))
    errors: list[str] = []
    with result_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RESULT_FIELDS:
            raise ValueError("Result columns differ from the agreed 18-column interface")
        rows = list(reader)
    if len(rows) != 672:
        errors.append(f"Expected 672 rows, got {len(rows)}")
    if not rows:
        raise ValueError("Result CSV is empty")
    previous_time = None
    previous_end = initial_state(p)
    previous_battery_power = 0.0
    previous_electro = 0.0
    previous_fuel = 0.0
    energy = {name.replace("_kw", "_kwh"): 0.0 for name in (*FLOW_COLUMNS, "load_kw", "pv_kw", "wind_kw")}
    max_balance_residual = 0.0
    for index, row in enumerate(rows, 1):
        label = f"row {index}"
        try:
            time = datetime.fromisoformat(row["timestamp_utc"])
            if time.tzinfo is None or time.utcoffset() != timedelta(0):
                raise ValueError("timestamp is not UTC")
            if previous_time is not None and time - previous_time != timedelta(minutes=15):
                raise ValueError("timestamp spacing is not 15 minutes")
            previous_time = time
            if row["strategy"] != "fixed_rule_battery_hydrogen_grid_15min":
                raise ValueError("unexpected strategy name")
            if row["step_cost"] != "":
                raise ValueError("step_cost must be blank until a cost definition is agreed")
            values = {}
            for col in NUMERIC_COLUMNS:
                value = float(row[col])
                if not math.isfinite(value):
                    raise ValueError(f"{col} is not finite")
                if col in NONNEGATIVE and value < -1e-7:
                    raise ValueError(f"{col} is negative")
                values[col] = value
            for col, upper in (
                ("grid_import_kw", grid["import_power_max_kw"]),
                ("battery_charge_kw", b["charge_power_max_kw"]),
                ("battery_discharge_kw", b["discharge_power_max_kw"]),
                ("electrolyzer_kw", el["power_max_kw"]),
                ("fuel_cell_kw", fc["power_max_kw"]),
            ):
                if values[col] > upper + 1e-7:
                    raise ValueError(f"{col} exceeds rating")
            if values["battery_charge_kw"] > 1e-7 and values["battery_discharge_kw"] > 1e-7:
                raise ValueError("battery charges and discharges together")
            if values["electrolyzer_kw"] > 1e-7 and values["fuel_cell_kw"] > 1e-7:
                raise ValueError("electrolyzer and fuel cell operate together")
            if 1e-7 < values["electrolyzer_kw"] < el["power_min_on_kw"] - 1e-7:
                raise ValueError("electrolyzer below minimum on-power")
            if 1e-7 < values["fuel_cell_kw"] < fc["power_min_on_kw"] - 1e-7:
                raise ValueError("fuel cell below minimum on-power")
            if values["unserved_kw"] > values["load_kw"] + 1e-7:
                raise ValueError("unserved load exceeds customer load")
            b_signed = values["battery_discharge_kw"] - values["battery_charge_kw"]
            for name, current, previous, rate in (
                ("battery", b_signed, previous_battery_power, b["net_power_ramp_kw_per_h"]),
                ("electrolyzer", values["electrolyzer_kw"], previous_electro, el["power_ramp_kw_per_h"]),
                ("fuel cell", values["fuel_cell_kw"], previous_fuel, fc["power_ramp_kw_per_h"]),
            ):
                if abs(current - previous) > rate * dt + 1e-7:
                    raise ValueError(f"{name} ramp violation")
            previous_battery_power = b_signed
            previous_electro = values["electrolyzer_kw"]
            previous_fuel = values["fuel_cell_kw"]
            if not close(values["battery_energy_start_kwh"], previous_end.battery_energy_kwh):
                raise ValueError("battery state discontinuity")
            if not close(values["hydrogen_start_kg"], previous_end.hydrogen_kg):
                raise ValueError("hydrogen state discontinuity")
            expected_energy = values["battery_energy_start_kwh"] + values["battery_charge_kw"] * b["charge_efficiency"] * dt - values["battery_discharge_kw"] * dt / b["discharge_efficiency"]
            expected_hydrogen = values["hydrogen_start_kg"] + values["electrolyzer_kw"] * dt / el["electricity_kwh_per_kg_h2"] - values["fuel_cell_kw"] * dt / fc["electric_output_kwh_per_kg_h2"]
            if not close(values["battery_energy_end_kwh"], expected_energy):
                raise ValueError("battery state equation mismatch")
            if not close(values["hydrogen_end_kg"], expected_hydrogen):
                raise ValueError("hydrogen state equation mismatch")
            if not (b["soc_min"] * b["energy_capacity_kwh"] - 1e-7 <= values["battery_energy_end_kwh"] <= b["soc_max"] * b["energy_capacity_kwh"] + 1e-7):
                raise ValueError("battery SOC bound violated")
            if not (h["min_kg"] - 1e-7 <= values["hydrogen_end_kg"] <= h["max_kg"] + 1e-7):
                raise ValueError("hydrogen bound violated")
            previous_end = type(previous_end)(values["battery_energy_end_kwh"], values["hydrogen_end_kg"], b_signed, values["electrolyzer_kw"], values["fuel_cell_kw"])
            supply = values["pv_kw"] + values["wind_kw"] + values["battery_discharge_kw"] + values["fuel_cell_kw"] + values["grid_import_kw"] + values["unserved_kw"]
            use = values["load_kw"] + values["battery_charge_kw"] + values["electrolyzer_kw"] + values["curtailment_kw"]
            max_balance_residual = max(max_balance_residual, abs(supply - use))
            if abs(supply - use) > 1e-5:
                raise ValueError("power balance mismatch")
            expected_status = "rule_unserved" if values["unserved_kw"] > 1e-7 else "rule_curtailment" if values["curtailment_kw"] > 1e-7 else "rule_ok"
            if row["solver_status"] != expected_status:
                raise ValueError("solver_status does not match flows")
            for col in energy:
                energy[col] += values[col.replace("_kwh", "_kw")] * dt
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"{label}: {exc}")
            if len(errors) >= 25:
                break
    if summary_path and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("steps") != 672 or not close(summary.get("duration_h", -1), 168.0):
            errors.append("Summary steps/duration mismatch")
        if summary.get("total_cost") is not None or summary.get("cost_status") != "not_implemented_no_agreed_price_or_degradation_cost":
            errors.append("Summary must declare cost unimplemented")
        for name, value in energy.items():
            if not close(value, summary.get("energy_kwh", {}).get(name, math.nan), tol=1e-4):
                errors.append(f"Summary energy mismatch: {name}")
    if source_path:
        first = datetime.fromisoformat(rows[0]["timestamp_utc"])
        source_rows, _ = read_15min_week(source_path, first)
        for index, (actual, original) in enumerate(zip(rows, source_rows), 1):
            if actual["timestamp_utc"] != original["timestamp_utc"]:
                errors.append(f"row {index}: source timestamp mismatch")
                break
            for name in ("load_kw", "pv_kw", "wind_kw"):
                if not close(float(actual[name]), original[name]):
                    errors.append(f"row {index}: source {name} mismatch")
                    break
            if errors:
                break
    return {
        "passed": not errors,
        "rows_checked": len(rows),
        "required_rows": 672,
        "expected_duration_h": 168.0,
        "max_abs_power_balance_residual_kw": max_balance_residual,
        "cost_status": "not_implemented",
        "energy_kwh_recalculated": energy,
        "errors": errors,
    }


def run_unit_tests(source_path: Path | None = None) -> dict:
    previous_source = os.environ.get("AIC_ENERGY_SOURCE_CSV")
    if source_path is not None:
        os.environ["AIC_ENERGY_SOURCE_CSV"] = str(source_path)
    suite = unittest.defaultTestLoader.discover(start_dir=str(Path(__file__).parent), pattern="test_rule_dispatch.py")
    output = io.StringIO()
    try:
        result = unittest.TextTestRunner(stream=output, verbosity=1).run(suite)
    finally:
        if previous_source is None:
            os.environ.pop("AIC_ENERGY_SOURCE_CSV", None)
        else:
            os.environ["AIC_ENERGY_SOURCE_CSV"] = previous_source
    return {
        "passed": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "details": output.getvalue(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a 15-minute 672-row rule result")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--params", type=Path, default=Path(__file__).with_name("candidate_parameters_15min.json"))
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--source", type=Path, help="Optional original 15-minute CSV")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    validation = validate(args.results, args.params, args.summary, args.source)
    tests = run_unit_tests(args.source)
    report = {"validation": validation, "unit_tests": tests, "passed": validation["passed"] and tests["passed"]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "rows_checked": validation["rows_checked"], "unit_tests_run": tests["tests_run"], "errors": validation["errors"]}, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
