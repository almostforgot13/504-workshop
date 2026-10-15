"""Boundary and interface tests for 15-minute fixed-rule dispatch."""

from copy import deepcopy
import csv
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from device_model import DeviceState, apply_dispatch, load_parameters, ramp_down_energy_kwsteps
from rule_dispatch import InfeasibleDispatchError, RESULT_FIELDS, fixed_rule, read_15min_week, run_week


PARAMS = load_parameters(Path(__file__).with_name("candidate_parameters_15min.json"))
REPO_SOURCE = Path(__file__).resolve().parents[3] / "data" / "processed" / "A_15min_virtual_2017_2019.csv"
SOURCE = Path(os.environ.get("AIC_ENERGY_SOURCE_CSV", str(REPO_SOURCE)))
START = datetime(2017, 1, 1, tzinfo=timezone.utc)


def blank_flow():
    return {
        "grid_import_kw": 0.0, "battery_charge_kw": 0.0, "battery_discharge_kw": 0.0,
        "electrolyzer_kw": 0.0, "fuel_cell_kw": 0.0,
        "curtailment_kw": 0.0, "unserved_kw": 0.0,
    }


class DeviceBoundaryTests(unittest.TestCase):
    def test_quarter_hour_energy_and_hydrogen_conversion(self):
        state = DeviceState(1000.0, 300.0)
        flow = blank_flow()
        flow.update(battery_charge_kw=100.0, electrolyzer_kw=55.0)
        after, residual = apply_dispatch(state, 0.0, 155.0, 0.0, flow, PARAMS)
        self.assertAlmostEqual(after.battery_energy_kwh, 1000 + 100 * 0.25 * PARAMS["battery"]["charge_efficiency"])
        self.assertAlmostEqual(after.hydrogen_kg, 300.25)
        self.assertAlmostEqual(residual, 0)

    def test_full_battery_and_tank_record_spill(self):
        state = DeviceState(1900.0, 540.0)
        flow = fixed_rule(state, 0.0, 1000.0, 0.0, PARAMS)
        self.assertAlmostEqual(flow["battery_charge_kw"], 0)
        self.assertAlmostEqual(flow["electrolyzer_kw"], 0)
        self.assertAlmostEqual(flow["curtailment_kw"], 1000)
        after, _ = apply_dispatch(state, 0.0, 1000.0, 0.0, flow, PARAMS)
        self.assertEqual(after, state)

    def test_empty_battery_and_tank_record_unserved(self):
        state = DeviceState(300.0, 60.0)
        flow = fixed_rule(state, 1000.0, 0.0, 0.0, PARAMS)
        self.assertAlmostEqual(flow["grid_import_kw"], 500)
        self.assertAlmostEqual(flow["unserved_kw"], 500)
        after, _ = apply_dispatch(state, 1000.0, 0.0, 0.0, flow, PARAMS)
        self.assertEqual(after, state)

    def test_maximum_power_and_minimum_on_are_enforced(self):
        state = DeviceState(1000.0, 300.0)
        flow = blank_flow()
        flow.update(grid_import_kw=501.0, unserved_kw=499.0)
        with self.assertRaisesRegex(ValueError, "rating"):
            apply_dispatch(state, 1000.0, 0.0, 0.0, flow, PARAMS)
        flow = blank_flow()
        flow.update(electrolyzer_kw=10.0, curtailment_kw=0.0)
        with self.assertRaisesRegex(ValueError, "minimum on-power"):
            apply_dispatch(state, 0.0, 10.0, 0.0, flow, PARAMS)
        flow = blank_flow()
        flow.update(fuel_cell_kw=10.0, curtailment_kw=0.0)
        with self.assertRaisesRegex(ValueError, "minimum on-power"):
            apply_dispatch(state, 10.0, 0.0, 0.0, flow, PARAMS)

    def test_startup_respects_ramp_and_minimum_on_power(self):
        state = DeviceState(300.0, 300.0)
        flow = fixed_rule(state, 1000.0, 0.0, 0.0, PARAMS)
        self.assertGreaterEqual(flow["fuel_cell_kw"], 50.0)
        self.assertLessEqual(flow["fuel_cell_kw"], 125.0)
        apply_dispatch(state, 1000.0, 0.0, 0.0, flow, PARAMS)
        low_deficit = fixed_rule(state, 30.0, 0.0, 0.0, PARAMS)
        self.assertEqual(low_deficit["fuel_cell_kw"], 0.0)
        full_battery = DeviceState(1900.0, 300.0)
        surplus = fixed_rule(full_battery, 0.0, 1000.0, 0.0, PARAMS)
        self.assertGreaterEqual(surplus["electrolyzer_kw"], 30.0)
        self.assertLessEqual(surplus["electrolyzer_kw"], 75.0)
        apply_dispatch(full_battery, 0.0, 1000.0, 0.0, surplus, PARAMS)
        small_surplus = fixed_rule(full_battery, 0.0, 20.0, 0.0, PARAMS)
        self.assertEqual(small_surplus["electrolyzer_kw"], 0.0)

    def test_sudden_renewable_loss_keeps_ramp_down_physically_supplied(self):
        state = DeviceState(1000.0, 300.0)
        for _ in range(4):
            flow = fixed_rule(state, 0.0, 800.0, 0.0, PARAMS)
            state, _ = apply_dispatch(state, 0.0, 800.0, 0.0, flow, PARAMS)
        self.assertLessEqual(
            max(0.0, -state.battery_net_power_kw - 125.0)
            + max(0.0, state.electrolyzer_power_kw - 75.0),
            PARAMS["grid"]["import_power_max_kw"] + 1e-7,
        )
        flow = fixed_rule(state, 0.0, 0.0, 0.0, PARAMS)
        self.assertEqual(flow["unserved_kw"], 0.0)
        self.assertLessEqual(flow["grid_import_kw"], 500.0)
        apply_dispatch(state, 0.0, 0.0, 0.0, flow, PARAMS)

    def test_preexisting_infeasible_state_is_reported(self):
        state = DeviceState(1000.0, 300.0, battery_net_power_kw=-500.0, electrolyzer_power_kw=300.0)
        with self.assertRaisesRegex(InfeasibleDispatchError, "Mandatory device demand") as raised:
            fixed_rule(state, 0.0, 0.0, 0.0, PARAMS)
        self.assertGreater(raised.exception.details["unserved_slack_required_kw"], 0.0)

    def test_electrolyzer_ramps_down_in_deficit(self):
        state = DeviceState(1000.0, 300.0, electrolyzer_power_kw=225.0)
        flow = fixed_rule(state, 300.0, 0.0, 0.0, PARAMS)
        self.assertGreaterEqual(flow["electrolyzer_kw"], 150.0 - 1e-7)
        self.assertEqual(flow["fuel_cell_kw"], 0.0)
        apply_dispatch(state, 300.0, 0.0, 0.0, flow, PARAMS)
        bad = blank_flow()
        bad["grid_import_kw"] = 300.0
        with self.assertRaisesRegex(ValueError, "ramp"):
            apply_dispatch(state, 300.0, 0.0, 0.0, bad, PARAMS)

    def test_fuel_cell_ramps_down_when_demand_drops(self):
        state = DeviceState(1000.0, 300.0, fuel_cell_power_kw=300.0)
        flow = fixed_rule(state, 0.0, 0.0, 0.0, PARAMS)
        self.assertGreaterEqual(flow["fuel_cell_kw"], 175.0 - 1e-7)
        self.assertEqual(flow["electrolyzer_kw"], 0.0)
        self.assertAlmostEqual(flow["battery_charge_kw"] + flow["curtailment_kw"], flow["fuel_cell_kw"])
        apply_dispatch(state, 0.0, 0.0, 0.0, flow, PARAMS)

    def test_charge_discharge_switch_needs_transition(self):
        charging = DeviceState(1000.0, 300.0, battery_net_power_kw=-250.0)
        flow = fixed_rule(charging, 500.0, 0.0, 0.0, PARAMS)
        self.assertGreaterEqual(flow["battery_charge_kw"], 125.0 - 1e-7)
        self.assertEqual(flow["battery_discharge_kw"], 0.0)
        apply_dispatch(charging, 500.0, 0.0, 0.0, flow, PARAMS)
        discharging = DeviceState(1000.0, 300.0, battery_net_power_kw=250.0)
        flow = fixed_rule(discharging, 0.0, 500.0, 0.0, PARAMS)
        self.assertGreaterEqual(flow["battery_discharge_kw"], 125.0 - 1e-7)
        self.assertEqual(flow["battery_charge_kw"], 0.0)
        apply_dispatch(discharging, 0.0, 500.0, 0.0, flow, PARAMS)

    def test_ramp_near_inventory_bounds_keeps_future_braking_room(self):
        p = deepcopy(PARAMS)
        state = DeviceState(301.0, 60.2)
        flow = fixed_rule(state, 1000.0, 0.0, 0.0, p)
        after, _ = apply_dispatch(state, 1000.0, 0.0, 0.0, flow, p)
        self.assertGreaterEqual(after.battery_energy_kwh, 300.0 - 1e-7)
        self.assertGreaterEqual(after.hydrogen_kg, 60.0 - 1e-7)
        future_battery_use = ramp_down_energy_kwsteps(after.battery_net_power_kw, 125.0) * 0.25 / p["battery"]["discharge_efficiency"]
        future_fuel_use = ramp_down_energy_kwsteps(after.fuel_cell_power_kw, 125.0, 50.0) * 0.25 / 17.875
        self.assertLessEqual(future_battery_use, after.battery_energy_kwh - 300.0 + 1e-6)
        self.assertLessEqual(future_fuel_use, after.hydrogen_kg - 60.0 + 1e-6)

    def test_nan_and_infinity_rejected_in_state_and_flow(self):
        state = DeviceState(1000.0, 300.0)
        for bad_value in (float("nan"), float("inf"), -1.0):
            flow = blank_flow()
            flow["grid_import_kw"] = bad_value
            with self.assertRaises(ValueError):
                apply_dispatch(state, 0.0, 0.0, 0.0, flow, PARAMS)
        with self.assertRaises(ValueError):
            apply_dispatch(DeviceState(float("nan"), 300.0), 0.0, 0.0, 0.0, blank_flow(), PARAMS)


class InputAndWeekTests(unittest.TestCase):
    def _tiny_csv(self, rows):
        temp = TemporaryDirectory()
        path = Path(temp.name) / "input.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["time", "split", "load_kw", "pv_kw", "wind_kw", "net_load_kw"])
            writer.writeheader()
            writer.writerows(rows)
        self.addCleanup(temp.cleanup)
        return path

    def test_input_rejects_missing_nan_infinity_negative_duplicate_and_gap(self):
        a = {"time": "2017-01-01T00:00:00+00:00", "split": "train", "load_kw": "100", "pv_kw": "0", "wind_kw": "20", "net_load_kw": "80"}
        b = {**a, "time": "2017-01-01T00:15:00+00:00"}
        cases = [
            [{**a, "load_kw": ""}],
            [{**a, "pv_kw": "NaN"}],
            [{**a, "wind_kw": "Infinity"}],
            [{**a, "load_kw": "-1"}],
            [a, a],
            [a, {**b, "time": "2017-01-01T00:30:00+00:00"}],
            [{**a, "net_load_kw": ""}],
        ]
        for rows in cases:
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    read_15min_week(self._tiny_csv(rows), START)

    def test_real_week_has_672_steps_and_continuous_states(self):
        if not SOURCE.exists():
            self.skipTest("A's source CSV is not available on this computer")
        source, flagged = read_15min_week(SOURCE, START)
        results, summary = run_week(source, PARAMS, flagged)
        self.assertEqual(len(results), 672)
        self.assertEqual(tuple(results[0].keys()), RESULT_FIELDS)
        self.assertEqual(summary["steps"], 672)
        self.assertEqual(summary["duration_h"], 168.0)
        self.assertIsNone(summary["total_cost"])
        self.assertTrue(all(row["step_cost"] == "" for row in results))
        for previous, current in zip(results, results[1:]):
            self.assertAlmostEqual(previous["battery_energy_end_kwh"], current["battery_energy_start_kwh"])
            self.assertAlmostEqual(previous["hydrogen_end_kg"], current["hydrogen_start_kg"])
            self.assertLessEqual(abs((current["battery_discharge_kw"] - current["battery_charge_kw"]) - (previous["battery_discharge_kw"] - previous["battery_charge_kw"])), 125.0 + 1e-7)
            self.assertLessEqual(abs(current["electrolyzer_kw"] - previous["electrolyzer_kw"]), 75.0 + 1e-7)
            self.assertLessEqual(abs(current["fuel_cell_kw"] - previous["fuel_cell_kw"]), 125.0 + 1e-7)
        self.assertGreater(summary["energy_kwh"]["unserved_kwh"], 0)
        self.assertGreater(summary["energy_kwh"]["curtailment_kwh"], 0)

    def test_generated_week_runs_without_external_dataset(self):
        fixture = []
        for index in range(672):
            ts = START + timedelta(minutes=15 * index)
            pv = 300.0 if 8 <= ts.hour < 17 else 0.0
            fixture.append({
                "time": ts.isoformat(), "split": "train", "load_kw": "800",
                "pv_kw": str(pv), "wind_kw": "250", "net_load_kw": str(800 - pv - 250),
            })
        source, flagged = read_15min_week(self._tiny_csv(fixture), START)
        rows, summary = run_week(source, PARAMS, flagged)
        self.assertEqual(len(rows), 672)
        self.assertEqual(summary["duration_h"], 168.0)
        self.assertTrue(all(row["step_cost"] == "" for row in rows))


if __name__ == "__main__":
    unittest.main()
