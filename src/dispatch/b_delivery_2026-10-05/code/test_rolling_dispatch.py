"""Focused checks for the forecast contract and measured-action correction."""
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from device_model import DeviceState, apply_dispatch, load_parameters
from rolling_dispatch import FLOW_NAMES, actual_flow, read_forecasts, solve_window


HERE = Path(__file__).resolve().parent
START = datetime(2019, 6, 15, tzinfo=timezone.utc)


class ForecastContractTests(unittest.TestCase):
    def make_file(self, path: Path, *, omit_last=False):
        fields = ["issue_time", "target_time", "horizon_step", "horizon_hours",
                  "q10_kw", "q50_kw", "q90_kw"]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for step in range(1, 96 if omit_last else 97):
                writer.writerow({"issue_time": START.isoformat(),
                                 "target_time": (START + timedelta(minutes=15 * (step - 1))).isoformat(),
                                 "horizon_step": step, "horizon_hours": step / 4,
                                 "q10_kw": -30, "q50_kw": -10, "q90_kw": 20})

    def test_96_step_negative_net_load_is_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forecast.csv"
            self.make_file(path)
            window = read_forecasts(path, START, 1)
            self.assertEqual(len(window[0]), 96)
            self.assertEqual(window[0][0]["q50_kw"], -10)

    def test_incomplete_window_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forecast.csv"
            self.make_file(path, omit_last=True)
            with self.assertRaisesRegex(ValueError, "96 horizons"):
                read_forecasts(path, START, 1)


class ActualCorrectionTests(unittest.TestCase):
    def test_blackout_correction_does_not_invent_customer_load(self):
        p = load_parameters(HERE / "candidate_parameters_15min.json")
        state = DeviceState(1000, 300, -400, 300, 0)
        planned = {"charge": 500.0, "discharge": 0.0,
                   "electrolyzer": 200.0, "fuel_cell": 0.0}
        actual = {"load_kw": 0.0, "pv_kw": 0.0, "wind_kw": 0.0}
        with self.assertRaisesRegex(ValueError, "device load"):
            actual_flow(planned, actual, p)
        action, _ = solve_window([0.0], p, state, customer_load_kw=0.0,
                                 target=planned, time_limit=5)
        flow = dict(zip(FLOW_NAMES, (action[k] for k in
                        ("grid", "charge", "discharge", "electrolyzer", "fuel_cell",
                         "curtailment", "unserved"))))
        next_state, residual = apply_dispatch(state, **actual, flow=flow, p=p)
        self.assertAlmostEqual(flow["unserved_kw"], 0)
        self.assertLessEqual(flow["grid_import_kw"], 500)
        self.assertAlmostEqual(residual, 0)
        self.assertGreater(next_state.battery_energy_kwh, state.battery_energy_kwh)


if __name__ == "__main__":
    unittest.main()
