import csv
import importlib.util
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_arms.py"
SPEC = importlib.util.spec_from_file_location("compare_arms", SCRIPT)
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


def empty_gate_inputs():
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seeds = defaultdict(lambda: defaultdict(set))
    return data, seeds


class GateTests(unittest.TestCase):
    def test_empty_gate_stops(self):
        data, seeds = empty_gate_inputs()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            self.assertFalse(COMPARE.write_gate(out, data, seeds))

            gate = out / "aggregated" / "regression_check.csv"
            self.assertTrue(gate.read_text().endswith("# GATE: STOP\n"))
            with gate.open() as fh:
                rows = list(csv.DictReader(
                    line for line in fh if not line.startswith("#")))
            self.assertEqual(rows, [])

    def test_multiseed_core_regression_stops(self):
        data, seeds = empty_gate_inputs()
        key = ("bursty", 8)
        data["A"][key]["goodput"] = [0.80, 0.81, 0.79]
        data["B"][key]["goodput"] = [0.55, 0.56, 0.54]
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(COMPARE.write_gate(Path(directory), data, seeds))

    def test_multiseed_difference_within_noise_passes(self):
        data, seeds = empty_gate_inputs()
        key = ("bursty", 8)
        data["A"][key]["goodput"] = [0.70, 0.80, 0.90]
        data["B"][key]["goodput"] = [0.72, 0.82, 0.92]
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(COMPARE.write_gate(Path(directory), data, seeds))

    def test_zero_failed_requests_are_identical(self):
        data, seeds = empty_gate_inputs()
        key = ("bursty", 8)
        data["A"][key]["throughput"] = [8.0]
        data["B"][key]["throughput"] = [8.0]
        data["A"][key]["failed"] = [0.0]
        data["B"][key]["failed"] = [0.0]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            self.assertTrue(COMPARE.write_gate(out, data, seeds))
            text = (out / "aggregated" / "regression_check.csv").read_text()
            self.assertIn("failed,0.0,1,0.0,1,,identical", text)


if __name__ == "__main__":
    unittest.main()
