import csv
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.gpu_probe import GpuProbe, parse_sample, summarize_samples


class GpuProbeTests(unittest.TestCase):
    def test_parses_only_valid_single_gpu_samples(self):
        self.assertEqual(parse_sample("50, 2048, 46080\n"), (50, 2048, 46080))
        for output in ("N/A, 100, 200", "101, 100, 200", "5, 300, 200", "5, 1, 2\n5, 1, 2"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                parse_sample(output)

    def test_summary_uses_sampled_utilization_and_peak_memory(self):
        samples = [{"gpu_utilization_pct": value, "memory_used_mib": (index + 1) * 1000,
                    "memory_total_mib": 46080} for index, value in enumerate([0, 50, 100])]
        summary = summarize_samples(samples)
        self.assertEqual(summary["gpu_utilization_mean_pct"], 50)
        self.assertEqual(summary["gpu_utilization_p50_pct"], 50)
        self.assertEqual(summary["gpu_utilization_p95_pct"], 95)
        self.assertEqual(summary["active_sample_fraction"], 2 / 3)
        self.assertEqual(summary["busy_sample_fraction"], 1 / 3)
        self.assertEqual(summary["peak_memory_used_mib"], 3000)
        self.assertEqual(summarize_samples([]), {"samples": 0})

    def test_csv_stays_local_until_probe_closes_and_queries_only_gpu_metrics(self):
        queried = threading.Event()
        def query(*args, **kwargs):
            queried.set()
            return SimpleNamespace(stdout="50, 2048, 46080\n")
        with tempfile.TemporaryDirectory() as temp, patch("scripts.gpu_probe.subprocess.run", side_effect=query) as run:
            output = Path(temp) / "stored" / "gpu.csv"
            with GpuProbe(output) as probe:
                self.assertTrue(queried.wait(2))
                self.assertFalse(output.exists())
                self.assertNotEqual(probe.local_path.parent, output.parent)
                with probe.stage("initialization"):
                    pass
            with output.open() as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]["memory_used_mib"]), 2048)
        self.assertIn("--query-gpu=utilization.gpu,memory.used,memory.total", run.call_args.args[0])
        self.assertIn("initialization", probe.summary()["stages"])

    def test_unavailable_probe_does_not_fail_inference(self):
        queried = threading.Event()
        def unavailable(*args, **kwargs):
            queried.set()
            raise FileNotFoundError("nvidia-smi")
        with tempfile.TemporaryDirectory() as temp, patch("scripts.gpu_probe.subprocess.run", side_effect=unavailable):
            with GpuProbe(Path(temp) / "gpu.csv") as probe:
                self.assertTrue(queried.wait(2))
                with probe.stage("native_inference"):
                    pass
        self.assertEqual(probe.summary()["samples"], 0)
        self.assertEqual(probe.summary()["collection_errors"], 1)


if __name__ == "__main__":
    unittest.main()
