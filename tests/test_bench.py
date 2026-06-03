"""Tests for GPU utility benchmark suite."""
import unittest
from src.memory_bandwidth import BandwidthResult, LatencyResult
from src.compute_bench import ComputeResult


class TestBandwidthResult(unittest.TestCase):
    def test_avg_bandwidth(self):
        result = BandwidthResult(
            device="Test GPU", size_mb=1024,
            read_gbps=4500, write_gbps=4200, copy_gbps=4000,
            iterations=100, elapsed_sec=1.0,
        )
        self.assertAlmostEqual(result.avg_bandwidth, 4233.333, places=0)

    def test_result_fields(self):
        result = BandwidthResult(
            device="MI300X", size_mb=4096,
            read_gbps=5000, write_gbps=4800, copy_gbps=4500,
            iterations=50, elapsed_sec=2.0,
        )
        self.assertEqual(result.device, "MI300X")
        self.assertEqual(result.size_mb, 4096)
        self.assertEqual(result.iterations, 50)


class TestComputeResult(unittest.TestCase):
    def test_efficiency(self):
        result = ComputeResult(
            device="MI300X", precision="fp32", matrix_size=4096,
            tflops=150.0, elapsed_sec=2.0, iterations=50,
        )
        eff = result.efficiency_pct
        self.assertGreater(eff, 0)
        self.assertLess(eff, 200)

    def test_precision_labeling(self):
        result = ComputeResult(
            device="MI300X", precision="fp16", matrix_size=4096,
            tflops=1000.0, elapsed_sec=1.0, iterations=50,
        )
        self.assertEqual(result.precision, "fp16")


class TestLatencyResult(unittest.TestCase):
    def test_latency_ordering(self):
        result = LatencyResult(
            device="Test", min_us=10.0, max_us=100.0,
            avg_us=25.0, p50_us=20.0, p99_us=90.0, iterations=1000,
        )
        self.assertLessEqual(result.min_us, result.p50_us)
        self.assertLessEqual(result.p50_us, result.p99_us)
        self.assertLessEqual(result.p99_us, result.max_us)


if __name__ == "__main__":
    unittest.main()
