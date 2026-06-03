"""
Tests for GPU Benchmark Suite
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import yaml


# --- Metrics Collector Tests ---

class TestMetricsCollector:
    """Test metrics collection and export."""

    def test_init(self, tmp_path):
        from src.metrics import MetricsCollector
        mc = MetricsCollector(output_dir=str(tmp_path))
        assert mc.output_dir.exists()

    def test_save_json(self, tmp_path):
        from src.metrics import MetricsCollector
        mc = MetricsCollector(output_dir=str(tmp_path))
        results = {
            "bandwidth": {
                "peak_read": {"value": 5210, "unit": "GB/s"},
            },
            "compute": {
                "fp16": {"value": 1307, "unit": "TFLOPS"},
            },
        }
        mc.save(results, formats=["json"])
        files = list(tmp_path.glob("benchmark_*.json"))
        assert len(files) == 1

        with open(files[0]) as f:
            data = json.load(f)
        assert "results" in data
        assert "timestamp" in data

    def test_save_csv(self, tmp_path):
        from src.metrics import MetricsCollector
        mc = MetricsCollector(output_dir=str(tmp_path))
        results = {
            "bandwidth": {
                "peak_read": {"value": 5210, "unit": "GB/s"},
            },
        }
        mc.save(results, formats=["csv"])
        files = list(tmp_path.glob("benchmark_*.csv"))
        assert len(files) == 1

    def test_compare(self):
        from src.metrics import MetricsCollector
        mc = MetricsCollector(output_dir="/tmp/test")
        baseline = {"bw": {"peak": {"value": 100, "unit": "GB/s"}}}
        current = {"bw": {"peak": {"value": 120, "unit": "GB/s"}}}
        comp = mc.compare(baseline, current)
        assert comp["bw"]["peak"]["delta"] == 20
        assert comp["bw"]["peak"]["delta_pct"] == 20.0

    def test_flatten(self):
        from src.metrics import MetricsCollector
        mc = MetricsCollector(output_dir="/tmp/test")
        rows = []
        mc._flatten(
            {"cat": {"metric": {"value": 42, "unit": "ns"}}},
            ["cat"],
            rows,
        )
        assert len(rows) == 1
        assert rows[0] == ["cat", "metric", 42, "ns"]

    def test_load(self, tmp_path):
        from src.metrics import MetricsCollector
        mc = MetricsCollector(output_dir=str(tmp_path))
        data = {"results": {"key": "value"}, "timestamp": "20240101"}
        path = tmp_path / "test.json"
        with open(path, "w") as f:
            json.dump(data, f)
        loaded = mc.load(str(path))
        assert loaded == {"key": "value"}


# --- Bandwidth Benchmark Tests ---

class TestBandwidthBenchmark:
    """Test bandwidth benchmarking (mocked)."""

    def test_init(self):
        from src.bandwidth import BandwidthBenchmark
        bb = BandwidthBenchmark(vendor="nvidia", iterations=10)
        assert bb.iterations == 10
        assert bb.vendor == "nvidia"

    def test_format_size(self):
        from src.bandwidth import BandwidthBenchmark
        assert BandwidthBenchmark._format_size(512) == "512KB"
        assert BandwidthBenchmark._format_size(1024) == "1MB"
        assert BandwidthBenchmark._format_size(4096) == "4MB"


# --- Compute Benchmark Tests ---

class TestComputeBenchmark:
    """Test compute benchmarking (mocked)."""

    def test_init(self):
        from src.compute import ComputeBenchmark
        cb = ComputeBenchmark(
            vendor="nvidia", iterations=10,
            dtypes=["fp16", "fp32"], matrix_sizes=[1024, 4096],
        )
        assert cb.iterations == 10
        assert len(cb.dtypes) == 2

    def test_dtype_map_exists(self):
        from src.compute import ComputeBenchmark
        assert "fp16" in ComputeBenchmark.DTYPE_MAP
        assert "fp32" in ComputeBenchmark.DTYPE_MAP
        assert "int8" in ComputeBenchmark.DTYPE_MAP


# --- Memory Latency Tests ---

class TestMemoryLatencyBenchmark:

    def test_init(self):
        from src.memory import MemoryLatencyBenchmark
        mlb = MemoryLatencyBenchmark(vendor="nvidia", iterations=50)
        assert mlb.iterations == 50

    def test_format_size(self):
        from src.memory import MemoryLatencyBenchmark
        assert MemoryLatencyBenchmark._format_size(512) == "512KB"
        assert MemoryLatencyBenchmark._format_size(1024) == "1MB"


# --- Kernel Latency Tests ---

class TestKernelLatencyBenchmark:

    def test_init(self):
        from src.latency import KernelLaunchBenchmark
        klb = KernelLaunchBenchmark(vendor="nvidia", iterations=100)
        assert klb.iterations == 100


# --- Visualizer Tests ---

class TestVisualizer:

    def test_init(self, tmp_path):
        from src.visualizer import BenchmarkVisualizer
        bv = BenchmarkVisualizer(output_dir=str(tmp_path))
        assert bv.output_dir.exists()


# --- Profiler Tests ---

class TestCUProfiler:

    def test_init(self):
        from src.profiler import CUProfiler
        cp = CUProfiler(vendor="nvidia", iterations=10)
        assert cp.iterations == 10


# --- Config YAML Tests ---

class TestConfigs:

    def test_bandwidth_config(self):
        with open("configs/bandwidth_test.yml") as f:
            cfg = yaml.safe_load(f)
        assert "buffer_sizes_mb" in cfg
        assert cfg["gpu_vendor"] == "auto"

    def test_compute_config(self):
        with open("configs/compute_test.yml") as f:
            cfg = yaml.safe_load(f)
        assert "matrix_sizes" in cfg
        assert "fp16" in cfg["dtypes"]


# --- Orchestrator Tests ---

class TestOrchestrator:

    def test_detect_gpu_returns_string(self):
        from src.benchmark import GPUBenchmarkOrchestrator
        with open("configs/compute_test.yml") as f:
            cfg = yaml.safe_load(f)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp:
            yaml.dump(cfg, tmp)
            tmp_path = tmp.name

        orch = GPUBenchmarkOrchestrator(tmp_path)
        vendor = orch.detect_gpu()
        assert isinstance(vendor, str)
        assert vendor in ("nvidia", "amd")
