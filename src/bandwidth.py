"""
HBM/DDR Memory Bandwidth Benchmark
Measures peak and sustained memory bandwidth for GPU HBM and DDR.
"""
import time
from typing import Optional

import numpy as np


class BandwidthBenchmark:
    """Measures GPU memory bandwidth (HBM/DDR)."""

    def __init__(
        self,
        vendor: str = "nvidia",
        iterations: int = 100,
        warmup: int = 10,
        buffer_sizes: Optional[list] = None,
    ):
        self.vendor = vendor
        self.iterations = iterations
        self.warmup = warmup
        self.buffer_sizes = buffer_sizes or [
            256, 512, 1024, 2048, 4096, 8192, 16384,  # MB
        ]
        self.device = self._get_device()

    def _get_device(self):
        try:
            import torch
            if torch.cuda.is_available():
                return torch.device("cuda:0")
        except ImportError:
            pass
        return None

    def run(self) -> dict:
        """Run bandwidth benchmark for all buffer sizes."""
        results = {
            "read_bandwidth": {},
            "write_bandwidth": {},
            "copy_bandwidth": {},
        }

        for size_mb in self.buffer_sizes:
            bw = self._measure_bandwidth(size_mb)
            results["read_bandwidth"][f"{size_mb}MB"] = {
                "value": round(bw["read"], 2),
                "unit": "GB/s",
            }
            results["write_bandwidth"][f"{size_mb}MB"] = {
                "value": round(bw["write"], 2),
                "unit": "GB/s",
            }
            results["copy_bandwidth"][f"{size_mb}MB"] = {
                "value": round(bw["copy"], 2),
                "unit": "GB/s",
            }

        # Peak bandwidth across all sizes
        peak_read = max(
            v["value"] for v in results["read_bandwidth"].values()
        )
        peak_write = max(
            v["value"] for v in results["write_bandwidth"].values()
        )
        peak_copy = max(
            v["value"] for v in results["copy_bandwidth"].values()
        )
        results["peak_read_bandwidth"] = {"value": peak_read, "unit": "GB/s"}
        results["peak_write_bandwidth"] = {"value": peak_write, "unit": "GB/s"}
        results["peak_copy_bandwidth"] = {"value": peak_copy, "unit": "GB/s"}

        return results

    def _measure_bandwidth(self, size_mb: int) -> dict:
        """Measure read/write/copy bandwidth for a given buffer size."""
        import torch

        size_bytes = size_mb * 1024 * 1024
        numel = size_bytes // 4  # float32

        dst = torch.empty(numel, dtype=torch.float32, device=self.device)
        src = torch.randn(numel, dtype=torch.float32, device=self.device)
        torch.cuda.synchronize()

        # Warmup
        for _ in range(self.warmup):
            dst.copy_(src)
        torch.cuda.synchronize()

        # Write bandwidth
        write_times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            dst.zero_()
            torch.cuda.synchronize()
            write_times.append(time.perf_counter() - start)

        # Read bandwidth (copy)
        copy_times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            dst.copy_(src)
            torch.cuda.synchronize()
            copy_times.append(time.perf_counter() - start)

        # Read bandwidth (reduce sum to force read)
        read_times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            _ = src.sum()
            torch.cuda.synchronize()
            read_times.append(time.perf_counter() - start)

        def to_gb(size_b, times):
            median = np.median(times)
            return (size_b / (1024 ** 3)) / median

        return {
            "read": to_gb(size_bytes, read_times),
            "write": to_gb(size_bytes, write_times),
            "copy": to_gb(size_bytes, copy_times),
        }
