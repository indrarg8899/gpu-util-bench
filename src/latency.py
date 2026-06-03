"""
Kernel Launch Latency Benchmark
Measures overhead for launching GPU kernels.
"""
import time
from typing import Optional

import numpy as np


class KernelLaunchBenchmark:
    """Measures GPU kernel launch latency."""

    def __init__(
        self,
        vendor: str = "nvidia",
        iterations: int = 10000,
        warmup: int = 1000,
    ):
        self.vendor = vendor
        self.iterations = iterations
        self.warmup = warmup

    def run(self) -> dict:
        """Measure kernel launch latency."""
        import torch

        results = {}
        device = torch.device("cuda:0")

        # 1. Empty kernel launch latency
        results["empty_kernel"] = self._measure_empty_kernel(device)

        # 2. Simple elementwise kernel launch
        results["elementwise_kernel"] = self._measure_elementwise(device)

        # 3. Kernel launch with increasing grid sizes
        results["grid_scaling"] = self._measure_grid_scaling(device)

        # 4. CUDA event-based measurement (more precise)
        results["event_based_latency"] = self._measure_event_based(device)

        # Summary
        latencies = [v.get("value", 0) for k, v in results.items() if k != "grid_scaling"]
        results["mean_kernel_latency"] = {
            "value": round(float(np.mean(latencies)), 2),
            "unit": "μs",
        }

        return results

    def _measure_empty_kernel(self, device) -> dict:
        """Measure time to launch an empty CUDA operation."""
        # Warmup
        for _ in range(self.warmup):
            torch.cuda.synchronize()
        torch.cuda.synchronize()

        times = []
        for _ in range(self.iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        median_us = np.median(times) * 1e6
        return {"value": round(median_us, 2), "unit": "μs"}

    def _measure_elementwise(self, device) -> dict:
        """Measure time to launch a simple elementwise kernel."""
        n = 1024
        a = torch.ones(n, device=device)
        b = torch.ones(n, device=device)

        # Warmup
        for _ in range(self.warmup):
            torch.cuda.synchronize()
            c = a + b
            torch.cuda.synchronize()

        times = []
        for _ in range(self.iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            c = a + b
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        median_us = np.median(times) * 1e6
        return {"value": round(median_us, 2), "unit": "μs"}

    def _measure_grid_scaling(self, device) -> dict:
        """Measure launch overhead across different tensor sizes."""
        sizes = [256, 1024, 4096, 16384, 65536, 262144]
        results = {}
        for n in sizes:
            a = torch.ones(n, device=device)
            b = torch.ones(n, device=device)

            for _ in range(self.warmup):
                torch.cuda.synchronize()
                c = a + b
                torch.cuda.synchronize()

            times = []
            for _ in range(self.iterations // 2):
                torch.cuda.synchronize()
                start = time.perf_counter()
                c = a + b
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            results[f"N={n}"] = {
                "value": round(float(np.median(times)) * 1e6, 2),
                "unit": "μs",
            }
        return results

    def _measure_event_based(self, device) -> dict:
        """Use CUDA events for precise timing."""
        torch.cuda.synchronize()
        for _ in range(self.warmup):
            torch.cuda.synchronize()
        torch.cuda.synchronize()

        times = []
        for _ in range(self.iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            torch.cuda.synchronize()
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            times.append(elapsed_ms * 1000)  # Convert to μs

        median_us = np.median(times)
        return {"value": round(median_us, 2), "unit": "μs"}
