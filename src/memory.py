"""
GPU Memory Allocation Latency Benchmark
Measures latency for GPU memory allocation at various block sizes.
"""
import time
from typing import Optional

import numpy as np


class MemoryLatencyBenchmark:
    """Measures GPU memory allocation latency."""

    def __init__(
        self,
        vendor: str = "nvidia",
        iterations: int = 1000,
        warmup: int = 100,
        block_sizes: Optional[list] = None,
    ):
        self.vendor = vendor
        self.iterations = iterations
        self.warmup = warmup
        self.block_sizes = block_sizes or [
            1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144,  # KB
        ]

    def run(self) -> dict:
        """Run memory allocation latency tests."""
        import torch

        results = {"allocation_latency": {}, "deallocation_latency": {}}

        for size_kb in self.block_sizes:
            alloc_times = []
            dealloc_times = []
            size_bytes = size_kb * 1024
            num_elements = size_bytes // 4  # float32

            # Warmup
            for _ in range(self.warmup):
                t = torch.empty(num_elements, dtype=torch.float32, device="cuda:0")
                del t
            torch.cuda.synchronize()

            # Measure allocation latency
            for _ in range(self.iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                t = torch.empty(num_elements, dtype=torch.float32, device="cuda:0")
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                alloc_times.append(elapsed)
                del t

            # Measure deallocation latency
            for _ in range(self.iterations):
                t = torch.empty(num_elements, dtype=torch.float32, device="cuda:0")
                torch.cuda.synchronize()
                start = time.perf_counter()
                del t
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                dealloc_times.append(elapsed)

            label = self._format_size(size_kb)
            alloc_ns = np.median(alloc_times) * 1e9
            dealloc_ns = np.median(dealloc_times) * 1e9

            results["allocation_latency"][label] = {
                "value": round(alloc_ns, 2),
                "unit": "ns",
            }
            results["deallocation_latency"][label] = {
                "value": round(dealloc_ns, 2),
                "unit": "ns",
            }

        # Summary stats
        alloc_values = [
            v["value"] for v in results["allocation_latency"].values()
        ]
        results["mean_allocation_latency"] = {
            "value": round(float(np.mean(alloc_values)), 2),
            "unit": "ns",
        }
        results["max_allocation_latency"] = {
            "value": round(float(np.max(alloc_values)), 2),
            "unit": "ns",
        }

        return results

    @staticmethod
    def _format_size(size_kb: int) -> str:
        if size_kb >= 1024:
            return f"{size_kb // 1024}MB"
        return f"{size_kb}KB"
