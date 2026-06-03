"""
Compute Throughput Benchmark (TFLOPS)
Measures FP16, FP32, FP64, and INT8 TFLOPS via GEMM.
"""
import time
from typing import Optional

import numpy as np


class ComputeBenchmark:
    """Measures GPU compute throughput in TFLOPS."""

    DTYPE_MAP = {
        "fp16": {"torch": "float16", "flops_per_elem": 2},
        "fp32": {"torch": "float32", "flops_per_elem": 2},
        "fp64": {"torch": "float64", "flops_per_elem": 2},
        "bf16": {"torch": "bfloat16", "flops_per_elem": 2},
        "int8": {"torch": "int8", "flops_per_elem": 2},
    }

    def __init__(
        self,
        vendor: str = "nvidia",
        iterations: int = 100,
        warmup: int = 10,
        dtypes: Optional[list] = None,
        matrix_sizes: Optional[list] = None,
    ):
        self.vendor = vendor
        self.iterations = iterations
        self.warmup = warmup
        self.dtypes = dtypes or ["fp16", "fp32"]
        self.matrix_sizes = matrix_sizes or [1024, 2048, 4096, 8192]
        self.device = self._get_device()

    def _get_device(self):
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return None

    def run(self) -> dict:
        """Run compute benchmarks for all dtypes and matrix sizes."""
        results = {}
        for dtype_name in self.dtypes:
            results[dtype_name] = {}
            dtype_info = self.DTYPE_MAP.get(dtype_name)
            if dtype_info is None:
                continue

            for size in self.matrix_sizes:
                tflops = self._measure_gemm_tflops(size, dtype_info)
                results[dtype_name][f"N={size}"] = {
                    "value": round(tflops, 2),
                    "unit": "TFLOPS",
                }

            # Peak TFLOPS for this dtype
            peak = max(
                v["value"] for v in results[dtype_name].values()
            )
            results[f"{dtype_name}_peak"] = {"value": peak, "unit": "TFLOPS"}

        return results

    def _measure_gemm_tflops(self, n: int, dtype_info: dict) -> float:
        """Measure GEMM TFLOPS for a given matrix size and dtype."""
        import torch

        torch_dtype = getattr(torch, dtype_info["torch"])
        # GEMM: C = A @ B where A, B are [N, N]
        # FLOPs = 2 * N^3
        flops = 2 * (n ** 3)

        A = torch.randn(n, n, dtype=torch_dtype, device=self.device)
        B = torch.randn(n, n, dtype=torch_dtype, device=self.device)
        torch.cuda.synchronize()

        # Warmup
        for _ in range(self.warmup):
            if dtype_info["torch"] == "int8":
                C = torch.matmul(A.to(torch.int32), B.to(torch.int32))
            else:
                _ = A @ B
        torch.cuda.synchronize()

        times = []
        for _ in range(self.iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            if dtype_info["torch"] == "int8":
                _ = torch.matmul(A.to(torch.int32), B.to(torch.int32))
            else:
                _ = A @ B
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        median_time = np.median(times)
        tflops = (flops / median_time) / 1e12
        return tflops
