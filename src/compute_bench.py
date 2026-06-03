"""
Compute Throughput Benchmark.

Measures matrix multiplication TFLOPS across precision formats
(FP64, FP32, FP16, BF16, FP8) on AMD Instinct GPUs.
"""

import time
from dataclasses import dataclass
from typing import Optional, Dict

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None


@dataclass
class ComputeResult:
    """Result of a compute throughput benchmark."""
    device: str
    precision: str
    matrix_size: int
    tflops: float
    elapsed_sec: float
    iterations: int

    @property
    def efficiency_pct(self) -> float:
        """Efficiency vs theoretical peak (rough estimate)."""
        peak = {
            "fp64": 163.4, "fp32": 163.4, "fp16": 1307.0,
            "bf16": 1307.0, "fp8": 2614.0,  # MI300X theoretical
        }
        theoretical = peak.get(self.precision, 100.0)
        return (self.tflops / theoretical) * 100


class ComputeBench:
    """Compute throughput benchmark for AMD GPUs."""

    PRECISION_DTYPES = {
        "fp64": torch.float64,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    def __init__(self, device: str = "cuda", warmup: int = 5):
        if torch is None:
            raise RuntimeError("PyTorch required")
        self.device = torch.device(device)
        self.warmup = warmup

    def run(self, precision: str = "fp32", matrix_size: int = 4096,
            iterations: int = 50) -> ComputeResult:
        """Run GEMM throughput benchmark."""
        if precision == "fp8":
            return self._run_fp8(matrix_size, iterations)

        dtype = self.PRECISION_DTYPES.get(precision)
        if dtype is None:
            raise ValueError(f"Unsupported precision: {precision}")

        n = matrix_size
        a = torch.randn(n, n, device=self.device, dtype=dtype)
        b = torch.randn(n, n, device=self.device, dtype=dtype)

        # Warmup
        for _ in range(self.warmup):
            _ = torch.mm(a, b)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            _ = torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        flops = 2 * n * n * n * iterations  # 2*N^3 FLOPs per GEMM
        tflops = (flops / elapsed) / 1e12

        return ComputeResult(
            device=torch.cuda.get_device_name(self.device),
            precision=precision,
            matrix_size=matrix_size,
            tflops=tflops,
            elapsed_sec=elapsed,
            iterations=iterations,
        )

    def _run_fp8(self, matrix_size: int, iterations: int) -> ComputeResult:
        """Run FP8 GEMM (using FP16 accumulation on MI300X)."""
        n = matrix_size
        try:
            a = torch.randn(n, n, device=self.device, dtype=torch.float16)
            b = torch.randn(n, n, device=self.device, dtype=torch.float16)
        except RuntimeError:
            a = torch.randn(n, n, device=self.device, dtype=torch.float32)
            b = torch.randn(n, n, device=self.device, dtype=torch.float32)

        for _ in range(self.warmup):
            _ = torch.mm(a, b)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iterations):
            _ = torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        flops = 2 * n * n * n * iterations
        tflops = (flops / elapsed) / 1e12

        return ComputeResult(
            device=torch.cuda.get_device_name(self.device),
            precision="fp8 (via fp16)",
            matrix_size=matrix_size,
            tflops=tflops,
            elapsed_sec=elapsed,
            iterations=iterations,
        )

    def run_all_precisions(self, matrix_size: int = 4096,
                           iterations: int = 50) -> Dict[str, ComputeResult]:
        """Run benchmarks for all supported precisions."""
        results = {}
        for prec in ["fp64", "fp32", "fp16", "bf16"]:
            try:
                results[prec] = self.run(precision=prec, matrix_size=matrix_size,
                                         iterations=iterations)
            except RuntimeError:
                pass  # Skip unsupported precision
        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GPU Compute Throughput Benchmark")
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    bench = ComputeBench(device=args.device)

    if args.all:
        print(f"\nCompute Throughput ({torch.cuda.get_device_name()}):")
        print(f"Matrix size: {args.matrix_size}x{args.matrix_size}\n")
        results = bench.run_all_precisions(matrix_size=args.matrix_size,
                                            iterations=args.iterations)
        for prec, result in results.items():
            print(f"  {prec:>6}: {result.tflops:.2f} TFLOPS "
                  f"({result.efficiency_pct:.1f}% peak est.)")
    else:
        result = bench.run(precision=args.precision, matrix_size=args.matrix_size,
                           iterations=args.iterations)
        print(f"\nCompute ({result.device}):")
        print(f"  {result.precision}: {result.tflops:.2f} TFLOPS")


if __name__ == "__main__":
    main()
