"""
PCIe Bandwidth Benchmark.

Measures host-to-device (H2D), device-to-host (D2H),
and device-to-device (D2D) transfer bandwidth.
"""

import time
from dataclasses import dataclass
from typing import List, Optional

try:
    import torch
except ImportError:
    torch = None


@dataclass
class PCIeResult:
    """PCIe bandwidth measurement result."""
    direction: str  # "H2D", "D2H", "D2D"
    size_mb: float
    bandwidth_gbps: float
    iterations: int
    elapsed_sec: float


class PCIeBandwidthBench:
    """PCIe transfer bandwidth benchmark."""

    def __init__(self, device: str = "cuda", warmup: int = 5):
        if torch is None:
            raise RuntimeError("PyTorch required")
        self.device = torch.device(device)
        self.warmup = warmup

    def run(self, direction: str = "H2D", size_mb: float = 1024,
            iterations: int = 100) -> PCIeResult:
        """Run PCIe bandwidth benchmark."""
        elements = int(size_mb * 1024 * 1024 / 4)
        data_bytes = elements * 4

        if direction == "H2D":
            return self._bench_h2d(elements, data_bytes, size_mb, iterations)
        elif direction == "D2H":
            return self._bench_d2h(elements, data_bytes, size_mb, iterations)
        elif direction == "D2D":
            return self._bench_d2d(elements, data_bytes, size_mb, iterations)
        else:
            raise ValueError(f"Unknown direction: {direction}")

    def _bench_h2d(self, elements, data_bytes, size_mb, iterations):
        """Benchmark host-to-device transfers."""
        host_tensor = torch.randn(elements)
        device_tensor = torch.empty(elements, device=self.device)

        # Warmup
        for _ in range(self.warmup):
            device_tensor.copy_(host_tensor)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iterations):
            device_tensor.copy_(host_tensor)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        return PCIeResult(
            direction="H2D", size_mb=size_mb,
            bandwidth_gbps=(data_bytes * iterations / elapsed) / 1e9,
            iterations=iterations, elapsed_sec=elapsed,
        )

    def _bench_d2h(self, elements, data_bytes, size_mb, iterations):
        """Benchmark device-to-host transfers."""
        host_tensor = torch.empty(elements)
        device_tensor = torch.randn(elements, device=self.device)

        for _ in range(self.warmup):
            host_tensor.copy_(device_tensor)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iterations):
            host_tensor.copy_(device_tensor)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        return PCIeResult(
            direction="D2H", size_mb=size_mb,
            bandwidth_gbps=(data_bytes * iterations / elapsed) / 1e9,
            iterations=iterations, elapsed_sec=elapsed,
        )

    def _bench_d2d(self, elements, data_bytes, size_mb, iterations):
        """Benchmark device-to-device copies."""
        src = torch.randn(elements, device=self.device)
        dst = torch.empty(elements, device=self.device)

        for _ in range(self.warmup):
            dst.copy_(src)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iterations):
            dst.copy_(src)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        return PCIeResult(
            direction="D2D", size_mb=size_mb,
            bandwidth_gbps=(data_bytes * iterations / elapsed) / 1e9,
            iterations=iterations, elapsed_sec=elapsed,
        )

    def run_all(self, size_mb: float = 1024,
                iterations: int = 100) -> List[PCIeResult]:
        """Run all transfer direction benchmarks."""
        results = []
        for direction in ["H2D", "D2H", "D2D"]:
            result = self.run(direction=direction, size_mb=size_mb,
                              iterations=iterations)
            results.append(result)
            print(f"  {direction}: {result.bandwidth_gbps:.1f} GB/s")
        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PCIe Bandwidth Benchmark")
    parser.add_argument("--size-mb", type=float, default=1024)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--direction", choices=["H2D", "D2H", "D2D", "all"],
                        default="all")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    bench = PCIeBandwidthBench(device=args.device)

    if args.direction == "all":
        print(f"\nPCIe Bandwidth ({torch.cuda.get_device_name()}):")
        print(f"Size: {args.size_mb:.0f} MB\n")
        bench.run_all(size_mb=args.size_mb, iterations=args.iterations)
    else:
        result = bench.run(direction=args.direction, size_mb=args.size_mb,
                           iterations=args.iterations)
        print(f"\nPCIe {args.direction}: {result.bandwidth_gbps:.1f} GB/s")


if __name__ == "__main__":
    main()
