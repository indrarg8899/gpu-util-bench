"""
GPU Memory Bandwidth Benchmark.

Measures HBM2/HBM3 bandwidth using various access patterns
on AMD Instinct GPUs.
"""

import time
from dataclasses import dataclass
from typing import Optional, List

try:
    import torch
except ImportError:
    torch = None


@dataclass
class BandwidthResult:
    """Result of a memory bandwidth benchmark."""
    device: str
    size_mb: float
    read_gbps: float
    write_gbps: float
    copy_gbps: float
    iterations: int
    elapsed_sec: float

    @property
    def avg_bandwidth(self) -> float:
        return (self.read_gbps + self.write_gbps + self.copy_gbps) / 3


@dataclass
class LatencyResult:
    """Kernel launch latency measurement."""
    device: str
    min_us: float
    max_us: float
    avg_us: float
    p50_us: float
    p99_us: float
    iterations: int


class MemoryBandwidthBench:
    """Memory bandwidth benchmark for AMD GPUs."""

    def __init__(self, device: str = "cuda", warmup: int = 10):
        if torch is None:
            raise RuntimeError("PyTorch required. Install with ROCm support.")
        self.device = torch.device(device)
        self.warmup = warmup

    def run(self, size_mb: float = 1024, iterations: int = 100) -> BandwidthResult:
        """Run full memory bandwidth benchmark."""
        elements = int(size_mb * 1024 * 1024 / 4)  # float32 elements

        # Warmup
        for _ in range(self.warmup):
            a = torch.randn(elements, device=self.device, dtype=torch.float32)
            b = torch.empty_like(a)
            b.copy_(a)
            del a, b
        torch.cuda.synchronize()

        data_bytes = elements * 4

        # Write benchmark (host to device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            a = torch.randn(elements, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
        write_time = time.perf_counter() - start

        # Read benchmark (device to host)
        a = torch.randn(elements, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            _ = a.cpu()
        torch.cuda.synchronize()
        read_time = time.perf_counter() - start

        # Copy benchmark (device to device)
        b = torch.empty_like(a)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            b.copy_(a)
        torch.cuda.synchronize()
        copy_time = time.perf_counter() - start

        total_time = write_time + read_time + copy_time

        device_name = torch.cuda.get_device_name(self.device)
        write_gbps = (data_bytes * iterations / write_time) / 1e9
        read_gbps = (data_bytes * iterations / read_time) / 1e9
        copy_gbps = (data_bytes * iterations / copy_time) / 1e9

        return BandwidthResult(
            device=device_name,
            size_mb=size_mb,
            read_gbps=read_gbps,
            write_gbps=write_gbps,
            copy_gbps=copy_gbps,
            iterations=iterations,
            elapsed_sec=total_time,
        )

    def run_sweep(self, sizes_mb: List[float] = None,
                  iterations: int = 100) -> List[BandwidthResult]:
        """Run bandwidth benchmark across multiple sizes."""
        if sizes_mb is None:
            sizes_mb = [16, 64, 256, 1024, 4096, 16384]

        results = []
        for size in sizes_mb:
            result = self.run(size_mb=size, iterations=iterations)
            results.append(result)
            print(f"  {size:>8} MB: R={result.read_gbps:.1f} "
                  f"W={result.write_gbps:.1f} C={result.copy_gbps:.1f} GB/s")
        return results

    def measure_latency(self, iterations: int = 1000) -> LatencyResult:
        """Measure kernel launch latency."""
        a = torch.randn(1024, device=self.device)

        # Warmup
        for _ in range(100):
            _ = a + a
        torch.cuda.synchronize()

        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = a + a
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1e6)

        latencies.sort()
        return LatencyResult(
            device=torch.cuda.get_device_name(self.device),
            min_us=latencies[0],
            max_us=latencies[-1],
            avg_us=sum(latencies) / len(latencies),
            p50_us=latencies[len(latencies) // 2],
            p99_us=latencies[int(len(latencies) * 0.99)],
            iterations=iterations,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GPU Memory Bandwidth Benchmark")
    parser.add_argument("--size-mb", type=float, default=1024)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--latency", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    bench = MemoryBandwidthBench(device=args.device)

    if args.latency:
        result = bench.measure_latency(iterations=args.iterations)
        print(f"\nKernel Latency ({result.device}):")
        print(f"  Min: {result.min_us:.1f} us")
        print(f"  Avg: {result.avg_us:.1f} us")
        print(f"  P50: {result.p50_us:.1f} us")
        print(f"  P99: {result.p99_us:.1f} us")
    elif args.sweep:
        print("\nMemory Bandwidth Sweep:")
        bench.run_sweep(iterations=args.iterations)
    else:
        result = bench.run(size_mb=args.size_mb, iterations=args.iterations)
        print(f"\nMemory Bandwidth ({result.device}):")
        print(f"  Size: {result.size_mb:.0f} MB")
        print(f"  Read:  {result.read_gbps:.1f} GB/s")
        print(f"  Write: {result.write_gbps:.1f} GB/s")
        print(f"  Copy:  {result.copy_gbps:.1f} GB/s")


if __name__ == "__main__":
    main()
