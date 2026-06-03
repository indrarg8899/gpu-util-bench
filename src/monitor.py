"""
GPU Utilization Monitor.

Real-time monitoring of GPU SM utilization, memory bandwidth,
temperature, and power draw using rocm-smi.
"""

import subprocess
import time
import csv
from dataclasses import dataclass
from typing import Optional, List, Generator
from pathlib import Path


@dataclass
class GPUStats:
    """Single GPU utilization snapshot."""
    timestamp: float
    gpu_id: int
    gpu_util_pct: float
    mem_util_pct: float
    vram_used_mb: float
    vram_total_mb: float
    temp_edge_c: float
    temp_junction_c: float
    power_w: float
    clock_mhz: float
    mem_clock_mhz: float


class Monitor:
    """AMD GPU utilization monitor using rocm-smi."""

    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self._validate_rocm_smi()

    def _validate_rocm_smi(self):
        try:
            subprocess.run(["rocm-smi", "--version"],
                           capture_output=True, timeout=5, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise RuntimeError("rocm-smi not available. Install ROCm toolkit.")

    def sample(self) -> GPUStats:
        """Take a single GPU utilization sample."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showuse", "--showtemp", "--showpower",
                 "--showmeminfo", "vram", "--showclocks", "--json"],
                capture_output=True, text=True, timeout=10
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("rocm-smi timed out")

        # Parse rocm-smi JSON output
        import json
        data = json.loads(result.stdout) if result.returncode == 0 else {}

        return GPUStats(
            timestamp=time.time(),
            gpu_id=self.gpu_id,
            gpu_util_pct=self._safe_float(data, "GPU use (%)"),
            mem_util_pct=self._safe_float(data, "GPU memory use (%)"),
            vram_used_mb=self._safe_float(data, "VRAM Total Memory (B)") / 1e6,
            vram_total_mb=self._safe_float(data, "VRAM Total Memory (B)") / 1e6,
            temp_edge_c=self._safe_float(data, "Temperature (Sensor edge) (C)"),
            temp_junction_c=self._safe_float(data, "Temperature (Sensor junction) (C)"),
            power_w=self._safe_float(data, "Average Graphics Package Power (W)"),
            clock_mhz=self._safe_float(data, "GFXCLK") / 1e6,
            mem_clock_mhz=self._safe_float(data, "MCLK") / 1e6,
        )

    def _safe_float(self, data: dict, key: str) -> float:
        """Safely extract float from rocm-smi output."""
        try:
            val = data.get(str(self.gpu_id), {}).get(key, 0)
            return float(val) if val else 0.0
        except (ValueError, TypeError):
            return 0.0

    def stream(self, interval: float = 1.0) -> Generator[GPUStats, None, None]:
        """Stream continuous GPU stats."""
        while True:
            yield self.sample()
            time.sleep(interval)

    def log_csv(self, output_path: str, duration_sec: float = 60.0,
                interval: float = 1.0) -> List[GPUStats]:
        """Log GPU stats to CSV file."""
        samples = []
        start = time.time()

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "gpu_id", "gpu_util_pct", "mem_util_pct",
                "vram_used_mb", "temp_edge_c", "temp_junction_c",
                "power_w", "clock_mhz", "mem_clock_mhz"
            ])

            while time.time() - start < duration_sec:
                stats = self.sample()
                samples.append(stats)
                writer.writerow([
                    f"{stats.timestamp:.3f}", stats.gpu_id,
                    f"{stats.gpu_util_pct:.1f}", f"{stats.mem_util_pct:.1f}",
                    f"{stats.vram_used_mb:.0f}", f"{stats.temp_edge_c:.1f}",
                    f"{stats.temp_junction_c:.1f}", f"{stats.power_w:.1f}",
                    f"{stats.clock_mhz:.0f}", f"{stats.mem_clock_mhz:.0f}"
                ])
                f.flush()
                time.sleep(interval)

        return samples

    def summary(self, duration_sec: float = 60.0, interval: float = 1.0) -> dict:
        """Collect stats and return summary."""
        samples = list(self.stream(interval=interval))
        if not samples:
            return {}

        n = len(samples)
        return {
            "samples": n,
            "gpu_util_avg": sum(s.gpu_util_pct for s in samples) / n,
            "gpu_util_max": max(s.gpu_util_pct for s in samples),
            "temp_avg": sum(s.temp_edge_c for s in samples) / n,
            "temp_max": max(s.temp_edge_c for s in samples),
            "power_avg": sum(s.power_w for s in samples) / n,
            "power_max": max(s.power_w for s in samples),
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GPU Utilization Monitor")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=0)
    parser.add_argument("--output", type=str, help="CSV output path")
    args = parser.parse_args()

    mon = Monitor(gpu_id=args.gpu)

    if args.output:
        print(f"Logging GPU {args.gpu} stats to {args.output}...")
        samples = mon.log_csv(args.output,
                              duration_sec=args.duration or 60,
                              interval=args.interval)
        print(f"Collected {len(samples)} samples")
    elif args.duration > 0:
        print(f"Monitoring GPU {args.gpu} for {args.duration}s...")
        start = time.time()
        for stats in mon.stream(interval=args.interval):
            elapsed = time.time() - start
            if elapsed > args.duration:
                break
            print(f"  [{elapsed:6.1f}s] GPU: {stats.gpu_util_pct:5.1f}% | "
                  f"Mem: {stats.mem_util_pct:5.1f}% | "
                  f"Temp: {stats.temp_edge_c:5.1f}C | "
                  f"Power: {stats.power_w:6.1f}W")
    else:
        stats = mon.sample()
        print(f"\nGPU {args.gpu} ({stats.vram_total_mb:.0f} MB VRAM):")
        print(f"  Utilization: {stats.gpu_util_pct:.1f}%")
        print(f"  Memory: {stats.mem_util_pct:.1f}%")
        print(f"  Temp (edge): {stats.temp_edge_c:.1f}C")
        print(f"  Temp (junction): {stats.temp_junction_c:.1f}C")
        print(f"  Power: {stats.power_w:.1f}W")
        print(f"  GFX Clock: {stats.clock_mhz:.0f} MHz")
        print(f"  Mem Clock: {stats.mem_clock_mhz:.0f} MHz")


if __name__ == "__main__":
    main()
