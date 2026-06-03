"""
Benchmark Visualization
Matplotlib-based plots for benchmark results.
"""
import os
from pathlib import Path
from typing import Optional

import numpy as np


class BenchmarkVisualizer:
    """Generates plots and charts from benchmark results."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_all(self, results: dict, prefix: str = ""):
        """Generate all available plots."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  matplotlib not available, skipping plots")
            return

        self.plt = plt

        if "bandwidth" in results:
            self.plot_bandwidth(results["bandwidth"], prefix)
        if "compute" in results:
            self.plot_compute(results["compute"], prefix)
        if "memory" in results:
            self.plot_memory_latency(results["memory"], prefix)
        if "latency" in results:
            self.plot_kernel_latency(results["latency"], prefix)
        if "profiler" in results:
            self.plot_profiler(results["profiler"], prefix)

        self.plt.close("all")

    def plot_bandwidth(self, data: dict, prefix: str = ""):
        """Plot bandwidth vs buffer size."""
        fig, ax = self.plt.subplots(figsize=(12, 6))

        categories = {}
        for key in ["read_bandwidth", "write_bandwidth", "copy_bandwidth"]:
            if key in data:
                categories[key] = data[key]

        for label, values in categories.items():
            sizes = []
            bw = []
            for k, v in sorted(values.items(), key=lambda x: int(x[0].replace("MB", ""))):
                sizes.append(k)
                bw.append(v["value"])
            ax.plot(sizes, bw, marker="o", label=label.replace("_bandwidth", "").title())

        ax.set_xlabel("Buffer Size")
        ax.set_ylabel("Bandwidth (GB/s)")
        ax.set_title("Memory Bandwidth vs Buffer Size")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = self.output_dir / f"{prefix}bandwidth.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {path}")

    def plot_compute(self, data: dict, prefix: str = ""):
        """Plot TFLOPS vs matrix size for each dtype."""
        fig, ax = self.plt.subplots(figsize=(12, 6))

        for dtype_name, dtype_data in data.items():
            if dtype_name.endswith("_peak"):
                continue
            if not isinstance(dtype_data, dict):
                continue
            sizes = []
            tflops = []
            for k, v in sorted(dtype_data.items()):
                if isinstance(v, dict) and "value" in v:
                    sizes.append(k)
                    tflops.append(v["value"])
            if sizes:
                ax.plot(sizes, tflops, marker="s", label=dtype_name.upper())

        ax.set_xlabel("Matrix Size (N)")
        ax.set_ylabel("TFLOPS")
        ax.set_title("Compute Throughput (GEMM) vs Matrix Size")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = self.output_dir / f"{prefix}compute.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {path}")

    def plot_memory_latency(self, data: dict, prefix: str = ""):
        """Plot memory allocation latency."""
        fig, ax = self.plt.subplots(figsize=(12, 6))

        for key in ["allocation_latency", "deallocation_latency"]:
            if key in data:
                labels = []
                times = []
                for k, v in sorted(data[key].items()):
                    labels.append(k)
                    times.append(v["value"])
                ax.plot(labels, times, marker="^", label=key.replace("_latency", "").title())

        ax.set_xlabel("Block Size")
        ax.set_ylabel("Latency (ns)")
        ax.set_title("GPU Memory Allocation Latency")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = self.output_dir / f"{prefix}memory_latency.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {path}")

    def plot_kernel_latency(self, data: dict, prefix: str = ""):
        """Plot kernel launch latency."""
        fig, ax = self.plt.subplots(figsize=(10, 6))

        labels = []
        times = []
        for key, val in data.items():
            if isinstance(val, dict) and "value" in val:
                labels.append(key)
                times.append(val["value"])

        bars = ax.barh(labels, times, color="steelblue", alpha=0.8)
        ax.set_xlabel("Latency (μs)")
        ax.set_title("Kernel Launch Latency")

        for bar, val in zip(bars, times):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}μs", va="center", fontsize=9)

        path = self.output_dir / f"{prefix}kernel_latency.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {path}")

    def plot_profiler(self, data: dict, prefix: str = ""):
        """Plot profiler results (occupancy, efficiency)."""
        fig, axes = self.plt.subplots(1, 2, figsize=(14, 6))

        # Occupancy pie chart
        if "occupancy" in data:
            occ = data["occupancy"]
            achieved = occ.get("estimated_occupancy_pct", 0)
            remaining = 100 - achieved
            axes[0].pie(
                [achieved, remaining],
                labels=["Achieved", "Theoretical Gap"],
                colors=["#2ecc71", "#e8e8e8"],
                autopct="%1.1f%%",
                startangle=90,
            )
            axes[0].set_title("CU Occupancy")

        # Memory efficiency bar chart
        if "memory_efficiency" in data:
            mem = data["memory_efficiency"]
            labels = []
            bw = []
            for k, v in mem.items():
                labels.append(k)
                bw.append(v.get("bandwidth_gb_s", 0))
            axes[1].bar(labels, bw, color="steelblue", alpha=0.8)
            axes[1].set_ylabel("Bandwidth (GB/s)")
            axes[1].set_title("Memory Bandwidth Efficiency")
            axes[1].tick_params(axis="x", rotation=45)

        fig.tight_layout()
        path = self.output_dir / f"{prefix}profiler.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {path}")

    def plot_comparison(self, baseline: dict, current: dict,
                        baseline_name: str = "Baseline",
                        current_name: str = "Current",
                        prefix: str = ""):
        """Plot comparison between two benchmark runs."""
        fig, ax = self.plt.subplots(figsize=(14, 7))

        categories = list(baseline.keys())
        x = np.arange(len(categories))
        width = 0.35

        b_vals = [baseline[c].get("value", 0) if isinstance(baseline[c], dict) else 0 for c in categories]
        c_vals = [current.get(c, {}).get("value", 0) if isinstance(current.get(c), dict) else 0 for c in categories]

        ax.bar(x - width / 2, b_vals, width, label=baseline_name, color="#3498db", alpha=0.8)
        ax.bar(x + width / 2, c_vals, width, label=current_name, color="#e74c3c", alpha=0.8)

        ax.set_ylabel("Value")
        ax.set_title(f"{baseline_name} vs {current_name}")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = self.output_dir / f"{prefix}comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Comparison plot saved: {path}")
