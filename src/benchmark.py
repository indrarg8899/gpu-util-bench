"""
GPU Benchmark Orchestrator
Main entry point for running all benchmark suites.
"""
import argparse
import json
import time
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .bandwidth import BandwidthBenchmark
from .compute import ComputeBenchmark
from .memory import MemoryLatencyBenchmark
from .latency import KernelLaunchBenchmark
from .profiler import CUProfiler
from .metrics import MetricsCollector
from .visualizer import BenchmarkVisualizer

console = Console()


class GPUBenchmarkOrchestrator:
    """Orchestrates all GPU benchmark suites."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.collector = MetricsCollector(
            output_dir=self.config.get("output_dir", "results")
        )
        self.visualizer = BenchmarkVisualizer(
            output_dir=self.config.get("output_dir", "results")
        )

    def detect_gpu(self) -> str:
        """Auto-detect GPU vendor."""
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                if "AMD" in name or "MI" in name:
                    return "amd"
                return "nvidia"
        except ImportError:
            pass
        return "nvidia"

    def run_all(self) -> dict:
        """Run all configured benchmarks."""
        vendor = self.config.get("gpu_vendor", "auto")
        if vendor == "auto":
            vendor = self.detect_gpu()
        console.print(f"[bold green]Detected GPU vendor: {vendor}[/]")

        results = {}
        iterations = self.config.get("iterations", 100)
        warmup = self.config.get("warmup_iterations", 10)

        suites = self.config.get("suites", ["bandwidth", "compute", "memory", "latency", "profiler"])

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            if "bandwidth" in suites:
                task = progress.add_task("Running bandwidth tests...", total=None)
                bw = BandwidthBenchmark(vendor=vendor, iterations=iterations, warmup=warmup)
                results["bandwidth"] = bw.run()
                progress.update(task, description="[green]Bandwidth tests complete")

            if "compute" in suites:
                task = progress.add_task("Running compute tests...", total=None)
                dtypes = self.config.get("dtypes", ["fp16", "fp32"])
                sizes = self.config.get("matrix_sizes", [4096])
                comp = ComputeBenchmark(
                    vendor=vendor, iterations=iterations, warmup=warmup,
                    dtypes=dtypes, matrix_sizes=sizes,
                )
                results["compute"] = comp.run()
                progress.update(task, description="[green]Compute tests complete")

            if "memory" in suites:
                task = progress.add_task("Running memory latency tests...", total=None)
                mem = MemoryLatencyBenchmark(vendor=vendor, iterations=iterations, warmup=warmup)
                results["memory"] = mem.run()
                progress.update(task, description="[green]Memory latency tests complete")

            if "latency" in suites:
                task = progress.add_task("Running kernel launch latency tests...", total=None)
                lat = KernelLaunchBenchmark(vendor=vendor, iterations=iterations, warmup=warmup)
                results["latency"] = lat.run()
                progress.update(task, description="[green]Kernel launch latency tests complete")

            if "profiler" in suites:
                task = progress.add_task("Running CU profiling...", total=None)
                prof = CUProfiler(vendor=vendor, iterations=iterations)
                results["profiler"] = prof.run()
                progress.update(task, description="[green]CU profiling complete")

        self._print_results_table(results)
        self.collector.save(results)
        self.visualizer.plot_all(results)
        return results

    def _print_results_table(self, results: dict):
        """Print results as a rich table."""
        table = Table(title="GPU Benchmark Results", show_lines=True)
        table.add_column("Test Category", style="cyan", no_wrap=True)
        table.add_column("Metric", style="white")
        table.add_column("Value", style="green", justify="right")
        table.add_column("Unit", style="yellow")

        for category, data in results.items():
            if isinstance(data, dict):
                for metric, value in data.items():
                    if isinstance(value, dict):
                        for k, v in value.items():
                            unit = v.get("unit", "")
                            val = v.get("value", v) if isinstance(v, dict) else v
                            table.add_row(category, f"{metric} / {k}", str(val), unit)
                    else:
                        table.add_row(category, metric, str(value), "")
            else:
                table.add_row(category, "", str(data), "")

        console.print(table)


def main():
    parser = argparse.ArgumentParser(description="GPU Benchmark Suite")
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Override output directory",
    )
    args = parser.parse_args()

    if args.output:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        cfg["output_dir"] = args.output
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
        yaml.dump(cfg, tmp)
        tmp.close()
        args.config = tmp.name

    console.print(Panel.fit(
        "[bold]GPU Utilization & Bandwidth Benchmark Suite[/]\n"
        "Measuring memory, compute, and latency across accelerators",
        title="🔬 gpu-util-bench",
    ))

    orchestrator = GPUBenchmarkOrchestrator(args.config)
    results = orchestrator.run_all()
    console.print("\n[bold green]✓ All benchmarks complete![/]")


if __name__ == "__main__":
    main()
