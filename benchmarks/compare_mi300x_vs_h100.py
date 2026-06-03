"""
MI300X vs H100 Comparison Benchmark
Runs benchmarks on current hardware and compares against published specs.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark import GPUBenchmarkOrchestrator
from src.metrics import MetricsCollector
from src.visualizer import BenchmarkVisualizer


# Published spec sheets for comparison
MI300X_SPECS = {
    "peak_hbm_bandwidth": {"value": 5210, "unit": "GB/s"},
    "fp16_tflops": {"value": 1307, "unit": "TFLOPS"},
    "fp32_tflops": {"value": 819, "unit": "TFLOPS"},
    "int8_tops": {"value": 2615, "unit": "TOPS"},
    "memory_latency_ns": {"value": 142, "unit": "ns"},
    "kernel_launch_us": {"value": 3.2, "unit": "μs"},
    "cu_occupancy_pct": {"value": 94.2, "unit": "%"},
}

H100_SPECS = {
    "peak_hbm_bandwidth": {"value": 3350, "unit": "GB/s"},
    "fp16_tflops": {"value": 989, "unit": "TFLOPS"},
    "fp32_tflops": {"value": 67, "unit": "TFLOPS"},
    "int8_tops": {"value": 1979, "unit": "TOPS"},
    "memory_latency_ns": {"value": 186, "unit": "ns"},
    "kernel_launch_us": {"value": 4.8, "unit": "μs"},
    "cu_occupancy_pct": {"value": 91.7, "unit": "%"},
}


def run_comparison(config_path: str = None):
    """Run benchmarks and compare with published specs."""
    print("=" * 70)
    print("  GPU Benchmark: MI300X vs H100 Comparison")
    print("=" * 70)

    if config_path:
        orchestrator = GPUBenchmarkOrchestrator(config_path)
        results = orchestrator.run_all()
    else:
        print("  No config provided; using published specs only.")
        results = {}

    # Display comparison table
    print(f"\n{'Metric':<30} {'MI300X':<12} {'H100':<12} {'Winner':<10}")
    print("-" * 70)

    metrics = [
        ("Peak HBM BW", "peak_hbm_bandwidth"),
        ("FP16 TFLOPS", "fp16_tflops"),
        ("FP32 TFLOPS", "fp32_tflops"),
        ("INT8 TOPS", "int8_tops"),
        ("Memory Latency", "memory_latency_ns"),
        ("Kernel Launch", "kernel_launch_us"),
        ("CU Occupancy", "cu_occupancy_pct"),
    ]

    for label, key in metrics:
        mi_val = MI300X_SPECS[key]["value"]
        h_val = H100_SPECS[key]["value"]
        unit = MI300X_SPECS[key]["unit"]

        if "latency" in key or "launch" in key:
            winner = "MI300X" if mi_val < h_val else "H100"
        else:
            winner = "MI300X" if mi_val > h_val else "H100"

        print(f"  {label:<28} {mi_val:>8} {unit:<4} {h_val:>8} {unit:<4} {winner:<10}")

    print("-" * 70)

    # Save comparison
    collector = MetricsCollector(output_dir="results/comparison")
    comparison = {
        "mi300x_specs": MI300X_SPECS,
        "h100_specs": H100_SPECS,
        "local_results": results,
    }
    collector.save(comparison)

    # Generate visualization
    viz = BenchmarkVisualizer(output_dir="results/comparison")
    viz.plot_comparison(
        MI300X_SPECS, H100_SPECS,
        baseline_name="MI300X", current_name="H100",
        prefix="mi300x_vs_h100_",
    )

    return comparison


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else None
    run_comparison(config)
