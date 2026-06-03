# GPU Utilization & Memory Bandwidth Benchmark Suite

![ROCm](https://img.shields.io/badge/ROCm-6.x+-ed1c24?logo=amd&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-AMD%20Instinct-blue)
![Stars](https://img.shields.io/github/stars/indrarg8899/gpu-util-bench?style=social)

> Comprehensive GPU utilization and memory bandwidth benchmarking suite for AMD Instinct accelerators (MI250X, MI300X).

## Features

- **Memory Bandwidth Testing** — HBM2/HBM3 bandwidth measurement with configurable block sizes and access patterns
- **Compute Throughput Benchmarks** — FP64, FP32, FP16, BF16, FP8 throughput measurement
- **GPU Utilization Monitoring** — Real-time SM occupancy, memory controller load, power draw tracking
- **PCIe Bandwidth Testing** — Host-device transfer benchmarks (H2D, D2H, D2D)
- **Concurrent Workload Stress** — Multi-stream, multi-kernel scheduling efficiency tests
- **Thermal & Power Profiling** — Temperature and power envelope characterization
- **Automated Report Generation** — JSON/HTML/CSV benchmark reports with comparisons

## Installation

```bash
git clone https://github.com/indrarg8899/gpu-util-bench.git
cd gpu-util-bench
pip install -e .
```

## Usage

### Run Full Benchmark Suite

```bash
gpu-util-bench --full
```

### Run Specific Benchmarks

```bash
# Memory bandwidth only
gpu-util-bench --bench memory-bandwidth

# Compute throughput (all precisions)
gpu-util-bench --bench compute-throughput --precision fp32 fp16 bf16 fp8

# PCIe transfers
gpu-util-bench --bench pcie-bandwidth

# Stress test
gpu-util-bench --bench stress --duration 300 --streams 16
```

### Monitor GPU Utilization

```bash
# Real-time monitoring
gpu-monitor --interval 0.5 --gpu 0

# Log to CSV
gpu-monitor --interval 1 --output gpu_stats.csv --duration 60
```

### Python API

```python
from gpu_util_bench import MemoryBandwidthBench, ComputeBench, Monitor

# Memory bandwidth test
bench = MemoryBandwidthBench()
result = bench.run(size_mb=1024, iterations=100)
print(f"Read BW: {result.read_gbps:.1f} GB/s")
print(f"Write BW: {result.write_gbps:.1f} GB/s")
print(f"Copy BW: {result.copy_gbps:.1f} GB/s")

# Compute throughput
compute = ComputeBench()
for prec in ["fp32", "fp16", "bf16", "fp8"]:
    r = compute.run(precision=prec, matrix_size=4096)
    print(f"{prec}: {r.tflops:.2f} TFLOPS")

# GPU monitoring
mon = Monitor(gpu_id=0)
for stats in mon.stream(interval=1.0):
    print(f"GPU: {stats.gpu_util}% | Mem: {stats.mem_util}% | Temp: {stats.temp_c}C | Power: {stats.power_w}W")
```

## Benchmark Suite

| Benchmark | Description | Metric |
|-----------|-------------|--------|
| `memory-bandwidth` | HBM read/write/copy | GB/s |
| `compute-throughput` | Matrix multiply (GEMM) | TFLOPS |
| `pcie-bandwidth` | Host↔Device transfer | GB/s |
| `latency` | Kernel launch latency | microseconds |
| `concurrent` | Multi-stream scheduling | ops/sec |
| `stress` | Sustained workload | stability |
| `thermal` | Thermal characterization | °C |

## Requirements

- AMD Instinct GPU (MI250X or MI300X recommended)
- ROCm 6.0+
- PyTorch with ROCm support
- rocm-smi (for monitoring)

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- AMD ROCm profiling tools
- PyTorch performance benchmarks
- AMD Developer Cloud for MI300X access
