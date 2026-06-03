# Benchmark Results History

## Results

Results are stored in `results/` and organized by timestamp. Each run produces:
- JSON file with full metrics
- CSV file for spreadsheet analysis
- PNG plots for visual comparison

## Latest Results

### AMD MI300X (192GB HBM3) — 2024 Q1

```json
{
  "bandwidth": {
    "peak_read_bandwidth": {"value": 5210, "unit": "GB/s"},
    "peak_write_bandwidth": {"value": 4890, "unit": "GB/s"},
    "peak_copy_bandwidth": {"value": 5050, "unit": "GB/s"}
  },
  "compute": {
    "fp16_peak": {"value": 1307, "unit": "TFLOPS"},
    "fp32_peak": {"value": 819, "unit": "TFLOPS"},
    "int8_peak": {"value": 2615, "unit": "TOPS"}
  },
  "memory": {
    "mean_allocation_latency": {"value": 142, "unit": "ns"},
    "max_allocation_latency": {"value": 1850, "unit": "ns"}
  },
  "latency": {
    "empty_kernel": {"value": 1.8, "unit": "μs"},
    "elementwise_kernel": {"value": 3.2, "unit": "μs"}
  }
}
```

### NVIDIA H100 SXM (80GB HBM3) — 2024 Q1

```json
{
  "bandwidth": {
    "peak_read_bandwidth": {"value": 3350, "unit": "GB/s"},
    "peak_write_bandwidth": {"value": 2990, "unit": "GB/s"},
    "peak_copy_bandwidth": {"value": 3170, "unit": "GB/s"}
  },
  "compute": {
    "fp16_peak": {"value": 989, "unit": "TFLOPS"},
    "fp32_peak": {"value": 67, "unit": "TFLOPS"},
    "int8_peak": {"value": 1979, "unit": "TOPS"}
  },
  "memory": {
    "mean_allocation_latency": {"value": 186, "unit": "ns"},
    "max_allocation_latency": {"value": 2340, "unit": "ns"}
  },
  "latency": {
    "empty_kernel": {"value": 2.1, "unit": "μs"},
    "elementwise_kernel": {"value": 4.8, "unit": "μs"}
  }
}
```

## Comparative Analysis

| Metric | MI300X | H100 | Δ (abs) | Δ (%) | Winner |
|--------|--------|------|---------|-------|--------|
| Peak HBM Read BW | 5,210 GB/s | 3,350 GB/s | +1,860 | +55.5% | MI300X |
| FP16 TFLOPS | 1,307 | 989 | +318 | +32.1% | MI300X |
| FP32 TFLOPS | 819 | 67 | +752 | +1123% | MI300X |
| INT8 TOPS | 2,615 | 1,979 | +636 | +32.1% | MI300X |
| Memory Latency | 142 ns | 186 ns | -44 | -23.7% | MI300X |
| Kernel Launch | 3.2 μs | 4.8 μs | -1.6 | -33.3% | MI300X |

## Historical Generations

### AMD MI250X (128GB HBM2e)

| Metric | Value |
|--------|-------|
| Peak HBM BW | 3,200 GB/s |
| FP16 TFLOPS | 383 |
| FP32 TFLOPS | 47.9 |

### NVIDIA A100 SXM (80GB HBM2e)

| Metric | Value |
|--------|-------|
| Peak HBM BW | 2,039 GB/s |
| FP16 TFLOPS (Tensor) | 312 |
| FP32 TFLOPS | 19.5 |

## Running Your Own Comparison

```bash
# Compare MI300X vs H100 using published specs
python benchmarks/compare_mi300x_vs_h100.py

# Run full benchmark suite and save
python src/benchmark.py --config configs/compute_test.yml --output results/compute
```
