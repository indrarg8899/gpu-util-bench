# GPU Utilization Bench

## Memory Bandwidth Tests

### HBM Bandwidth Measurement
```bash
gpu-util-bench --bench memory-bandwidth --size-mb 1024
```

### Bandwidth Sweep (16MB to 16GB)
```bash
gpu-util-bench --bench memory-bandwidth --sweep
```

### Expected Results (MI300X)
- **HBM3 Read**: ~5,300 GB/s theoretical, ~4,500 GB/s measured
- **HBM3 Write**: ~5,300 GB/s theoretical, ~4,200 GB/s measured
- **HBM3 Copy**: ~5,300 GB/s theoretical, ~4,000 GB/s measured

## Compute Throughput Tests

### FP32 GEMM
```bash
gpu-util-bench --bench compute --precision fp32 --matrix-size 8192
```

### All Precisions
```bash
gpu-util-bench --bench compute --all
```

### Expected Results (MI300X)
| Precision | Theoretical TFLOPS | Typical Achievable |
|-----------|--------------------|--------------------|
| FP64      | 163.4             | ~140-155           |
| FP32      | 163.4             | ~150-160           |
| FP16      | 1,307             | ~1,000-1,200       |
| BF16      | 1,307             | ~1,000-1,200       |
| FP8       | 2,614             | ~1,800-2,200       |

## PCIe Bandwidth Tests

```bash
gpu-util-bench --bench pcie --size-mb 1024
```

## GPU Monitoring

```bash
# Real-time
gpu-monitor --interval 0.5

# CSV logging
gpu-monitor --output stats.csv --duration 300
```

## Tips for Accurate Results

1. **Warm up** the GPU before benchmarking (run 5-10 iterations first)
2. **Use pinned memory** for H2D/D2H tests for better PCIe bandwidth
3. **Disable ECC** if possible for higher peak bandwidth
4. **Run on a cool GPU** — thermal throttling affects results
5. **Use `GPU_MAX_HW_QUEUES=1`** for deterministic single-stream results
