# GPU Benchmark Methodology

## Overview

This document describes the methodology used in GPU Utilization & Bandwidth Benchmark Suite to ensure reproducible, accurate, and comparable results across different GPU architectures.

## Test Environment

All benchmarks are run under the following conditions:

- **OS**: Ubuntu 22.04 LTS or later
- **Driver**: Latest stable driver (NVIDIA 535+ or AMD ROCm 6.0+)
- **CUDA/ROCm**: CUDA 12.x or ROCm 6.x
- **Python**: 3.10+
- **PyTorch**: 2.1.0+

## Measurement Principles

### Timing

- **High-resolution timing**: All measurements use `time.perf_counter()` which provides nanosecond resolution on Linux
- **CUDA synchronization**: `torch.cuda.synchronize()` is called before and after each timed region to ensure all GPU work completes
- **CUDA events**: For kernel launch latency, `torch.cuda.Event(enable_timing=True)` provides GPU-side timing independent of host-GPU transfer overhead

### Warmup

Each benchmark performs a configurable warmup phase (default: 10-1000 iterations) to:
- Prime GPU caches (L1, L2, texture)
- Allow clock frequency boosting (boost clocks take a few iterations to reach peak)
- Ensure memory pages are faulted and mapped

### Statistical Treatment

- **Median** is used as the central tendency measure (robust to outliers)
- **Iterations**: At least 50 iterations per measurement point; 100-10000 for latency measurements
- Results include min/max/percentile spread for reliability assessment

## Bandwidth Testing

### Method

1. Allocate source and destination buffers of specified size
2. Warmup by performing copies
3. Measure read bandwidth via `tensor.sum()` (forces all elements to be read)
4. Measure write bandwidth via `tensor.zero_()` (forces all elements to be written)
5. Measure copy bandwidth via `dst.copy_(src)` (bidirectional)
6. Calculate bandwidth: `size_bytes / median_time`

### Buffer Sizes

Standard buffer sizes range from 256 MB to 32 GB, covering:
- L2 cache-resident working sets (< 50 MB)
- HBM working sets (50 MB - 16 GB)
- Beyond-HBM working sets (> 16 GB)

### Expected Results

| GPU | Peak HBM BW | Typical Achieved |
|-----|-------------|-------------------|
| MI300X | 5,300 GB/s | ~5,210 GB/s |
| H100 SXM | 3,350 GB/s | ~3,200 GB/s |
| A100 SXM | 2,039 GB/s | ~1,950 GB/s |

## Compute Throughput (TFLOPS)

### Method

1. Allocate NxN matrices A and B
2. Measure time for `C = A @ B` (GEMM)
3. Calculate theoretical FLOPs: `2 * N^3` (multiply + accumulate per output element)
4. TFLOPS = FLOPs / (median_time * 1e12)

### Data Types

| Type | Precision | TFLOPS Definition |
|------|-----------|-------------------|
| FP16 | Half | 2 FLOPs per element |
| FP32 | Single | 2 FLOPs per element |
| FP64 | Double | 2 FLOPs per element |
| INT8 | Integer | 2 operations per element |

### Tensor Core Usage

When available (Volta+), PyTorch automatically uses tensor cores for FP16 GEMM. Results reflect actual achieved performance including tensor core acceleration.

## Memory Allocation Latency

### Method

1. For each block size (1KB - 256MB):
   a. Time `torch.empty(N, device='cuda')` for allocation
   b. Time `del tensor` for deallocation
2. Repeat 1000 times and report median

### Interpretation

- Low block sizes (1-16 KB): Measures CUDA memory allocator overhead
- Medium block sizes (256 KB - 4 MB): Mix of allocator + page fault
- Large block sizes (> 64 MB): Dominated by page fault and HBM allocation

## Kernel Launch Latency

### Method

- Empty kernel: Time `torch.cuda.synchronize()` alone measures host-GPU sync overhead
- Elementwise kernel: `C = A + B` measures minimal kernel launch + execution
- Event-based: `torch.cuda.Event` pairs isolate GPU-side timing

### Expected Values

| GPU | Launch Latency |
|-----|----------------|
| MI300X | 2-4 μs |
| H100 | 3-6 μs |
| A100 | 4-8 μs |

## CU Profiling

### Occupancy Estimation

SM occupancy is estimated by comparing achieved TFLOPS against theoretical peak:
```
occupancy = (achieved_tflops / theoretical_peak_tflops) * 100
```

### Cache Behavior

Cache efficiency is estimated by comparing sequential access patterns (high L2 hit rate) against random access patterns (lower hit rate):
```
cache_speedup = random_access_time / sequential_access_time
```

A ratio closer to 1.0 indicates better cache utilization.

## Error Margins

- Bandwidth: ±2% (limited by memory controller scheduling jitter)
- TFLOPS: ±3% (dependent on GEMM kernel implementation)
- Latency measurements: ±10% (subject to OS scheduling and interrupt noise)
- CU occupancy: ±15% (estimated, not hardware counter based)

## Reproducibility

To reproduce results:
1. Set `CUDA_VISIBLE_DEVICES=0` to fix GPU selection
2. Set `CUDA_LAUNCH_BLOCKING=1` for kernel launch latency (disables async)
3. Run on a dedicated node with no other GPU workloads
4. Use `nice -n -20` for consistent CPU scheduling
