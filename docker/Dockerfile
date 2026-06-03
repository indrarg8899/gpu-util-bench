FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

LABEL maintainer="indrarg8899"
LABEL description="GPU Utilization & Bandwidth Benchmark Suite"
LABEL version="1.0.0"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create results directory
RUN mkdir -p results

# Environment
ENV PYTHONPATH=/app
ENV CUDA_LAUNCH_BLOCKING=0

# Default: run full benchmark
ENTRYPOINT ["python3", "src/benchmark.py"]
CMD ["--config", "configs/compute_test.yml"]
