[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "gpu-util-bench"
version = "1.0.0"
description = "GPU utilization and memory bandwidth benchmark suite for AMD Instinct"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.9"
authors = [{name = "indrarg8899"}]
keywords = ["gpu", "benchmark", "rocm", "amd", "mi300x", "bandwidth"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Topic :: System :: Benchmark",
]

[project.scripts]
gpu-util-bench = "src.memory_bandwidth:main"
gpu-monitor = "src.monitor:main"
gpu-pcie-bench = "src.pcie_bench:main"

[project.optional-dependencies]
dev = ["pytest", "flake8"]
