# Multi-GPU Benchmarking and Monitoring

This repository provides Python scripts to benchmark multi-GPU performance and monitor system usage. It supports:

- **Synthetic Workload:** Large random matrix multiplications on each GPU.
- **Deep-Learning Workload:** Training a dummy PyTorch neural network on each GPU.

Both workloads run for a configurable duration or number of iterations. While the workload runs, the scripts track and log system stats:

- **GPU Utilization & Memory (all GPUs)** using `GPUtil`.
- **CPU Utilization & RAM Usage** using `psutil`.

All data (timestamp, CPU%, RAM%, each GPU’s load% and memory usage) is logged to a CSV file. After execution, time-series plots (matplotlib) are generated for CPU usage and for each GPU’s utilization and memory usage.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
