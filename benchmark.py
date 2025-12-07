#!/usr/bin/env python3
"""
benchmark.py: Main script to run GPU benchmarks (synthetic or deep-learning workloads).
Logs CPU/GPU usage to CSV and generates plots.

Dependencies (see requirements.txt): torch, psutil, GPUtil, matplotlib
"""
import time
import threading
import argparse
import pandas as pd
import sys

from monitor import get_cpu_ram, get_gpu_stats
from synthetic_workload import synthetic_compute_loop
from dl_workload import dl_training_loop
from plotter import plot_results

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU Benchmarking")
    subparsers = parser.add_subparsers(dest="mode", required=True,
                                       help="Choose 'synthetic' or 'dl' workload mode")

    # Synthetic workload parameters
    syn_parser = subparsers.add_parser('synthetic', help='Run synthetic matrix multiplication workload')
    syn_parser.add_argument('--matrix-size', type=int, default=1000,
                            help='Size of the square matrices (default: 1000)')
    syn_parser.add_argument('--iterations', type=int, default=100,
                            help='Number of matrix multiplications to perform')
    syn_parser.add_argument('--duration', type=int, default=60,
                            help='Maximum duration (seconds) to run the test')
    syn_parser.add_argument('--interval', type=float, default=1.0,
                            help='Logging interval in seconds for sampling stats')

    # Deep-learning workload parameters
    dl_parser = subparsers.add_parser('dl', help='Run dummy deep-learning workload')
    dl_parser.add_argument('--batch-size', type=int, default=32,
                            help='Batch size for training')
    dl_parser.add_argument('--epochs', type=int, default=5,
                            help='Number of epochs (full passes) to run')
    dl_parser.add_argument('--duration', type=int, default=60,
                            help='Maximum duration (seconds) to run the test')
    dl_parser.add_argument('--interval', type=float, default=1.0,
                            help='Logging interval in seconds for sampling stats')

    return parser.parse_args()

def main():
    args = parse_args()
    gpu_count = 0
    try:
        import torch
        if not torch.cuda.is_available():
            print("CUDA not available. At least one GPU is required.")
            sys.exit(1)
        gpu_count = torch.cuda.device_count()
    except ImportError:
        print("PyTorch not installed. Please install torch from requirements.")
        sys.exit(1)

    print(f"Detected {gpu_count} GPU(s). Running in '{args.mode}' mode.")

    # Compute end time for the test
    start_time = time.time()
    end_time = start_time + args.duration

    # Launch one thread per GPU for the selected workload
    threads = []
    for gpu_id in range(gpu_count):
        if args.mode == 'synthetic':
            t = threading.Thread(
                target=synthetic_compute_loop,
                args=(gpu_id, args.matrix_size, args.iterations, end_time)
            )
        else:  # deep learning mode
            t = threading.Thread(
                target=dl_training_loop,
                args=(gpu_id, args.batch_size, args.epochs, end_time)
            )
        t.start()
        threads.append(t)

    # Prepare CSV logging
    csv_file = 'benchmark_log.csv'
    # Write header if file does not exist or is empty
    header = ['timestamp', 'cpu_percent', 'ram_percent']
    # Add columns for each GPU: util%, mem_used_MB, mem_free_MB
    for i in range(gpu_count):
        header += [f'gpu{i}_util_percent', f'gpu{i}_mem_used_MB', f'gpu{i}_mem_free_MB']
    # Initialize CSV
    with open(csv_file, 'a', newline='') as f:
        if f.tell() == 0:
            f.write(','.join(header) + '\n')

    # Monitoring loop: record stats until time is up
    print("Starting monitoring. Press Ctrl+C to stop early.")
    try:
        while time.time() < end_time:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cpu_perc, ram_perc = get_cpu_ram()  # CPU%, RAM%
            # Get GPU stats: list of dicts (one per GPU)
            gpu_stats = get_gpu_stats()

            # Prepare CSV row data
            row = [timestamp, f"{cpu_perc:.1f}", f"{ram_perc:.1f}"]
            for gpu in gpu_stats:
                # GPU load is returned 0-100 already, memory in MB
                row += [f"{gpu['util']:.1f}", f"{gpu['memory_used']}", f"{gpu['memory_free']}"]

            # Append to CSV
            with open(csv_file, 'a', newline='') as f:
                f.write(','.join(map(str, row)) + '\n')

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user.")
    finally:
        # Wait for workload threads to finish
        for t in threads:
            t.join(timeout=1)
        print("All workload threads completed.")

    # Plot the results
    print("Generating plots...")
    plot_results(csv_file, gpu_count)
    print("Benchmark completed. Logged to", csv_file)

if __name__ == "__main__":
    main()
