"""
plotter.py: Plot CPU and GPU usage over time from the logged CSV data.
Produces line plots (matplotlib) showing CPU%, RAM%, and each GPU's utilization and memory.
"""
import pandas as pd
import matplotlib.pyplot as plt


def plot_results(csv_file, gpu_count):
    # Read the logged CSV data
    df = pd.read_csv(csv_file)

    # Convert timestamp column to datetime if needed (optional, here assume it's just indices)
    # Plot CPU utilization and RAM usage
    times = pd.to_datetime(df['timestamp'])
    plt.figure(figsize=(8, 4))
    plt.plot(times, df['cpu_percent'], label='CPU Util %')
    plt.plot(times, df['ram_percent'], label='RAM Util %')
    plt.xlabel('Time')
    plt.ylabel('Usage (%)')
    plt.title('CPU and RAM Usage Over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig('cpu_ram_usage.png')

    # Plot each GPU's utilization and memory usage
    for i in range(gpu_count):
        plt.figure(figsize=(8, 4))
        util_col = f'gpu{i}_util_percent'
        mem_used_col = f'gpu{i}_mem_used_MB'
        plt.plot(times, df[util_col], label=f'GPU{i} Util %')
        plt.plot(times, df[mem_used_col] / df[mem_used_col].max()
                 * 100, label=f'GPU{i} Mem Used % (scaled)')
        plt.xlabel('Time')
        plt.ylabel('Percent')
        plt.title(f'GPU {i} Utilization and Memory Usage')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'gpu{i}_usage.png')
