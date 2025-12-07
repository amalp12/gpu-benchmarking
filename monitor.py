"""
monitor.py: Utility functions to sample CPU and GPU statistics.
Uses psutil for CPU/RAM and GPUtil for NVIDIA GPU stats.
"""
import psutil
import GPUtil
import time


def get_cpu_ram():
    """
    Return current total CPU usage (%) and RAM usage (%).
    Uses psutil.cpu_percent and psutil.virtual_memory.
    """
    # psutil.cpu_percent(interval=0) gives usage since last call; use a small sleep for accuracy if needed
    cpu_percent = psutil.cpu_percent(interval=0.5)
    ram_percent = psutil.virtual_memory().percent
    return cpu_percent, ram_percent


def get_gpu_stats():
    """
    Return a list of dicts with 'id', 'util', 'memory_used', 'memory_free' for each GPU.
    GPUtil.getGPUs() returns GPU objects with attributes like load (0-1) and memory usage.
    """
    stats = []
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        stats.append({
            'id': gpu.id,
            # convert to percentage (GPUtil.load is 0-1):contentReference[oaicite:4]{index=4}
            'util': gpu.load * 100,
            # in MB (total GPU memory allocated by active contexts):contentReference[oaicite:5]{index=5}
            'memory_used': gpu.memoryUsed,
            'memory_free': gpu.memoryFree        # in MB
        })
    return stats
