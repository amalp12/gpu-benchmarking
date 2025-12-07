"""
synthetic_workload.py: Perform a synthetic compute-intensive workload on a specified GPU.
Each loop does a large matrix multiplication using PyTorch to fully utilize the GPU.
"""
import torch


def synthetic_compute_loop(gpu_id, matrix_size, max_iterations, end_time):
    """
    Compute loop: on GPU `gpu_id`, multiply two random matrices of shape (matrix_size x matrix_size).
    Runs until either max_iterations is reached or time > end_time.

    GPU is set by torch.cuda.set_device, and operations use the default CUDA stream.
    """
    torch.cuda.set_device(gpu_id)  # select GPU
    # Pre-allocate tensors to possibly reuse memory (optional)
    for i in range(max_iterations):
        if time.time() > end_time:
            break
        # Create random matrices on the GPU
        a = torch.randn((matrix_size, matrix_size), device=f'cuda:{gpu_id}')
        b = torch.randn((matrix_size, matrix_size), device=f'cuda:{gpu_id}')
        c = torch.matmul(a, b)  # perform matrix multiplication on GPU
        # Optionally do something with c to prevent lazy evaluation (not needed here)
        # This loop heavily loads the GPU with compute (similar to many synthetic benchmarks).
