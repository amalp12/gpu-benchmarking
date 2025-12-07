"""
dl_workload.py: Perform a dummy deep-learning workload (training a small neural network) on a GPU.
This simulates a more realistic training process including data transfer, forward/backward passes.
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim


def dl_training_loop(gpu_id, batch_size, epochs, end_time):
    """
    Training loop: on GPU `gpu_id`, train a simple network on random data.
    Runs until `epochs` are done or time > end_time.
    """
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')

    # Define a simple model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop with random data
    epoch = 0
    while epoch < epochs and time.time() < end_time:
        # Generate random input batch (100 features) and random targets
        inputs = torch.randn(batch_size, 100, device=device)
        targets = torch.randint(0, 10, (batch_size,), device=device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch += 1
        # Looping like this simulates typical mini-batch training on GPU, including compute and memory ops
