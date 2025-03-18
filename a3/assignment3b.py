#!/usr/bin/env python3
import os
import time
import torch
import torch.distributed as dist

# Read environment variables set by torchrun
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

# Initializes the default (global) process group
dist.init_process_group(backend="nccl")

# Limit GPU allocation of this process to only one GPU
torch.cuda.set_device(local_rank)

N = 2 ** 30 # ~1.1 billion elements
parameters = torch.ones((N,), dtype=torch.float32, device="cuda")
gradients = torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")
print(f"[Python] rank={rank} | Initial parameters[0]={parameters[0].item()}")

LEARNING_RATE = 0.1
ROOT_RANK = 0 # Central rank for parameter updates

# Warmup
print(f"[Python] rank={rank} | Starting warmup")
for _ in range(5):
    dist.reduce(gradients, dst=ROOT_RANK, op=dist.ReduceOp.AVG)
print(f"[Python] rank={rank} | Warmup complete")
gradients = torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")


# Force a CUDA synchronization point before measuring time
torch.cuda.synchronize()
# Record the start time
start = time.time()
dist.reduce(gradients, dst=ROOT_RANK, op=dist.ReduceOp.AVG)
# On root rank, compute the average gradient and update parameters
if rank == ROOT_RANK:
    # gradients /= world_size # Average the gradients
    parameters -= LEARNING_RATE * gradients # SGD update

# Broadcast updated parameters to all ranks
dist.broadcast(parameters, src=ROOT_RANK)
# Force a CUDA synchronization again to ensure the operation is completed before measuring the end time.
torch.cuda.synchronize()
# Measure end time
end = time.time()
elapsed_seconds = end - start


expected_param = 1.0 - LEARNING_RATE * (world_size - 1) / 2
assert torch.allclose(
parameters[0],
torch.tensor(expected_param, device="cuda")
), f"[Python] rank={rank} | Parameter mismatch: expected {expected_param}, got {parameters[0].item()}"

total_bytes = parameters.nelement() * 4 # convert elements to bytes
total_gbs = total_bytes / (1024**3) # convert to GB
throughput = total_gbs / elapsed_seconds # GB/s
print(f"[Python] rank={rank} | transferred {total_gbs:.2}GB | throughput={throughput:.4}GB/s")