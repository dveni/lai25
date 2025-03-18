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
tensor = torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")
print(f"[Python] rank={rank} | tensor={tensor}")

# Warmup
print(f"[Python] rank={rank} | Starting warmup")
for _ in range(5):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
print(f"[Python] rank={rank} | Warmup complete")

N = 2 ** 30 # ~1.1 billion elements
tensor = torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")

# Force a CUDA synchronization point before measuring time
torch.cuda.synchronize()
# Record the start time
start = time.time()
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
# Force a CUDA synchronization again to ensure the operation is completed before measuring the end time.
torch.cuda.synchronize()
# Measure end time
end = time.time()
elapsed_seconds = end - start

expected_val = world_size*(world_size-1)/2
# print(f"{tensor.shape=}")
# print(f"{tensor[0]=}")
# print(f"{tensor[1]=}")
assert torch.allclose(
tensor,
torch.full_like(tensor, expected_val)
), f"[Python] rank={rank} | all-Reduce mismatch: expected {expected_val}, got {tensor[0].item()} in first element."


total_bytes = tensor.nelement() * 4 # convert elements to bytes
total_gbs = total_bytes / (1024**3) # convert to GB
throughput = total_gbs / elapsed_seconds # GB/s
print(f"[Python] rank={rank} | transferred {total_gbs:.2}GB | throughput={throughput:.4}GB/s")


async_op = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
while not async_op.is_completed():
    print(f"{rank}|", end='', flush=True) # Print the rank number without a newline to simulate CPU work
    time.sleep(0.1) # Wait for 0.1 seconds