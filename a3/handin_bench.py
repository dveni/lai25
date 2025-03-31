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

ordered_node_groups = [
    dist.new_group([0,1,2,3]),
    dist.new_group([4,5,6,7]),
    dist.new_group([8,9,10,11]),
    dist.new_group([12,13,14,15]),
]

unordered_node_groups = [
    dist.new_group([0,4,8,12]),
    dist.new_group([1,5,9,13]),
    dist.new_group([2,6,10,14]),
    dist.new_group([3,7,11,15]),
]


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


# Total parameters size remains the same but splits across ranks
N = 2 ** 30 # ~1.1 billion elements
# Each process starts with data of its rank
tensor = torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")



# Synchronize before starting communication
dist.barrier()
send_start = time.time()
# Executes the reduce op on the group to which this process belongs.
dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=node_groups[rank // 4])
# More unnatural group
# dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=node_groups[rank % 4])
torch.cuda.synchronize() # shouldn't be needed but .wait() is not behaving as expected.

send_end = time.time()
elapsed_seconds = send_end - send_start
total_bytes = tensor.nelement() * 4 # convert elements to bytes
total_gbs = total_bytes / (1024**3) # convert to GB
throughput = total_gbs / elapsed_seconds # GB/s
print(f"[Python] rank={rank} | transferred {total_gbs:.2}GB | throughput={throughput:.4}GB/s | tensor.mean()={tensor.mean()}")
print("-------------------------------------------------")