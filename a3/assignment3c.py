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

# Total parameters size remains the same but splits across ranks
N = 2 ** 27 # ~0.13 billion elements
## ----- AllGather
# Each process starts with a unique subset of data
send_tensor= torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")
# Create a tensor to store the data collected from all rank
recv_tensor = torch.zeros((world_size, N), dtype=torch.float32, device="cuda")
dist.all_gather(tensor_list=[recv_tensor[i] for i in range(world_size)],
tensor=send_tensor)

print(f"[Python] rank={rank} | recv_tensor={recv_tensor.mean(dim=0)}")