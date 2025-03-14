#!/usr/bin/env python3
import os
import socket
import torch
import torch.distributed as dist
# Read environment variables set by torchrun
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
# Initializes the default (global) process group
dist.init_process_group(backend="nccl")
# Limit GPU allocation of this process to only one GPU
torch.cuda.set_device(local_rank)
# Create a float32 tensor on each rank with a single element of value 'rank' and move it to the GPU.
local_tensor = torch.tensor([rank], dtype=torch.float32).cuda()
print(f"[Python] rank={rank} | local_tensor={local_tensor.item()}")
# Perform a sum operation across all ranks.
dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)
print(f"[Python] rank={rank} | local_tensor_after_all_reduce={local_tensor.item()}")
# Cleanup
dist.destroy_process_group()