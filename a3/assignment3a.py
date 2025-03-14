#!/usr/bin/env python3
import os
import socket
import torch
import torch.distributed as dist
# Read environment variables that we set in the sbatch script
master_addr = os.environ.get("MASTER_ADDR", "N/A")
master_port = os.environ.get("MASTER_PORT", "N/A")
world_size = int(os.environ.get("WORLD_SIZE", "N/A"))
foobar = os.environ.get("FOOBAR", "N/A")
# Read environment variables set by SLURM
rank = int(os.environ["SLURM_PROCID"])
local_rank = int(os.environ["SLURM_LOCALID"])
# Read the information form the node
hostname = socket.gethostname()
# Each process prints a final message to confirm it didn't get stuck
print(f"[Python] rank={rank} | host={hostname} | {master_addr}:{master_port} | {world_size} | {foobar} ")   

# Initializes the default (global) process group
dist.init_process_group(
backend="nccl", # the NCCL backend requires a GPU on each process
init_method=f"tcp://{master_addr}:{master_port}",
world_size=world_size,
rank=rank,
)
# Limit GPU allocation of this process to only one GPU
torch.cuda.set_device(local_rank)
# Create a float32 tensor on each rank with a single element of value 'rank' and move it to the GPU.
local_tensor = torch.tensor([rank], dtype=torch.float32).cuda()
print(f"[Python] rank={rank} | local_rank={local_rank} | host={hostname} | local_tensor={local_tensor.item()}")
# Perform a sum operation across all ranks.
dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)
print(f"[Python] rank={rank} | local_rank={local_rank} | host={hostname} | local_tensor_after_all_reduce={local_tensor.item()}")
# Cleanup
dist.destroy_process_group()