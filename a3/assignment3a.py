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