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

# Define send and receive ranks
send_rank = (rank + 1) % world_size
recv_rank = (rank - 1 + world_size) % world_size

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

# Create a tensor to send: filled with the sender's rank
send_tensor = torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")
# Prepare a tensor to receive data
recv_tensor = torch.zeros(N, dtype=torch.float32, device="cuda")

# Synchronize before starting communication
dist.barrier()
# Start send and receive
send_start = time.time()
print(f"[Python] rank={rank} | Sending data to rank={send_rank}")
send_req = dist.isend(tensor=send_tensor, dst=send_rank)
print(f"[Python] rank={rank} | Receiving data from rank={recv_rank}")
recv_req = dist.irecv(tensor=recv_tensor, src=recv_rank)
# Wait for both send and receive to complete
print(f"[Python] rank={rank} | Waiting for send_req and recv_req to complete")
send_req.wait()
recv_req.wait()

print(f"[Python] rank={rank} is_complete={send_req.is_completed()}", flush=True)
print(f"[Python] rank={rank} is_complete={recv_req.is_completed()}", flush=True)
torch.cuda.synchronize() # shouldn't be needed but .wait() is not behaving as expected.

send_end = time.time()
elapsed_seconds = send_end - send_start



total_bytes = recv_tensor.nelement() * 4 # convert elements to bytes
total_gbs = total_bytes / (1024**3) # convert to GB
throughput = total_gbs / elapsed_seconds # GB/s
print(f"[Python] rank={rank} | transferred {total_gbs:.2}GB | throughput={throughput:.4}GB/s")