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



# Total parameters size remains the same but splits across ranks
N = 2 ** 27 # ~0.13 billion elements
## ----- AllGather
# Each process starts with a unique subset of data
send_tensor= torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")
# Create a tensor to store the data collected from all rank
recv_tensor = torch.zeros((world_size, N), dtype=torch.float32, device="cuda")

# Synchronize before starting communication
dist.barrier()

send_start = time.time()
dist.all_gather(tensor_list=[recv_tensor[i] for i in range(world_size)],
tensor=send_tensor)
torch.cuda.synchronize() # shouldn't be needed but .wait() is not behaving as expected.

send_end = time.time()
elapsed_seconds = send_end - send_start
total_bytes = recv_tensor.nelement() * 4 # convert elements to bytes
total_gbs = total_bytes / (1024**3) # convert to GB
throughput = total_gbs / elapsed_seconds # GB/s
print(f"[Python] rank={rank} | transferred {total_gbs:.2}GB | throughput={throughput:.4}GB/s")

print(f"[Python] rank={rank} | recv_tensor={recv_tensor.mean(dim=0)}")


print("-------------------------------------------------")
print("REDUCE SCATTER\n")
# Reduce scatter
device="cuda"
# Each process starts with the full dataset
send_tensor = torch.full((world_size * N,), fill_value=rank, dtype=torch.float32, device=device)
# And we create a tensor to hold 1/world_size part of it.
recv_tensor = torch.zeros((N,), dtype=torch.float32, device=device)

# Synchronize before starting communication
dist.barrier()
send_start = time.time()

dist.reduce_scatter(output=recv_tensor, input_list=[send_tensor[i*N:(i+1)*N] for i in range(world_size)], op=dist.ReduceOp.SUM)
torch.cuda.synchronize() # shouldn't be needed but .wait() is not behaving as expected.

send_end = time.time()
elapsed_seconds = send_end - send_start
total_bytes = recv_tensor.nelement() * 4 # convert elements to bytes
total_gbs = total_bytes / (1024**3) # convert to GB
throughput = total_gbs / elapsed_seconds # GB/s
print(f"[Python] rank={rank} | transferred {total_gbs:.2}GB | throughput={throughput:.4}GB/s")

print(f"[Python ReduceScatter] rank={rank} | transferred {total_gbs:.2}GB | throughput={throughput:.4}GB/s | recv_tensor.mean()={recv_tensor.mean()}")

