from utils import init_distributed, create_batch, check, compare_tensors
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed import ReduceOp

rank, local_rank, world_size = init_distributed()


# Define global parameters
global_batch_size = 128 # Must be divisible by world_size (e.g., if world_size=16, each gets 8)
local_batch_size = global_batch_size // world_size
input_dim = 64
output_dim = 32
seed = 42

class CustomLinearLayer(nn.Module):
    """
    A linear layer.
    weight matrix W has shape [in_dim, out_dim]
    activation matrix X has shape [bsz, in_dim]
    out = X @ W which as shape [bsz, out_dim]
    """
    def __init__(self, weight: torch.Tensor):
        super(CustomLinearLayer, self).__init__()
        self.W = nn.Parameter(weight)
        self.in_dim = weight.shape[0]
        self.out_dim = weight.shape[1]
    
    def forward(self, X):
        local_bsz = X.shape[0]
        check(X, (local_bsz, self.in_dim))
        # Batched matrix-vector multiplication
        # this could be replaced with matmul or @
        X = torch.einsum("bi,ij->bj", X, self.W)
        check(X, (local_bsz, self.out_dim))
        return X


### Part 1: We compute the reference weight on a single GPU.
def single_step(seed=42, device="cuda") -> torch.Tensor:
    """
    Educational example of performing a single gradient step.
    """
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    # Generate a weight matrix
    initial_weight = torch.randn(input_dim, output_dim)
    # Create the custom linear model using the provided weight matrix.
    model = CustomLinearLayer(initial_weight).to(device)
    # Set up the SGD optimizer with learning rate 0.5
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    # Create the loss function
    loss_fn = nn.MSELoss(reduction="mean")
    # Create a synthetic batch of data with global_batch_size elements
    inputs, targets = create_batch(global_batch_size, input_dim, output_dim, seed=seed, device=device)
    check(inputs, (global_batch_size, input_dim))
    check(targets, (global_batch_size, output_dim))
    # Perform a forward pass through the model we defined above.
    outputs = model(inputs)
    check(outputs, (global_batch_size, output_dim))
    # Compute the MSE loss using loss_fn defined above by taking the average over the target and batch dimension.
    loss = loss_fn(outputs, targets)
    check(loss, [])
    # Reset gardients of all parameters to 0
    optimizer.zero_grad()
    # compute gradients
    loss.backward()
    # perform a parameter update
    optimizer.step()
    # Return the updated weight matrix (detached from the computation graph).
    return initial_weight, model.W.detach()


def single_step_with_grad_accumulation(seed=42, device="cuda", accumulation_steps: int = 4) -> torch.Tensor:
    """
    Educational example of performing a single gradient step with gradient accumulation.
    """
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    # Generate a weight matrix
    initial_weight = torch.randn(input_dim, output_dim)
    # Create the custom linear model using the provided weight matrix.
    model = CustomLinearLayer(initial_weight).to(device)
    # Set up the SGD optimizer with learning rate 0.5
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    # Create the loss function
    loss_fn = nn.MSELoss(reduction="mean")
    # Create a synthetic batch of data with global_batch_size elements
    inputs, targets = create_batch(global_batch_size, input_dim, output_dim, seed=seed, device=device)
    check(inputs, (global_batch_size, input_dim))
    check(targets, (global_batch_size, output_dim))

    # Calculate the micro batch size
    micro_batch_size = global_batch_size // accumulation_steps

    # Perform a forward pass through the model we defined above.
    outputs = model(inputs)
    check(outputs, (global_batch_size, output_dim))
    # Compute the MSE loss using loss_fn defined above by taking the average over the target and batch dimension.
    loss = loss_fn(outputs, targets)
    check(loss, [])
    # Reset gardients of all parameters to 0
    optimizer.zero_grad()
    # Perform gradient accumulation over multiple smaller batches
    for i in range(accumulation_steps):
        # Calculate the start and end indices for this micro-batch
        start_idx = i * micro_batch_size
        end_idx = start_idx + micro_batch_size
        # Slice the original inputs and targets to get this micro-batch
        micro_inputs = inputs[start_idx:end_idx]
        micro_targets = targets[start_idx:end_idx]
        check(micro_inputs, (micro_batch_size, input_dim))
        check(micro_targets, (micro_batch_size, output_dim))
        # Perform a forward pass through the model
        micro_outputs = model(micro_inputs)
        check(micro_outputs, (micro_batch_size, output_dim))
        # Compute the loss for this micro-batch
        micro_loss = loss_fn(micro_outputs, micro_targets)
        check(micro_loss, [])
        # Scale the loss to maintain the same gradient magnitude regardless of accumulation steps. It is numerically advantagous to divide by the number of steps before computing the sum.
        scaled_loss = micro_loss / accumulation_steps
        # Compute gradients (backward pass)
        # The gradients are accumulated (summed) in param.grad
        scaled_loss.backward()
    # After accumulating gradients from all micro-batches, update parameters
    optimizer.step()
    # Return updated weight matrix
    return model.W.detach()

### Part 3: We compute the updated weight using data parallelism
def data_parallel_single_step(seed=42, device="cuda") -> torch.Tensor:
    """
    Educational example of performing a single gradient step using data parallelism.
    Each process handles a subset of the global batch.
    """
    # Set the seed for reproducibility
    # We need to ensure all processes start with the same weight
    torch.manual_seed(seed)
    # Generate a weight matrix
    initial_weight = torch.randn(input_dim, output_dim)
    # Alternatively we could broadcast the tensor from rank 0 to all other processes
    # initial_weight = initial_weight.to(device)
    # dist.broadcast(initial_weight, src=0)
    # Create the custom linear model using the provided weight matrix.
    model = CustomLinearLayer(initial_weight).to(device)
    # Set up the SGD optimizer with learning rate 0.5
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    # Create the loss function
    loss_fn = nn.MSELoss(reduction="mean")

    # Create a synthetic batch of data with the same seed across all workers
    # Then each process will handle a subset of the data based on rank
    full_inputs, full_targets = create_batch(global_batch_size, input_dim, output_dim, seed=seed, device=device)
    # Calculate start and end indices for this process's portion of data
    start_idx = rank * local_batch_size
    end_idx = start_idx + local_batch_size
    # Get local batch by slicing the full batch based on rank
    local_inputs = full_inputs[start_idx:end_idx]
    local_targets = full_targets[start_idx:end_idx]
    check(local_inputs, (local_batch_size, input_dim))
    check(local_targets, (local_batch_size, output_dim))
    # Reset gradients before forward/backward pass
    optimizer.zero_grad()
    # Perform a forward pass through the model with the local batch
    local_outputs = model(local_inputs)
    check(local_outputs, (local_batch_size, output_dim))
    # Compute the MSE loss for the local batch
    local_loss = loss_fn(local_outputs, local_targets)
    check(local_loss, [])
    # Compute gradients (backward pass)
    local_loss.backward()

    # print(f"[Rank {rank}] Gradient before all_reduce: {model.W.grad}")
    # Synchronize gradients across all processes
    for param in model.parameters():
        print(f"[Rank {rank}] param: {param}")
        print(f"[Rank {rank}] Gradient before all_reduce: {param.grad}")
        # Sum the gradients across all processes
        dist.all_reduce(param.grad, op=ReduceOp.SUM)
        print(f"[Rank {rank}] Gradient after all_reduce: {param.grad}")
        # Average the gradients by dividing by world_size
        param.grad.div_(world_size) # Good to know: in pytorch func_ are in-place operations.
    # Perform parameter update - all processes will have the same update now
    optimizer.step()
    # Return the updated weight matrix
    return model.W.detach()



class BroadcastParallel(torch.autograd.Function):
    """
    Identity in forward pass, AllReduce in backward pass.
    Identical to function f in the Megatron paper (https://arxiv.org/abs/1909.08053).
    """
    @staticmethod
    def forward(ctx, x):
        # The forward pass is an identity because we assume the data is already sharded.
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if world_size == 1:
            return grad_output

        dist.all_reduce(grad_output, op=ReduceOp.SUM)
        return grad_output


class GatherParallel(torch.autograd.Function):
    """Gather in forward pass, split in backward pass."""
    @staticmethod
    def forward(ctx, x):
        if world_size == 1:
            return x

        x = x.contiguous()
        x_list = [torch.empty_like(x) for _ in range(world_size)]
        x_list[rank] = x
        dist.all_gather(x_list, x)
        out = torch.cat(x_list, dim=-1).contiguous()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        if world_size == 1:
            return grad_output

        local_dim = grad_output.shape[-1] // world_size
        grad_output_split = torch.split(grad_output, local_dim, dim=-1)
        return grad_output_split[rank].contiguous()

class FullColumnParallelLinear(nn.Module):
    """
    A column-parallel linear layer with a final gather.
    
    weight matrix W has shape [in_dim, out_dim]
    activation matrix X has shape [bsz, in_dim]
    
    out = X @ W which as shape [bsz, out_dim]
    
    In a column-parallel linear layer, the weight matrix is now split along the
    output (column) dimension such that each rank holds a shard of shape
    [in_dim, out_dim / world_size]. The results are all [bsz, out_dim]
    and need to be summed across all ranks.
    
    Parameters:
        weight (torch.Tensor): Full weight matrix of shape (out_dim, in_dim).
        world_size (int): Total number of processes.
        rank (int): The current process rank.
    """

    def __init__(self, weight: torch.Tensor, world_size: int, rank: int):
        super(FullColumnParallelLinear, self).__init__()
        in_dim, out_dim = weight.shape
        assert out_dim % world_size == 0, "out_dim must be divisible by world_size"
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.world_size = world_size
        self.rank = rank
        self.local_out_dim = out_dim // world_size
        
        # Partition the weight along columns for this rank.
        start = rank * self.local_out_dim
        end = start + self.local_out_dim
        self.W = nn.Parameter(weight[:, start:end].clone().contiguous())
    
    def forward(self, X):
        """
        Forward pass for the column-parallel linear layer.
        Parameters:
        X (torch.Tensor): Input tensor of shape (batch_size, in_dim).
        Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_dim).
        """
        local_bsz = X.shape[0]
        check(X, (local_bsz, self.in_dim))
        
        # distribute X in the forward pass and collect the errors from all ranks in the backward pass
        X = BroadcastParallel.apply(X)

        # Batched matrix multiplication
        local_out = torch.einsum("bi,ij->bj", X, self.W).contiguous()
        check(local_out, (local_bsz, self.local_out_dim))
        # Collect the outputs of the linear map in the forward pass and keep only the rank specific errors in the backward pass.
        out = GatherParallel.apply(local_out)
        check(out, (local_bsz, self.out_dim))
        return out

def full_column_parallel_single_step(seed=42, device="cuda"):
    """
    Performs one gradient update using tensor parallelism with column-wise splitting.
    """
    torch.manual_seed(seed)
    initial_weight = torch.randn(input_dim, output_dim)
    # Instantiate the column-parallel linear model.
    model = FullColumnParallelLinear(initial_weight, world_size, rank).to(device)
    # Set up optimiser as previously
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    loss_fn = nn.MSELoss(reduction="mean")
    # Get a batch
    full_inputs, full_targets = create_batch(global_batch_size, input_dim, output_dim, seed=seed, device=device)
    # Forward pass: Each process slices its corresponding input columns.
    outputs = model(full_inputs)
    check(outputs, (global_batch_size, output_dim))
    # Compute the loss using the full output.
    loss = loss_fn(outputs, full_targets)
    # Backward pass.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Each rank now holds its shard of the updated weight of shape (input_dim, local_output_dim).
    local_updated_weight = model.W.detach()
    check(local_updated_weight, (input_dim, output_dim // world_size))
    # Gather weight shards from all processes along the column dimension.
    # To do so we first need to create a list of zero tensors on the GPU so we can call all_gather
    weight_shards = [torch.zeros(input_dim, local_updated_weight.shape[1], device=device) for _ in range(world_size)]
    dist.all_gather(weight_shards, local_updated_weight)
    # Concatenate the shards along columns to reconstruct the full updated weight.
    global_updated_weight = torch.cat(weight_shards, dim=1)
    check(global_updated_weight, (input_dim, output_dim))
    return global_updated_weight





if rank == 0:
    print(f"[Rank {rank}] Compute the updated matrix which should be different from the initial weight matrix.")
    initial_weight, updated_weight = single_step()
    compare_tensors(initial_weight, updated_weight.cpu())
else:
    # On all other ranks we create a tensor placeholder so we can distribute the updated_weight to all ranks
    updated_weight = torch.zeros(input_dim, output_dim, device="cuda")
    # distribute updated weight to all ranks to enable a comparison with the baseline later on
    dist.broadcast(updated_weight, src=0)


if rank == 0:
    print(f"[Rank {rank}] Compute the updated weight using batch accumulation. They should match.")
    batch_accum_weight = single_step_with_grad_accumulation()
    compare_tensors(updated_weight.cpu(), batch_accum_weight.cpu())

# if rank == 0:
#     print(f"[Rank {rank}] Compute the updated weight using data parallelism.")
print(f"[Rank {rank}] Compute the updated weight using data parallelism.")
data_parallel_weight = data_parallel_single_step()
# Compare on all ranks
compare_tensors(updated_weight.cpu(), data_parallel_weight.cpu(), prefix="DataParallel")

if rank == 0:
    print(f"[Rank {rank}] Compute the updated weight using tensor parallelism (FullColumnParallelLinear).")
column_parallel_weight = full_column_parallel_single_step()
compare_tensors(updated_weight.cpu(), column_parallel_weight.cpu(), prefix="FullColumnParallel")


# Cleanup
print(f"[Rank {rank}] done")
dist.destroy_process_group()
