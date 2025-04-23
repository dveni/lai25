from utils import init_distributed, create_batch, check, compare_tensors
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist


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


if rank == 0:
    print(f"[Rank {rank}] Compute the updated matrix which should be different from the initial weight matrix.")
    initial_weight, updated_weight = single_step()
    compare_tensors(initial_weight, updated_weight.cpu())
else:
    # On all other ranks we create a tensor placeholder so we can distribute the updated_weight to all ranks
    updated_weight = torch.zeros(input_dim, output_dim, device="cuda")
    # distribute updated weight to all ranks to enable a comparison with the baseline later on
    dist.broadcast(updated_weight, src=0)

dist.barrier()
# Cleanup
dist.destroy_process_group()
print(f"[Rank {rank}] done")
