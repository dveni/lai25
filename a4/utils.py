import os
import torch
import torch.distributed as dist

def init_distributed():
    """
    Initialise the distributed environment.
    Assumes that environment variables RANK, LOCAL_RANK, and WORLD_SIZE are set.
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    node_id = os.environ.get("SLURM_NODEID", "N/A")
    
    # Set the current device for this process
    torch.cuda.set_device(local_rank)
    
    # Initialise the process group with NCCL backend (requires Nvidia GPUs)
    dist.init_process_group(backend="nccl")
    
    print(f"[Distributed Init] Rank {rank} initialized on {node_id} on GPU {local_rank}.")
    dist.barrier()
    if rank == 0:
        print(f"[Rank {rank}] All ranks ready!")
    return rank, local_rank, world_size

def create_batch(batch_size, input_dim, output_dim, seed=42, device="cuda"):
    """
    Create synthetic input and target tensors for a batch.
    
    Parameters:
        batch_size (int): Number of examples in the batch
        input_dim (int): Dimension of each input example
        seed (int): Random seed for reproducibility
        device (str): Device to create the tensors on ("cuda" or "cpu")

    Returns:
        tuple: (inputs, targets)
        - inputs: Tensor of shape (batch_size, input_dim) containing random values
        - targets: Tensor of shape (batch_size,) for regression tasks
    """
    torch.manual_seed(seed)
    # Create input tensor
    inputs = torch.randn(batch_size, input_dim, device=device)
    # Create regression targets
    targets = torch.randn(batch_size, output_dim, device=device)
    
    return inputs, targets

def check(tensor, expected_shape):
    """
    Checks that the tensor's shape matches the expected shape.
    
    Parameters:
        tensor (torch.Tensor): The tensor to check.
        expected_shape (list or tuple): The expected shape. For any dimension,
        a value <= 0 is treated as a wildcard.
    
    Raises:
        AssertionError: If tensor is not a torch.Tensor or if the actual shape
        does not match the expected shape.
    
    Example:
        >>> x = torch.randn(32, 64, 10)
        >>> check(x, [32, 64, 10]) # Passes - exact match
        >>> check(x, [32, -1, 10]) # Passes - middle dimension treated as wildcard
        >>> check(x, [32, 128, 10]) # Raises AssertionError - middle dimension mismatch
    """
    if tensor is None:
        return
    
    # Ensure the provided tensor is a torch.Tensor.
    assert isinstance(tensor, torch.Tensor), "SHAPE GUARD: Provided tensor is not a torch.Tensor!"

    # Detach the tensor to avoid issues with in-place modifications (useful for torch.compile).
    actual_shape = list(tensor.detach().shape)
    
    # Ensure expected_shape is a list or tuple.
    assert isinstance(expected_shape, (list, tuple)), "SHAPE GUARD: expected_shape must be a list or tuple!"
    assert len(expected_shape) == len(actual_shape), (
    f"SHAPE GUARD: Expected shape length {len(expected_shape)} but got {len(actual_shape)} with shape {actual_shape}."
    )

    for idx, (actual_dim, expected_dim) in enumerate(zip(actual_shape, expected_shape)):
        # Skip the check for dimensions with a non-positive expected size (wildcard).
        if expected_dim <= 0:
            continue
        assert actual_dim == expected_dim, (
            f"SHAPE GUARD: At dimension {idx}: expected {expected_dim}, but got {actual_dim} (actual shape: {actual_shape})."
            )


def compare_tensors(tensor1, tensor2, tol=1e-5, prefix=""):
    """
    Simple comparison of two tensors, printing basic difference statistics.
    
    Parameters:
        tensor1 (torch.Tensor): First tensor to compare
        tensor2 (torch.Tensor): Second tensor to compare
        tol (float): Tolerance for considering values as close
    """
    # Calculate differences
    abs_diff = (tensor1 - tensor2).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    
    # Simple check if tensors are close
    is_close = torch.allclose(tensor1, tensor2, rtol=tol, atol=tol)
    
    # Print brief comparison summary
    rank = dist.get_rank() if dist.is_initialized() else 0
    prefix = f"[{prefix}]" if prefix else ""
    print(f"{prefix}[Rank {rank}] Tensors match: {is_close} | Max diff: {max_diff:.6e} | Mean diff: {mean_diff:.6e}", flush=True)