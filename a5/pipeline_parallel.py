import copy
import torch

from typing import List

from torch import nn
from torch import distributed as dist

# Q7: Complete the conditions of the `if` statements within each `operation`
def pipeline_communicate(operation, pp_process_group, tensor=None, shapes=None):

    # NOTE(tj.solergibert) `src` & `dest` MUST be global ranks, hence we do "src = dist.get_global_rank..."

    pp_rank = dist.get_rank(pp_process_group)
    pp_prev_rank = pp_rank - 1
    pp_next_rank = pp_rank + 1
    
    pp_is_first_stage = pp_rank == 0
    pp_is_last_stage = pp_rank == dist.get_world_size(pp_process_group) - 1
    

    if operation == 'recv_forward':
        if pp_is_first_stage: return None
        print(f"[{dist.get_global_rank()}] - creating tensor recv_forward")
        tensor = torch.empty(shapes, requires_grad=True, device="cuda")
        print(f"[{dist.get_global_rank()}] - tensor created")
        src = dist.get_global_rank(pp_process_group, pp_prev_rank)
        print(f"[{dist.get_global_rank()}] - src rank {src}")
    
    elif operation == 'send_forward':
        if pp_is_last_stage: return
        dest = dist.get_global_rank(pp_process_group, pp_next_rank)
    
    elif operation == 'recv_backward':
        if pp_is_last_stage: return None
        tensor = torch.empty(shapes, requires_grad=True, device="cuda")
        src = dist.get_global_rank(pp_process_group, pp_next_rank)
    
    elif operation == 'send_backward':
        if pp_is_first_stage: return
        dest = dist.get_global_rank(pp_process_group, pp_prev_rank)
    
    print(f"[{dist.get_global_rank()}] - a")
    print_shapes = shapes if shapes else tensor.shape
    print(f"[{dist.get_global_rank()}] - b")
    is_send = operation.startswith('send')
    print(f"[{dist.get_global_rank()}] - c")
    peer_rank = dest if is_send else src
    print(f"[{dist.get_global_rank()}] - d")

    print(f"[{dist.get_global_rank()}] - {operation} to/from rank {peer_rank} with shape {print_shapes}")

    op = dist.P2POp(dist.isend if is_send else dist.irecv, tensor, peer_rank)
    print(f"[{dist.get_global_rank()}] - {operation} op created")
    [req.wait() for req in dist.batch_isend_irecv([op])]
    print(f"[{dist.get_global_rank()}] - {operation} op completed")
    torch.cuda.synchronize()
    return tensor if not is_send else None

# Q5: Develop this function
def distribute_layers(num_layers: int, pp_rank: int, pp_world_size: int) -> List:
        """
        Distribute model layers across GPUs as evenly as possible.
        Returns a list with the layer indices that should be processed by this GPU.
        """
        # TODO
        layers_per_stage = num_layers // pp_world_size
        layers_in_current_stage = list(range(pp_rank * layers_per_stage, (pp_rank + 1) * layers_per_stage))
        return layers_in_current_stage

class PipelineStage(nn.Module):
    """
    Implements pipeline parallelism by distributing model layers across multiple GPUs.
    Each GPU processes a subset of the model's layers in a pipeline fashion.
    """
    def __init__(self, model, number_of_layers, pp_rank, pp_world_size):
        super().__init__()
        # Determine which layers should be assigned to this GPU
        self.layer_distribution = distribute_layers(number_of_layers, pp_rank, pp_world_size)
        # Assign relevant decoder layers to this GPU
        self.pp_stage_layers = nn.ModuleList([copy.deepcopy(model.layers[i]) for i in self.layer_distribution])
    
    def forward(self, x):
        for layer in self.pp_stage_layers:
            x = layer(x)
        return x
    
    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        """
        Backward pass for this pipeline stage.
        Computes gradients for assigned layers using received gradient from next stage.
        """
        if input_tensor is not None: input_tensor.retain_grad()
        if output_tensor_grad is None:
            output_tensor_grad = torch.ones_like(output_tensor, memory_format=torch.preserve_format)
        torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad, retain_graph=False, create_graph=False)
        return input_tensor.grad if input_tensor is not None else None