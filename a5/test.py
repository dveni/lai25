import click
import os
import random

import torch
import torch.distributed as dist

from model import MyDummyModel
from pipeline_parallel import pipeline_communicate, PipelineStage
from utils import build_device_mesh

@click.command()
@click.option("--pp", default=4)
@click.option("--global_batch_size", default=16)
@click.option("--micro_batch_size", default=2)
@click.option("--sequence_length", default=8192)
@click.option("--hidden_size", default=4096)
@click.option("--intermediate_size", default=14336)
@click.option("--number_of_layers", default=12)
@click.option("--seed", default=1234)
def main(pp: int,
         global_batch_size: int,
         micro_batch_size: int,
         sequence_length: int,
         hidden_size: int,
         intermediate_size: int,
         number_of_layers: int,
         seed: int
         ):
    ###################################################
    ###################### Init #######################
    ###################################################
    if not torch.cuda.is_available():
        raise RuntimeError("No GPUs detected!")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"Rank {rank} of {world_size} on local rank {local_rank}")

    # Init process groups
    dist.init_process_group("nccl")
    device_mesh = build_device_mesh(rank, world_size, pp)

    # Init environment
    # Q1: Seeding
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.set_device(local_rank)
    torch.cuda.manual_seed_all(seed)
    device_id = torch.device(f"cuda:{local_rank}")

    # Q2: Assert args
    assert number_of_layers % pp == 0, f"number_of_layers {number_of_layers} must be divisible by pp {pp}"
    assert global_batch_size % micro_batch_size == 0, f"global_batch_size {global_batch_size} must be divisible by micro_batch_size {micro_batch_size}"
    
    number_of_microbatches = global_batch_size // micro_batch_size

    input = torch.rand((global_batch_size, sequence_length, hidden_size))
    input = torch.split(input, split_size_or_sections=micro_batch_size)

    train_dl_iterator = iter(input)
    ###################################################    
    print("Finished initializing the environment")
    ###################################################
    ###################### No PP ######################
    ###################################################
    output_tensors_no_pp = [] # NOTE(tj.solergibert) To check PP vs NON-PP outputs!
    
    # Q3: Model definition & shape of the intermidiate tensors
    model = MyDummyModel(number_of_layers, hidden_size, intermediate_size).cuda()
    tensor_shapes = (micro_batch_size, sequence_length, hidden_size) # TODO

    for _ in range(number_of_microbatches):
        # Q4: 1. Fetch a batch of data from the dataloader
        batch = next(train_dl_iterator)
        # 2. Move it to the GPU
        batch = batch.to(device_id)
        # 3. Compute the forward pass
        output = model(batch)


        output_tensors_no_pp.append(output.detach().clone()) # NOTE(tj.solergibert) To check PP vs NON-PP outputs!
        # 4. Compute the backward pass
        loss = output.mean()
        loss.backward()

        
    
    ################################################### 
    torch.cuda.synchronize()
    dist.barrier()
    print("Finished forward pass without pipeline parallelism")
    ###################################################
    ######################## PP #######################
    ###################################################

    # Q5: Pipeline stage definition and `distribute_layers` function
    model_stage = PipelineStage(model, number_of_layers, device_mesh["pp"].get_local_rank(), pp).cuda()

    # Q6: Which ranks require the training dataloader?
    if device_mesh["pp"].get_local_rank() == 0:
        train_dl_iterator = iter(input)
    else:
        train_dl_iterator = None
    
    # Store tensors to recreate computation graph during backward pass + To check PP vs NON-PP outputs!
    input_tensors, output_tensors, output_tensors_pp = [], [], []

    for _ in range(number_of_microbatches): # All forward passes
        input_tensor = pipeline_communicate(operation='recv_forward', pp_process_group=device_mesh["pp"].get_group(), shapes=tensor_shapes)
        # Q8: 1. Fetch a batch from the dataloader if needed
        if device_mesh["pp"].get_local_rank() == 0:
            batch = next(train_dl_iterator)
        else:
            batch = None
        # 2. Move the batch from the dataloader OR the activations from the previous PP stage to the GPU
        input_tensor = input_tensor.to(device_id).require_grad_() if batch is None else batch.to(device_id).require_grad_()
        # 3. Compute the forward pass
        output = model_stage(input_tensor)
        pipeline_communicate(operation='send_forward', pp_process_group=device_mesh["pp"].get_group(), tensor=output)
        
        output_tensors_pp.append(output.detach().clone()) # NOTE(tj.solergibert) To check PP vs NON-PP outputs!
        
        # Compute loss on the last stage
        # 4. Compute the loss in the required stage
        if device_mesh["pp"].get_local_rank() == pp - 1:
            output = output.mean()

        # Save tensors to reconstruct computation graph during backward pass
        input_tensors.append(input_tensor)
        output_tensors.append(output)

    for _ in range(number_of_microbatches): # All backward passes
        output_tensor_grad = pipeline_communicate(operation='recv_backward', pp_process_group=device_mesh["pp"].get_group(), shapes=tensor_shapes)
        # Retrieve saved tensors in FIFO order to match forward pass sequence
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        input_tensor_grad = model_stage.backward(input_tensor, output_tensor, output_tensor_grad)
        pipeline_communicate(operation='send_backward', pp_process_group=device_mesh["pp"].get_group(), tensor=input_tensor_grad)

    dist.barrier()

    # Q9: Check the model outputs in the required rank
    if device_mesh["pp"].get_local_rank() == pp - 1:
        for output_no_pp, output_pp in zip(output_tensors_no_pp, output_tensors_pp):
            torch.testing.assert_close(output_no_pp, output_pp)
    dist.barrier()

    # Q10: Check the grads of the required layers. Remember that we store the `layer_idx` in each layer of the model
    for pp_stage_layer in model_stage.pp_stage_layers:
        layer_idx = pp_stage_layer.layer_idx
        torch.testing.assert_close(pp_stage_layer.fc1.weight.grad, model.layers[layer_idx].fc1.weight.grad)
        torch.testing.assert_close(pp_stage_layer.fc2.weight.grad, model.layers[layer_idx].fc2.weight.grad)
    ################################################### 
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
    