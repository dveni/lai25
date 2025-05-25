import os
import time
import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from utils import inspect_model

from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist


from dataset import CollatorForCLM, ParquetDataset
from model import Transformer, TransformerModelArgs, TransformerBlock
from utils import build_lr_scheduler, clip_grad_norm_, get_args, get_num_params, get_num_flop_per_token, init_logger, logger, PRECISION_STR_TO_DTYPE, set_default_dtype, init_distributed

import transformer_engine.pytorch as te
from transformer_engine.common import recipe

from torchao.quantization import float8_weight_only, quantize_
from torchao.float8 import convert_to_float8_training  
from torchao.prototype.low_bit_optim import AdamW8bit
from torchao.float8.config import Float8LinearConfig

from torch import nn
import subprocess
import itertools
from dataclasses import replace

# fix random seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def apply_compile(model: nn.Module):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = torch.compile(transformer_block, fullgraph=True)
        model.layers.register_module(layer_id, transformer_block)

    logger.info("Compiling each TransformerBlock with torch.compile")

def train(args):
  # Init
  ddp_rank, ddp_local_rank, world_size = init_distributed()
  device = f"cuda:{ddp_local_rank}"

  master_process = ddp_rank == 0
  model_dtype = PRECISION_STR_TO_DTYPE[args.model_dtype]
  
  if master_process:
    logger.info(f"Experiment args: {args}")
    logger.info(f"Distributed training with {world_size} processes on device {device}")
  
  logger.info(f"FSDP rank: {ddp_rank}, Local rank: {ddp_local_rank}, World size: {world_size}")

  # Set up DataLoader
  logger.info("Setting up DataLoaders...")
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
  train_ds = ParquetDataset(args.dataset, tokenizer, args.sequence_length, args.batch_size*args.training_steps)
  train_collator = CollatorForCLM(args.sequence_length, tokenizer.pad_token_id)
  train_dl = DataLoader(train_ds,
                        batch_size=args.batch_size,
                        collate_fn=train_collator,
                        num_workers=4,
                        pin_memory=True,
                        shuffle=False,
                        sampler=DistributedSampler(train_ds))
  # train_dl_iterator = iter(train_dl)

  # Set up Model
  if master_process:
    logger.info("Setting up Model...")
  model_config = TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        vocab_size=tokenizer.vocab_size,
        seq_len=args.sequence_length,
    )
  

  print(torch.cuda.memory_summary())

  with torch.device("meta"):
    with set_default_dtype(model_dtype):
      model = Transformer(model_config)

  assert not (args.quantization and args.quantization_torchao)

  if args.quantization:
    logger.info("Quantizing model weights with transformer engine...")
    # Create an FP8 recipe. Note: All input args are optional.
    fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)

  if args.quantization_torchao:
    logger.info("Quantizing model weights with torchao...")
    # quantize_(model, float8_weight_only())
    config = Float8LinearConfig()
    if args.enable_fsdp_float8_all_gather:
      config = replace(config, enable_fsdp_float8_all_gather = True)
    if args.force_recompute_fp8_weight_in_bwd:
      config = replace(config, force_recompute_fp8_weight_in_bwd = True)
    convert_to_float8_training(model, config=config)



  

  if master_process:
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--> model has {total_params / 1e6} Million params\n")
  
  if args.compile:
    logger.info("Using `torch.compile`")
    # model = torch.compile(model, fullgraph=True)
    apply_compile(model)
  print(torch.cuda.memory_summary())


  logger.info("Sharding model...")
  for layer in model.layers:
        if isinstance(layer, TransformerBlock):
          fully_shard(layer)
  fully_shard(model)
  for tensor in itertools.chain(model.parameters(), model.buffers()):
    assert tensor.device == torch.device("meta")
  model.to_empty(device="cuda")
  # model.reset_parameters()

  inspect_model(model)

  print(torch.cuda.memory_summary())
  

  model.train()

  # Build Optimizers & LR Scheduler
  if args.quantize_optimizer:
    logger.info("Quantizing optimizer with torchao...")
    # Use AdamW8bit from torchao
    if args.fused_optimizer:
      raise NotImplementedError("Fused optimizer is not supported with torchao quantization")
    optimizer = AdamW8bit(model.parameters(), lr=args.learning_rate)
  else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=args.fused_optimizer)

  lr_scheduler = build_lr_scheduler(optimizer, args.lr_warmup_steps)

  # Utils
  num_flop_per_token = get_num_flop_per_token(
      get_num_params(model, exclude_embedding=True),
      model_config,
  )

  ntokens_since_last_log = 0
  ntraining_tokens_since_last_log = 0
  time_last_log = time.perf_counter()

  logger.info("Starting training!")
  train_step = 0
  # while train_step < args.training_steps:
  train_steps = []
  losses = []
  tokens_per_second_list = []
  training_tokens_per_second_list = []
  mfus = []
  tflops_list = []
  memory_summaries = []

  training_start = time.perf_counter()
  for i, (input_ids, labels) in enumerate(train_dl):
    train_step += 1
    if train_step > args.training_steps:
      break

    # Profiling
    if args.profile and args.profile_step_start == train_step:
      torch.cuda.cudart().cudaProfilerStart()
      torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

    ntokens_since_last_log += args.batch_size * args.sequence_length * world_size
    num_items_in_batch = labels.ne(-100).sum()
    ntraining_tokens_since_last_log += num_items_in_batch * world_size
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    if args.quantization:
      # Enable autocasting for the forward pass
      with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
          logits = model(input_ids)
    else:
      logits = model(input_ids)

    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(0, 1), reduction="sum")
    # del input_ids, labels
    loss = loss / num_items_in_batch
    del logits
    loss.backward()

    # Clip gradients
    clip_grad_norm_(model.parameters(), args.grad_max_norm)

    optimizer.step()
    lr_scheduler.step()

    # Logging
    if (train_step == 1 or train_step % args.logging_frequency == 0):
      time_delta = time.perf_counter() - time_last_log
      # tokens per second per device, abbreviated as tps
      tps = ntokens_since_last_log / time_delta 
      mfu = 100 * num_flop_per_token * tps / 989e12 / world_size 
      tflops = num_flop_per_token * tps / 1e12
      training_tps = ntraining_tokens_since_last_log / time_delta

      if master_process:
        vram_peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        logger.info(f"Step: {train_step} | Loss: {loss.item():.2f} | Tokens per second: {tps:.2f} | Training tokens per second (%): {100*training_tps/tps:.2f} | MFU (%): {mfu:.2f} | TFLOPs: {tflops:.2f} | VRAM peak: {vram_peak/(1024 ** 3):.2f} GB")
        train_steps.append(train_step)
        losses.append(loss.item())
        tokens_per_second_list.append(tps)
        training_tokens_per_second_list.append(training_tps)
        mfus.append(mfu)
        tflops_list.append(tflops)
        memory_summaries.append(vram_peak)
      ntokens_since_last_log = 0
      ntraining_tokens_since_last_log = 0
      time_last_log = time.perf_counter()
    
    # Profiling
    if args.profile and args.profile_step_end == train_step:
      torch.cuda.cudart().cudaProfilerStop()

  training_end = time.perf_counter()
  logger.info("Training completed")
  if master_process:
    # Save lists
    # import numpy as np
    # np.savez("train_steps.npz", train_steps=np.array(train_steps))
    # np.savez("losses.npz", losses=np.array(losses))
    # np.savez("tokens_per_second.npz", tokens_per_second=np.array(tokens_per_second_list))
    # np.savez("training_tokens_per_second.npz", training_tokens_per_second=np.array(training_tokens_per_second_list))
    # np.savez("mfus.npz", mfus=np.array(mfus))
    # np.savez("tflops.npz", tflops=np.array(tflops_list))
    # with open("memory_summaries.txt", "w") as f:
    #   for memory_summary in memory_summaries:
    #     f.write(memory_summary + "\n\n")

    # Save arguments in a json dict
    args_dict = vars(args)
    args_dict["train_steps"] = train_steps
    args_dict["losses"] = losses
    args_dict["tokens_per_second"] = tokens_per_second_list
    args_dict["training_tokens_per_second"] = training_tokens_per_second_list
    args_dict["mfus"] = mfus
    args_dict["tflops"] = tflops_list
    args_dict["memory_summaries"] = memory_summaries
    args_dict["training_start"] = training_start
    args_dict["training_end"] = training_end
    args_dict["training_duration"] = training_end - training_start
    name = f"compile_{args.compile}_quantization_{args.quantization_torchao}_fused_optimizer_{args.fused_optimizer}_quantized_optimizer_{args.quantize_optimizer}_world_size_{world_size}_batch_size_{args.batch_size}_enable_fsdp_float8_all_gather_{args.enable_fsdp_float8_all_gather}_force_recompute_fp8_weight_in_bwd_{args.force_recompute_fp8_weight_in_bwd}.json"


    def make_json_serializable(obj):
      """Convert non-serializable objects to serializable ones"""
      if hasattr(obj, 'tolist'):  # For tensors
          return obj.tolist() if obj.numel() > 1 else obj.item()
      elif hasattr(obj, '__dict__'):  # For custom objects
          return obj.__dict__
      elif isinstance(obj, (list, tuple)):
          return [make_json_serializable(item) for item in obj]
      elif isinstance(obj, dict):
          return {key: make_json_serializable(value) for key, value in obj.items()}
      else:
          return obj
    # Convert args_dict to a serializable format
    args_dict = make_json_serializable(args_dict)

    os.makedirs("/iopsstor/scratch/cscs/$USER/lai25/a2/results", exist_ok=True)
    path = os.path.join("results", name)
    with open(path, "w") as f:
      json.dump(args_dict, f, indent=4)

  dist.destroy_process_group()


if __name__ == "__main__":
  init_logger()
  args = get_args()
  train(args)