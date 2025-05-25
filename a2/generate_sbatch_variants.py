import os
from itertools import product

base_script = """#!/bin/bash

#SBATCH --account=a-large-sc
#SBATCH --partition=normal
#SBATCH --time=00:14:59
#SBATCH --job-name=lsai
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node={gpus}
#SBATCH --environment=myenv
#SBATCH --no-requeue

set -eo pipefail

echo "START TIME: $(date)"
echo "[sbatch-master] running on $(hostname)"
echo "[sbatch-master] SLURM_NODELIST: $SLURM_NODELIST"
echo "[sbatch-master] SLURM_NNODES: $SLURM_NNODES"
echo "[sbatch-master] SLURM_NODEID: $SLURM_NODEID"

export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=12345

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ASSIGNMENT_DIR="/iopsstor/scratch/cscs/$USER/lai25/a2"

TRAINING_CMD="
torchrun \\
--nnodes=$SLURM_NNODES \\
--node_rank=$SLURM_NODEID \\
--nproc_per_node={nproc_per_node} \\
--master_addr=$MASTER_ADDR \\
--master_port=$MASTER_PORT \\
$ASSIGNMENT_DIR/train.py \\
--sequence-length 4096 \\
--batch-size 1 \\
--learning-rate 5e-5 \\
--lr-warmup-steps 100 \\
--training-steps 1000 \\
{extra_flags}
"

srun bash -c "$TRAINING_CMD"

echo "END TIME: $(date)"
"""

os.makedirs("generated_jobs_niklas", exist_ok=True)

# (nodes, gpus_per_node) combinations
gpu_configs = [(4, 4)]
# gpu_configs = [(1, 4), (2, 4), (3, 4), (4, 4)]
# gpu_configs = [(1, 1), (1, 4), (2, 4), (3, 4), (4, 4)]

torchao_options = [True, False]
compile_options = [True, False]

for (nodes, gpus), torchao, compile in product(gpu_configs, torchao_options, compile_options):
    nproc = gpus  # same number of processes as GPUs per node
    if torchao:
        for allgather, fp8recompute, quant_opt in product([True, False], repeat=3):
            flags = ["--quantization_torchao"]
            if allgather:
                flags.append("--enable_fsdp_float8_all_gather")
            if fp8recompute:
                flags.append("--force_recompute_fp8_weight_in_bwd")
            if quant_opt:
                flags.append("--quantize_optimizer")
            if compile:
                flags.append("--compile")

            jobname = (
                f"sbatch_nodes{nodes}_gpus{gpus}_"
                f"torchao_ag{int(allgather)}_fp8{int(fp8recompute)}_qopt{int(quant_opt)}_compile{int(compile)}.sh"
            )
            full_script = base_script.format(
                nodes=nodes,
                gpus=gpus,
                nproc_per_node=nproc,
                extra_flags=" \\\n".join(flags)
            )
            with open(f"generated_jobs_niklas/{jobname}", "w") as f:
                f.write(full_script)

    else:
        for fused_opt in [True, False]:
            flags = []
            if fused_opt:
                flags.append("--fused-optimizer")
            if compile:
                flags.append("--compile")

            jobname = (
                f"sbatch_nodes{nodes}_gpus{gpus}_"
                f"notorchao_fopt{int(fused_opt)}_compile{int(compile)}.sh"
            )
            full_script = base_script.format(
                nodes=nodes,
                gpus=gpus,
                nproc_per_node=nproc,
                extra_flags=" \\\n".join(flags)
            )
            with open(f"generated_jobs_niklas/{jobname}", "w") as f:
                f.write(full_script)
