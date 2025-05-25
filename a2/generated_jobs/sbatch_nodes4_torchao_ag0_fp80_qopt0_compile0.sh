#!/bin/bash

#SBATCH --account=a-large-sc
#SBATCH --partition=normal
#SBATCH --time=00:14:59
#SBATCH --job-name=lsai
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
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
torchrun \
--nnodes=$SLURM_NNODES \
--node_rank=$SLURM_NODEID \
--nproc_per_node=4 \
--master_addr=$MASTER_ADDR \
--master_port=$MASTER_PORT \
$ASSIGNMENT_DIR/train.py \
--sequence-length 4096 \
--batch-size 1 \
--learning-rate 5e-5 \
--lr-warmup-steps 100 \
--training-steps 1000 \
--quantization_torchao
"

srun bash -c "$TRAINING_CMD"

echo "END TIME: $(date)"
