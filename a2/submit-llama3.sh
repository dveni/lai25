#!/bin/bash

#SBATCH --account=a-large-sc
#SBATCH --partition=normal
#SBATCH --time=00:14:59
#SBATCH --job-name=lsai
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --environment=myenv #/iopsstor/scratch/cscs/dveranieto/ngc_pt_jan.toml     # Vanilla 25.01 PyTorch NGC Image 
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs


set -eo pipefail


echo "START TIME: $(date)"
echo "[sbatch-master] running on $(hostname)"
echo "[sbatch-master] SLURM_NODELIST: $SLURM_NODELIST"
echo "[sbatch-master] SLURM_NNODES: $SLURM_NNODES"
echo "[sbatch-master] SLURM_NODEID: $SLURM_NODEID"

# The defined environment vars will be shared with the other compute nodes.
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=12345

# Set up ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ASSIGNMENT_DIR="/iopsstor/scratch/cscs/$USER/lai25/a2"

TRAINING_CMD="
torchrun \
--nnodes="${SLURM_NNODES}" \
--node_rank=\$SLURM_NODEID \
--nproc_per_node=1 \
--master_addr="${MASTER_ADDR}" \
--master_port="${MASTER_PORT}" \
$ASSIGNMENT_DIR/train.py \
--sequence-length 4096 \
--batch-size 1 \
--learning-rate 5e-5 \
--lr-warmup-steps 100 \
--training-steps 1000 \
--quantization_torchao \
--compile \
--quantize_optimizer \
--enable_fsdp_float8_all_gather \
--force_recompute_fp8_weight_in_bwd \
--training-steps=20
"
# --fused-optimizer \

srun bash -c "$TRAINING_CMD"

echo "END TIME: $(date)"