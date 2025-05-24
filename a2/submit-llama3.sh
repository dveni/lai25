#!/bin/bash

#SBATCH --account=a-large-sc
#SBATCH --time=00:14:59
#SBATCH --job-name=lsai
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=myenv #/iopsstor/scratch/cscs/dveranieto/ngc_pt_jan.toml     # Vanilla 25.01 PyTorch NGC Image 
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

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

CMD_PREFIX="numactl --membind=0-3"

# cd $ASSIGNMENT_DIR

# TRAINING_CMD="uv run train.py \
# TRAINING_CMD="python3 $ASSIGNMENT_DIR/train.py \
#     --sequence-length 4096 \
#     --batch-size 1 \
#     --learning-rate 5e-5 \
#     --lr-warmup-steps 100 \
#     --training-steps 1000 \
#     --compile \
#     --fused-optimizer \
#     --quantization_torchao
#     "
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
--compile \
--fused-optimizer \
"
# --quantization_torchao

srun --cpus-per-task $SLURM_CPUS_PER_TASK bash -c "$CMD_PREFIX $TRAINING_CMD"

echo "END TIME: $(date)"