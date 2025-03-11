#!/bin/bash

#SBATCH --account=a-large-sc
#SBATCH --time=00:14:59
#SBATCH --job-name=lsai
#SBATCH --output=/iopsstor/scratch/cscs/%u/assignment-2/logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=/iopsstor/scratch/cscs/dveranieto/ngc_pt_jan.toml     # Vanilla 25.01 PyTorch NGC Image 
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

echo "START TIME: $(date)"

# Set up ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ASSIGNMENT_DIR="/iopsstor/scratch/cscs/$USER/assignment-2"

CMD_PREFIX="numactl --membind=0-3"

# cd $ASSIGNMENT_DIR

TRAINING_CMD="python3 $ASSIGNMENT_DIR/train.py \
# TRAINING_CMD="uv run train.py \
    --sequence-length 2048 \
    --batch-size 1 \
    --learning-rate 5e-5 \
    --lr-warmup-steps 100 \
    --training-steps 1000 \
    "

srun --cpus-per-task $SLURM_CPUS_PER_TASK bash -c "$CMD_PREFIX $TRAINING_CMD"

echo "END TIME: $(date)"