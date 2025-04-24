#!/bin/bash
#SBATCH --job-name=LSAIE_a5
#SBATCH --account=a-large-sc
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:10:00
#SBATCH --output=output_5_%j.log
#SBATCH --error=output_5_%j.err
#SBATCH --environment=myenv
# Stop the script if a command fails or if an undefined variable is used
set -eo pipefail
# The sbatch script is executed by only one node.
echo "[sbatch-master] running on $(hostname)"
echo "[sbatch-master] SLURM_NODELIST: $SLURM_NODELIST"
echo "[sbatch-master] SLURM_NNODES: $SLURM_NNODES"
echo "[sbatch-master] SLURM_NODEID: $SLURM_NODEID"

# The defined environment vars will be shared with the other compute nodes.
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=12345 # Choose an unused port
export FOOBAR=666
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))

WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))

echo "[sbatch-master] execute command on compute nodes"
# The command that will run on each process
CMD="
# print current environment variables
echo \"[srun] world_size=\$(WORLD_SIZE) rank=\$SLURM_PROCID host=\$(hostname) noderank=\$SLURM_NODEID localrank=\$SLURM_LOCALID wrong_host=$(hostname)\"
# run the script
torchrun \
--nnodes="${SLURM_NNODES}" \
--node_rank=\$SLURM_NODEID \
--nproc_per_node=4 \
--master_addr="${MASTER_ADDR}" \
--master_port="${MASTER_PORT}" \
/iopsstor/scratch/cscs/$USER/lai25/a5/test.py
"

# Submits the CMD to all the processes on all the nodes.
srun bash -c "$CMD"
echo "[sbatch-master] task finished"
