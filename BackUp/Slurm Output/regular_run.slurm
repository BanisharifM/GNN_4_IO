#!/bin/bash
#SBATCH --job-name=my_job_name              # Job name
#SBATCH --output=my_job_output_%j.out       # Output file (%j expands to jobId)
#SBATCH --error=my_job_error_%j.err         # Error file (%j expands to jobId)
#SBATCH --nodes=1                           # Number of nodes (adjust as needed)
#SBATCH --ntasks=2                          # Total number of tasks (GPUs across all nodes)
#SBATCH --ntasks-per-node=2                 # Number of tasks (GPUs) per node
#SBATCH --cpus-per-task=16                  # Number of CPU cores per task (adjust as needed)
#SBATCH --mem=128g                          # Total memory per node
#SBATCH --time=6:00:00                      # Maximum time limit
#SBATCH --partition=gpuA100x8               # Partition name
#SBATCH --gres=gpu:2                        # Number of GPUs per node
#SBATCH --account=bdau-delta-gpu            # Account name

# Load necessary modules
module load python/3.11.6
module load cuda/11.8                       # Load CUDA 11.8

# Activate your virtual environment
source /u/mbanisharifdehkordi/env4/bin/activate

# Set environment variables for distributed training
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=29501  # Set this to a consistent value
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Debugging output
echo "WORLD_SIZE: $WORLD_SIZE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "RANK: $RANK"
echo "LOCAL_RANK: $LOCAL_RANK"

# Use the full path to the Python executable from your virtual environment
PYTHON_EXEC=/u/mbanisharifdehkordi/env4/bin/python

# Launch the training script using srun with the environment variables
srun --export=ALL $PYTHON_EXEC /u/mbanisharifdehkordi/Github/GNN_4_IO/gnn_training104.py
