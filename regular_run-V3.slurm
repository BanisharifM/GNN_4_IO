#!/bin/bash
#SBATCH --job-name=my_job_name              # Job name
#SBATCH --output=my_job_output_%j.out       # Output file (%j expands to jobId)
#SBATCH --error=my_job_error_%j.err         # Error file (%j expands to jobId)
#SBATCH --nodes=2                           # Number of nodes
#SBATCH --ntasks=4                          # Total number of tasks (GPUs across all nodes)
#SBATCH --ntasks-per-node=2                 # Number of tasks (GPUs) per node
#SBATCH --cpus-per-task=16                  # Number of CPU cores per task
#SBATCH --mem=128g                          # Total memory per node
#SBATCH --time=6:00:00                      # Maximum time limit
#SBATCH --partition=gpuA100x8               # Partition name
#SBATCH --gres=gpu:2                        # Number of GPUs per node
#SBATCH --account=bdau-delta-gpu            # Account name

# Load necessary modules
module load python/3.11.6
module load cuda/11.8                       # Load CUDA 11.8

# Source the virtual environment
source /u/mbanisharifdehkordi/env4/bin/activate

# Set MASTER_ADDR and MASTER_PORT
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)  # Get the first node
export MASTER_PORT=12345  # You can use any free port

# Print some information about the job
echo "Job started on $(date)"
echo "Running on node $(hostname)"
echo "Using the following GPUs:"
nvidia-smi

# Run the distributed training script
srun /u/mbanisharifdehkordi/env4/bin/python /u/mbanisharifdehkordi/Github/GNN_4_IO/gnn_training104.py

# Print job end time
echo "Job ended on $(date)"
