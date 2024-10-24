#!/bin/bash
#SBATCH --job-name=my_job_name              # Job name
#SBATCH --output=my_job_output_%j.out       # Output file (%j expands to jobId)
#SBATCH --error=my_job_error_%j.err         # Error file (%j expands to jobId)
#SBATCH --nodes=2                           # Number of nodes
#SBATCH --ntasks=16                         # Total number of tasks (adjust as needed)
#SBATCH --ntasks-per-node=8                 # Number of tasks per node
#SBATCH --cpus-per-task=16                  # Number of CPU cores per task (adjust as needed)
#SBATCH --mem=64g                          # Total memory per node (adjust as needed)
#SBATCH --time=48:00:00                     # Maximum time limit hrs:min:sec
#SBATCH --partition=gpuA100x8               # Partition name
#SBATCH --gres=gpu:1                        # Number of GPUs per node
#SBATCH --account=bdau-delta-gpu            # Account name

# Load necessary modules
module load python/3.11.6
module load cuda/11.8                       # Load CUDA 11.8

# Activate your virtual environment
source ~/env4/bin/activate

# Verify Python executable
which python

# Print some information about the job
echo "Job started on $(date)"
echo "Running on node $(hostname)"
echo "Using the following GPUs:"
nvidia-smi

# Run your Python script with the absolute path to Python executable
# Use torch.distributed.launch for multi-node training
srun ~/env4/bin/python -m torch.distributed.launch --nproc_per_node=8 /u/mbanisharifdehkordi/Github/Darshan-logs-Processing/gnn_training48.py

# Print job end time
echo "Job ended on $(date)"
