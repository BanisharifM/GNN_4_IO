#!/bin/bash
#SBATCH --job-name=my_job_name              # Job name
#SBATCH --output=my_job_output_%j.out       # Output file (%j expands to jobId)
#SBATCH --error=my_job_error_%j.err         # Error file (%j expands to jobId)
#SBATCH --ntasks=1                          # Number of tasks (usually 1 for GPU jobs)
#SBATCH --cpus-per-task=4                   # Number of CPU cores per task (adjust as needed)
#SBATCH --mem=16g                           # Total memory per node (adjust as needed)
#SBATCH --time=01:00:00                     # Maximum time limit hrs:min:sec
#SBATCH --partition=gpuA100x4-interactive   # Partition name
#SBATCH --gres=gpu:4                        # Number of GPUs per node
#SBATCH --account=bdau-delta-gpu 

# Load necessary modules
module load python/3.11.6
module load cuda/11.0                       # Load CUDA if needed

# Activate your virtual environment
source ~/myenv/bin/activate

# Print some information about the job
echo "Job started on $(date)"
echo "Running on node $(hostname)"
echo "Using the following GPUs:"
nvidia-smi

# Run your Python script
srun python3 gnn_training_test.py

# Print job end time
echo "Job ended on $(date)"

