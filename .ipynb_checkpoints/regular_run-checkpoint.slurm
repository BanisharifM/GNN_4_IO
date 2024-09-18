#!/bin/bash
#SBATCH --job-name=GNN_Training
#SBATCH --output=my_job_output_%j.out
#SBATCH --error=my_job_error_%j.err
#SBATCH --nodes=1                           
#SBATCH --ntasks=1                          
#SBATCH --ntasks-per-node=1                 
#SBATCH --cpus-per-task=8                   
#SBATCH --mem=128g                           
#SBATCH --time=24:00:00                      
#SBATCH --partition=gpuA100x8               
#SBATCH --gres=gpu:1                        
#SBATCH --account=bdau-delta-gpu            

# Load necessary modules
module load python/3.11.6
module load cuda/11.8

# Activate your virtual environment
source /u/mbanisharifdehkordi/env4/bin/activate

# Set environment variables for distributed training
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=localhost  # Use localhost for single node
export MASTER_PORT=29511      # Use a different, unused port
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# NCCL settings
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_DEBUG=INFO

# Debugging output
echo "WORLD_SIZE: $WORLD_SIZE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "RANK: $RANK"
echo "LOCAL_RANK: $LOCAL_RANK"

# Run the Python script with srun
srun --export=ALL python /u/mbanisharifdehkordi/Github/GNN_4_IO/gnn_training105.py
