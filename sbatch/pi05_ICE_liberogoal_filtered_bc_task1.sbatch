#!/bin/bash
#SBATCH -J pi05_liberogoal_filtered_bc_task1_lora          # Job name
#SBATCH -N 1 --ntasks-per-node=8        # Number of nodes and tasks per node
#SBATCH --gres=gpu:h100:2               # Request 8xH100 GPUs
#SBATCH --mem-per-cpu=128G                # Memory per core
#SBATCH -t 8:00:00                     # Duration of the job
#SBATCH -o ./output/%x-%j.out           # Combined output and error messages file, with job name

cd /home/hice1/zchen927/scratch/openpi  # Change to openpi directory

source $HOME/.local/bin/env # source the uv env

# dataset environment variables
export OPENPI_DATA_HOME=/home/hice1/zchen927/scratch/openpi/assets # set the cache directory
export HF_LEROBOT_HOME=/storage/cedar/cedar0/cedarp-dxu345-0/zhenyang/datasets
export HF_DATASETS_CACHE=/home/hice1/zchen927/scratch/datasets/cache # set the cache directory

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 # train with 95% of GPU memory

uv run scripts/train.py \
    pi05_liberogoal_filtered_bc_lora \
    --exp-name=liberogoal_filtered_bc_lora_batch256_task1 \
    --data.repo_id=lerobot_filtered_bc_libero_goal_task1_1024 \
    --fsdp-devices=2 \
    --batch-size=256 \
    --resume