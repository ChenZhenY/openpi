#!/bin/bash
#SBATCH -J pi05_liberogoal_interpolation_50steps_inter2_1122         # Job name
#SBATCH -N 1 --ntasks-per-node=8        # Number of nodes and tasks per node
#SBATCH --gres=gpu:h100:2               # Request 8xH100 GPUs
#SBATCH --mem-per-cpu=128G                # Memory per core
#SBATCH -t 3:00:00                     # Duration of the job
#SBATCH -o ./output/%x-%j.out           # Combined output and error messages file, with job name

cd /storage/home/hcoda1/2/zchen927/openpi  # Change to openpi directory

source $HOME/.local/bin/env # source the uv env

# dataset environment variables
export OPENPI_DATA_HOME=/storage/scratch1/2/zchen927/assets # set the cache directory (scratch will be deleted every 2 months)
export HF_LEROBOT_HOME=/storage/cedar/cedar0/cedarp-dxu345-0/zhenyang/datasets
export HF_DATASETS_CACHE=/storage/scratch1/2/zchen927/datasets/cache # set the cache directory (scratch will be deleted every 2 months)

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 # train with 95% of GPU memory

uv run scripts/train.py \
    pi05_libero_cotraining_interpolation_1122 \
    --exp-name=liberogoal_pi_libero_cotraining_interpolation_50steps_inter2_1122 \
    --fsdp-devices=2 \
    --batch-size=128 \
    --resume