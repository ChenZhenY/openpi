#!/bin/bash
#SBATCH -J libero_finetune              # Job name
#SBATCH -N 1 --ntasks-per-node=8        # Number of nodes and tasks per node
#SBATCH --gres=gpu:h100:8               # Request 8xH100 GPUs
#SBATCH --mem-per-cpu=128G                # Memory per core
#SBATCH -t 2:00:00                     # Duration of the job
#SBATCH -o ./output/%x-%j.out           # Combined output and error messages file, with job name

cd /home/hice1/zchen927/scratch/openpi  # Change to openpi directory

source $HOME/.local/bin/env # source the uv env

# dataset environment variables
export OPENPI_DATA_HOME=/home/hice1/zchen927/scratch/openpi/assets # set the cache directory
export HF_LEROBOT_HOME=/home/hice1/zchen927/scratch/datasets/lerobot # set the output directory
export HF_DATASETS_CACHE=/home/hice1/zchen927/scratch/datasets/cache # set the cache directory
# export LEROBOT_HOME=/home/hice1/zchen927/scratch/datasets/lerobot # set the output directory

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 # train with 90% of GPU memory
export GOOGLE_APPLICATION_CREDENTIALS=/home/hice1/zchen927/scratch/openpi/assets/openpi-preview.json

uv run scripts/train.py \
    pi05_libero \
    --exp-name=libero_finetune \
    --fsdp-devices=8 \
    --batch-size=256 \
    --resume
    # --overwrite