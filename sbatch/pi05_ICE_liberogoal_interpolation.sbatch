#!/bin/bash
#SBATCH -J pi05_liberogoal_interpolation_30steps_1206_gen         # Job name
#SBATCH -N 1 --ntasks-per-node=8        # Number of nodes and tasks per node
#SBATCH --gres=gpu:H100:2               # Request 8xH100 GPUs
#SBATCH --mem-per-cpu=128G                # Memory per core
#SBATCH -t 4:00:00                     # Duration of the job, 2h-> ~1k steps
#SBATCH -o ./output/%x-%j.out           # Combined output and error messages file, with job name

cd /home/hice1/zchen927/scratch/openpi  # Change to openpi directory

source $HOME/.local/bin/env # source the uv env

# dataset environment variables
export OPENPI_DATA_HOME=/home/hice1/zchen927/scratch/openpi/assets # set the cache directory
export HF_LEROBOT_HOME=/storage/cedar/cedar0/cedarp-dxu345-0/zhenyang/datasets
export HF_DATASETS_CACHE=/home/hice1/zchen927/scratch/datasets/cache # set the cache directory

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 # train with 95% of GPU memory

# uv run scripts/train.py \
#     pi05_libero_cotraining_interpolation_1122 \
#     --exp-name=liberogoal_pi_libero_cotraining_interpolation_50steps_1127 \
#     --fsdp-devices=2 \
#     --batch-size=128 \
#     --resume

# uv run scripts/train.py \
#     pi05_libero_cotraining_interpolation_1122 \
#     --exp-name=liberogoal_pi_libero_cotraining_interpolation_30steps_test \
#     --fsdp-devices=2 \
#     --batch-size=128

CONFIG_NAME=pi05_libero_cotraining_interpolation
EXP_NAME=pi05_liberogoal_interpolation_30steps_1206_gen
# REPO_IDS="${1:-pi_libero_lerobot lerobot_interpolation_50steps_1112}"

# CONFIG_NAME=pi05_libero_cotraining_interpolation
# EXP_NAME=pi05_liberogoal_interpolation_30steps_1206_new
# REPO_IDS="${1:-pi_libero_lerobot lerobot_interpolation_50steps_1112}"

FSDP_DEVICES=2
BATCH_SIZE=128

uv run scripts/train.py \
    $CONFIG_NAME \
    --exp-name=$EXP_NAME \
    --fsdp-devices=$FSDP_DEVICES \
    --batch-size=$BATCH_SIZE
