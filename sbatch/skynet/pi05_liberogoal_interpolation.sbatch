#!/bin/bash
#SBATCH -Agts-ylin715-paid
#SBATCH -J pi05_liberogoal_interpolation_MI_sampled         # Job name
#SBATCH -N 1 # node
#SBATCH --gres=gpu:H100:2               # Request how many H100 GPUs per node
#SBATCH --mem-per-cpu=8G                # Memory per core
#SBATCH -t 8:00:00                     # Duration of the job
#SBATCH -o ./output/%x-%j.out           # Combined output and error messages file, with job name

# PACE recomeend not setting --ntasks-per-node   #SBSTCH -q embers  #SBATCH --gpus=H100:2    # Request how many H100 GPUs, total node will be gpus/gres

BATCH_SIZE=64
FSDP_DEVICES=2
EXPT_NAME=liberogoal_pi_interpolation_MI_sampled_0109
RESUME=False

cd /storage/home/hcoda1/2/zchen927/p-dxu345-0/openpi  # Change to openpi directory
source $HOME/.local/bin/env # source the uv env

# dataset environment variables
export OPENPI_DATA_HOME=/storage/home/hcoda1/2/zchen927/p-dxu345-0/openpi/assets # set the cache directory (scratch will be deleted every 2 months)
export HF_LEROBOT_HOME=/storage/cedar/cedar0/cedarp-dxu345-0/zhenyang/datasets
export HF_DATASETS_CACHE=/storage/home/hcoda1/2/zchen927/p-dxu345-0/datasets/cache # set the cache directory (scratch will be deleted every 2 months)

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 # train with 95% of GPU memory

uv run scripts/train.py \
    pi05_libero_cotraining_interpolation \
    --exp-name=$EXPT_NAME \
    --fsdp-devices=$FSDP_DEVICES \
    --batch-size=$BATCH_SIZE \
    $(if [ "$RESUME" = "True" ]; then echo "--resume"; fi)