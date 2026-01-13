#!/bin/bash
#SBATCH -J pi05_liberogoal_interpolation_SRBC         # Job name
#SBATCH --partition=overcap
#SBATCH --nodes=1   # NOTE: try to use GPUs on the same node to reduce communication overhead
#SBATCH --gres=gpu:a40:4               # Request how many H100 GPUs per node
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-gpu=32G               # RAM per GPU Memory per core
#SBATCH -t 16:00:00                     # Duration of the job
#SBATCH -o ./logs/%A/task_%a/out.log
#SBATCH -e ./logs/%A/task_%a/err.log
#SBATCH --exclude='xaea-12,dendrite,synapse,uniblab'

BATCH_SIZE=64
FSDP_DEVICES=4
EXPT_NAME=liberogoal_pi_interpolation1122_SRBC_0113
CONFIG_NAME=pi05_libero_cotraining_SRBC
RESUME=False

cd /coc/flash7/zhenyang/openpi  # Change to openpi directory
source $HOME/.bashrc
source $HOME/.local/bin/env # source the uv env

# dataset environment variables
export OPENPI_DATA_HOME=/coc/flash7/zhenyang/openpi/assets # set the cache directory (scratch will be deleted every 2 months)
export HF_LEROBOT_HOME=/coc/cedarp-dxu345-0/zhenyang/datasets
export HF_DATASETS_CACHE=/coc/flash7/zhenyang/data/cache # set the cache directory (scratch will be deleted every 2 months)

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 # train with 95% of GPU memory

uv run scripts/train.py \
    $CONFIG_NAME \
    --exp-name=$EXPT_NAME \
    --fsdp-devices=$FSDP_DEVICES \
    --batch-size=$BATCH_SIZE \
    $(if [ "$RESUME" = "True" ]; then echo "--resume"; fi)