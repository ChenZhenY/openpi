#!/bin/bash
#SBATCH -J pi05_droid_multidataset         # Job name
#SBATCH --partition=overcap
#SBATCH --nodes=1   # NOTE: try to use GPUs on the same node to reduce communication overhead
#SBATCH --gres=gpu:a40:4               # Request how many GPUs per node
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-gpu=32G               # RAM per GPU Memory per core
#SBATCH -t 12:00:00                     # Duration of the job
#SBATCH -o ./logs/%A/task_%a/out.log
#SBATCH -e ./logs/%A/task_%a/err.log
#SBATCH --exclude='xaea-12,dendrite,synapse,uniblab,puma'

BATCH_SIZE=32 # 32 for DROID (larger action_dim=32)
FSDP_DEVICES=2
EXPT_NAME=droid_6tasks_0118
CONFIG_NAME=pi05_droid_finetune_multidataset
RESUME=False

cd /coc/flash7/zhenyang/openpi  # Change to openpi directory
source $HOME/.bashrc
source $HOME/.local/bin/env # source the uv env

# dataset environment variables
export OPENPI_DATA_HOME=/coc/flash7/zhenyang/openpi/assets # set the cache directory
export HF_LEROBOT_HOME=/coc/cedarp-dxu345-0/zhenyang/datasets
export HF_DATASETS_CACHE=/coc/flash7/zhenyang/data/cache # set the cache directory

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 # train with 90% of GPU memory

uv run scripts/train.py \
    $CONFIG_NAME \
    --exp-name=$EXPT_NAME \
    --fsdp-devices=$FSDP_DEVICES \
    --batch-size=$BATCH_SIZE \
    $(if [ "$RESUME" = "True" ]; then echo "--resume"; fi)

