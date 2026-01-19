#!/bin/bash
#SBATCH -J pi05_liberogoal_SRBC_0116_0.5         # Job name
#SBATCH --partition=overcap
#SBATCH --nodes=1   # NOTE: try to use GPUs on the same node to reduce communication overhead
#SBATCH --gres=gpu:a40:4               # Request how many H100 GPUs per node
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-gpu=32G               # RAM per GPU Memory per core
#SBATCH -t 12:00:00                     # Duration of the job
#SBATCH -o ./logs/%A/task_%a/out.log
#SBATCH -e ./logs/%A/task_%a/err.log
#SBATCH --exclude='xaea-12,dendrite,synapse,uniblab,puma'

BATCH_SIZE=32 # 32 for 4GPUs
FSDP_DEVICES=2
EXPT_NAME=liberogoal_pi_SRBC_success_vs_mi_0.5_random_0116_4k
CONFIG_NAME=pi05_libero_cotraining_SRBC
RESUME=True

cd /coc/flash7/zhenyang/openpi  # Change to openpi directory
source $HOME/.bashrc
source $HOME/.local/bin/env # source the uv env

# dataset environment variables
export OPENPI_DATA_HOME=/coc/flash7/zhenyang/openpi/assets # set the cache directory (scratch will be deleted every 2 months)
export HF_LEROBOT_HOME=/coc/cedarp-dxu345-0/zhenyang/datasets
export HF_DATASETS_CACHE=/coc/flash7/zhenyang/data/cache # set the cache directory (scratch will be deleted every 2 months)

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 # train with 95% of GPU memory

# Force NFS cache refresh (list parent directories to populate cache)
# ls -la /coc/cedarp-dxu345-0/zhenyang/datasets/ > /dev/null 2>&1
# ls -la /coc/cedarp-dxu345-0/zhenyang/datasets/pi_libero_lerobot/ > /dev/null 2>&1
# ls -la /coc/cedarp-dxu345-0/zhenyang/datasets/pi_libero_lerobot/meta/ > /dev/null 2>&1
# ls -la /coc/cedarp-dxu345-0/zhenyang/datasets/lerobot_interpolation_50steps_1112/ > /dev/null 2>&1
# ls -la /coc/cedarp-dxu345-0/zhenyang/datasets/lerobot_interpolation_50steps_1112/meta/ > /dev/null 2>&1
# ls -la /coc/cedarp-dxu345-0/zhenyang/datasets/0116_sampled_800_success_vs_mi_1.0_random/ > /dev/null 2>&1
# ls -la /coc/cedarp-dxu345-0/zhenyang/datasets/0116_sampled_800_success_vs_mi_1.0_random/meta/ > /dev/null 2>&1
# sleep 2

uv run scripts/train.py \
    $CONFIG_NAME \
    --exp-name=$EXPT_NAME \
    --fsdp-devices=$FSDP_DEVICES \
    --batch-size=$BATCH_SIZE \
    $(if [ "$RESUME" = "True" ]; then echo "--resume"; fi)