#!/bin/bash

DATASET_NAME=lerobot_filtered_bc_libero_goal_task1_1024

cd /home/hice1/zchen927/scratch/openpi
source $HOME/.local/bin/env

export OPENPI_DATA_HOME=/home/hice1/zchen927/scratch/openpi/assets # set the cache directory
export HF_LEROBOT_HOME=/storage/cedar/cedar0/cedarp-dxu345-0/zhenyang/datasets
export HF_DATASETS_CACHE=/home/hice1/zchen927/scratch/datasets/cache # set the cache directory

for chunk in $HF_LEROBOT_HOME/$DATASET_NAME/data/chunk-*; do
    uv run fix_lerobot_dataset.py $chunk --fix
done

uv run scripts/compute_norm_stats.py \
    --config-name=pi05_liberogoal_filtered_bc_lora \
    --repo-id=$DATASET_NAME
# uv run scripts/compute_norm_stats.py --config-name pi05_liberogoal_filtered_bc_task1
