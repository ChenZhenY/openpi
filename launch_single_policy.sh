CUDA_VISIBLE_DEVICES=1 \
uv run scripts/serve_policy.py \
    --port 8001 \
    policy:checkpoint \
    --policy.config=pi05_liberogoal_filtered_bc_lora \
    --policy.dir=/research/data/zhenyang/openpi/checkpoints/pi05_liberogoal_task1_batch256_it5k
    # --policy.config.data.repo_id=lerobot_filtered_bc_libero_goal_task0_1024