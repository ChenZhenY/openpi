#!/bin/bash
#SBATCH --job-name=end_to_end
#SBATCH --output=logs/end_to_end_%j.out
#SBATCH --error=logs/end_to_end_%j.err
#SBATCH --partition=rl2-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --gpus-per-node="l40s:2"
#SBATCH --nodelist=dynamics
#SBATCH --mem-per-gpu=128
#SBATCH --array=0-2

set -e

BATCH_SIZES=(1 2 4)
BATCH_SIZE=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

#TODO: ablate RTC
ACTION_BROKER_TYPE=SYNC
RTC_S_MIN=5
RTC_D_INIT=3


echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "======================================"

# Get the hostnames of the allocated nodes
HOSTS=($(scontrol show hostnames $SLURM_JOB_NODELIST))
NODE=${HOSTS[0]}

# Ensure cleanup on exit
cleanup() {
    echo "Cleaning up..."
    if [ ! -z "$SERVER_JOB_PID" ] && kill -0 $SERVER_JOB_PID 2>/dev/null; then
        echo "Stopping server process (PID: $SERVER_JOB_PID)"
        kill $SERVER_JOB_PID
        wait $SERVER_JOB_PID 2>/dev/null || true
    fi
    echo "Cleanup complete"
}
trap cleanup EXIT INT TERM

# --- Step 1: Launch the server on the first node ---
srun --ntasks=1 --gpus-per-node="l40s:1" --cpus-per-task=2 --overlap --exact -w $NODE bash -c "
    echo 'Starting server on $NODE... for batch_size=$BATCH_SIZE'
    source ~/.bashrc
    source .venv/bin/activate
    uv run scripts/serve_policy.py --env=LIBERO --batch-size=$BATCH_SIZE --log-dir=logs/server
" &
SERVER_JOB_PID=$!
echo "Server launched (PID $SERVER_JOB_PID). Waiting for it to initialize..."

# --- Step 2: Run the client on the second node ---
if [ "$ACTION_BROKER_TYPE" = "RTC" ]; then
    ACTION_CHUNK_BROKER_FLAGS="--action-chunk-broker.broker-type RTC --action-chunk-broker.s-min $RTC_S_MIN --action-chunk-broker.d-init $RTC_D_INIT"
elif [ "$ACTION_BROKER_TYPE" = "SYNC" ]; then
    ACTION_CHUNK_BROKER_FLAGS="--action-chunk-broker.broker-type SYNC"
else
    echo "Invalid action broker type: $ACTION_BROKER_TYPE"
    exit 1
fi

for NUM_ROBOTS in 1 5 10 20 30 40; do
    srun --ntasks=1 --gpus-per-node="l40s:1" --cpus-per-task=40 --overlap --exact -w $NODE bash -c "
        echo 'Starting client on $NODE... for num_robots=$NUM_ROBOTS'
        source scripts/libero_client.sh
        ./examples/libero/.venv/bin/python examples/libero/main_multi_robot_runtime.py \
            --host $NODE \
            --num-robots $NUM_ROBOTS \
            --task-suite-name libero_10 \
            --num-trials-per-robot 10 \
            --action-horizon 10 \
            --control-hz 20 \
            --max-steps 520 \
            --output-dir data/libero/benchmark_end_to_end/batch_size_${BATCH_SIZE}_num_robots_${NUM_ROBOTS}_broker_type_${ACTION_BROKER_TYPE} \
            --progress-type logging \
            --log-dir logs/client \
            --overwrite \
            ${ACTION_CHUNK_BROKER_FLAGS}
    "
done
echo "======================================"
echo "Completed run with args.action-chunk-broker.broker-type=$ACTION_BROKER_TYPE"
echo "======================================"
