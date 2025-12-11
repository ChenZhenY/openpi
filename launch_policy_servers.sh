#!/bin/bash

# Launch policy servers on GPUs 0-7 for EIC GPUs!
# Each server will run on a different GPU and port
# Example:
# ./launch_policy_servers.sh start --checkpoint /research/data/zhenyang/openpi/checkpoints/liberogoal_filtered_bc_lora_4k pi05_liberogoal_filtered_bc_lora

# Configuration
BASE_PORT=8000
ENV_MODE="LIBERO"
LOG_DIR="policy_logs"
CHECKPOINT_DIR=""
CONFIG_NAME=""

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to launch a policy server
launch_server() {
    local gpu_id=$1
    local port=$2
    local log_file="$LOG_DIR/policy_gpu${gpu_id}_port${port}.log"
    
    echo "Launching policy server on GPU $gpu_id, port $port..."
    echo "Log file: $log_file"
    
    # Build the command based on whether checkpoint is provided
    if [ -n "$CHECKPOINT_DIR" ] && [ -n "$CONFIG_NAME" ]; then
        echo "Using checkpoint: $CHECKPOINT_DIR with config: $CONFIG_NAME"
        # Launch the server in background with GPU assignment and checkpoint
        CUDA_VISIBLE_DEVICES=$gpu_id \
        uv run scripts/serve_policy.py \
            --port $port \
            policy:checkpoint \
            --policy.config=$CONFIG_NAME \
            --policy.dir=$CHECKPOINT_DIR \
            > "$log_file" 2>&1 &
    else
        echo "Using default policy for environment: $ENV_MODE"
        # Launch the server in background with GPU assignment (default behavior)
        CUDA_VISIBLE_DEVICES=$gpu_id \
        uv run scripts/serve_policy.py \
            --env $ENV_MODE \
            --port $port \
            > "$log_file" 2>&1 &
    fi
    
    # Store the PID
    local pid=$!
    echo "Server PID: $pid"
    echo "$pid" > "$LOG_DIR/policy_gpu${gpu_id}.pid"
    
    # Wait a moment for the server to start
    sleep 2
    
    # Check if the process is still running
    if kill -0 $pid 2>/dev/null; then
        echo "✓ Policy server on GPU $gpu_id (port $port) started successfully"
    else
        echo "✗ Failed to start policy server on GPU $gpu_id"
        return 1
    fi
}

# Function to stop all servers
stop_servers() {
    echo "Stopping all policy servers..."
    for i in {0..7}; do
        pid_file="$LOG_DIR/policy_gpu${i}.pid"
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if kill -0 $pid 2>/dev/null; then
                echo "Stopping server on GPU $i (PID: $pid)"
                kill $pid
                # Wait a moment for graceful shutdown
                sleep 1
                # Force kill if still running
                if kill -0 $pid 2>/dev/null; then
                    echo "Force killing server on GPU $i (PID: $pid)"
                    kill -9 $pid
                fi
            fi
            rm -f "$pid_file"
        fi
    done
    echo "All servers stopped."
}

# Function to check server status
check_status() {
    echo "Checking server status..."
    for i in {0..7}; do
        port=$((BASE_PORT + i))
        pid_file="$LOG_DIR/policy_gpu${i}.pid"
        
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if kill -0 $pid 2>/dev/null; then
                echo "✓ GPU $i: Running (PID: $pid, Port: $port)"
            else
                echo "✗ GPU $i: Not running (stale PID file)"
            fi
        else
            echo "✗ GPU $i: No PID file found"
        fi
    done
}

# Handle command line arguments
case "${1:-start}" in
    "start")
        # Parse additional arguments for checkpoint
        if [ "$2" = "--checkpoint" ] && [ -n "$3" ] && [ -n "$4" ]; then
            CHECKPOINT_DIR="$3"
            CONFIG_NAME="$4"
            echo "Starting policy servers on GPUs 0-7 with checkpoint..."
            echo "Checkpoint directory: $CHECKPOINT_DIR"
            echo "Config name: $CONFIG_NAME"
        else
            echo "Starting policy servers on GPUs 0-7..."
            echo "Environment: $ENV_MODE"
        fi
        echo "Base port: $BASE_PORT"
        echo "Log directory: $LOG_DIR"
        echo ""
        
        # Launch servers on GPUs 0-7
        for i in {0..7}; do
            port=$((BASE_PORT + i))
            launch_server $i $port
        done
        
        echo ""
        echo "All servers launched! Check status with: $0 status"
        echo "Stop servers with: $0 stop"
        ;;
    "stop")
        stop_servers
        ;;
    "status")
        check_status
        ;;
    "restart")
        stop_servers
        sleep 2
        # Pass through checkpoint arguments if provided
        if [ "$2" = "--checkpoint" ] && [ -n "$3" ] && [ -n "$4" ]; then
            $0 start --checkpoint "$3" "$4"
        else
            $0 start
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart} [--checkpoint <checkpoint_dir> <config_name>]"
        echo ""
        echo "Commands:"
        echo "  start   - Launch policy servers on GPUs 0-7 (default)"
        echo "  stop    - Stop all running policy servers"
        echo "  status  - Check status of all servers"
        echo "  restart - Stop and restart all servers"
        echo ""
        echo "Options:"
        echo "  --checkpoint <checkpoint_dir> <config_name> - Use specific checkpoint instead of default policy"
        echo ""
        echo "Examples:"
        echo "  $0 start  # Use default LIBERO policy"
        echo "  $0 start --checkpoint /path/to/checkpoint pi05_liberogoal_filtered_bc_lora"
        echo ""
        echo "Server configuration:"
        echo "  Environment: $ENV_MODE"
        echo "  Ports: $BASE_PORT-$((BASE_PORT + 7))"
        echo "  Logs: $LOG_DIR/"
        exit 1
        ;;
esac
