"""
Replay debug data from a saved episode.

This script loads saved debug data (observations, noise, actions) and replays them
in the LIBERO environment. There are two modes:

1. --use_saved_actions: Directly use the saved output_actions from debug data
2. Default: Send saved observation and noise to policy to reproduce actions

Usage:
    # Use saved actions directly (fastest, guaranteed deterministic)
    python examples/libero/replay_debug_data.py \
        --debug_data_dir data/libero/multi_robot_videos/0/0_libero_10_8_success \
        --use_saved_actions

    # Re-infer actions from policy with saved noise (verifies reproducibility)
    python examples/libero/replay_debug_data.py \
        --debug_data_dir data/libero/multi_robot_videos/0/0_libero_10_8_success \
        --host localhost --port 8080

The script will:
1. Load metadata to get task info
2. Load debug data chunks
3. For each chunk, either use saved actions or re-infer from policy
4. Execute the actions in the environment
5. Save a replay video and report success/failure
"""

import argparse
import json
import logging
import pathlib
import time
from dataclasses import dataclass
from typing import List, Optional

import imageio
import numpy as np
from libero.libero import benchmark
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from openpi_client import websocket_client_policy as _websocket_client_policy
from examples.libero import utils
from examples.libero.env import LiberoSimEnvironment

LIBERO_ENV_RESOLUTION = 256


@dataclass
class ReplayConfig:
    """Configuration for replay."""
    debug_data_dir: pathlib.Path
    host: str
    port: int
    seed: int = 7
    resize_size: int = 224
    num_steps_wait: int = 10
    max_steps: int = 500
    control_hz: int = 20
    action_horizon: int = 50  # Number of actions per chunk (model's action horizon)
    action_dim: int = 7  # Actual robot action dimension (6 DoF + gripper for LIBERO)
    output_video: Optional[str] = None
    use_saved_actions: bool = False  # If True, use saved output_actions directly


def load_metadata(debug_data_dir: pathlib.Path) -> dict:
    """Load episode metadata."""
    metadata_path = debug_data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path) as f:
        return json.load(f)


def load_debug_chunks(debug_data_dir: pathlib.Path) -> List[dict]:
    """Load all debug data chunks in order."""
    chunk_dir = debug_data_dir / "debug_data"
    if not chunk_dir.exists():
        raise FileNotFoundError(f"Debug data directory not found: {chunk_dir}")
    
    chunk_files = sorted(chunk_dir.glob("chunk_*.npy"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {chunk_dir}")
    
    chunks = []
    for chunk_file in chunk_files:
        data = np.load(chunk_file, allow_pickle=True).item()
        chunks.append(data)
    
    return chunks


def unflatten_debug_data(flat_data: dict) -> dict:
    """Convert flattened debug data back to nested structure.
    
    Special handling for 'raw_obs' which contains keys with '/' in them
    (like 'observation/image'). These should NOT be split further.
    """
    result = {}
    for key, value in flat_data.items():
        # Special handling for raw_obs - only split on first '/'
        if key.startswith("raw_obs/"):
            if "raw_obs" not in result:
                result["raw_obs"] = {}
            # The rest of the key (after 'raw_obs/') should be kept as-is
            inner_key = key[len("raw_obs/"):]
            result["raw_obs"][inner_key] = value
        else:
            # Normal nested structure handling
            parts = key.split("/")
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
    return result


def create_observation_from_debug(debug_data: dict, prompt: str, step: int) -> dict:
    """Create observation dict from debug data for policy inference.
    
    Prefers 'raw_obs' (the exact observation before any transforms) if available,
    otherwise falls back to 'obs_before_preprocess' with reverse transform.
    """
    # Prefer raw_obs if available (exact observation before any transforms)
    if "raw_obs" in debug_data:
        raw_obs = debug_data["raw_obs"]
        # raw_obs is the exact dict that was passed to policy.infer()
        # Just update the step counter
        obs = dict(raw_obs)
        obs["step"] = step
        return obs
    
    raise ValueError("No raw_obs found in debug data")


def get_noise_from_debug(debug_data: dict) -> np.ndarray:
    """Extract noise from debug data."""
    noise = debug_data.get("noise")
    if noise is None:
        raise ValueError("No noise found in debug data")
    # Remove batch dimension if present
    if noise.ndim == 3 and noise.shape[0] == 1:
        noise = noise[0]
    return noise


def get_saved_actions_from_debug(debug_data: dict, action_dim: int = 7) -> np.ndarray:
    """Extract saved final actions from debug data.
    
    Prefers 'final_actions' (post-processed, unnormalized actions ready for robot)
    over 'output_actions' (raw model output before unnormalization).
    
    Args:
        debug_data: Debug data dictionary containing 'final_actions' or 'output_actions'
        action_dim: The actual robot action dimension (default: 7 for LIBERO)
    
    Returns:
        Actions with shape (action_horizon, action_dim)
    """
    # Prefer final_actions (post-processed) over output_actions (raw model output)
    actions = debug_data.get("final_actions")
    if actions is None:
        # Fallback to output_actions for older debug data
        actions = debug_data.get("output_actions")
        if actions is None:
            raise ValueError("No final_actions or output_actions found in debug data")
        # output_actions needs dimension slicing since it's padded
        # Remove batch dimension if present
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        # Extract only the actual action dimensions (first action_dim values)
        actions = actions[:, :action_dim]
    else:
        # final_actions is already the correct shape, just remove batch dim if present
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
    return actions


def replay_episode(
    config: ReplayConfig,
    policy: Optional[_websocket_client_policy.WebsocketClientPolicy],
    console: Console,
) -> bool:
    """Replay a single episode from debug data.
    
    Args:
        config: Replay configuration
        policy: Policy client (only needed if not using saved actions)
        console: Rich console for output
    
    Returns:
        True if episode was successful, False otherwise.
    """
    # Load metadata and chunks
    metadata = load_metadata(config.debug_data_dir)
    chunks = load_debug_chunks(config.debug_data_dir)
    
    console.print(f"[bold blue]Loaded {len(chunks)} debug chunks[/bold blue]")
    console.print(f"  Task Suite: {metadata['task_suite_name']}")
    console.print(f"  Task ID: {metadata['task_id']}")
    console.print(f"  Original Success: {metadata['success']}")
    console.print(f"  Mode: {'Using saved actions' if config.use_saved_actions else 'Re-inferring from policy'}")
    console.print()
    
    # Setup environment
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[metadata["task_suite_name"]]()
    task = task_suite.get_task(metadata["task_id"])
    
    # Get initial state for this episode
    all_initial_states = task_suite.get_task_init_states(metadata["task_id"])
    episode_idx = metadata.get("episode_idx", 1) - 1  # episode_idx is 1-based after increment
    if episode_idx >= len(all_initial_states):
        episode_idx = 0
    initial_state = all_initial_states[episode_idx:episode_idx+1]
    
    env_raw, task_description = utils._get_libero_env(
        task, LIBERO_ENV_RESOLUTION, seed=config.seed
    )
    
    env = LiberoSimEnvironment(
        env=env_raw,
        task_description=task_description,
        initial_states=initial_state,
        resize_size=config.resize_size,
        num_steps_wait=config.num_steps_wait,
        max_episode_steps=config.max_steps,
        control_hz=config.control_hz,
    )
    
    console.print(f"[bold green]Environment initialized[/bold green]")
    console.print(f"  Task: {task_description}")
    console.print()
    
    # Reset environment
    env.reset()
    
    # Replay loop
    frames: List[np.ndarray] = []
    chunk_idx = 0
    action_idx = 0
    current_actions = None
    step = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_progress = progress.add_task("[cyan]Replaying episode...", total=None)
        ran_out_of_chunks = False
        last_action = None
        
        while not env.is_episode_complete() and step < config.max_steps:
            # Get observation for frame capture
            obs = env.get_observation()
            frames.append(obs["observation/image"])
            
            # Determine which action to use
            if ran_out_of_chunks:
                # Keep using last action after running out of chunks
                action = last_action
            elif current_actions is None or action_idx >= len(current_actions):
                if chunk_idx >= len(chunks):
                    # Ran out of chunks
                    if not ran_out_of_chunks:
                        console.print("[yellow]Warning: Ran out of debug chunks, using last action[/yellow]")
                        ran_out_of_chunks = True
                    if last_action is not None:
                        action = last_action
                    else:
                        console.print("[red]No actions available, stopping[/red]")
                        break
                else:
                    # Load next chunk
                    chunk_data = unflatten_debug_data(chunks[chunk_idx])
                    
                    if config.use_saved_actions:
                        # Use saved output_actions directly (extract only action_dim dimensions)
                        current_actions = get_saved_actions_from_debug(chunk_data, config.action_dim)
                    else:
                        # Re-infer from policy with saved noise
                        if policy is None:
                            raise ValueError("Policy is required when not using saved actions")
                        noise = get_noise_from_debug(chunk_data)
                        
                        # Create observation from debug data
                        debug_obs = create_observation_from_debug(chunk_data, task_description, step)
                        
                        # Call policy with the saved noise
                        response = policy.infer(debug_obs, noise=noise)
                        # Policy response already has correct action_dim from post-processing
                        current_actions = response["actions"]
                    
                    chunk_idx += 1
                    action_idx = 0
                    
                    progress.update(task_progress, description=f"[cyan]Step {step}, Chunk {chunk_idx}/{len(chunks)}")
                    
                    # Get action from the new chunk
                    action = current_actions[action_idx]
                    action_idx += 1
            else:
                # Get next action from current chunk
                action = current_actions[action_idx]
                action_idx += 1
            
            # Remember last action for when we run out of chunks
            last_action = action
            
            # Apply action
            env.apply_action({"actions": action})
            step += 1
    
    # Capture final frame
    if not env.is_episode_complete():
        obs = env.get_observation()
        frames.append(obs["observation/image"])
    
    success = env.current_success
    
    # Save video
    output_path = config.output_video
    if output_path is None:
        output_path = config.debug_data_dir / "replay.mp4"
    else:
        output_path = pathlib.Path(output_path)
    
    console.print(f"\n[bold]Saving replay video to {output_path}[/bold]")
    imageio.mimwrite(
        str(output_path),
        [np.asarray(f) for f in frames],
        fps=config.control_hz,
    )
    
    # Cleanup
    env.close()
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Replay debug data from a saved episode"
    )
    parser.add_argument(
        "--debug_data_dir",
        type=str,
        required=True,
        help="Path to the episode directory containing metadata.json and debug_data/",
    )
    parser.add_argument(
        "--use_saved_actions",
        action="store_true",
        help="Use saved output_actions directly instead of re-inferring from policy",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Policy server host (only needed if not using --use_saved_actions)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Policy server port (only needed if not using --use_saved_actions)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for environment",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default=None,
        help="Output video path (default: <debug_data_dir>/replay.mp4)",
    )
    
    args = parser.parse_args()
    
    console = Console()
    
    config = ReplayConfig(
        debug_data_dir=pathlib.Path(args.debug_data_dir),
        host=args.host,
        port=args.port,
        seed=args.seed,
        max_steps=args.max_steps,
        output_video=args.output_video,
        use_saved_actions=args.use_saved_actions,
    )
    
    console.print("[bold magenta]═══════════════════════════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]                    Debug Data Replay                       [/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════════════════════════[/bold magenta]")
    console.print()
    
    policy = None
    if not config.use_saved_actions:
        # Connect to policy server
        console.print(f"[bold]Connecting to policy server at {config.host}:{config.port}...[/bold]")
        policy = _websocket_client_policy.WebsocketClientPolicy(
            host=config.host,
            port=config.port,
        )
        console.print("[green]Connected![/green]")
        console.print()
    else:
        console.print("[bold]Using saved actions (no policy server needed)[/bold]")
        console.print()
    
    # Run replay
    try:
        success = replay_episode(config, policy, console)
        
        console.print()
        console.print("[bold magenta]═══════════════════════════════════════════════════════════[/bold magenta]")
        if success:
            console.print("[bold green]                    REPLAY SUCCESS ✓                        [/bold green]")
        else:
            console.print("[bold red]                    REPLAY FAILURE ✗                        [/bold red]")
        console.print("[bold magenta]═══════════════════════════════════════════════════════════[/bold magenta]")
        
    except Exception as e:
        console.print(f"[bold red]Error during replay: {e}[/bold red]")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

