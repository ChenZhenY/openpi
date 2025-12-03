from __future__ import annotations

import dataclasses
import logging
import pathlib
import multiprocessing

import numpy as np
from libero.libero import benchmark
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.libero import utils
from examples.libero.env import LiberoSimEnvironment
from examples.libero.subscribers.metadata_saver import MetadataSaver
from examples.libero.subscribers.video_saver import VideoSaver

LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8080
    resize_size: int = 224
    action_horizon: int = (
        10  # Action horizon for ActionChunkBroker (matches Libero model config)
    )
    latency_ms: list[float] = dataclasses.field(
        default_factory=list
    )  # Optional per-robot artificial latency (ms); length <= num_robots

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_robot: int = 10  # Number of rollouts per robot per task
    max_steps: int = 300  # Maximum number of control steps per episode

    #################################################################################################################
    # Multi-robot / threading parameters
    #################################################################################################################
    num_robots: int = 9  # Number of always-running sims (robots)
    control_hz: float = 20.0  # Target control frequency for each sim

    #################################################################################################################
    # ActionChunkBroker / RTC parameters
    #################################################################################################################
    use_rtc: bool = False
    s: int = 5
    d: int = 4

    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 7  # Random Seed (for reproducibility)
    video_out_path: str = "data/libero/multi_robot_videos"
    overwrite: bool = False


def _latency_for_robot(args: Args, robot_idx: int) -> float:
    """Return the latency (in ms) to use for a given robot index."""
    if not args.latency_ms:
        return 0.0
    if robot_idx < len(args.latency_ms):
        return float(args.latency_ms[robot_idx])
    # If fewer latencies than robots, repeat the last value
    return float(args.latency_ms[-1])


def create_runtime(args: Args, robot_idx: int, task_id: int) -> _runtime.Runtime:
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    logging.info(
        "Setting up multi-robot LIBERO runtime over suite '%s' with %d tasks, num_robots=%d, trials_per_robot=%d",
        args.task_suite_name,
        num_tasks_in_suite,
        args.num_robots,
        args.num_trials_per_robot,
    )

    task = task_suite.get_task(task_id)
    all_initial_states = task_suite.get_task_init_states(task_id)

    if len(all_initial_states) == 0:
        logging.warning("Task %d has no initial states; skipping", task_id)
        raise ValueError(f"Task {task_id} has no initial states")

    logging.info(
        "Task %d: %d initial states, launching up to %d robots",
        task_id,
        len(all_initial_states),
        args.num_robots,
    )
    # Build per-robot initial states for this task.
    # Each robot gets num_trials_per_robot episodes, cycling through available initial states if needed.
    n_init = len(all_initial_states)
    per_robot_states: list[np.ndarray] = []
    for i in range(args.num_robots):
        idxs = [
            (i * args.num_trials_per_robot + ep_idx) % n_init
            for ep_idx in range(args.num_trials_per_robot)
        ]
        per_robot_states.append(all_initial_states[idxs])
    init_states_robot = per_robot_states[robot_idx]

    # Create LIBERO env for this robot
    env_raw, task_description = utils._get_libero_env(
        task,
        LIBERO_ENV_RESOLUTION,
        seed=args.seed + robot_idx,
    )
    env = LiberoSimEnvironment(
        env=env_raw,
        task_description=task_description,
        initial_states=init_states_robot,
        resize_size=args.resize_size,
        num_steps_wait=args.num_steps_wait,
        max_episode_steps=args.max_steps,
        latency_ms=_latency_for_robot(args, robot_idx),
        control_hz=args.control_hz,
    )

    # Number of episodes this robot will run equals number of its assigned initial states
    num_episodes = len(init_states_robot)

    # Create policy agent for this robot (with optional per-robot latency)
    ws_client = _websocket_client_policy.WebsocketClientPolicy(
        args.host,
        args.port,
    )
    broker = action_chunk_broker.ActionChunkBroker(
        policy=ws_client,
        action_horizon=args.action_horizon,
        is_rtc=args.use_rtc,
        s=args.s,
        d=args.d,
    )
    agent = _policy_agent.PolicyAgent(policy=broker)

    runtime = _runtime.Runtime(
        environment=env,
        agent=agent,
        subscribers=[
            MetadataSaver(
                out_dir=pathlib.Path(args.video_out_path)
                / str(robot_idx)
                / args.task_suite_name,
                environment=env,
                action_chunk_broker=broker,
            ),
            VideoSaver(
                out_dir=pathlib.Path(args.video_out_path)
                / str(robot_idx)
                / args.task_suite_name,
            ),
            # ProgressSubscriber() # TODO
        ],
        max_hz=args.control_hz,
        num_episodes=num_episodes,
        max_episode_steps=env._max_episode_steps,  # type: ignore[attr-defined]
    )
    return runtime


def _robot_wrapper(args: Args, robot_idx: int, task_id: int) -> None:
    runtime = create_runtime(args, robot_idx, task_id)
    runtime.run()
    runtime.close()


def run_robots(args: Args, robot_indices: list[int], task_id: int) -> None:
    # TODO: rich multiprocessing progress
    processes = [
        multiprocessing.Process(target=_robot_wrapper, args=(args, robot_idx, task_id))
        for robot_idx in robot_indices
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()


def main(args: Args) -> None:
    if not args.overwrite and pathlib.Path(args.video_out_path).exists():
        raise ValueError(f"Output path {args.video_out_path} already exists")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    logging.info(
        "Setting up multi-robot LIBERO runtime over suite '%s' with %d tasks, num_robots=%d, trials_per_robot=%d",
        args.task_suite_name,
        num_tasks_in_suite,
        args.num_robots,
        args.num_trials_per_robot,
    )

    # TODO: should just distribute all tasks across robots
    # global_results: list[dict] = []

    for task_id in range(num_tasks_in_suite):
        run_robots(args, list(range(args.num_robots)), task_id)
        break

        # Aggregate per-task stats
        # task_episodes = sum(r["episodes"] for r in task_results)
        # task_successes = sum(r["successes"] for r in task_results)
        # task_success_rate = (
        #     (task_successes / task_episodes) if task_episodes > 0 else 0.0
        # )

        # logging.info(
        #     "Task %d complete: episodes=%d, successes=%d (%.1f%%)",
        #     task_id,
        #     task_episodes,
        #     task_successes,
        #     task_success_rate * 100.0,
        # )

        # TODO: compute, print, and save task-level stats
        # Build and save matrix video + JSON for this task from robot_envs
        # try:
        #     # Save JSON results for this task
        #     out_dir_json = pathlib.Path(args.video_out_path) / args.task_suite_name
        #     out_dir_json.mkdir(parents=True, exist_ok=True)
        #     json_path = out_dir_json / f"task_{task_id}_results.json"

        #     robots_json: list[dict] = []
        #     for env, r_idx in zip(robot_envs, robot_indices):
        #         episodes_json: list[dict] = []
        #         for ep_idx, success in enumerate(env.episode_results):
        #             num_frames = (
        #                 len(env.episode_frames[ep_idx])
        #                 if ep_idx < len(env.episode_frames)
        #                 else 0
        #             )
        #             episodes_json.append(
        #                 {
        #                     "episode_idx": ep_idx,
        #                     "success": bool(success),
        #                     "num_frames": num_frames,
        #                 }
        #             )
        #         robots_json.append(
        #             {
        #                 "robot_idx": r_idx,
        #                 "latency_ms": _latency_for_robot(args, r_idx),
        #                 "episodes": episodes_json,
        #             }
        #         )

        #     task_json = {
        #         "task_id": task_id,
        #         "task_suite_name": args.task_suite_name,
        #         "num_robots": args.num_robots,
        #         "num_trials_per_robot": args.num_trials_per_robot,
        #         "summary": {
        #             "episodes": task_episodes,
        #             "successes": task_successes,
        #             "success_rate": task_success_rate,
        #         },
        #         "robots": robots_json,
        #     }

        #     with json_path.open("w") as f:
        #         json.dump(task_json, f, indent=2)
        #     logging.info("Saved JSON results for task %d to %s", task_id, json_path)
        # except Exception as e:
        #     logging.error("Failed to create matrix video for task %d: %s", task_id, e)

        # for r in task_results:
        #     r_with_task = dict(r)
        #     r_with_task["task_id"] = task_id
        #     global_results.append(r_with_task)

    # Aggregate global stats across all tasks
    # TODO: global stats
    # total_episodes = sum(r["episodes"] for r in global_results)
    # total_successes = sum(r["successes"] for r in global_results)
    # total_success_rate = (
    #     (total_successes / total_episodes) if total_episodes > 0 else 0.0
    # )

    # logging.info("=== Multi-robot LIBERO evaluation over entire suite complete ===")
    # logging.info(
    #     "Total episodes: %d, total successes: %d (%.1f%%)",
    #     total_episodes,
    #     total_successes,
    #     total_success_rate * 100.0,
    # )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # allows multiple processes with envs
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
