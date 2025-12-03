from __future__ import annotations

import pandas as pd
import json
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
from dataclasses import dataclass, asdict, field

from examples.libero import utils
from examples.libero.env import LiberoSimEnvironment
from examples.libero.subscribers.metadata_saver import MetadataSaver
from examples.libero.subscribers.video_saver import VideoSaver

LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclass
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
    latency_ms: list[float] = field(
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
    output_dir: str = "data/libero/multi_robot_videos"
    overwrite: bool = False


def _latency_for_robot(args: Args, robot_idx: int) -> float:
    """Return the latency (in ms) to use for a given robot index."""
    if not args.latency_ms:
        return 0.0
    if robot_idx < len(args.latency_ms):
        return float(args.latency_ms[robot_idx])
    # If fewer latencies than robots, repeat the last value
    return float(args.latency_ms[-1])


def create_runtime(args: Args, robot_idx: int, job: Job) -> _runtime.Runtime:
    # Create LIBERO env for this robot
    # TODO: cache this if slow

    env_raw, task_description = utils._get_libero_env(
        job.task,
        LIBERO_ENV_RESOLUTION,
        seed=args.seed + robot_idx,
    )
    env = LiberoSimEnvironment(
        env=env_raw,
        task_description=task_description,
        initial_states=job.initial_state,
        resize_size=args.resize_size,
        num_steps_wait=args.num_steps_wait,
        max_episode_steps=args.max_steps,
        latency_ms=_latency_for_robot(args, robot_idx),
        control_hz=args.control_hz,
    )

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
                out_dir=pathlib.Path(args.output_dir)
                / str(robot_idx)
                / (args.task_suite_name + "_" + str(job.task_id)),
                environment=env,
                action_chunk_broker=broker,
                task_suite_name=job.task_suite_name,
                task_id=job.task_id,
                robot_idx=robot_idx,
            ),
            VideoSaver(
                out_dir=pathlib.Path(args.output_dir)
                / str(robot_idx)
                / (args.task_suite_name + "_" + str(job.task_id)),
            ),
            # ProgressSubscriber() # TODO
        ],
        max_hz=args.control_hz,
        num_episodes=1,
        max_episode_steps=env._max_episode_steps,  # type: ignore[attr-defined]
    )
    return runtime


def _robot_worker(args: Args, robot_idx: int, jobs: list[Job]) -> None:
    """Worker process that handles jobs for a specific robot index."""
    for job in jobs:
        runtime = create_runtime(args, robot_idx, job)
        runtime.run()
        runtime.close()


def run_robots(args: Args, jobs: list[Job]) -> None:
    # TODO: rich multiprocessing progress
    # Distribute jobs across robots (round-robin)
    jobs_per_robot: list[list[Job]] = [[] for _ in range(args.num_robots)]
    for idx, job in enumerate(jobs):
        jobs_per_robot[idx % args.num_robots].append(job)

    # Create one process per robot, each with its index (0 to num_robots-1)
    processes = []
    for robot_idx in range(args.num_robots):
        p = multiprocessing.Process(
            target=_robot_worker,
            args=(args, robot_idx, jobs_per_robot[robot_idx]),
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()


@dataclass
class Job:
    task: benchmark.Task
    task_suite_name: str
    task_id: int
    initial_state: np.ndarray


def create_jobs(args: Args) -> list[Job]:
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

    jobs: list[Job] = []
    for task_id in range(num_tasks_in_suite):
        task = task_suite.get_task(task_id)
        all_initial_states = task_suite.get_task_init_states(task_id)

        if len(all_initial_states) < args.num_trials_per_robot:
            logging.error(
                "Task %d has less initial states than trials per robot; skipping",
                task_id,
            )
            continue

        for i in range(args.num_trials_per_robot):
            initial_state = all_initial_states[
                None, i
            ]  # FIXME: LiberoSimEnvironment expects a batch of initial states, clean up these semantics
            job = Job(
                task=task,
                task_suite_name=args.task_suite_name,
                task_id=task_id,
                initial_state=initial_state,
            )
            jobs.append(job)

    logging.info("Created %d jobs", len(jobs))

    return jobs


# TODO: refactor in metadata dataclass and put with metadata_saver
@dataclass
class Result:
    success: bool
    robot_idx: int
    task_suite_name: str
    task_id: int

    @classmethod
    def from_metadata_file(cls, metadata_file: pathlib.Path) -> Result:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            return cls(
                success=metadata["success"],
                robot_idx=metadata["robot_idx"],
                task_suite_name=metadata["task_suite_name"],
                task_id=metadata["task_id"],
            )


def aggregate_results(output_path: pathlib.Path) -> None:
    metadata_files = list(output_path.glob("**/metadata.json"))

    results: list[Result] = []
    for metadata_file in metadata_files:
        result = Result.from_metadata_file(metadata_file)
        results.append(result)

    results_df = pd.DataFrame([asdict(result) for result in results])
    results_df.to_csv(output_path / "results.csv", index=False)
    summary = results_df.groupby(["task_suite_name", "task_id"]).agg(
        {
            "success": "mean",
        }
    )
    summary.reset_index().to_csv(output_path / "summary.csv", index=False)
    # TODO: rich
    print(summary)
    print("Total success rate: ", summary["success"].mean())


def main(args: Args) -> None:
    if not args.overwrite and pathlib.Path(args.output_dir).exists():
        raise ValueError(f"Output path {args.output_dir} already exists")

    np.random.seed(args.seed)

    jobs = create_jobs(args)
    run_robots(args, jobs)
    aggregate_results(pathlib.Path(args.output_dir))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # allows multiple processes with envs
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
