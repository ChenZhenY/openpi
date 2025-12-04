from __future__ import annotations

import pandas as pd
import json
import logging
import pathlib
import multiprocessing
import shutil

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
    control_hz: int = 20  # Target control frequency for each sim #NOTE: int because this is the fps of the video

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


# TODO: make explicit
def _latency_for_robot(args: Args, robot_idx: int) -> float:
    """Return the latency (in ms) to use for a given robot index."""
    if not args.latency_ms:
        return 0.0
    if robot_idx < len(args.latency_ms):
        return float(args.latency_ms[robot_idx])
    # If fewer latencies than robots, repeat the last value
    return float(args.latency_ms[-1])


def init_worker(args: Args, counter) -> None:
    global robot_idx, ws_client, broker, agent
    with counter.get_lock():
        robot_idx = counter.value
        counter.value += 1

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


def create_runtime(args: Args, job: Job) -> _runtime.Runtime:
    env_raw, task_description = utils._get_libero_env(
        job.task,
        LIBERO_ENV_RESOLUTION,
        seed=args.seed + robot_idx,
    )
    env = LiberoSimEnvironment(
        env=env_raw,
        task_description=task_description,
        initial_states=job.initial_states,
        resize_size=args.resize_size,
        num_steps_wait=args.num_steps_wait,
        max_episode_steps=args.max_steps,
        latency_ms=_latency_for_robot(args, robot_idx),
        control_hz=args.control_hz,
    )

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
            # ProgressSubscriber() # TODO
        ],
        max_hz=args.control_hz,
        num_episodes=len(job.initial_states),
        max_episode_steps=env._max_episode_steps,  # type: ignore[attr-defined]
    )
    return runtime


def _robot_worker(args: Args, job: Job) -> None:
    """Worker process that handles jobs for a specific robot index."""
    runtime = create_runtime(args, job)
    runtime.run()
    runtime.close()


def run_robots(args: Args, jobs: list[Job]) -> None:
    counter = multiprocessing.Value("i", 0)  # for assigning robot indices

    # TODO: rich multiprocessing progress
    with multiprocessing.Pool(
        processes=args.num_robots, initializer=init_worker, initargs=(args, counter)
    ) as pool:
        pool.starmap(_robot_worker, [(args, job) for job in jobs])


@dataclass
class Job:
    """A job is a task with a batch of episodes."""

    task_suite_name: str

    task: benchmark.Task
    task_id: int
    initial_states: np.ndarray  # batch, state_dim


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

        initial_states = all_initial_states[: args.num_trials_per_robot]
        job = Job(
            task=task,
            task_suite_name=args.task_suite_name,
            task_id=task_id,
            initial_states=initial_states,
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
    if args.overwrite:
        shutil.rmtree(args.output_dir)
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    jobs = create_jobs(args)
    run_robots(args, jobs)
    aggregate_results(pathlib.Path(args.output_dir))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # allows multiple processes with envs
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
