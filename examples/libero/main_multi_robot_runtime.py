from __future__ import annotations

import dataclasses
import logging
import math
import pathlib
import threading
import time
from typing import List

import imageio
import json
import numpy as np
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.benchmark import Task
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import action_chunk_broker
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from PIL import Image, ImageDraw, ImageFont


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(
    task: Task, resolution: int, seed: int
) -> tuple[OffScreenRenderEnv, str]:
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


class LiberoSimEnvironment(_environment.Environment):
    """Wraps OffScreenRenderEnv into the openpi_client.runtime.Environment interface.

    This environment:
    - Uses pre-collected initial states from LIBERO.
    - On reset(), loads the next initial state and waits for objects to settle.
    - get_observation() returns the dict expected by the Libero VLA policy.
    - apply_action() steps the underlying simulator with the provided action.
    """

    def __init__(
        self,
        env: OffScreenRenderEnv,
        task_description: str,
        initial_states: np.ndarray,
        *,
        resize_size: int = 224,
        num_steps_wait: int = 10,
        max_episode_steps: int = 300,
        latency_ms: float = 0.0,
        control_hz: float = 100.0,
    ) -> None:
        self._env = env
        self._task_description = task_description
        self._initial_states = initial_states
        self._resize_size = resize_size
        self._num_steps_wait = num_steps_wait
        self._max_episode_steps = max_episode_steps
        self._latency_ms = latency_ms
        self._control_hz = control_hz

        self._episode_idx = 0
        self._done = True
        self._step_counter = 0
        self._last_obs = None
        self._episode_results: list[bool] = []
        self._episode_frames: list[list[np.ndarray]] = []
        self._current_frames: list[np.ndarray] = []
        self._current_success = False

    def reset(self) -> None:
        """Reset environment to next initial state and wait for object stabilization."""
        if self._episode_idx >= len(self._initial_states):
            # Loop around if more episodes are requested than initial states.
            self._episode_idx = 0

        self._env.reset()
        obs = self._env.set_init_state(self._initial_states[self._episode_idx])

        # Let objects fall / settle
        for _ in range(self._num_steps_wait):
            obs, _, _, _ = self._env.step(LIBERO_DUMMY_ACTION)

        self._last_obs = obs
        self._done = False
        self._step_counter = 0
        self._current_frames = []
        self._current_success = False
        self._episode_idx += 1

    def is_episode_complete(self) -> bool:
        return self._done

    def get_observation(self) -> dict:
        if self._last_obs is None:
            raise RuntimeError("Observation is not set. Call reset() first.")

        obs = self._last_obs

        # IMPORTANT: rotate 180 degrees to match train preprocessing
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, self._resize_size, self._resize_size)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, self._resize_size, self._resize_size)
        )

        # Record frame for video (use the main agentview image)
        self._current_frames.append(img)

        return {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            ),
            "prompt": str(self._task_description),
        }

    def apply_action(self, action: dict) -> None:
        """Take one or more low-level action steps in the LIBERO simulator.

        To simulate latency affecting the environment, we optionally repeat
        the same action for multiple simulator steps based on latency_ms and
        control_hz, so higher latency results in fewer distinct decisions per
        unit of simulated time.
        """
        # ActionChunkBroker returns a dict with key "actions"
        act = action["actions"]

        # Always execute at least one step with the new action
        obs, _, done, info = self._env.step(act.tolist())
        self._last_obs = obs
        self._step_counter += 1

        if done:
            self._current_success = True

        # Compute additional steps to simulate latency (repeat same action)
        if not done and self._latency_ms > 0.0 and self._control_hz > 0.0:
            extra_steps = int(round((self._latency_ms / 1000.0) * self._control_hz))
            for _ in range(extra_steps):
                if self._step_counter >= self._max_episode_steps or done:
                    break
                obs, _, done, info = self._env.step(act.tolist())
                self._last_obs = obs
                self._step_counter += 1

                # Record a frame for each extra latency step so pauses show up in video
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(
                        img, self._resize_size, self._resize_size
                    )
                )
                self._current_frames.append(img)

                if done:
                    self._current_success = True

        if done or self._step_counter >= self._max_episode_steps:
            self._done = True
            self._episode_results.append(self._current_success)
            # Store frames for this episode
            self._episode_frames.append(self._current_frames)
            self._current_frames = []

    @property
    def episode_results(self) -> list[bool]:
        """Per-episode success flags accumulated so far."""
        return self._episode_results

    @property
    def episode_frames(self) -> list[list[np.ndarray]]:
        """Per-episode list of recorded frames (each frame is HxWxC uint8)."""
        return self._episode_frames


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


def _partition_initial_states(
    initial_states: np.ndarray, num_parts: int
) -> List[np.ndarray]:
    """Partition initial states into num_parts slices (round-robin)."""
    parts: list[list[np.ndarray]] = [[] for _ in range(num_parts)]
    for idx, state in enumerate(initial_states):
        parts[idx % num_parts].append(state)
    return [
        np.stack(p) if p else np.empty((0,) + initial_states.shape[1:]) for p in parts
    ]


def _latency_for_robot(args: Args, robot_idx: int) -> float:
    """Return the latency (in ms) to use for a given robot index."""
    if not args.latency_ms:
        return 0.0
    if robot_idx < len(args.latency_ms):
        return float(args.latency_ms[robot_idx])
    # If fewer latencies than robots, repeat the last value
    return float(args.latency_ms[-1])


def _create_policy_agent(args: Args, robot_idx: int) -> _policy_agent.PolicyAgent:
    """Create a PolicyAgent that uses WebsocketClientPolicy + ActionChunkBroker."""
    latency = _latency_for_robot(args, robot_idx)
    ws_client = _websocket_client_policy.WebsocketClientPolicy(
        args.host,
        args.port,
        latency_ms=latency,
    )
    broker = action_chunk_broker.ActionChunkBroker(
        policy=ws_client,
        action_horizon=args.action_horizon,
        is_rtc=args.use_rtc,
        s=args.s,
        d=args.d,
    )
    return _policy_agent.PolicyAgent(policy=broker)


def _run_robot(
    robot_idx: int,
    env: LiberoSimEnvironment,
    agent: _policy_agent.PolicyAgent,
    num_episodes: int,
    control_hz: float,
    results_list: list,
    lock: threading.Lock,
) -> None:
    """Worker function: run a single robot in its own Runtime at control_hz."""
    runtime = _runtime.Runtime(
        environment=env,
        agent=agent,
        subscribers=[],
        max_hz=control_hz,
        num_episodes=num_episodes,
        max_episode_steps=env._max_episode_steps,  # type: ignore[attr-defined]
    )
    start_time = time.time()
    runtime.run()
    elapsed = time.time() - start_time

    # Aggregate results
    with lock:
        successes = sum(env.episode_results)
        episodes = len(env.episode_results)
        logging.info(
            f"[Robot {robot_idx}] Finished {episodes} episodes, "
            f"successes={successes} ({(successes / episodes * 100.0) if episodes > 0 else 0.0:.1f}%), "
            f"wall_time={elapsed:.1f}s"
        )
        results_list.append(
            {
                "robot_idx": robot_idx,
                "episodes": episodes,
                "successes": successes,
                "success_rate": (successes / episodes) if episodes > 0 else 0.0,
                "wall_time": elapsed,
            }
        )


def main(args: Args) -> None:
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

    global_results: list[dict] = []

    for task_id in range(num_tasks_in_suite):
        task = task_suite.get_task(task_id)
        all_initial_states = task_suite.get_task_init_states(task_id)

        if len(all_initial_states) == 0:
            logging.warning("Task %d has no initial states; skipping", task_id)
            continue

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
        for robot_idx in range(args.num_robots):
            idxs = [
                (robot_idx * args.num_trials_per_robot + ep_idx) % n_init
                for ep_idx in range(args.num_trials_per_robot)
            ]
            per_robot_states.append(all_initial_states[idxs])

        # Create environments and agents for each robot
        robots: list[threading.Thread] = []
        task_results: list[dict] = []
        results_lock = threading.Lock()
        robot_envs: list[LiberoSimEnvironment] = []
        robot_indices: list[int] = []

        for robot_idx in range(args.num_robots):
            init_states_robot = per_robot_states[robot_idx]
            if init_states_robot.shape[0] == 0:
                logging.warning(
                    "Task %d, robot %d has no initial states assigned; skipping",
                    task_id,
                    robot_idx,
                )
                continue

            # Create LIBERO env for this robot
            env_raw, task_description = _get_libero_env(
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

            # Create policy agent for this robot (with optional per-robot latency)
            agent = _create_policy_agent(args, robot_idx)

            # Number of episodes this robot will run equals number of its assigned initial states
            num_episodes_robot = init_states_robot.shape[0]

            thread = threading.Thread(
                target=_run_robot,
                args=(
                    robot_idx,
                    env,
                    agent,
                    num_episodes_robot,
                    args.control_hz,
                    task_results,
                    results_lock,
                ),
                daemon=True,
            )
            robots.append(thread)
            robot_envs.append(env)
            robot_indices.append(robot_idx)

        # Start all robots for this task
        logging.info(
            "Task %d: starting %d robot runtimes (target control_hz=%.1f)...",
            task_id,
            len(robots),
            args.control_hz,
        )
        for t in robots:
            t.start()

        # Wait for all robots to finish for this task
        for t in robots:
            t.join()

        # Aggregate per-task stats
        task_episodes = sum(r["episodes"] for r in task_results)
        task_successes = sum(r["successes"] for r in task_results)
        task_success_rate = (
            (task_successes / task_episodes) if task_episodes > 0 else 0.0
        )

        logging.info(
            "Task %d complete: episodes=%d, successes=%d (%.1f%%)",
            task_id,
            task_episodes,
            task_successes,
            task_success_rate * 100.0,
        )

        # Build and save matrix video + JSON for this task from robot_envs
        try:
            # Flatten frames per robot across all its episodes, and track episode indices
            # as well as the last frame index for each episode (to detect episode end).
            per_robot_flat_frames: list[list[np.ndarray]] = []
            per_robot_flat_episode_ids: list[list[int]] = []
            per_robot_episode_last_indices: list[list[int]] = []
            for env in robot_envs:
                frames_flat: list[np.ndarray] = []
                episode_ids_flat: list[int] = []
                episode_last_indices: list[int] = []
                for ep_idx, ep_frames in enumerate(env.episode_frames):
                    if not ep_frames:
                        continue

                    for frame in ep_frames:
                        frames_flat.append(frame)
                        episode_ids_flat.append(ep_idx)
                    last_idx = len(frames_flat) - 1
                    episode_last_indices.append(last_idx)
                per_robot_flat_frames.append(frames_flat)
                per_robot_flat_episode_ids.append(episode_ids_flat)
                per_robot_episode_last_indices.append(episode_last_indices)

            if per_robot_flat_frames:
                # Determine grid layout
                num_robots_task = len(per_robot_flat_frames)
                cols = int(math.ceil(math.sqrt(num_robots_task)))
                rows = int(math.ceil(num_robots_task / cols))

                # Determine max length across robots
                max_len = max(len(f) for f in per_robot_flat_frames)

                # If no frames, skip
                if max_len > 0:
                    # Assume all frames share same H, W, C
                    sample_frame = next(
                        f for frames in per_robot_flat_frames for f in frames if frames
                    )
                    frame_h, frame_w, frame_c = sample_frame.shape

                    matrix_frames: list[np.ndarray] = []
                    black = np.zeros_like(sample_frame)

                    for t_idx in range(max_len):
                        canvas = np.zeros(
                            (rows * frame_h, cols * frame_w, frame_c),
                            dtype=sample_frame.dtype,
                        )
                        for r_idx in range(num_robots_task):
                            frames_r = per_robot_flat_frames[r_idx]
                            ep_ids_r = per_robot_flat_episode_ids[r_idx]
                            if t_idx < len(frames_r):
                                frm = frames_r[t_idx]
                                ep_id = ep_ids_r[t_idx]
                            else:
                                frm = black
                                ep_id = None
                            row = r_idx // cols
                            col = r_idx % cols
                            y0 = row * frame_h
                            x0 = col * frame_w
                            canvas[y0 : y0 + frame_h, x0 : x0 + frame_w, :] = frm

                        # Optionally overlay per-robot details in each cell
                        img_pil = Image.fromarray(canvas)
                        draw = ImageDraw.Draw(img_pil)
                        try:
                            font = ImageFont.load_default()
                        except Exception:  # pragma: no cover
                            font = None

                        for r_idx in range(num_robots_task):
                            ep_ids_r = per_robot_flat_episode_ids[r_idx]
                            if t_idx < len(ep_ids_r):
                                ep_id = ep_ids_r[t_idx]
                            else:
                                ep_id = None
                            if ep_id is None:
                                continue

                            # Map from local index to global robot index and latency
                            robot_idx_global = robot_indices[r_idx]
                            latency_for_robot = _latency_for_robot(
                                args, robot_idx_global
                            )

                            # Episode success flag, if available
                            env = robot_envs[r_idx]
                            success_flag = None
                            if ep_id < len(env.episode_results):
                                success_flag = bool(env.episode_results[ep_id])

                            row = r_idx // cols
                            col = r_idx % cols
                            y0 = row * frame_h
                            x0 = col * frame_w

                            # Example label (no S/F here): "R0 ep1 200ms"
                            label = f"R{robot_idx_global} ep{ep_id} {latency_for_robot:.0f}ms"

                            # Small dark rectangle for readability (wider for more text)
                            rect_w, rect_h = 120, 14
                            draw.rectangle(
                                [x0, y0, x0 + rect_w, y0 + rect_h],
                                fill=(0, 0, 0, 160),
                            )
                            draw.text(
                                (x0 + 2, y0 + 1),
                                label,
                                fill=(255, 255, 255),
                                font=font,
                            )

                            # Draw a colored border to indicate success / failure
                            # only at the end of the episode.
                            if success_flag is not None and ep_id < len(
                                per_robot_episode_last_indices[r_idx]
                            ):
                                last_idx_for_ep = per_robot_episode_last_indices[r_idx][
                                    ep_id
                                ]
                                # Highlight on the last few frames of the episode
                                if (
                                    t_idx >= last_idx_for_ep - 4
                                    and t_idx <= last_idx_for_ep
                                ):
                                    border_color = (
                                        (0, 255, 0) if success_flag else (255, 0, 0)
                                    )
                                    # Top and bottom
                                    draw.rectangle(
                                        [x0, y0, x0 + frame_w - 1, y0 + 2],
                                        outline=None,
                                        fill=border_color,
                                    )
                                    draw.rectangle(
                                        [
                                            x0,
                                            y0 + frame_h - 3,
                                            x0 + frame_w - 1,
                                            y0 + frame_h - 1,
                                        ],
                                        outline=None,
                                        fill=border_color,
                                    )
                                    # Left and right
                                    draw.rectangle(
                                        [x0, y0, x0 + 2, y0 + frame_h - 1],
                                        outline=None,
                                        fill=border_color,
                                    )
                                    draw.rectangle(
                                        [
                                            x0 + frame_w - 3,
                                            y0,
                                            x0 + frame_w - 1,
                                            y0 + frame_h - 1,
                                        ],
                                        outline=None,
                                        fill=border_color,
                                    )
                        canvas = np.asarray(img_pil)

                        matrix_frames.append(canvas)

                    # Ensure output directory exists
                    out_dir = pathlib.Path(args.video_out_path) / args.task_suite_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"task_{task_id}_matrix.mp4"

                    imageio.mimwrite(
                        out_path,
                        [np.asarray(x) for x in matrix_frames],
                        fps=10,
                    )
                    logging.info(
                        "Saved matrix video for task %d to %s", task_id, out_path
                    )

            # Save JSON results for this task
            out_dir_json = pathlib.Path(args.video_out_path) / args.task_suite_name
            out_dir_json.mkdir(parents=True, exist_ok=True)
            json_path = out_dir_json / f"task_{task_id}_results.json"

            robots_json: list[dict] = []
            for env, r_idx in zip(robot_envs, robot_indices):
                episodes_json: list[dict] = []
                for ep_idx, success in enumerate(env.episode_results):
                    num_frames = (
                        len(env.episode_frames[ep_idx])
                        if ep_idx < len(env.episode_frames)
                        else 0
                    )
                    episodes_json.append(
                        {
                            "episode_idx": ep_idx,
                            "success": bool(success),
                            "num_frames": num_frames,
                        }
                    )
                robots_json.append(
                    {
                        "robot_idx": r_idx,
                        "latency_ms": _latency_for_robot(args, r_idx),
                        "episodes": episodes_json,
                    }
                )

            task_json = {
                "task_id": task_id,
                "task_suite_name": args.task_suite_name,
                "num_robots": args.num_robots,
                "num_trials_per_robot": args.num_trials_per_robot,
                "summary": {
                    "episodes": task_episodes,
                    "successes": task_successes,
                    "success_rate": task_success_rate,
                },
                "robots": robots_json,
            }

            with json_path.open("w") as f:
                json.dump(task_json, f, indent=2)
            logging.info("Saved JSON results for task %d to %s", task_id, json_path)
        except Exception as e:
            logging.error("Failed to create matrix video for task %d: %s", task_id, e)

        for r in task_results:
            r_with_task = dict(r)
            r_with_task["task_id"] = task_id
            global_results.append(r_with_task)

    # Aggregate global stats across all tasks
    total_episodes = sum(r["episodes"] for r in global_results)
    total_successes = sum(r["successes"] for r in global_results)
    total_success_rate = (
        (total_successes / total_episodes) if total_episodes > 0 else 0.0
    )

    logging.info("=== Multi-robot LIBERO evaluation over entire suite complete ===")
    logging.info(
        "Total episodes: %d, total successes: %d (%.1f%%)",
        total_episodes,
        total_successes,
        total_success_rate * 100.0,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
