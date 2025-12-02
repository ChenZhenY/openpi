from openpi_client.runtime import environment as _environment
import numpy as np
from openpi_client import image_tools
from libero.libero.envs import OffScreenRenderEnv
from examples.libero import utils
from typing import List

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


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
        self._episode_results: List[bool] = []
        self._episode_frames: List[List[np.ndarray]] = []
        self._current_frames: List[np.ndarray] = []
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
                    utils._quat2axisangle(obs["robot0_eef_quat"]),
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
    def episode_results(self) -> List[bool]:
        """Per-episode success flags accumulated so far."""
        return self._episode_results

    @property
    def episode_frames(self) -> List[List[np.ndarray]]:
        """Per-episode list of recorded frames (each frame is HxWxC uint8)."""
        return self._episode_frames
