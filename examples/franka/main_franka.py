#!/usr/bin/env python3
"""
Main script for running a remote pi policy on a real Franka Grey Robot in 1209 (RL2 RobotEnv),
following the LIBERO inference style.

Key responsibilities:
1. Create the real Franka RobotEnv with cameras.
2. Get observations and format them according to the LIBERO / pi data config.
3. Connect to the remote policy server via WebsocketClientPolicy.
4. Execute actions in short horizons (replan_steps).
5. Record and save replay videos.
6. Expose a clean tyro-based CLI similar to examples/libero/main.py.
"""

import collections
import dataclasses
import logging
import math
import pathlib
import time
from typing import Dict

import imageio
import numpy as np
import tyro

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

# Make sure your PYTHONPATH includes gello_software, or adjust imports below
from gello.robots.panda_deoxys_simple import PandaRobot
from gello.rl2_env import RobotEnv


# --------------------------------------------------------------------------------------
# Camera configuration for the real robot setup
# --------------------------------------------------------------------------------------
# You can edit this dict to match your actual camera setup.
CAMERA_CONFIG_DICT: Dict[str, Dict] = {
    # Example Kinect as "agentview"
    "agentview": {
        "sn": "001039114912",  # Kinect serial (example)
        "type": "Kinect",
    },
    # Example Zed as "wrist"
    "wrist": {
        "sn": 14620168,  # ZED serial (example)
        "type": "Zed",
        # Optional resize config used by the camera driver itself (if supported)
        "resize": True,
        "resize_resolution": (640, 576),
    },
}


# --------------------------------------------------------------------------------------
# Argument dataclass (user interface)
# --------------------------------------------------------------------------------------
@dataclasses.dataclass
class Args:
    ###########################################################################
    # Model server parameters
    ###########################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 256  # image size before feeding to pi policy
    replan_steps: int = 5   # action horizon length before replanning

    ###########################################################################
    # Real robot / environment parameters
    ###########################################################################
    max_steps: int = 400      # max control steps per rollout
    num_episodes: int = 1     # number of rollouts to run
    control_rate_hz: float = 100.0  # passed to RobotEnv
    confirm_first_action: bool = True  # optional safety confirmation

    # Natural language prompt (must match what the pi policy expects)
    prompt: str = (
        # "pick up the organge cup and place it on the top of the drawer"
        "pick up the pink cup and hang it on the mug tree"
    )
    # pick up the pink cup and hang it on the mug tree (red cup)
    # pick up the blue bowl and place it on the top of the drawer (organge cup)


    ###########################################################################
    # Utils
    ###########################################################################
    video_out_path: str = "data/franka/videos"  # where to save replay videos


# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------
def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Same helper as in examples/libero/main.py.

    Converts a quaternion (x, y, z, w) to axis-angle representation (3,).
    """
    quat = quat.copy()
    # clip quaternion scalar part
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def axisangle2quat(vecs):
    """
    input array [batch_size, 3]
    """
    assert len(vecs.shape) == 2
    assert vecs.shape[-1] == 3
    angles = np.linalg.norm(vecs, axis=1, keepdims=True) # [b, 1]

    quaternions = np.zeros([vecs.shape[0], 4])
    for i in range(vecs.shape[0]):
        angle = angles[i]
        vec = vecs[i]
        if math.isclose(angles[i], 0.0):
            return np.array([0.0, 0.0, 0.0, 1.0])
    
        axis = vec / angle

        q = np.zeros(4)
        q[3] = np.cos(angle / 2.0)
        q[:3] = axis * np.sin(angle / 2.0)
        quaternions[i] = q
    return quaternions

def _create_cameras_from_config(camera_config_dict: Dict[str, Dict]):
    """
    Instantiate camera drivers from CAMERA_CONFIG_DICT.

    Returns:
        cam_dict: {camera_name: CameraDriver}
    """
    cam_dict = {}
    for cam_name, cam_config in camera_config_dict.items():
        cam_type = cam_config["type"]
        if cam_type == "Zed":
            from gello.cameras.zed_camera import ZedCamera

            cam_dict[cam_name] = ZedCamera(cam_name, cam_config)
        elif cam_type == "RealSense":
            from gello.cameras.realsense_camera import RealSenseCamera

            cam_dict[cam_name] = RealSenseCamera(
                device_id=cam_config["sn"], enable_depth=True
            )
        elif cam_type == "Kinect":
            from gello.cameras.kinect_camera import KinectCamera

            cam_dict[cam_name] = KinectCamera(cam_name, cam_config)
        else:
            raise ValueError(f"Unsupported camera type: {cam_type}")
    return cam_dict


def _preprocess_images(obs: Dict[str, np.ndarray], resize_size: int):
    """
    Preprocess agentview and wrist images to match LIBERO data config.

    Steps:
    1. Take 'agentview_image' and 'wrist_image' from RobotEnv.
    2. Rotate 180 degrees ([::-1, ::-1]) to match LIBERO preprocessing.
    3. Resize with padding to a square `resize_size x resize_size`.
    4. Convert to uint8.

    Returns:
        img, wrist_img: preprocessed uint8 arrays of shape (H, W, 3).
    """
    # These keys must exist in RobotEnv.get_obs()
    agentview = obs["agentview_image"]
    wrist = obs["wrist_image"]

    # Rotate 180 degrees to match LIBERO train preprocessing
    agentview = np.ascontiguousarray(agentview[::-1, ::-1])
    wrist = np.ascontiguousarray(wrist[::-1, ::-1])

    agentview = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(agentview, resize_size, resize_size)
    )
    wrist = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist, resize_size, resize_size)
    )
    return agentview, wrist


def _build_state_vector(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Build the state vector expected by the pi policy.

    Mirrors LIBERO:
        [eef_pos (3), eef_axis_angle (3), gripper_qpos (2)] -> shape (8,)

    The RobotEnv already provides:
        - "eef_pos"
        - "eef_axis_angle" (or we can recompute from "eef_quat")
        - "gripper_position" (2D: [g, -g])

    Returns:
        state: np.ndarray of shape (8,)
    """
    eef_pos = np.asarray(obs["eef_pos"])             # (3,)
    # You can either use the precomputed axis-angle or recompute from quat:
    if "eef_axis_angle" in obs:
        eef_axis_angle = np.asarray(obs["eef_axis_angle"])
    else:
        eef_quat = np.asarray(obs["eef_quat"])
        eef_axis_angle = _quat2axisangle(eef_quat)

    gripper_qpos = np.asarray(obs["gripper_position"])  # (2,)

    state = np.concatenate((eef_pos, eef_axis_angle, gripper_qpos), axis=0)
    return state


def _create_robot_env(args: Args) -> RobotEnv:
    """
    Instantiate the real Franka robot client + cameras + RobotEnv.
    """
    logging.info("Initializing cameras...")
    cam_dict = _create_cameras_from_config(CAMERA_CONFIG_DICT)
    logging.info("Cameras initialized: %s", list(cam_dict.keys()))

    logging.info("Initializing PandaRobot client...")
    robot_client = PandaRobot("OSC_POSE", gripper_type="robotiq")
    logging.info("Robot client initialized.")

    env = RobotEnv(
        robot=robot_client,
        camera_dict=cam_dict,
        control_rate_hz=args.control_rate_hz,
        save_depth_obs=False,  # set True if you want depth in obs
    )
    logging.info("RobotEnv created.")

    return env


# --------------------------------------------------------------------------------------
# Main evaluation loop
# --------------------------------------------------------------------------------------
def eval_franka(args: Args) -> None:
    """
    Run inference on the real Franka using a remote pi policy server.

    The loop mirrors examples/libero/main.py:
      - get obs
      - preprocess images + state
      - send to policy server
      - get action chunk, execute replan_steps actions
      - record replay video
    """
    # Prepare output directory for videos
    video_dir = pathlib.Path(args.video_out_path)
    video_dir.mkdir(parents=True, exist_ok=True)

    # Connect to policy server
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logging.info(f"Connected to policy server at ws://{args.host}:{args.port}")

    # Create real robot environment
    env = _create_robot_env(args)

    try:
        for episode_idx in range(args.num_episodes):
            logging.info(f"========== Episode {episode_idx + 1} / {args.num_episodes} ==========")

            # Reset robot, if supported
            if hasattr(env.robot(), "reset"):
                logging.info("Resetting robot to home pose...")
                env.robot().reset()

            # Initial observation
            obs = env.get_obs()

            # Action buffer (short horizon planning)
            action_plan = collections.deque()

            # For video replay: store preprocessed agentview images
            replay_images = []

            # Optional safety: confirm first action from policy
            confirmed_first_action = not args.confirm_first_action

            t = 0
            while t < args.max_steps:
                # Preprocess images at every step (for consistent replay)
                img, wrist_img = _preprocess_images(obs, args.resize_size)
                replay_images.append(img)

                if not action_plan:
                    # Build state vector
                    state = _build_state_vector(obs)

                    # Prepare model input element (matches LIBERO data config)
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": state,
                        "prompt": str(args.prompt),
                    }

                    # Query remote policy for a chunk of actions
                    resp = client.infer(element)
                    action_chunk = resp["actions"]
                    action_chunk = np.asarray(action_chunk)

                    if action_chunk.ndim == 1:
                        action_chunk = np.expand_dims(action_chunk, axis=0)  # (T=1, D)

                    # we predict axis angle, but controller expects Quterion
                    # assert action_chunk.shape == (,7)
                    pos = action_chunk[:, :3]
                    rot = action_chunk[:, 3:6]
                    gripper_action = action_chunk[:, 6]
                    quat = axisangle2quat(rot)

                    if gripper_action.ndim == 1:
                        gripper_action = np.expand_dims(gripper_action, axis=1)

                    action_chunk = np.hstack((pos, quat, gripper_action))

                    assert (
                        len(action_chunk) >= args.replan_steps
                    ), f"Policy predicted {len(action_chunk)} steps, but replan_steps={args.replan_steps}"

                    # Keep only the first replan_steps actions
                    action_plan.extend(action_chunk[: args.replan_steps])

                    logging.info(
                        "Received %d actions from policy, will execute %d steps before replanning",
                        len(action_chunk),
                        args.replan_steps,
                    )

                # Take the next action from the plan
                action = np.asarray(action_plan.popleft(), dtype=np.float32)

                # Optional: show the first action for manual confirmation
                if not confirmed_first_action:
                    input(
                        f"\nFirst action from policy: {action}\n"
                        f"Press <Enter> to execute, or Ctrl+C to abort.\n"
                    )
                    confirmed_first_action = True

                # Execute action in real robot environment
                # RobotEnv.step returns (obs, action_executed)
                obs, _ = env.step(action)
                t += 1

            # Save replay video
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            video_path = video_dir / f"franka_rollout_{episode_idx:03d}_{timestamp}.mp4"
            imageio.mimwrite(
                video_path,
                [np.asarray(frame) for frame in replay_images],
                fps=10,
            )
            logging.info("Saved replay video to %s", video_path)

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received, stopping control loop...")

    finally:
        # Clean up cameras, robot, etc.
        logging.info("Closing environment and cameras...")
        env.close()
        logging.info("Done.")


# --------------------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_franka)
