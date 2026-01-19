# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import json
import os
import queue
import signal
import threading
import time
from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro
import websockets.sync.client

faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "20036094" # "<your_camera_id>"  # e.g., "24259877"
    right_camera_id: str = "21497414" # "<your_camera_id>"  # e.g., "24514023"
    wrist_camera_id: str = "14620168" # "<your_camera_id>"  # e.g., "13062452"

    # Policy parameters
    external_camera: str | None = (
        None  # which external camera should be fed to the policy, choose from ["left", "right"]
    )

    # Rollout parameters
    max_timesteps: int = 600
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )

    # WebUI mode parameters
    webui_mode: bool = False  # Enable WebUI mode for voice control
    webui_host: str = "localhost"  # WebUI server host
    webui_port: int = 8080  # WebUI server port


class WebUIPromptReceiver:
    """Receives prompts from WebUI server via WebSocket.
    
    This class connects to the WebUI server and listens for new task commands.
    When a new prompt is received, it interrupts the current rollout.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        self._uri = f"ws://{host}:{port}/ws/robot"
        self._prompt_queue: queue.Queue[str] = queue.Queue()
        self._interrupt_flag = threading.Event()
        self._ws = None
        self._running = False
        self._thread = None
        self._current_task: str | None = None
        
    def start(self):
        """Start the WebSocket listener thread."""
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        """Stop the WebSocket listener."""
        self._running = False
        if self._ws:
            self._ws.close()
        if self._thread:
            self._thread.join(timeout=2.0)
            
    def _listen_loop(self):
        """Main loop for WebSocket listener."""
        while self._running:
            try:
                print(f"Connecting to WebUI server at {self._uri}...")
                self._ws = websockets.sync.client.connect(self._uri)
                print("Connected to WebUI server!")
                
                while self._running:
                    try:
                        message = self._ws.recv()
                        data = json.loads(message)
                        
                        if data.get("type") == "new_task":
                            prompt = data.get("prompt")
                            if prompt:
                                print(f"\n[WebUI] New task received: {prompt}")
                                self._prompt_queue.put(prompt)
                                self._interrupt_flag.set()
                                
                    except websockets.ConnectionClosed:
                        print("[WebUI] Connection closed, reconnecting...")
                        break
                        
            except Exception as e:
                print(f"[WebUI] Connection error: {e}, retrying in 5s...")
                time.sleep(5)
                
    def get_prompt(self, timeout: float | None = None) -> str | None:
        """Get the next prompt from the queue.
        
        Args:
            timeout: How long to wait for a prompt. None means wait forever.
            
        Returns:
            The prompt string, or None if timeout expired.
        """
        try:
            prompt = self._prompt_queue.get(timeout=timeout)
            self._current_task = prompt
            return prompt
        except queue.Empty:
            return None
            
    def should_interrupt(self) -> bool:
        """Check if current rollout should be interrupted."""
        return self._interrupt_flag.is_set()
        
    def clear_interrupt(self):
        """Clear the interrupt flag after handling it."""
        self._interrupt_flag.clear()
        
    def send_status(self, status: str, task: str | None = None):
        """Send status update to WebUI."""
        if self._ws:
            try:
                self._ws.send(json.dumps({
                    "type": "status_update",
                    "status": status,
                    "task": task or self._current_task,
                }))
            except Exception:
                pass
                
    def send_step_update(self, step: int, max_steps: int):
        """Send step progress update to WebUI."""
        if self._ws:
            try:
                self._ws.send(json.dumps({
                    "type": "step_update",
                    "step": step,
                    "max_steps": max_steps,
                }))
            except Exception:
                pass
                
    def send_task_complete(self, message: str = "Task completed"):
        """Notify WebUI that task is complete."""
        if self._ws:
            try:
                self._ws.send(json.dumps({
                    "type": "task_complete",
                    "message": message,
                }))
            except Exception:
                pass


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def main(args: Args):
    # Make sure external camera is specified by user -- we only use one external camera for the policy
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    print("Created the droid env!")

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    # Initialize WebUI prompt receiver if in WebUI mode
    prompt_receiver: WebUIPromptReceiver | None = None
    if args.webui_mode:
        prompt_receiver = WebUIPromptReceiver(args.webui_host, args.webui_port)
        prompt_receiver.start()
        print(f"WebUI mode enabled. Waiting for commands from {args.webui_host}:{args.webui_port}")

    df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    while True:
        # Get instruction from WebUI or terminal
        if args.webui_mode and prompt_receiver:
            print("Waiting for task from WebUI...")
            instruction = prompt_receiver.get_prompt(timeout=None)
            if instruction is None:
                continue
            prompt_receiver.clear_interrupt()
            prompt_receiver.send_status("executing", instruction)
        else:
            instruction = input("Enter instruction: ")

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None
        interrupted_by_new_task = False

        # Prepare to save video of rollout
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video = []
        bar = tqdm.tqdm(range(args.max_timesteps))
        print(f"Running rollout for: {instruction}")
        print("Press Ctrl+C to stop early." if not args.webui_mode else "Send new task from WebUI to interrupt.")
        
        for t_step in bar:
            start_time = time.time()
            try:
                # Check for WebUI interrupt (new task received)
                if args.webui_mode and prompt_receiver and prompt_receiver.should_interrupt():
                    print("\n[WebUI] Interrupted by new task!")
                    interrupted_by_new_task = True
                    break

                # Get the current observation
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    # Save the first observation to disk
                    save_to_disk=t_step == 0,
                )

                video.append(curr_obs[f"{args.external_camera}_image"])

                # Send step update to WebUI
                if args.webui_mode and prompt_receiver:
                    prompt_receiver.send_step_update(t_step + 1, args.max_timesteps)

                # Send websocket request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                    # and improve latency.
                    request_data = {
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(
                            curr_obs[f"{args.external_camera}_image"], 224, 224
                        ),
                        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": instruction,
                    }

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        # this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
                        pred_action_chunk = policy_client.infer(request_data)["actions"]
                    assert pred_action_chunk.shape == (10, 8)

                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Binarize gripper action
                if action[-1].item() > 0.5:
                    # action[-1] = 1.0
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    # action[-1] = 0.0
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                # clip all dimensions of action to [-1, 1]
                action = np.clip(action, -1, 1)

                env.step(action)

                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

        # Handle rollout completion
        if args.webui_mode and prompt_receiver:
            if interrupted_by_new_task:
                # Don't save video or ask for success rating, just continue to next task
                prompt_receiver.send_task_complete("Task interrupted by new command")
                continue
            else:
                prompt_receiver.send_task_complete("Task completed")
                prompt_receiver.send_status("idle")

        # Save video
        if len(video) > 0:
            video = np.stack(video)
            save_filename = "video_" + timestamp
            ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")
        else:
            save_filename = "no_video"

        # Skip success rating in WebUI mode
        if args.webui_mode:
            continue

        success: str | float | None = None
        while not isinstance(success, float):
            success = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
            )
            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0

            success = float(success) / 100
            if not (0 <= success <= 1):
                print(f"Success must be a number in [0, 100] but got: {success * 100}")

        df = df.append(
            {
                "success": success,
                "duration": t_step,
                "video_filename": save_filename,
            },
            ignore_index=True,
        )

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
        env.reset()

    # Cleanup
    if prompt_receiver:
        prompt_receiver.stop()

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        # Note the "left" below refers to the left camera in the stereo pair.
        # The model is only trained on left stereo cams, so we only feed those.
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    # Drop the alpha dimension
    left_image = left_image[..., :3]
    right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # Convert to RGB
    left_image = left_image[..., ::-1]
    right_image = right_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    # Save the images to disk so that they can be viewed live while the robot is running
    # Create one combined image to make live viewing easy
    if save_to_disk:
        combined_image = np.concatenate([left_image, wrist_image, right_image], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
