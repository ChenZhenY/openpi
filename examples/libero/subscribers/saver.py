import csv
import logging
import pathlib
import time
import json
import imageio
import numpy as np

from typing import List
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override
from openpi_client import action_chunk_broker
from examples.libero.env import LiberoSimEnvironment
from dataclasses import dataclass, fields

logger = logging.getLogger(__name__)


@dataclass
class Timestamp:
    timestamp: float
    action_chunk_index: int
    action_chunk_current_step: int

    @classmethod
    def to_csv(cls, timestamps: List["Timestamp"], filepath: pathlib.Path) -> None:
        """Save a list of Timestamps to a CSV file."""
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[field.name for field in fields(cls)])
            writer.writeheader()
            for ts in timestamps:
                writer.writerow(
                    {
                        "timestamp": ts.timestamp,
                        "action_chunk_index": ts.action_chunk_index,
                        "action_chunk_current_step": ts.action_chunk_current_step,
                    }
                )

    @classmethod
    def from_csv(cls, filepath: pathlib.Path) -> List["Timestamp"]:
        """Load a list of Timestamps from a CSV file."""
        timestamps = []
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamps.append(
                    cls(
                        timestamp=float(row["timestamp"]),
                        action_chunk_index=int(row["action_chunk_index"]),
                        action_chunk_current_step=int(row["action_chunk_current_step"]),
                    )
                )
        return timestamps


class Saver(_subscriber.Subscriber):
    """Saves episode data."""

    # TODO: probably pass metadata with dataclass
    def __init__(
        self,
        out_dir: pathlib.Path,
        environment: LiberoSimEnvironment,
        action_chunk_broker: action_chunk_broker.ActionChunkBroker,
        task_suite_name: str,
        task_id: int,
        robot_idx: int,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._task_suite_name = task_suite_name
        self._task_id = task_id
        self._robot_idx = robot_idx
        self._environment = environment
        self._action_chunk_broker = action_chunk_broker
        self._timestamps: List[Timestamp] = []
        self._images: List[np.ndarray] = []
        self._control_hz = environment.control_hz

    @override
    def on_episode_start(self) -> None:
        self._timestamps = []
        self._action_chunk_indices = []
        self._images = []

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        self._timestamps.append(
            Timestamp(
                timestamp=time.time(),
                action_chunk_index=action["action_chunk_index"],
                action_chunk_current_step=action["action_chunk_current_step"],
            )
        )
        self._images.append(observation["observation/image"])

    @override
    def on_episode_end(self) -> None:
        out_folder = self._get_out_folder()

        self._save_metadata(out_folder)
        self._save_timestamps(out_folder)
        self._save_action_chunks(out_folder)
        self._save_video(out_folder)

    def _get_out_folder(self) -> pathlib.Path:
        robot_folder = self._out_dir / str(self._robot_idx)
        pathlib.Path(robot_folder).mkdir(parents=True, exist_ok=True)

        existing = list(robot_folder.iterdir())
        next_idx = (
            max([int(p.name.split("_")[0]) for p in existing if p.is_dir()], default=-1)
            + 1
        )
        success_str = "success" if self._environment.current_success else "failure"
        out_folder = (
            robot_folder
            / f"{next_idx}_{self._task_suite_name}_{self._task_id}_{success_str}"
        )
        pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)
        return pathlib.Path(out_folder)

    def _save_metadata(self, out_folder: pathlib.Path) -> None:
        logger.info(f"Saving metadata to {out_folder / 'metadata.json'}")
        with open(out_folder / "metadata.json", "w") as f:
            metadata = {
                "task_suite_name": self._task_suite_name,
                "task_id": self._task_id,
                "robot_idx": self._robot_idx,
                "success": self._environment.current_success,
            }
            json.dump(metadata, f, indent=4)

    def _save_timestamps(self, out_folder: pathlib.Path) -> None:
        logger.info(f"Saving timestamps to {out_folder / 'timestamps.csv'}")
        Timestamp.to_csv(self._timestamps, out_folder / "timestamps.csv")

    def _save_action_chunks(self, out_folder: pathlib.Path) -> None:
        logger.info(f"Saving action chunks to {out_folder / 'action_chunks.csv'}")
        action_chunk_broker.ActionChunk.to_csv(
            self._action_chunk_broker.action_chunks, out_folder / "action_chunks.csv"
        )

    def _save_video(self, out_folder: pathlib.Path) -> None:
        logger.info(f"Saving video to {out_folder / 'out.mp4'}")
        imageio.mimwrite(
            out_folder / "out.mp4",
            [np.asarray(x) for x in self._images],
            fps=self._control_hz,  # NOTE: saving in control hz fps for now
        )
