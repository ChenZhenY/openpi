import logging
import pathlib
import time
import json

from typing import List
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override
from openpi_client import action_chunk_broker
from examples.libero.env import LiberoSimEnvironment
from dataclasses import asdict, dataclass


@dataclass
class Timestamp:
    timestamp: float
    action_chunk_index: int
    action_chunk_current_step: int


class MetadataSaver(_subscriber.Subscriber):
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

    @override
    def on_episode_start(self) -> None:
        self._timestamps = []
        self._action_chunk_indices = []

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        self._timestamps.append(
            Timestamp(
                timestamp=time.time(),
                action_chunk_index=action["action_chunk_index"],
                action_chunk_current_step=action["action_chunk_current_step"],
            )
        )

    # TODO: "folder/robot_idx/<index>_<task_suite_name>_<task_id>_<success>/metadata.json"
    @override
    def on_episode_end(self) -> None:
        existing = list(self._out_dir.glob("metadata_[0-9]*.json"))
        next_idx = max([int(p.stem.split("_")[1]) for p in existing], default=-1) + 1
        out_path = self._out_dir / f"metadata_{next_idx}.json"

        logging.info(f"Saving metadata to {out_path}")
        with open(self._out_dir / "metadata.json", "w") as f:
            metadata = {
                "task_suite_name": self._task_suite_name,
                "task_id": self._task_id,
                "robot_idx": self._robot_idx,
                "timestamps": [asdict(t) for t in self._timestamps],
                "action_chunks": [
                    asdict(ac) for ac in self._action_chunk_broker.action_chunks
                ],
                "success": self._environment.current_success,
            }
            json.dump(metadata, f, indent=4)
