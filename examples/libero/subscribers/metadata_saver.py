import logging
import pathlib
import time
import json

from typing import List
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override
from openpi_client import action_chunk_broker
from examples.libero.env import LiberoSimEnvironment
from dataclasses import asdict


class MetadataSaver(_subscriber.Subscriber):
    """Saves episode data."""

    def __init__(
        self,
        out_dir: pathlib.Path,
        environment: LiberoSimEnvironment,
        action_chunk_broker: action_chunk_broker.ActionChunkBroker,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._environment = environment
        self._action_chunk_broker = action_chunk_broker
        self._timestamps: List[float] = []
        self._action_chunk_indices: List[int] = []

    @override
    def on_episode_start(self) -> None:
        self._timestamps = []
        self._action_chunk_indices = []

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        self._timestamps.append(time.time())
        self._action_chunk_indices.append(
            self._action_chunk_broker.current_action_chunk.chunk_index
        )

    @override
    def on_episode_end(self) -> None:
        logging.info(f"Saving metadata to {self._out_dir / 'action_chunks.json'}")
        with open(self._out_dir / "action_chunks.json", "w") as f:
            metadata = {
                "timestamps": self._timestamps,
                "action_chunk_indices": self._action_chunk_indices,
                "action_chunks": [
                    asdict(ac) for ac in self._action_chunk_broker.action_chunks
                ],
                "success": self._environment.current_success,
            }
            json.dump(metadata, f)
