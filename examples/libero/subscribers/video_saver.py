import logging
import pathlib

import imageio
import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


class VideoSaver(_subscriber.Subscriber):
    """Saves episode data."""

    def __init__(self, out_dir: pathlib.Path, subsample: int = 1) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._images: list[np.ndarray] = []
        self._subsample = subsample

    @override
    def on_episode_start(self) -> None:
        self._images = []

    # TODO: make observation a dataclass
    @override
    def on_step(self, observation: dict, action: dict) -> None:
        self._images.append(observation["observation/image"])

    # TODO: "folder/robot_idx/<index>_<task_suite_name>_<task_id>_<success>/out_<index>.mp4"
    @override
    def on_episode_end(self) -> None:
        logging.info(f"Saving video to {self._out_dir / 'out.mp4'}")
        imageio.mimwrite(
            self._out_dir / "out.mp4",
            [np.asarray(x) for x in self._images[:: self._subsample]],
            fps=50 // max(1, self._subsample),
        )
