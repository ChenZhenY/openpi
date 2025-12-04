from __future__ import annotations

import json
import pathlib
import tyro

from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip
from dataclasses import dataclass
import logging
from examples.libero import logging_config
from examples.libero.schemas import Timestamp
from examples.libero.subscribers.saver import Result

from typing import Tuple
import math

import subprocess

from typing import Dict, List

LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclass
class Args:
    output_dir: str = "data/libero/multi_robot_videos"


def get_grid_dimensions(num_videos: int) -> Tuple[int, int]:
    cols = int(math.ceil(math.sqrt(num_videos)))
    rows = int(math.ceil(num_videos / cols))

    return rows, cols


def get_grid_index(video_id: int, grid_dimensions: Tuple[int, int]) -> Tuple[int, int]:
    rows, cols = grid_dimensions
    return video_id // cols, video_id % cols


@dataclass
class Video:
    path: pathlib.Path
    row: int
    col: int
    start_time: float


def get_video_size(path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
        capture_output=True,
        text=True,
    )
    info = json.loads(result.stdout)
    stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    return stream["width"], stream["height"]


def grid_videos(videos, output, cols, rows, duration):
    """
    videos: list of (path, row, col, start_time)
    """
    clip0 = VideoFileClip(str(videos[0].path))
    w, h = clip0.size

    clips = []
    for video in videos:
        logging.info(
            f"Processing video: path={video.path}, grid_position=({video.row}, {video.col}), start_time={video.start_time:.3f}s"
        )
        clip = VideoFileClip(str(video.path))
        clip = clip.set_position((video.col * w, video.row * h)).set_start(
            video.start_time
        )
        clips.append(clip)

    bg = ColorClip(size=(cols * w, rows * h), color=(0, 0, 0), duration=duration)

    final = CompositeVideoClip([bg] + clips, size=(cols * w, rows * h))
    final.write_videofile(str(output), codec="libx264")


def load_timestamps(output_path: pathlib.Path) -> Dict[pathlib.Path, List[Timestamp]]:
    paths = list(output_path.glob("**/timestamps.csv"))
    return {path.parent: Timestamp.from_csv(path) for path in paths}


def load_results(output_path: pathlib.Path) -> Dict[pathlib.Path, Result]:
    paths = list(output_path.glob("**/metadata.json"))
    return {path.parent: Result.from_json(path) for path in paths}


def combine_videos(output_path: pathlib.Path) -> None:
    video_paths = list(output_path.glob("**/out.mp4"))
    timestamps = load_timestamps(output_path)
    results = load_results(output_path)
    min_timestamp = min(
        [min([t.timestamp for t in timestamps[path]]) for path in timestamps]
    )

    rows, cols = get_grid_dimensions(4)
    videos = [
        Video(
            path,
            *get_grid_index(results[path.parent].robot_idx, (rows, cols)),
            start_time=timestamps[path.parent][0].timestamp - min_timestamp,
        )
        for i, path in enumerate(video_paths)
    ]

    grid_videos(videos, output_path / "combined.mp4", rows, cols, duration=10)


def main(args: Args) -> None:
    combine_videos(pathlib.Path(args.output_dir))


if __name__ == "__main__":
    logging_config.setup_logging()
    main(tyro.cli(Args))
