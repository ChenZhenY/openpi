from __future__ import annotations

import json
import logging
import pathlib

from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip
from dataclasses import dataclass

from typing import Tuple
import math

import subprocess

from examples.libero.subscribers.metadata_saver import Timestamp
from typing import Dict, List

LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


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


# def annotate_videos(output_path: pathlib.Path) -> None:
#     video_paths = list(output_path.glob("**/out_*.mp4"))
#     for video_path in video_paths:
#         metadata_path = video_path.parent / "metadata.json"

#         with open(metadata_path, "r") as f:
#             metadata = json.load(f)
#             # FIXME: for now, assume indices are correct
#             action_chunk_indices = metadata["action_chunk_indices"]

#             action_chunks = metadata["action_chunks"]

#         replay_images = np.array(imageio.mimread(video_path))
#         frame_metadata = []

#         visualized_frames = visualize.add_action_chunk_visualization(
#             replay_images,
#             frame_metadata,
#             time_window=10,
#         )

#         new_path = video_path.with_suffix("_annotated.mp4")
#         imageio.mimwrite(
#             new_path, [np.asarray(x) for x in visualized_frames], fps=10
#         )  # TODO fps


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
        print(video.path, video.col, video.row, video.start_time)
        clip = VideoFileClip(str(video.path))
        clip = clip.set_position((video.col * w, video.row * h)).set_start(
            video.start_time
        )
        clips.append(clip)

    bg = ColorClip(size=(cols * w, rows * h), color=(0, 0, 0), duration=duration)

    final = CompositeVideoClip([bg] + clips, size=(cols * w, rows * h))
    final.write_videofile(str(output), codec="libx264")


def load_timestamps(output_path: pathlib.Path) -> Dict[pathlib.Path, List[Timestamp]]:
    paths = list(output_path.glob("**/metadata.json"))
    return {
        path.parent: [Timestamp(**t) for t in json.load(open(path))["timestamps"]]
        for path in paths
    }


def load_robot_idx(output_path: pathlib.Path) -> Dict[pathlib.Path, int]:
    paths = list(output_path.glob("**/metadata.json"))
    return {path.parent: json.load(open(path))["robot_idx"] for path in paths}


def combine_videos(output_path: pathlib.Path) -> None:
    video_paths = list(output_path.glob("**/out.mp4"))
    timestamps = load_timestamps(output_path)
    robot_idx = load_robot_idx(output_path)
    min_timestamp = min(
        [min([t.timestamp for t in timestamps[path]]) for path in timestamps]
    )

    # video_paths = list(output_path.glob("**/out_*_annotated.mp4")) # TODO: eventually combine annotated videos
    rows, cols = get_grid_dimensions(4)
    videos = [
        Video(
            path,
            *get_grid_index(robot_idx[path.parent], (rows, cols)),
            start_time=timestamps[path.parent][0].timestamp - min_timestamp,
        )
        for i, path in enumerate(video_paths)
    ]

    grid_videos(videos, output_path / "combined.mp4", rows, cols, duration=10)


def main() -> None:
    # annotate_videos(pathlib.Path("data/libero/multi_robot_videos"))
    combine_videos(pathlib.Path("data/libero/multi_robot_videos"))


if __name__ == "__main__":
    main()
    logging.basicConfig(level=logging.INFO, force=True)
