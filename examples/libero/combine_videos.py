from __future__ import annotations

import json
import pathlib
import tyro

from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip, ImageClip
from dataclasses import dataclass
from examples.libero import logging_config
from examples.libero.schemas import Timestamp
from examples.libero.subscribers.saver import Result

from typing import Tuple
import math
from tqdm import tqdm
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
    result: Result
    clip: VideoFileClip = None


def get_video_size(path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
        capture_output=True,
        text=True,
    )
    info = json.loads(result.stdout)
    stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    return stream["width"], stream["height"]


def annotate_video(
    clip: VideoFileClip, result: Result, border_duration: float = 0.5
) -> VideoFileClip:
    """
    Annotate video with setup information and success/failure border.

    Args:
        clip: Video clip to annotate
        result: Metadata about the episode
        border_duration: Duration of the colored border at the end (in seconds)

    Returns:
        Annotated video clip
    """
    w, h = clip.size

    # Create text overlay with setup information using PIL
    text = (
        f"Robot {result.robot_idx} | {result.task_suite_name} | Task {result.task_id}"
    )

    # Create a PIL image for the text
    img = Image.new("RGBA", (w, 30), color=(0, 0, 0, 180))
    draw = ImageDraw.Draw(img)

    # Try to use a default font, fall back to default if unavailable
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    draw.text((5, 5), text, font=font, fill=(255, 255, 255, 255))

    # Convert PIL image to numpy array and create ImageClip
    txt_array = np.array(img)
    txt_clip = (
        ImageClip(txt_array).set_duration(clip.duration).set_position(("left", "top"))
    )

    # Create colored border for success/failure indicator at the end
    border_color = (0, 255, 0) if result.success else (255, 0, 0)  # Green or Red
    border_thickness = 5

    # Create border clips (top, bottom, left, right)
    border_start = max(0, clip.duration - border_duration)
    top_border = (
        ColorClip(size=(w, border_thickness), color=border_color)
        .set_position(("center", "top"))
        .set_start(border_start)
        .set_duration(border_duration)
    )
    bottom_border = (
        ColorClip(size=(w, border_thickness), color=border_color)
        .set_position(("center", "bottom"))
        .set_start(border_start)
        .set_duration(border_duration)
    )
    left_border = (
        ColorClip(size=(border_thickness, h), color=border_color)
        .set_position(("left", "center"))
        .set_start(border_start)
        .set_duration(border_duration)
    )
    right_border = (
        ColorClip(size=(border_thickness, h), color=border_color)
        .set_position(("right", "center"))
        .set_start(border_start)
        .set_duration(border_duration)
    )

    # Composite the video with text and borders
    annotated = CompositeVideoClip(
        [clip, txt_clip, top_border, bottom_border, left_border, right_border]
    )

    return annotated


def grid_videos(videos, output, cols, rows, duration):
    """
    videos: list of Video objects with pre-annotated clips
    """
    w, h = videos[0].clip.size

    clips = []
    for video in videos:
        clip = video.clip.set_position((video.col * w, video.row * h)).set_start(
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

    num_robots = len(set([result.robot_idx for result in results.values()]))
    rows, cols = get_grid_dimensions(num_robots)

    # Create video metadata
    videos = [
        Video(
            path,
            *get_grid_index(results[path.parent].robot_idx, (rows, cols)),
            start_time=timestamps[path.parent][0].timestamp - min_timestamp,
            result=results[path.parent],
        )
        for i, path in enumerate(video_paths)
    ]

    # Load and annotate all videos
    for video in tqdm(videos, desc="Loading and annotating videos"):
        clip = VideoFileClip(str(video.path))
        video.clip = annotate_video(clip, video.result)

    grid_videos(videos, output_path / "combined.mp4", cols, rows, duration=10)


def main(args: Args) -> None:
    combine_videos(pathlib.Path(args.output_dir))


if __name__ == "__main__":
    logging_config.setup_logging()
    main(tyro.cli(Args))
