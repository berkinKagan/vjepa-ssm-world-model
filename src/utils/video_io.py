from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    filename: str
    fps: float
    frame_count: int
    duration: float


@dataclass(frozen=True)
class ClipWindow:
    index: int
    start_frame: int
    end_frame: int
    frame_indices: list[int]
    start_time: float
    end_time: float


def open_video_reader(video_path: str | Path):
    from decord import VideoReader, cpu

    return VideoReader(str(video_path), ctx=cpu(0))


def get_video_info(video_path: str | Path) -> VideoInfo:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    reader = open_video_reader(path)
    fps = float(reader.get_avg_fps())
    if fps <= 0:
        raise ValueError(f"Could not determine FPS for video: {path}")
    frame_count = len(reader)
    duration = frame_count / fps if frame_count else 0.0
    return VideoInfo(path=path, filename=path.name, fps=fps, frame_count=frame_count, duration=duration)


def sample_clip_windows(video_info: VideoInfo, clip_frames: int, frame_stride: int, clip_stride_seconds: float, max_clips: int | None) -> list[ClipWindow]:
    if clip_frames <= 0:
        raise ValueError("clip_frames must be positive")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if clip_stride_seconds <= 0:
        raise ValueError("clip_stride_seconds must be positive")
    if video_info.frame_count <= 0:
        raise ValueError("Video has no frames")
    clip_span = (clip_frames - 1) * frame_stride + 1
    max_start = max(0, video_info.frame_count - clip_span)
    stride_frames = max(1, round(clip_stride_seconds * video_info.fps))
    starts = list(range(0, max_start + 1, stride_frames))
    if not starts:
        starts = [0]
    if max_clips is not None and max_clips > 0:
        starts = starts[:max_clips]
    windows = []
    for index, start_frame in enumerate(starts):
        frame_indices = [min(start_frame + offset * frame_stride, video_info.frame_count - 1) for offset in range(clip_frames)]
        end_frame = frame_indices[-1]
        windows.append(ClipWindow(index=index, start_frame=start_frame, end_frame=end_frame, frame_indices=frame_indices, start_time=start_frame / video_info.fps, end_time=(end_frame + 1) / video_info.fps))
    return windows


def read_clip(video_path: str | Path, frame_indices: list[int]) -> torch.Tensor:
    if not frame_indices:
        raise ValueError("frame_indices cannot be empty")
    reader = open_video_reader(video_path)
    batch = reader.get_batch(frame_indices).asnumpy()
    return torch.from_numpy(batch)
