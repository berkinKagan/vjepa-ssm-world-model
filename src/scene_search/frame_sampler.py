from pathlib import Path

from PIL import Image

from utils.paths import ensure_directory, resolve_project_path


def sample_clip_frames(video_path: str | Path, clip: dict, output_dir: str | Path, frames_per_clip: int) -> list[str]:
    resolved_video = resolve_project_path(video_path)
    if not resolved_video.exists():
        raise FileNotFoundError(f"Original video file not found: {resolved_video}")
    start_time = float(clip["start_time"])
    end_time = float(clip["end_time"])
    times = representative_times(start_time, end_time, max(1, frames_per_clip))
    output_root = ensure_directory(resolve_project_path(output_dir) / resolved_video.stem)
    try:
        return sample_with_decord(resolved_video, clip, output_root, times)
    except Exception:
        return sample_with_cv2(resolved_video, clip, output_root, times)


def sample_with_decord(video_path: Path, clip: dict, output_root: Path, times: list[float]) -> list[str]:
    from utils.video_io import open_video_reader

    reader = open_video_reader(video_path)
    fps = float(reader.get_avg_fps())
    frame_count = len(reader)
    paths = []
    for frame_number, time_value in enumerate(times):
        frame_index = min(max(0, round(time_value * fps)), max(0, frame_count - 1))
        frame = reader.get_batch([frame_index]).asnumpy()[0]
        image = Image.fromarray(frame)
        destination = frame_destination(output_root, clip, frame_number)
        image.save(destination, quality=90)
        paths.append(str(destination))
    return paths


def sample_with_cv2(video_path: Path, clip: dict, output_root: Path, times: list[float]) -> list[str]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    paths = []
    for frame_number, time_value in enumerate(times):
        frame_index = min(max(0, round(time_value * fps)), max(0, frame_count - 1))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        destination = frame_destination(output_root, clip, frame_number)
        image.save(destination, quality=90)
        paths.append(str(destination))
    capture.release()
    return paths


def frame_destination(output_root: Path, clip: dict, frame_number: int) -> Path:
    start_time = float(clip["start_time"])
    end_time = float(clip["end_time"])
    label = "segment" if "segment_index" in clip else "clip"
    index = int(clip.get("segment_index", clip["index"]))
    return output_root / f"{label}_{index:04d}_{start_time:.2f}_{end_time:.2f}_{frame_number:02d}.jpg"


def representative_times(start_time: float, end_time: float, count: int) -> list[float]:
    if count == 1:
        return [(start_time + end_time) / 2.0]
    if count == 3:
        span = max(0.0, end_time - start_time)
        return [start_time + span * 0.1, start_time + span * 0.5, start_time + span * 0.9]
    span = max(0.0, end_time - start_time)
    return [start_time + span * (index + 1) / (count + 1) for index in range(count)]
