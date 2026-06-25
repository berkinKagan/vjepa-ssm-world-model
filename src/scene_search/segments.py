from pathlib import Path

from utils.paths import resolve_project_path


def get_video_duration(video_path: str | Path) -> float:
    path = resolve_project_path(video_path)
    try:
        return duration_with_cv2(path)
    except Exception:
        return duration_with_decord(path)


def duration_with_cv2(path: Path) -> float:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or frame_count <= 0:
        raise ValueError(f"Could not read duration for video: {path}")
    return frame_count / fps


def duration_with_decord(path: Path) -> float:
    from utils.video_io import open_video_reader

    reader = open_video_reader(path)
    fps = float(reader.get_avg_fps())
    if fps <= 0:
        raise ValueError(f"Could not read FPS for video: {path}")
    return len(reader) / fps


def dense_video_segments(video_path: str | Path, segment_length_seconds: float, segment_stride_seconds: float, min_segment_seconds: float) -> tuple[list[dict], float]:
    duration = get_video_duration(video_path)
    segments = []
    start = 0.0
    index = 0
    while start < duration:
        end = min(start + segment_length_seconds, duration)
        if end - start >= min_segment_seconds:
            segments.append({"index": index, "segment_index": index, "start_time": start, "end_time": end})
            index += 1
        start += segment_stride_seconds
    return segments, duration


def coverage_diagnostics(segments: list[dict], video_duration: float) -> dict:
    if not segments or video_duration <= 0:
        return {"video_duration": video_duration, "segment_count": len(segments), "first_segment_start": None, "last_segment_end": None, "coverage_ratio": 0.0, "warning": "Scene index does not cover the full video."}
    first_start = float(segments[0]["start_time"])
    last_end = float(max(segment["end_time"] for segment in segments))
    ratio = max(0.0, min(1.0, (last_end - first_start) / video_duration))
    diagnostics = {"video_duration": video_duration, "segment_count": len(segments), "first_segment_start": first_start, "last_segment_end": last_end, "coverage_ratio": ratio}
    if ratio < 0.95:
        diagnostics["warning"] = "Scene index does not cover the full video."
    return diagnostics
