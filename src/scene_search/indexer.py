from datetime import datetime, timezone
from pathlib import Path

from scene_search.captioner import SceneCaptioner
from scene_search.frame_sampler import sample_clip_frames
from scene_search.schemas import SceneRecord
from scene_search.segments import coverage_diagnostics, dense_video_segments, get_video_duration
from scene_search.storage import load_scene_index_metadata, save_scene_index
from scene_search.text_embedder import TextEmbedder
from storage.load_embeddings import load_embedding_artifact, load_metadata
from utils.paths import resolve_project_path


def build_scene_index(config: dict, embedding_path: str | Path, progress_callback=None) -> dict:
    resolved_embedding = resolve_project_path(embedding_path)
    cached = cached_scene_index(config, resolved_embedding)
    if cached is not None:
        return cached
    artifact = load_embedding_artifact(resolved_embedding)
    metadata = extract_metadata(artifact, resolved_embedding, config)
    video_path = metadata.get("video_path", "")
    segments, video_duration = build_segments(config, metadata, video_path)
    if not segments:
        raise ValueError("No scene segments could be built for scene indexing")
    diagnostics = coverage_diagnostics(segments, video_duration)
    metadata_path = find_metadata_path(resolved_embedding, config)
    captioner = SceneCaptioner(config)
    records = []
    total = len(segments)
    for index, segment in enumerate(segments, start=1):
        try:
            frame_paths = sample_clip_frames(video_path, segment, config["frame_cache_dir"], frames_per_segment(config))
        except Exception:
            frame_paths = []
        caption, backend, model = captioner.caption_clip(frame_paths, metadata, segment)
        segment_index = int(segment["segment_index"])
        record = SceneRecord(scene_id=f"{resolved_embedding.stem}_segment_{segment_index:04d}", video_path=str(resolve_project_path(video_path)), embedding_path=str(resolved_embedding), metadata_path=str(metadata_path) if metadata_path else None, segment_index=segment_index, start_time=float(segment["start_time"]), end_time=float(segment["end_time"]), caption=caption, frame_paths=frame_paths, caption_backend=backend, ollama_model=model, created_at=datetime.now(timezone.utc).isoformat(), segment_source=config.get("segment_source", "dense_video"), clip_index=segment.get("clip_index"))
        records.append(record)
        if progress_callback is not None:
            progress_callback(index, total)
    embedder = TextEmbedder(config["text_embedding_model"], config["device"])
    caption_embeddings = embedder.encode([record.caption for record in records])
    index_metadata = {**diagnostics, "segment_source": config.get("segment_source", "dense_video"), "segment_length_seconds": float(config.get("segment_length_seconds", 8.0)), "segment_stride_seconds": float(config.get("segment_stride_seconds", 8.0)), "min_segment_seconds": float(config.get("min_segment_seconds", 2.0)), "frames_per_segment": frames_per_segment(config), "embedding_model": config["text_embedding_model"], "caption_backend": config["caption_backend"], "ollama_model": config.get("ollama_model"), "reused": False}
    output = save_scene_index(records, caption_embeddings, config["scene_index_dir"], resolved_embedding.stem, index_metadata)
    output["embedding_backend"] = embedder.backend
    return output


def cached_scene_index(config: dict, embedding_path: Path) -> dict | None:
    index_path = resolve_project_path(config["scene_index_dir"]) / f"{embedding_path.stem}.jsonl"
    embeddings_path = index_path.with_suffix(".npy")
    metadata_path = index_path.with_suffix(".json")
    if not index_path.exists() or not embeddings_path.exists() or not metadata_path.exists():
        return None
    metadata = load_scene_index_metadata(index_path)
    if not reusable_index(config, metadata):
        return None
    return {**metadata, "index_file": str(index_path), "embeddings_file": str(embeddings_path), "reused": True}


def reusable_index(config: dict, metadata: dict) -> bool:
    expected = {"segment_source": config.get("segment_source", "dense_video"), "segment_length_seconds": float(config.get("segment_length_seconds", 8.0)), "segment_stride_seconds": float(config.get("segment_stride_seconds", 8.0)), "min_segment_seconds": float(config.get("min_segment_seconds", 2.0)), "frames_per_segment": frames_per_segment(config), "embedding_model": config["text_embedding_model"], "caption_backend": config["caption_backend"], "ollama_model": config.get("ollama_model")}
    for key, value in expected.items():
        if metadata.get(key) != value:
            return False
    return float(metadata.get("coverage_ratio", 0.0)) >= 0.95


def build_segments(config: dict, metadata: dict, video_path: str) -> tuple[list[dict], float]:
    source = config.get("segment_source", "dense_video")
    if source == "dense_video":
        return dense_video_segments(video_path, float(config.get("segment_length_seconds", 8.0)), float(config.get("segment_stride_seconds", 8.0)), float(config.get("min_segment_seconds", 2.0)))
    if source == "vjepa_clip_times":
        segments = segments_from_metadata(metadata)
        duration = float(metadata.get("duration") or get_video_duration(video_path))
        return segments, duration
    raise ValueError(f"Unsupported segment_source: {source}")


def segments_from_metadata(metadata: dict) -> list[dict]:
    clips = metadata.get("clips") or clips_from_times(metadata.get("clip_times", []))
    segments = []
    for index, clip in enumerate(clips):
        clip_index = int(clip.get("index", index))
        segments.append({"index": clip_index, "segment_index": index, "clip_index": clip_index, "start_time": float(clip["start_time"]), "end_time": float(clip["end_time"])})
    return segments


def frames_per_segment(config: dict) -> int:
    return int(config.get("frames_per_segment", config.get("frames_per_clip", 1)))


def extract_metadata(artifact, embedding_path: Path, config: dict) -> dict:
    if isinstance(artifact, dict) and isinstance(artifact.get("metadata"), dict):
        return artifact["metadata"]
    metadata_path = find_metadata_path(embedding_path, config)
    if metadata_path:
        return load_metadata(metadata_path)
    raise ValueError(f"No metadata found for embedding: {embedding_path}")


def find_metadata_path(embedding_path: Path, config: dict) -> Path | None:
    candidate = resolve_project_path(config["metadata_dir"]) / f"{embedding_path.stem}.json"
    return candidate if candidate.exists() else None


def clips_from_times(clip_times: list) -> list[dict]:
    clips = []
    for index, pair in enumerate(clip_times):
        clips.append({"index": index, "start_time": float(pair[0]), "end_time": float(pair[1])})
    return clips
