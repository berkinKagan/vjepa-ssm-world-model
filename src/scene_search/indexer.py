import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scene_search.captioner import SceneCaptioner
from scene_search.frame_sampler import sample_clip_frames
from scene_search.latent_aligner import ensure_text_latent_aligner
from scene_search.schemas import SceneRecord
from scene_search.segments import coverage_diagnostics, dense_video_segments, get_video_duration
from scene_search.segment_latents import attach_segment_latents
from scene_search.storage import load_scene_index_metadata, load_scene_records, save_scene_index
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
        record = SceneRecord(scene_id=f"{resolved_embedding.stem}_segment_{segment_index:04d}", video_path=str(resolve_project_path(video_path)), embedding_path=str(resolved_embedding), metadata_path=str(metadata_path) if metadata_path else None, segment_index=segment_index, start_time=float(segment["start_time"]), end_time=float(segment["end_time"]), caption=caption["searchable_summary"], raw_caption=caption["raw_caption"], clean_caption=caption["clean_caption"], searchable_summary=caption["searchable_summary"], frame_paths=frame_paths, caption_backend=backend, ollama_model=model, created_at=datetime.now(timezone.utc).isoformat(), segment_source=config.get("segment_source", "dense_video"), clip_index=segment.get("clip_index"))
        records.append(record)
        if progress_callback is not None:
            progress_callback(index, total)
    embedder = TextEmbedder(config["text_embedding_model"], config["device"])
    caption_embeddings = embedder.encode([record.searchable_summary for record in records])
    latent_path = resolve_project_path(config["scene_index_dir"]) / f"{resolved_embedding.stem}_vjepa_latents.npy"
    latent_info = attach_segment_latents(resolved_embedding, metadata, segments, latent_path)
    aligner_status = ensure_text_latent_aligner(config, caption_embeddings, latent_info.get("latents_file")) if bool(config.get("train_text_latent_aligner", True)) else {"status": "disabled", "checkpoint": None}
    index_metadata = {**diagnostics, **latent_info, "segment_latents_file": latent_info.get("latents_file"), "aligner_status": aligner_status, "segment_source": config.get("segment_source", "dense_video"), "segment_length_seconds": float(config.get("segment_length_seconds", 8.0)), "segment_stride_seconds": float(config.get("segment_stride_seconds", 8.0)), "min_segment_seconds": float(config.get("min_segment_seconds", 2.0)), "frames_per_segment": frames_per_segment(config), "embedding_model": config["text_embedding_model"], "caption_backend": config["caption_backend"], "ollama_model": config.get("ollama_model"), "reused": False}
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
    if has_failed_captions(index_path):
        return None
    metadata = ensure_cached_aligner(config, metadata, embeddings_path, metadata_path)
    return {**metadata, "index_file": str(index_path), "embeddings_file": str(embeddings_path), "reused": True}


def ensure_cached_aligner(config: dict, metadata: dict, embeddings_path: Path, metadata_path: Path) -> dict:
    if not bool(config.get("train_text_latent_aligner", True)):
        return metadata
    status = metadata.get("aligner_status")
    checkpoint = status.get("checkpoint") if isinstance(status, dict) else None
    if checkpoint and resolve_project_path(checkpoint).exists():
        return metadata
    latent_path = metadata.get("segment_latents_file") or metadata.get("latents_file")
    if not latent_path or not embeddings_path.exists():
        return metadata
    caption_embeddings = np.load(embeddings_path)
    metadata = {**metadata, "aligner_status": ensure_text_latent_aligner(config, caption_embeddings, latent_path)}
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def reusable_index(config: dict, metadata: dict) -> bool:
    expected = {"segment_source": config.get("segment_source", "dense_video"), "segment_length_seconds": float(config.get("segment_length_seconds", 8.0)), "segment_stride_seconds": float(config.get("segment_stride_seconds", 8.0)), "min_segment_seconds": float(config.get("min_segment_seconds", 2.0)), "frames_per_segment": frames_per_segment(config), "embedding_model": config["text_embedding_model"], "caption_backend": config["caption_backend"], "ollama_model": config.get("ollama_model")}
    for key, value in expected.items():
        if metadata.get(key) != value:
            return False
    return float(metadata.get("coverage_ratio", 0.0)) >= 0.95


def has_failed_captions(index_path: Path) -> bool:
    failed_markers = ["HTTP Error 400", "Ollama HTTP 400", "Bad Request", "No Ollama vision model is available"]
    try:
        for record in load_scene_records(index_path):
            text = " ".join(str(record.get(key, "")) for key in ["caption", "raw_caption", "clean_caption", "searchable_summary"])
            if any(marker in text for marker in failed_markers):
                return True
    except Exception:
        return True
    return False


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
