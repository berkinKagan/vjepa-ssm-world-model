from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from storage.save_embeddings import save_embedding_artifact, save_metadata
from utils.paths import DEFAULT_CONFIG_PATH, ensure_directory, load_config, resolve_config_path, resolve_project_path
from utils.video_io import ClipWindow, get_video_info, read_clip, sample_clip_windows
from vjepa.model_loader import VJEPAModel, load_pretrained_vjepa
from vjepa.preprocess import PreprocessConfig

ProgressCallback = Callable[[int, int], None]


@dataclass
class ExtractionResult:
    embeddings_path: Path
    metadata_path: Path
    metadata: dict


class VJEPAExtractor:
    def __init__(self, config: dict, model: VJEPAModel | None = None):
        self.config = config
        self.video_config = config["video"]
        self.storage_config = config["storage"]
        self.preprocess_config = PreprocessConfig(clip_frames=int(self.video_config["clip_frames"]), frame_stride=int(self.video_config["frame_stride"]), image_size=int(self.video_config["image_size"]), mean=tuple(self.video_config["mean"]), std=tuple(self.video_config["std"]))
        self.model = model or load_pretrained_vjepa(config)

    @classmethod
    def from_config_file(cls, config_path: str | Path = DEFAULT_CONFIG_PATH):
        return cls(load_config(config_path))

    def extract_video(self, video_path: str | Path, output_path: str | Path | None = None, metadata_path: str | Path | None = None, progress_callback: ProgressCallback | None = None) -> ExtractionResult:
        resolved_video_path = resolve_project_path(video_path)
        video_info = get_video_info(resolved_video_path)
        windows = sample_clip_windows(video_info, int(self.video_config["clip_frames"]), int(self.video_config["frame_stride"]), float(self.video_config["clip_stride_seconds"]), self.video_config.get("max_clips"))
        embeddings = []
        for index, window in enumerate(windows, start=1):
            frames = read_clip(resolved_video_path, window.frame_indices)
            features = self.model.encode(frames, self.preprocess_config)
            embeddings.append(remove_batch_dimension(features))
            if progress_callback is not None:
                progress_callback(index, len(windows))
        sequence = torch.stack(embeddings, dim=0)
        resolved_output_path = self.resolve_output_path(resolved_video_path, output_path)
        resolved_metadata_path = self.resolve_metadata_path(resolved_video_path, metadata_path)
        metadata = self.build_metadata(video_info, windows, sequence, resolved_output_path, resolved_metadata_path)
        save_embedding_artifact(sequence, metadata, resolved_output_path)
        save_metadata(metadata, resolved_metadata_path)
        return ExtractionResult(embeddings_path=resolved_output_path, metadata_path=resolved_metadata_path, metadata=metadata)

    def resolve_output_path(self, video_path: Path, output_path: str | Path | None) -> Path:
        if output_path is not None:
            return resolve_project_path(output_path)
        embeddings_dir = resolve_config_path(self.config, "embeddings_dir")
        ensure_directory(embeddings_dir)
        suffix = "." + self.storage_config.get("format", "pt").lstrip(".")
        return embeddings_dir / f"{video_path.stem}{suffix}"

    def resolve_metadata_path(self, video_path: Path, metadata_path: str | Path | None) -> Path:
        if metadata_path is not None:
            return resolve_project_path(metadata_path)
        metadata_dir = resolve_config_path(self.config, "metadata_dir")
        ensure_directory(metadata_dir)
        return metadata_dir / f"{video_path.stem}.json"

    def build_metadata(self, video_info, windows: list[ClipWindow], embeddings: torch.Tensor, embeddings_path: Path, metadata_path: Path) -> dict:
        clip_metadata = [{"index": window.index, "start_frame": window.start_frame, "end_frame": window.end_frame, "start_time": window.start_time, "end_time": window.end_time, "num_frames": len(window.frame_indices)} for window in windows]
        return {"video_filename": video_info.filename, "video_path": str(video_info.path), "fps": video_info.fps, "duration": video_info.duration, "frame_count": video_info.frame_count, "clips": clip_metadata, "clip_times": [[clip["start_time"], clip["end_time"]] for clip in clip_metadata], "frames_per_clip": int(self.video_config["clip_frames"]), "frame_stride": int(self.video_config["frame_stride"]), "clip_stride_seconds": float(self.video_config["clip_stride_seconds"]), "num_clips": len(windows), "embedding_shape": list(embeddings.shape), "embedding_layout": infer_embedding_layout(embeddings), "temporal_axis": 0, "model": {"backend": self.model.backend, "name": self.model.model_name, "device": str(self.model.device), "precision": self.model.precision}, "config": {"video": self.video_config, "storage": self.storage_config}, "embedding_file": str(embeddings_path), "metadata_file": str(metadata_path)}


def remove_batch_dimension(features: torch.Tensor) -> torch.Tensor:
    if features.ndim >= 1 and features.shape[0] == 1:
        return features.squeeze(0)
    return features


def infer_embedding_layout(embeddings: torch.Tensor) -> str:
    if embeddings.ndim == 3:
        return "clip, token, channel"
    if embeddings.ndim == 4:
        return "clip, time_or_depth, token, channel"
    return "clip, ..."
