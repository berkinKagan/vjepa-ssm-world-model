from pathlib import Path

import numpy as np
import torch

from storage.load_embeddings import load_embedding_artifact
from utils.paths import resolve_project_path


def attach_segment_latents(embedding_path: str | Path | None, metadata: dict, segments: list[dict], output_path: str | Path) -> dict:
    if not embedding_path:
        return {"latents_file": None, "latent_dim": None, "matched_count": 0}
    resolved_embedding = resolve_project_path(embedding_path)
    artifact = load_embedding_artifact(resolved_embedding)
    values = extract_embeddings(artifact)
    if values is None:
        return {"latents_file": None, "latent_dim": None, "matched_count": 0}
    pooled = pool_vjepa_embeddings(values)
    clip_times = clip_times_from_metadata(metadata, pooled.shape[0])
    if not clip_times:
        return {"latents_file": None, "latent_dim": int(pooled.shape[-1]), "matched_count": 0}
    segment_latents = []
    for segment in segments:
        segment_latents.append(match_segment_latent(segment, clip_times, pooled))
    matrix = torch.stack(segment_latents).float().numpy()
    destination = resolve_project_path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, matrix.astype("float32"))
    return {"latents_file": str(destination), "latent_dim": int(matrix.shape[-1]), "matched_count": int(matrix.shape[0])}


def extract_embeddings(artifact):
    if isinstance(artifact, dict):
        return artifact.get("embeddings")
    return artifact


def pool_vjepa_embeddings(values) -> torch.Tensor:
    tensor = values.detach().cpu().float() if isinstance(values, torch.Tensor) else torch.as_tensor(values).float()
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 3:
        return tensor.mean(dim=1)
    dims = tuple(range(1, tensor.ndim - 1))
    return tensor.mean(dim=dims)


def clip_times_from_metadata(metadata: dict, count: int) -> list[tuple[float, float]]:
    if metadata.get("clip_times"):
        return [(float(item[0]), float(item[1])) for item in metadata["clip_times"][:count]]
    clips = metadata.get("clips", [])
    return [(float(item["start_time"]), float(item["end_time"])) for item in clips[:count]]


def match_segment_latent(segment: dict, clip_times: list[tuple[float, float]], pooled: torch.Tensor) -> torch.Tensor:
    midpoint = (float(segment["start_time"]) + float(segment["end_time"])) / 2.0
    centers = torch.tensor([(start + end) / 2.0 for start, end in clip_times], dtype=torch.float32)
    distances = torch.abs(centers - midpoint)
    nearest = int(torch.argmin(distances).item())
    return pooled[nearest]
