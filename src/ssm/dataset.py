from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from storage.load_embeddings import load_embedding_artifact, load_metadata
from utils.paths import PROJECT_ROOT, resolve_project_path


@dataclass
class EmbeddingSequence:
    embedding_path: Path
    metadata_path: Path | None
    metadata: dict
    values: torch.Tensor


@dataclass(frozen=True)
class WindowSample:
    sequence_index: int
    start: int


@dataclass
class NormalizationStats:
    mean: torch.Tensor
    std: torch.Tensor

    def to_dict(self) -> dict:
        return {"mean": self.mean.cpu(), "std": self.std.cpu(), "enabled": True}


class LatentWindowDataset(Dataset):
    def __init__(self, sequences: list[EmbeddingSequence], samples: list[WindowSample], context_len: int, pred_horizon: int, normalizer: NormalizationStats | None = None):
        self.sequences = sequences
        self.samples = samples
        self.context_len = context_len
        self.pred_horizon = pred_horizon
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        sequence = self.sequences[sample.sequence_index].values
        context_end = sample.start + self.context_len
        target_end = context_end + self.pred_horizon
        context = sequence[sample.start:context_end]
        target = sequence[context_end:target_end]
        if self.normalizer is not None:
            context = normalize_values(context, self.normalizer)
            target = normalize_values(target, self.normalizer)
        return context, target

    @property
    def latent_dim(self) -> int:
        return int(self.sequences[0].values.shape[-1])


def load_embedding_sequences(config: dict) -> list[EmbeddingSequence]:
    embeddings_dir = resolve_project_path(config["embeddings_dir"])
    metadata_dir = resolve_project_path(config["metadata_dir"])
    pooling_mode = config.get("pooling_mode", "mean")
    configured_files = config.get("embedding_files")
    files = [resolve_project_path(path) for path in configured_files] if configured_files else list_embedding_files(embeddings_dir)
    sequences = [load_embedding_sequence(path, metadata_dir, pooling_mode) for path in files]
    return [sequence for sequence in sequences if sequence.values.shape[0] >= int(config["context_len"]) + int(config["pred_horizon"])]


def list_embedding_files(embeddings_dir: str | Path) -> list[Path]:
    directory = resolve_project_path(embeddings_dir)
    if not directory.exists():
        return []
    suffixes = {".pt", ".npy"}
    return [path for path in sorted(directory.glob("*")) if path.is_file() and path.suffix.lower() in suffixes]


def load_embedding_sequence(embedding_path: str | Path, metadata_dir: str | Path | None, pooling_mode: str) -> EmbeddingSequence:
    path = resolve_project_path(embedding_path)
    artifact = load_embedding_artifact(path)
    metadata = {}
    raw_embeddings = artifact
    if isinstance(artifact, dict):
        raw_embeddings = artifact.get("embeddings")
        metadata = artifact.get("metadata", {})
    if raw_embeddings is None:
        raise ValueError(f"No embeddings found in artifact: {path}")
    metadata_path = find_metadata_path(path, metadata_dir)
    if metadata_path is not None:
        metadata = {**metadata, **load_metadata(metadata_path)}
    values = pool_embeddings(to_float_tensor(raw_embeddings), pooling_mode)
    return EmbeddingSequence(embedding_path=path, metadata_path=metadata_path, metadata=metadata, values=values)


def find_metadata_path(embedding_path: Path, metadata_dir: str | Path | None) -> Path | None:
    if metadata_dir is None:
        return None
    candidate = resolve_project_path(metadata_dir) / f"{embedding_path.stem}.json"
    return candidate if candidate.exists() else None


def to_float_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).float()
    return torch.as_tensor(value).float()


def pool_embeddings(embeddings: torch.Tensor, pooling_mode: str) -> torch.Tensor:
    if embeddings.ndim == 2:
        return embeddings
    if embeddings.ndim < 2:
        raise ValueError(f"Embeddings must have at least 2 dimensions, got {list(embeddings.shape)}")
    if pooling_mode == "flatten":
        return embeddings.reshape(embeddings.shape[0], -1)
    if pooling_mode == "cls":
        if embeddings.ndim == 3:
            return embeddings[:, 0, :]
        return embeddings.reshape(embeddings.shape[0], -1, embeddings.shape[-1])[:, 0, :]
    if pooling_mode == "mean":
        if embeddings.ndim == 3:
            return embeddings.mean(dim=1)
        dims = tuple(range(1, embeddings.ndim - 1))
        return embeddings.mean(dim=dims)
    raise ValueError(f"Unsupported pooling mode: {pooling_mode}")


def build_window_samples(sequences: list[EmbeddingSequence], context_len: int, pred_horizon: int) -> list[WindowSample]:
    samples = []
    span = context_len + pred_horizon
    for sequence_index, sequence in enumerate(sequences):
        limit = sequence.values.shape[0] - span + 1
        for start in range(max(0, limit)):
            samples.append(WindowSample(sequence_index=sequence_index, start=start))
    return samples


def build_train_val_datasets(config: dict, normalizer: NormalizationStats | None = None) -> tuple[LatentWindowDataset, LatentWindowDataset, list[EmbeddingSequence]]:
    sequences = load_embedding_sequences(config)
    if not sequences:
        raise ValueError("No embedding sequences are long enough for the configured context_len and pred_horizon")
    context_len = int(config["context_len"])
    pred_horizon = int(config["pred_horizon"])
    samples = build_window_samples(sequences, context_len, pred_horizon)
    if not samples:
        raise ValueError("No sliding-window samples could be built from the available embeddings")
    train_samples, val_samples = split_samples(samples, float(config["validation_split"]), int(config["seed"]))
    train_dataset = LatentWindowDataset(sequences, train_samples, context_len, pred_horizon, normalizer)
    val_dataset = LatentWindowDataset(sequences, val_samples, context_len, pred_horizon, normalizer)
    return train_dataset, val_dataset, sequences


def split_samples(samples: list[WindowSample], validation_split: float, seed: int) -> tuple[list[WindowSample], list[WindowSample]]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(samples), generator=generator).tolist()
    shuffled = [samples[index] for index in order]
    if len(shuffled) <= 1 or validation_split <= 0:
        return shuffled, []
    val_count = int(round(len(shuffled) * validation_split))
    val_count = min(max(1, val_count), len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def compute_normalization_stats(dataset: LatentWindowDataset) -> NormalizationStats:
    chunks = []
    for sample in dataset.samples:
        sequence = dataset.sequences[sample.sequence_index].values
        end = sample.start + dataset.context_len + dataset.pred_horizon
        chunks.append(sequence[sample.start:end])
    values = torch.cat(chunks, dim=0)
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False).clamp_min(1e-6)
    return NormalizationStats(mean=mean, std=std)


def normalize_values(values: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    return (values - stats.mean) / stats.std


def denormalize_values(values: torch.Tensor, stats: NormalizationStats | None) -> torch.Tensor:
    if stats is None:
        return values
    return values * stats.std.to(values.device) + stats.mean.to(values.device)


def normalizer_from_checkpoint(checkpoint: dict) -> NormalizationStats | None:
    payload = checkpoint.get("normalization")
    if not payload or not payload.get("enabled", False):
        return None
    return NormalizationStats(mean=payload["mean"].float(), std=payload["std"].float())


def relative_to_project(path: str | Path) -> str:
    resolved = resolve_project_path(path)
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)
