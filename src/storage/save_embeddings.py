import json
from pathlib import Path

import numpy as np
import torch

from utils.paths import ensure_directory


def save_embedding_artifact(embeddings: torch.Tensor, metadata: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    ensure_directory(path.parent)
    suffix = path.suffix.lower()
    if suffix == ".pt":
        torch.save({"embeddings": embeddings, "clip_times": torch.tensor(metadata["clip_times"], dtype=torch.float32), "metadata": metadata}, path)
    elif suffix == ".npy":
        np.save(path, embeddings.numpy())
    else:
        raise ValueError("Embedding output path must end with .pt or .npy")
    return path


def save_metadata(metadata: dict, metadata_path: str | Path) -> Path:
    path = Path(metadata_path)
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return path
