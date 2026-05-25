import json
from pathlib import Path

import numpy as np
import torch


def load_embedding_artifact(path: str | Path):
    artifact_path = Path(path)
    suffix = artifact_path.suffix.lower()
    if suffix == ".pt":
        return torch.load(artifact_path, map_location="cpu")
    if suffix == ".npy":
        return np.load(artifact_path)
    raise ValueError("Embedding path must end with .pt or .npy")


def load_metadata(path: str | Path) -> dict:
    metadata_path = Path(path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
