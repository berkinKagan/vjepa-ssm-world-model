import json
from pathlib import Path

import numpy as np

from scene_search.schemas import SceneRecord
from utils.paths import ensure_directory, resolve_project_path


def save_scene_index(records: list[SceneRecord], embeddings: np.ndarray, index_dir: str | Path, name: str, metadata: dict | None = None) -> dict:
    directory = ensure_directory(resolve_project_path(index_dir))
    jsonl_path = directory / f"{name}.jsonl"
    embeddings_path = directory / f"{name}.npy"
    metadata_path = directory / f"{name}.json"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict()) + "\n")
    np.save(embeddings_path, embeddings.astype("float32"))
    payload = {"index_file": str(jsonl_path), "embeddings_file": str(embeddings_path), "num_scenes": len(records)}
    if metadata:
        payload.update(metadata)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def load_scene_index_metadata(index_path: str | Path) -> dict:
    path = resolve_project_path(index_path)
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_scene_records(index_path: str | Path) -> list[dict]:
    path = resolve_project_path(index_path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_scene_index(index_path: str | Path) -> tuple[list[dict], np.ndarray]:
    path = resolve_project_path(index_path)
    records = load_scene_records(path)
    embeddings = np.load(path.with_suffix(".npy"))
    return records, embeddings


def list_scene_indexes(index_dir: str | Path) -> list[str]:
    directory = ensure_directory(resolve_project_path(index_dir))
    return [str(path) for path in sorted(directory.glob("*.jsonl"))]
