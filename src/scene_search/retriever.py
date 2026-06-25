from pathlib import Path

import numpy as np

from scene_search.reranker import rerank_results
from scene_search.schemas import SceneSearchResult
from scene_search.storage import list_scene_indexes, load_scene_index
from scene_search.text_embedder import TextEmbedder
from utils.paths import resolve_project_path


def search_scene_index(config: dict, query: str, index_path: str | Path | None = None, top_k: int | None = None) -> dict:
    selected_index = resolve_index_path(config, index_path)
    records, embeddings = load_scene_index(selected_index)
    embedder = TextEmbedder(config["text_embedding_model"], config["device"])
    query_embedding = embedder.encode([query])[0]
    scores = cosine_scores(query_embedding, embeddings)
    count = min(int(top_k or config["top_k"]), len(records))
    indices = np.argsort(-scores)[:count]
    results = [make_result(rank + 1, float(scores[index]), records[index]) for rank, index in enumerate(indices)]
    if bool(config.get("enable_llm_reranking", False)):
        results = rerank_results(config, query, results)
    return {"query": query, "index_file": str(resolve_project_path(selected_index)), "top_k": count, "embedding_backend": embedder.backend, "results": [result.to_dict() for result in results]}


def resolve_index_path(config: dict, index_path: str | Path | None) -> Path:
    if index_path:
        return resolve_project_path(index_path)
    indexes = list_scene_indexes(config["scene_index_dir"])
    if not indexes:
        raise ValueError("No scene indexes found. Build an index first.")
    return resolve_project_path(indexes[-1])


def cosine_scores(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    query = query_embedding / max(np.linalg.norm(query_embedding), 1e-8)
    matrix = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    return matrix @ query


def make_result(rank: int, score: float, record: dict) -> SceneSearchResult:
    segment_index = int(record.get("segment_index", record.get("clip_index", 0)))
    clip_index = record.get("clip_index")
    return SceneSearchResult(rank=rank, score=score, scene_id=record["scene_id"], caption=record["caption"], clip_index=int(clip_index) if clip_index is not None else None, segment_index=segment_index, start_time=float(record["start_time"]), end_time=float(record["end_time"]), video_path=record["video_path"], frame_paths=list(record.get("frame_paths", [])), embedding_path=record.get("embedding_path"))
