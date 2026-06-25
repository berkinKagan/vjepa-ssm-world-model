import numpy as np


def deduplicate_results(results: list[dict], caption_embeddings: np.ndarray | None, threshold: float) -> list[dict]:
    kept = []
    kept_indices = []
    for result in results:
        index = int(result.get("segment_index", result.get("clip_index", 0)))
        duplicate = False
        for kept_result, kept_index in zip(kept, kept_indices):
            if overlapping(result, kept_result):
                duplicate = True
            elif caption_embeddings is not None and index < len(caption_embeddings) and kept_index < len(caption_embeddings):
                duplicate = cosine(caption_embeddings[index], caption_embeddings[kept_index]) >= threshold
            if duplicate:
                break
        if not duplicate:
            kept.append(result)
            kept_indices.append(index)
    for rank, result in enumerate(kept, start=1):
        result["rank"] = rank
    return kept


def overlapping(left: dict, right: dict) -> bool:
    left_start = float(left["start_time"])
    left_end = float(left["end_time"])
    right_start = float(right["start_time"])
    right_end = float(right["end_time"])
    overlap = min(left_end, right_end) - max(left_start, right_start)
    return overlap > 0


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = max(np.linalg.norm(left) * np.linalg.norm(right), 1e-8)
    return float(np.dot(left, right) / denom)
