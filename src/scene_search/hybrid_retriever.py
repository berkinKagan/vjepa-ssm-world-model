from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.run_manager import resolve_device
from scene_search.deduplicator import deduplicate_results
from scene_search.latent_aligner import load_aligner
from scene_search.reranker import rerank_results
from scene_search.retriever import cosine_scores, resolve_index_path
from scene_search.storage import load_scene_index, load_scene_index_metadata
from scene_search.text_embedder import TextEmbedder
from utils.paths import resolve_project_path


def hybrid_search_scene_index(config: dict, query: str, index_path: str | Path | None = None, top_k: int | None = None) -> dict:
    selected_index = resolve_index_path(config, index_path)
    metadata = load_scene_index_metadata(selected_index)
    enforce_coverage(config, metadata)
    records, caption_embeddings = load_scene_index(selected_index)
    embedder = TextEmbedder(config["text_embedding_model"], config["device"])
    query_embedding = embedder.encode([query])[0]
    caption_scores = cosine_scores(query_embedding, caption_embeddings)
    latent_scores, aligner_status, mode = latent_score_vector(config, metadata, query_embedding, len(records)) if config.get("retrieval_mode", "hybrid") == "hybrid" else (None, {"status": "disabled"}, "caption_only")
    final_scores = combine_scores(config, caption_scores, latent_scores, mode)
    candidate_count = min(int(config.get("candidate_k", top_k or config["top_k"])), len(records))
    top_count = min(int(top_k or config["top_k"]), len(records))
    indices = np.argsort(-final_scores)[:candidate_count]
    results = [make_hybrid_result(rank + 1, index, records[index], caption_scores, latent_scores, final_scores, mode) for rank, index in enumerate(indices)]
    if bool(config.get("enable_deduplication", True)):
        results = deduplicate_results(results, caption_embeddings, float(config["duplicate_caption_threshold"]))
    reranking_status = "disabled"
    if bool(config.get("enable_llm_reranking", False)):
        reranked = rerank_results(config, query, [dict_to_result(item) for item in results])
        results = [result.to_dict() for result in reranked]
        reranking_status = "applied"
    return {"query": query, "index_file": str(resolve_project_path(selected_index)), "top_k": top_count, "candidate_k": candidate_count, "retrieval_mode": mode, "aligner_status": aligner_status.get("status"), "reranking_status": reranking_status, "diagnostics": metadata, "results": results[:top_count]}


def enforce_coverage(config: dict, metadata: dict) -> None:
    if not bool(config.get("coverage_required", True)):
        return
    ratio = float(metadata.get("coverage_ratio", 0.0))
    if ratio < float(config.get("min_coverage_ratio", 0.95)):
        raise ValueError("Scene index does not cover the full video.")


def latent_score_vector(config: dict, metadata: dict, query_embedding: np.ndarray, count: int) -> tuple[np.ndarray | None, dict, str]:
    aligner_status = metadata.get("aligner_status", {"status": "missing"})
    checkpoint = aligner_status.get("checkpoint") if isinstance(aligner_status, dict) else None
    latents_file = metadata.get("segment_latents_file")
    if not checkpoint or not latents_file:
        return None, aligner_status if isinstance(aligner_status, dict) else {"status": "missing"}, "hybrid_without_aligner_fallback"
    try:
        device = resolve_device(config["device"])
        model, _ = load_aligner(checkpoint, config["device"])
        latents = np.load(resolve_project_path(latents_file)).astype("float32")
        with torch.inference_mode():
            query = torch.from_numpy(query_embedding).float().unsqueeze(0).to(device)
            projected = F.normalize(model(query), dim=-1).squeeze(0).cpu().numpy()
        scores = cosine_scores(projected, latents[:count])
        return scores, aligner_status, "hybrid_caption_vjepa"
    except Exception:
        return None, {"status": "load_failed"}, "hybrid_without_aligner_fallback"


def combine_scores(config: dict, caption_scores: np.ndarray, latent_scores: np.ndarray | None, mode: str) -> np.ndarray:
    if latent_scores is None:
        return normalize_scores(caption_scores)
    caption = normalize_scores(caption_scores)
    latent = normalize_scores(latent_scores)
    return float(config["caption_weight"]) * caption + float(config["latent_weight"]) * latent


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    low = float(np.min(scores))
    high = float(np.max(scores))
    if high - low < 1e-8:
        return np.zeros_like(scores)
    return (scores - low) / (high - low)


def make_hybrid_result(rank: int, index: int, record: dict, caption_scores: np.ndarray, latent_scores: np.ndarray | None, final_scores: np.ndarray, mode: str) -> dict:
    caption_score = float(caption_scores[index])
    latent_score = float(latent_scores[index]) if latent_scores is not None else None
    segment_index = int(record.get("segment_index", record.get("clip_index", index)))
    return {"rank": rank, "score": float(final_scores[index]), "caption_score": caption_score, "latent_score": latent_score, "final_score": float(final_scores[index]), "retrieval_mode": mode, "scene_id": record["scene_id"], "caption": record.get("clean_caption") or record["caption"], "searchable_summary": record.get("searchable_summary") or record.get("caption", ""), "clip_index": record.get("clip_index"), "segment_index": segment_index, "start_time": float(record["start_time"]), "end_time": float(record["end_time"]), "video_path": record["video_path"], "frame_paths": list(record.get("frame_paths", [])), "embedding_path": record.get("embedding_path")}


def dict_to_result(item: dict):
    from scene_search.schemas import SceneSearchResult

    return SceneSearchResult(rank=int(item["rank"]), score=float(item["score"]), scene_id=item["scene_id"], caption=item["caption"], searchable_summary=item.get("searchable_summary", item["caption"]), clip_index=item.get("clip_index"), segment_index=int(item["segment_index"]), start_time=float(item["start_time"]), end_time=float(item["end_time"]), video_path=item["video_path"], frame_paths=list(item.get("frame_paths", [])), embedding_path=item.get("embedding_path"), caption_score=item.get("caption_score"), latent_score=item.get("latent_score"), final_score=item.get("final_score"), retrieval_mode=item.get("retrieval_mode"))
