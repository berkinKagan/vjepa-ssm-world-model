from dataclasses import asdict, dataclass


@dataclass
class OllamaModelInfo:
    name: str
    size: int | None = None
    modified_at: str | None = None
    vision_capable: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SceneRecord:
    scene_id: str
    video_path: str
    embedding_path: str | None
    metadata_path: str | None
    segment_index: int
    start_time: float
    end_time: float
    caption: str
    raw_caption: str
    clean_caption: str
    searchable_summary: str
    frame_paths: list[str]
    caption_backend: str
    ollama_model: str | None
    created_at: str
    segment_source: str
    clip_index: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SceneSearchResult:
    rank: int
    score: float
    scene_id: str
    caption: str
    searchable_summary: str
    clip_index: int | None
    segment_index: int
    start_time: float
    end_time: float
    video_path: str
    frame_paths: list[str]
    embedding_path: str | None
    caption_score: float | None = None
    latent_score: float | None = None
    final_score: float | None = None
    rerank_score: float | None = None
    retrieval_mode: str | None = None
    original_rank: int | None = None
    original_score: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)
