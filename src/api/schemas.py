from pydantic import BaseModel


class ExtractResponse(BaseModel):
    job_id: str
    status: str


class StatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    embedding_path: str | None = None
    metadata_path: str | None = None
    error: str | None = None


class VideoListResponse(BaseModel):
    videos: list[str]
