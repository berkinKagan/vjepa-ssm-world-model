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


class FileListResponse(BaseModel):
    files: list[str]


class SSMJobResponse(BaseModel):
    job_id: str
    status: str


class SSMStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    stage: str | None = None
    embedding_path: str | None = None
    metadata_path: str | None = None
    checkpoint_path: str | None = None
    prediction_path: str | None = None
    prediction_shape: list[int] | None = None
    error: str | None = None
