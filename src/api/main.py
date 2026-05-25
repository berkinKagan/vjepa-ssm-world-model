import shutil
import uuid
from pathlib import Path
from threading import Lock

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import ExtractResponse, StatusResponse, VideoListResponse
from utils.paths import DEFAULT_CONFIG_PATH, PROJECT_ROOT, ensure_directory, load_config, resolve_config_path, resolve_project_path
from vjepa.extractor import VJEPAExtractor

app = FastAPI(title="V-JEPA Latent Dynamics Phase 1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CONFIG = load_config(DEFAULT_CONFIG_PATH)
WEB_DIR = PROJECT_ROOT / "web"
JOBS: dict[str, dict] = {}
JOBS_LOCK = Lock()
EXTRACTOR: VJEPAExtractor | None = None
EXTRACTOR_LOCK = Lock()


@app.post("/extract", response_model=ExtractResponse)
async def extract(background_tasks: BackgroundTasks, file: UploadFile | None = File(default=None), video_path: str | None = Form(default=None)) -> ExtractResponse:
    selected_path = save_upload(file) if file is not None and file.filename else resolve_input_video(video_path)
    job_id = uuid.uuid4().hex
    set_job(job_id, {"status": "queued", "progress": 0.0, "message": "queued", "video_path": str(selected_path)})
    background_tasks.add_task(run_extraction_job, job_id, str(selected_path))
    return ExtractResponse(job_id=job_id, status="queued")


@app.get("/status/{job_id}", response_model=StatusResponse)
def status(job_id: str) -> StatusResponse:
    job = get_job(job_id)
    return StatusResponse(job_id=job_id, status=job["status"], progress=job.get("progress", 0.0), message=job.get("message", ""), embedding_path=job.get("embedding_path"), metadata_path=job.get("metadata_path"), error=job.get("error"))


@app.get("/metadata/{job_id}")
def metadata(job_id: str) -> dict:
    job = get_job(job_id)
    if "metadata" not in job:
        raise HTTPException(status_code=404, detail="Metadata is not available for this job")
    return job["metadata"]


@app.get("/videos", response_model=VideoListResponse)
def videos() -> VideoListResponse:
    sample_dir = resolve_config_path(CONFIG, "sample_videos_dir")
    ensure_directory(sample_dir)
    suffixes = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    names = [str(path.relative_to(PROJECT_ROOT)) for path in sorted(sample_dir.rglob("*")) if path.is_file() and path.suffix.lower() in suffixes]
    return VideoListResponse(videos=names)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


def save_upload(file: UploadFile) -> Path:
    uploads_dir = resolve_config_path(CONFIG, "uploads_dir")
    ensure_directory(uploads_dir)
    name = Path(file.filename or "upload.mp4").name
    destination = uploads_dir / f"{uuid.uuid4().hex}_{name}"
    with destination.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    return destination


def resolve_input_video(video_path: str | None) -> Path:
    if not video_path:
        raise HTTPException(status_code=400, detail="Provide either a video upload or a video_path")
    path = resolve_project_path(video_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")
    return path


def run_extraction_job(job_id: str, video_path: str) -> None:
    try:
        update_job(job_id, status="running", progress=0.0, message="loading model and video")
        extractor = get_extractor()
        result = extractor.extract_video(video_path, progress_callback=lambda done, total: update_job(job_id, status="running", progress=done / total if total else 0.0, message=f"processed {done}/{total} clips"))
        update_job(job_id, status="completed", progress=1.0, message="completed", embedding_path=str(result.embeddings_path), metadata_path=str(result.metadata_path), metadata=result.metadata)
    except Exception as exc:
        update_job(job_id, status="failed", progress=0.0, message="failed", error=str(exc))


def get_extractor() -> VJEPAExtractor:
    global EXTRACTOR
    with EXTRACTOR_LOCK:
        if EXTRACTOR is None:
            EXTRACTOR = VJEPAExtractor(CONFIG)
        return EXTRACTOR


def set_job(job_id: str, values: dict) -> None:
    with JOBS_LOCK:
        JOBS[job_id] = values


def update_job(job_id: str, **values) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(values)


def get_job(job_id: str) -> dict:
    with JOBS_LOCK:
        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail="Job not found")
        return dict(JOBS[job_id])


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
