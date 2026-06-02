import shutil
import uuid
from pathlib import Path
from threading import Lock

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import ExtractResponse, FileListResponse, SSMJobResponse, SSMStatusResponse, StatusResponse, VideoListResponse
from experiments.benchmark import run_latent_benchmark
from experiments.run_manager import DEFAULT_SSM_CONFIG_PATH, load_ssm_config
from experiments.run_manager import save_json
from ssm.predictor import predict_future_latents
from ssm.trainer import train_ssm_forecaster
from utils.paths import DEFAULT_CONFIG_PATH, PROJECT_ROOT, ensure_directory, load_config, resolve_config_path, resolve_project_path
from vjepa.extractor import VJEPAExtractor

app = FastAPI(title="V-JEPA Latent Dynamics")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CONFIG = load_config(DEFAULT_CONFIG_PATH)
SSM_CONFIG = load_ssm_config(DEFAULT_SSM_CONFIG_PATH)
WEB_DIR = PROJECT_ROOT / "web"
JOBS: dict[str, dict] = {}
JOBS_LOCK = Lock()
SSM_JOBS: dict[str, dict] = {}
SSM_JOBS_LOCK = Lock()
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


@app.get("/embeddings", response_model=FileListResponse)
def embeddings() -> FileListResponse:
    return FileListResponse(files=list_relative_files(resolve_project_path(SSM_CONFIG["embeddings_dir"]), {".pt", ".npy"}))


@app.get("/checkpoints", response_model=FileListResponse)
def checkpoints() -> FileListResponse:
    return FileListResponse(files=list_relative_files(resolve_project_path(SSM_CONFIG["checkpoints_dir"]), {".pt"}))


@app.post("/ssm/train", response_model=SSMJobResponse)
async def ssm_train(background_tasks: BackgroundTasks, embedding: str | None = Form(default=None), config_path: str | None = Form(default=None)) -> SSMJobResponse:
    job_id = uuid.uuid4().hex
    set_ssm_job(job_id, {"status": "queued", "progress": 0.0, "message": "queued", "embedding": embedding})
    background_tasks.add_task(run_ssm_training_job, job_id, embedding, config_path)
    return SSMJobResponse(job_id=job_id, status="queued")


@app.post("/pipeline/run", response_model=SSMJobResponse)
async def pipeline_run(background_tasks: BackgroundTasks, file: UploadFile | None = File(default=None), video_path: str | None = Form(default=None), embedding: str | None = Form(default=None), config_path: str | None = Form(default=None)) -> SSMJobResponse:
    selected_video = None
    if not embedding:
        selected_video = save_upload(file) if file is not None and file.filename else resolve_input_video(video_path)
    job_id = uuid.uuid4().hex
    set_ssm_job(job_id, {"status": "queued", "progress": 0.0, "message": "queued", "stage": "queued", "embedding": embedding, "video_path": str(selected_video) if selected_video is not None else None})
    background_tasks.add_task(run_pipeline_job, job_id, str(selected_video) if selected_video is not None else None, embedding, config_path)
    return SSMJobResponse(job_id=job_id, status="queued")


@app.get("/ssm/status/{job_id}", response_model=SSMStatusResponse)
def ssm_status(job_id: str) -> SSMStatusResponse:
    return ssm_status_response(job_id, get_ssm_job(job_id))


@app.get("/pipeline/status/{job_id}", response_model=SSMStatusResponse)
def pipeline_status(job_id: str) -> SSMStatusResponse:
    return ssm_status_response(job_id, get_ssm_job(job_id))


@app.post("/ssm/predict", response_model=SSMJobResponse)
async def ssm_predict(background_tasks: BackgroundTasks, embedding: str = Form(...), checkpoint: str = Form("outputs/checkpoints/best.pt"), output: str | None = Form(default=None), config_path: str | None = Form(default=None)) -> SSMJobResponse:
    job_id = uuid.uuid4().hex
    set_ssm_job(job_id, {"status": "queued", "progress": 0.0, "message": "queued", "embedding": embedding, "checkpoint": checkpoint})
    background_tasks.add_task(run_ssm_prediction_job, job_id, embedding, checkpoint, output, config_path)
    return SSMJobResponse(job_id=job_id, status="queued")


@app.get("/ssm/metrics/{job_id}")
def ssm_metrics(job_id: str) -> dict:
    job = get_ssm_job(job_id)
    if "metrics" not in job:
        raise HTTPException(status_code=404, detail="Metrics are not available for this job")
    return job["metrics"]


@app.get("/pipeline/results/{job_id}")
def pipeline_results(job_id: str) -> dict:
    job = get_ssm_job(job_id)
    if "metrics" not in job:
        raise HTTPException(status_code=404, detail="Pipeline results are not available for this job")
    return job["metrics"]


@app.post("/benchmark/run", response_model=SSMJobResponse)
async def benchmark_run(background_tasks: BackgroundTasks, embedding: str = Form(...), steps: int = Form(5), config_path: str | None = Form(default=None)) -> SSMJobResponse:
    job_id = uuid.uuid4().hex
    set_ssm_job(job_id, {"status": "queued", "progress": 0.0, "message": "queued", "stage": "benchmark", "embedding": embedding})
    background_tasks.add_task(run_benchmark_job, job_id, embedding, steps, config_path)
    return SSMJobResponse(job_id=job_id, status="queued")


@app.get("/benchmark/status/{job_id}", response_model=SSMStatusResponse)
def benchmark_status(job_id: str) -> SSMStatusResponse:
    return ssm_status_response(job_id, get_ssm_job(job_id))


@app.get("/benchmark/results/{job_id}")
def benchmark_results(job_id: str) -> dict:
    job = get_ssm_job(job_id)
    if "metrics" not in job:
        raise HTTPException(status_code=404, detail="Benchmark results are not available for this job")
    return job["metrics"]


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


def run_ssm_training_job(job_id: str, embedding: str | None, config_path: str | None) -> None:
    try:
        config = load_ssm_config(config_path or DEFAULT_SSM_CONFIG_PATH)
        if embedding:
            config = {**config, "embedding_files": [embedding]}
        update_ssm_job(job_id, status="running", progress=0.0, message="loading embeddings")
        result = train_ssm_forecaster(config, progress_callback=lambda epoch, total, metrics: update_ssm_job(job_id, status="running", progress=epoch / total if total else 0.0, message=f"epoch {epoch}/{total}", metrics={"latest_validation": metrics}))
        metrics = {"training": result.training_metrics, "validation": result.validation_metrics, "model_type": result.model_type, "latent_dim": result.latent_dim}
        update_ssm_job(job_id, status="completed", progress=1.0, message="completed", checkpoint_path=str(result.best_checkpoint), metrics=metrics)
    except Exception as exc:
        update_ssm_job(job_id, status="failed", progress=0.0, message="failed", error=str(exc))


def run_pipeline_job(job_id: str, video_path: str | None, embedding: str | None, config_path: str | None) -> None:
    try:
        config = load_ssm_config(config_path or DEFAULT_SSM_CONFIG_PATH)
        metadata_path = None
        metadata = None
        if embedding:
            embedding_path = resolve_project_path(embedding)
            update_ssm_job(job_id, status="running", stage="latents", progress=0.1, message="using existing latents", embedding_path=str(embedding_path))
        else:
            if not video_path:
                raise ValueError("Provide an existing embedding or a video input")
            update_ssm_job(job_id, status="running", stage="extraction", progress=0.02, message="extracting V-JEPA latents")
            extractor = get_extractor()
            extraction = extractor.extract_video(video_path, progress_callback=lambda done, total: update_ssm_job(job_id, status="running", stage="extraction", progress=0.05 + 0.35 * (done / total if total else 0.0), message=f"extracted {done}/{total} clips"))
            embedding_path = extraction.embeddings_path
            metadata_path = extraction.metadata_path
            metadata = extraction.metadata
            update_ssm_job(job_id, embedding_path=str(embedding_path), metadata_path=str(metadata_path))
        train_config = {**config, "embedding_files": [str(embedding_path)]}
        update_ssm_job(job_id, status="running", stage="training", progress=0.42, message="training latent forecaster")
        training = train_ssm_forecaster(train_config, progress_callback=lambda epoch, total, metrics: update_ssm_job(job_id, status="running", stage="training", progress=0.42 + 0.43 * (epoch / total if total else 0.0), message=f"training epoch {epoch}/{total}", metrics={"latest_validation": metrics}))
        update_ssm_job(job_id, status="running", stage="prediction", progress=0.88, message="predicting future latents", checkpoint_path=str(training.best_checkpoint))
        prediction = predict_future_latents(embedding_path, training.best_checkpoint, None, config)
        metrics = {"embedding_file": str(embedding_path), "metadata_file": str(metadata_path) if metadata_path is not None else None, "checkpoint_file": str(training.best_checkpoint), "prediction": prediction, "training": training.training_metrics, "validation": training.validation_metrics, "model_type": training.model_type, "latent_dim": training.latent_dim, "metadata": metadata}
        update_ssm_job(job_id, status="completed", stage="completed", progress=1.0, message="completed", embedding_path=str(embedding_path), metadata_path=str(metadata_path) if metadata_path is not None else None, checkpoint_path=str(training.best_checkpoint), prediction_path=prediction["prediction_file"], prediction_shape=prediction["prediction_shape"], metrics=metrics)
    except Exception as exc:
        update_ssm_job(job_id, status="failed", progress=0.0, message="failed", error=str(exc))


def run_ssm_prediction_job(job_id: str, embedding: str, checkpoint: str, output: str | None, config_path: str | None) -> None:
    try:
        update_ssm_job(job_id, status="running", progress=0.2, message="loading checkpoint")
        config = load_ssm_config(config_path or DEFAULT_SSM_CONFIG_PATH)
        result = predict_future_latents(embedding, checkpoint, output, config)
        update_ssm_job(job_id, status="completed", progress=1.0, message="completed", prediction_path=result["prediction_file"], prediction_shape=result["prediction_shape"], checkpoint_path=result["checkpoint"], metrics=result)
    except Exception as exc:
        update_ssm_job(job_id, status="failed", progress=0.0, message="failed", error=str(exc))


def run_benchmark_job(job_id: str, embedding: str, steps: int, config_path: str | None) -> None:
    try:
        config = load_ssm_config(config_path or DEFAULT_SSM_CONFIG_PATH)
        update_ssm_job(job_id, status="running", stage="benchmark", progress=0.1, message="running latent benchmark", embedding_path=str(resolve_project_path(embedding)))
        result = run_latent_benchmark(embedding, config, steps)
        benchmark_path = resolve_project_path("outputs/benchmarks") / f"{job_id}.json"
        result["benchmark_file"] = str(save_json(result, benchmark_path))
        update_ssm_job(job_id, status="completed", stage="completed", progress=1.0, message="completed", embedding_path=str(resolve_project_path(embedding)), metrics=result)
    except Exception as exc:
        update_ssm_job(job_id, status="failed", progress=0.0, message="failed", error=str(exc))


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


def set_ssm_job(job_id: str, values: dict) -> None:
    with SSM_JOBS_LOCK:
        SSM_JOBS[job_id] = values


def update_ssm_job(job_id: str, **values) -> None:
    with SSM_JOBS_LOCK:
        SSM_JOBS[job_id].update(values)


def get_ssm_job(job_id: str) -> dict:
    with SSM_JOBS_LOCK:
        if job_id not in SSM_JOBS:
            raise HTTPException(status_code=404, detail="SSM job not found")
        return dict(SSM_JOBS[job_id])


def ssm_status_response(job_id: str, job: dict) -> SSMStatusResponse:
    return SSMStatusResponse(job_id=job_id, status=job["status"], progress=job.get("progress", 0.0), message=job.get("message", ""), stage=job.get("stage"), embedding_path=job.get("embedding_path"), metadata_path=job.get("metadata_path"), checkpoint_path=job.get("checkpoint_path"), prediction_path=job.get("prediction_path"), prediction_shape=job.get("prediction_shape"), error=job.get("error"))


def list_relative_files(directory: Path, suffixes: set[str]) -> list[str]:
    ensure_directory(directory)
    return [str(path.relative_to(PROJECT_ROOT)) for path in sorted(directory.glob("*")) if path.is_file() and path.suffix.lower() in suffixes]


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
