# vjepa-latent-dynamics

Phase 1 research prototype for extracting frozen V-JEPA latent video embeddings from local videos.

This repository currently implements only the V-JEPA feature extraction pipeline. It does not include the later SSM/Mamba temporal prediction work yet, but the saved artifact stores clip embeddings as a temporal sequence so Phase 2 can consume them directly.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The default model backend uses Hugging Face Transformers with `facebook/vjepa2-vitl-fpc64-256`. The first extraction downloads the model weights into your Hugging Face cache.

## Extract Embeddings

```bash
python scripts/extract_vjepa_embeddings.py --video data/sample_videos/example.mp4 --output outputs/embeddings/example.pt
```

Useful options:

```bash
python scripts/extract_vjepa_embeddings.py --video path/to/video.mp4 --config configs/vjepa.yaml --max-clips 4 --clip-stride-seconds 2.0
```

## Run Web Demo

```bash
uvicorn api.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

The page lets you upload a video or select one from `data/sample_videos`, start extraction, poll status, and inspect metadata plus saved output paths.

## API

```text
POST /extract
GET /status/{job_id}
GET /metadata/{job_id}
GET /videos
```

`POST /extract` accepts multipart form data with either:

- `file`: uploaded video file
- `video_path`: local video path relative to the repository root or absolute

## Configuration

The default config lives at `configs/vjepa.yaml`.

Key settings:

- `model.backend`: `transformers` or `torchhub`
- `model.name`: Hugging Face model id
- `model.torchhub_name`: PyTorch Hub entrypoint when `model.backend` is `torchhub`
- `video.clip_frames`: frames per sampled clip
- `video.frame_stride`: spacing between frames inside each clip
- `video.clip_stride_seconds`: spacing between sampled clips
- `video.max_clips`: maximum sampled clips per video
- `storage.format`: `pt` or `npy`

## Saved Artifact

For `.pt` output, the saved file contains:

```text
embeddings: Tensor shaped [num_clips, ...]
clip_times: Tensor shaped [num_clips, 2]
metadata: JSON-compatible metadata dictionary
```

The temporal axis is axis `0`, one entry per sampled clip. This is the intended handoff to the future latent dynamics phase.

Metadata JSON is also saved under `outputs/metadata`.

## References

- Meta V-JEPA 2 repository: https://github.com/facebookresearch/vjepa2
- Hugging Face V-JEPA 2 model: https://huggingface.co/facebook/vjepa2-vitl-fpc64-256
