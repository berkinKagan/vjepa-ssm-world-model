# vjepa-latent-dynamics

Research prototype for V-JEPA latent video dynamics.

Phase 1 extracts frozen V-JEPA video embeddings from local videos. Phase 2 trains a latent sequence model over those saved embeddings to forecast future V-JEPA latent states.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The default model backend uses Hugging Face Transformers with `facebook/vjepa2-vitl-fpc64-256`. The first extraction downloads the model weights into your Hugging Face cache.

Mamba support is optional. The repository falls back to a GRU forecaster when `mamba-ssm` is not installed or not supported on your system.

```bash
pip install -e ".[mamba]"
```

If Mamba installation fails, keep using the default install and set `model_type: gru` in `configs/ssm.yaml`.

## Phase 1: Extract Embeddings

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

The UI has two tabs:

- `Pipeline`: choose existing latents, upload a video, or select a sample video, then run extraction, SSM training, and future-latent prediction from one button.
- `Benchmark`: choose an embedding file and run a small latent benchmark that reports loading time, model build time, train-step time, inference time, model type, device, and sequence shape.

If an existing embedding is selected in the Pipeline tab, V-JEPA extraction is skipped.

## Phase 2: Train Latent Dynamics

Phase 1 saves V-JEPA latents as a sequence over sampled clips:

```text
z_1, z_2, ..., z_T
```

Phase 2 consumes only those saved embedding files. It never reads raw video. If an embedding is shaped `[num_clips, num_tokens, dim]`, the dataset can pool tokens into `[num_clips, dim]` with `mean`, `cls`, or experimental `flatten` pooling.

The forecaster learns:

```text
z_t ... z_{t+context_len-1} -> z_{t+context_len} ... z_{t+context_len+pred_horizon-1}
```

Train:

```bash
python scripts/train_ssm_forecaster.py --config configs/ssm.yaml
```

Evaluate:

```bash
python scripts/evaluate_ssm_forecaster.py --checkpoint outputs/checkpoints/best.pt --config configs/ssm.yaml
```

Predict future latents:

```bash
python scripts/predict_future_latents.py --embedding outputs/embeddings/example.pt --checkpoint outputs/checkpoints/best.pt --output outputs/predictions/example_future.pt
```

Training saves:

- `outputs/checkpoints/best.pt`
- `outputs/checkpoints/latest.pt`
- `outputs/checkpoints/config.yaml`
- `outputs/checkpoints/normalization.pt`
- `outputs/checkpoints/training_metrics.json`
- `outputs/checkpoints/validation_metrics.json`

Prediction saves a `.pt` file with predicted future latents, source context, checkpoint path, source embedding path, and prediction shape.

## API

```text
POST /extract
GET /status/{job_id}
GET /metadata/{job_id}
GET /videos
GET /embeddings
GET /checkpoints
POST /pipeline/run
GET /pipeline/status/{job_id}
GET /pipeline/results/{job_id}
POST /ssm/train
GET /ssm/status/{job_id}
POST /ssm/predict
GET /ssm/metrics/{job_id}
POST /benchmark/run
GET /benchmark/status/{job_id}
GET /benchmark/results/{job_id}
```

`POST /extract` accepts multipart form data with either:

- `file`: uploaded video file
- `video_path`: local video path relative to the repository root or absolute

## Configuration

The V-JEPA extraction config lives at `configs/vjepa.yaml`.

Key settings:

- `model.backend`: `transformers` or `torchhub`
- `model.name`: Hugging Face model id
- `model.torchhub_name`: PyTorch Hub entrypoint when `model.backend` is `torchhub`
- `video.clip_frames`: frames per sampled clip
- `video.frame_stride`: spacing between frames inside each clip
- `video.clip_stride_seconds`: spacing between sampled clips
- `video.max_clips`: maximum sampled clips per video
- `storage.format`: `pt` or `npy`

The SSM/Mamba config lives at `configs/ssm.yaml`.

Key settings:

- `embeddings_dir`: directory of saved Phase 1 embeddings
- `metadata_dir`: directory of saved Phase 1 metadata
- `checkpoints_dir`: checkpoint and training metric output directory
- `predictions_dir`: future latent prediction output directory
- `model_type`: `mamba` or `gru`
- `latent_dim`: inferred when `null`
- `hidden_dim`: temporal model hidden size
- `context_len`: number of past latent steps
- `pred_horizon`: number of future latent steps
- `pooling_mode`: `mean`, `cls`, or `flatten`
- `normalize`: dataset-level latent normalization
- `mse_weight` and `cosine_weight`: combined forecasting loss weights

## Saved Artifact

For `.pt` output, the saved file contains:

```text
embeddings: Tensor shaped [num_clips, ...]
clip_times: Tensor shaped [num_clips, 2]
metadata: JSON-compatible metadata dictionary
```

The temporal axis is axis `0`, one entry per sampled clip. This is the intended handoff to the future latent dynamics phase.

Metadata JSON is also saved under `outputs/metadata`.

Dangerous-event anticipation is planned for Phase 3. A future risk head can be added on top of the temporal model state, but Phase 2 stays focused on self-supervised future latent prediction.

## References

- Meta V-JEPA 2 repository: https://github.com/facebookresearch/vjepa2
- Hugging Face V-JEPA 2 model: https://huggingface.co/facebook/vjepa2-vitl-fpc64-256
