# vjepa-latent-dynamics

Research prototype for V-JEPA latent video dynamics.

Phase 1 extracts frozen V-JEPA video embeddings from local videos. Phase 2 trains a latent sequence model over those saved embeddings to forecast future V-JEPA latent states.
Phase 3 adds text-to-scene search over already processed videos using clip captions and text embeddings.

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
PYTHONPATH=src python -m uvicorn api.main:app --host 127.0.0.1 --port 8002
```

Then open:

```text
http://127.0.0.1:8002
```

Start Ollama before using the scene-search demo:

```bash
ollama serve
```

Confirm the local models:

```bash
ollama list
```

Expected local models:

- `llama3.2-vision:11b`
- `llama3.1:8b`

The simplified page lets you upload a video, type a natural-language scene query, click `Search Video`, and view the best matching scenes. The backend automatically extracts V-JEPA features, samples frames, captions clips with local Ollama, builds the scene index, searches captions, and returns matching timestamps.

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

## Phase 3: Text-To-Scene Search

Phase 3 does not compare text directly with V-JEPA embeddings. V-JEPA latents are not text-aligned by default. Instead, the scene-search pipeline builds a caption index:

```text
uploaded video
-> create dense full-video segments
-> sample representative frames from each segment
-> caption frames with Ollama vision model or placeholder backend
-> embed captions with a text embedding model
-> search captions with a text query
-> return matching clips and timestamps
```

Ollama is used as the local LLM/VLM backend:

```text
http://127.0.0.1:11434
```

List available Ollama models:

```bash
python scripts/list_ollama_models.py
```

Build a scene index:

```bash
python scripts/build_scene_index.py --config configs/scene_search.yaml --embedding outputs/embeddings/example.pt
```

Search indexed scenes:

```bash
python scripts/search_scene.py --config configs/scene_search.yaml --query "find the scene where someone enters the room" --top-k 5
```

Default scene segmentation:

- `segment_source`: `dense_video`
- `segment_length_seconds`: `8.0`
- `segment_stride_seconds`: `8.0`
- `min_segment_seconds`: `2.0`
- `frames_per_segment`: `1`

For a 1731-second video, this creates about 217 searchable segments and indexes the full video instead of only the V-JEPA sampled clip times.

Scene index outputs:

- `outputs/scene_index/*.jsonl`
- `outputs/scene_index/*.npy`
- `outputs/frame_cache/**/*.jpg`

Each scene row contains video path, embedding path, metadata path, segment index, start/end time, caption, sampled frame paths, caption backend, Ollama model, creation time, and segment source.

Repeated uploads of the same video reuse the same content-hash upload path. Existing V-JEPA artifacts and valid dense scene indexes are reused when their settings match.

If no vision-capable Ollama model is available, indexing falls back to placeholder captions based on metadata so search infrastructure can still run.

## API

```text
POST /extract
GET /status/{job_id}
GET /metadata/{job_id}
GET /videos
GET /embeddings
GET /checkpoints
GET /ollama/status
GET /ollama/models
POST /demo/search-video
GET /demo/status/{job_id}
GET /demo/results/{job_id}
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
POST /scene/index
GET /scene/index/status/{job_id}
POST /scene/search
GET /scene/search/results/{job_id}
GET /scene/indexes
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

The scene-search config lives at `configs/scene_search.yaml`.

Key settings:

- `scene_index_dir`: scene JSONL and caption embedding output directory
- `frame_cache_dir`: sampled frame output directory
- `caption_backend`: `ollama_vision` or `placeholder`
- `ollama_base_url`: local Ollama server URL
- `ollama_model`: default local vision model, `llama3.2-vision:11b`
- `ollama_text_model`: default local text reranking model, `llama3.1:8b`
- `text_embedding_model`: sentence-transformers model for captions
- `segment_source`: `dense_video` by default, or `vjepa_clip_times` for legacy indexing
- `segment_length_seconds`: dense segment duration
- `segment_stride_seconds`: dense segment stride
- `min_segment_seconds`: minimum final partial segment duration
- `frames_per_segment`: representative frames per dense segment
- `top_k`: default search result count
- `enable_llm_reranking`: optional Ollama reranking after vector retrieval

## Saved Artifact

For `.pt` output, the saved file contains:

```text
embeddings: Tensor shaped [num_clips, ...]
clip_times: Tensor shaped [num_clips, 2]
metadata: JSON-compatible metadata dictionary
```

The temporal axis is axis `0`, one entry per sampled clip. This is the intended handoff to the future latent dynamics phase.

Metadata JSON is also saved under `outputs/metadata`.

Dangerous-event anticipation is planned for a later phase. A future risk head can be added on top of the temporal model state, but Phase 2 stays focused on self-supervised future latent prediction and Phase 3 stays focused on text-to-scene retrieval.

## References

- Meta V-JEPA 2 repository: https://github.com/facebookresearch/vjepa2
- Hugging Face V-JEPA 2 model: https://huggingface.co/facebook/vjepa2-vitl-fpc64-256
