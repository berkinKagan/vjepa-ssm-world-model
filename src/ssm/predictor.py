from pathlib import Path

import torch

from experiments.run_manager import load_ssm_config, resolve_device, resolve_ssm_path
from ssm.dataset import denormalize_values, load_embedding_sequence, normalizer_from_checkpoint, normalize_values
from ssm.model import build_forecaster


def predict_future_latents(embedding_path: str | Path, checkpoint_path: str | Path, output_path: str | Path | None = None, config: str | Path | dict | None = None) -> dict:
    checkpoint = torch.load(resolve_ssm_path(checkpoint_path), map_location="cpu")
    loaded_config = resolve_prediction_config(config, checkpoint)
    runtime_config = {**loaded_config, "model_type": checkpoint.get("actual_model_type", loaded_config.get("model_type", "gru"))}
    normalizer = normalizer_from_checkpoint(checkpoint)
    sequence = load_embedding_sequence(embedding_path, runtime_config["metadata_dir"], runtime_config.get("pooling_mode", "mean"))
    context_len = int(runtime_config["context_len"])
    if sequence.values.shape[0] < context_len:
        raise ValueError(f"Embedding sequence has {sequence.values.shape[0]} steps but context_len is {context_len}")
    latent_dim = int(checkpoint.get("latent_dim", sequence.values.shape[-1]))
    if sequence.values.shape[-1] != latent_dim:
        raise ValueError(f"Embedding latent_dim {sequence.values.shape[-1]} does not match checkpoint latent_dim {latent_dim}")
    device = resolve_device(runtime_config["device"])
    model = build_forecaster(runtime_config, latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    context = sequence.values[-context_len:]
    if normalizer is not None:
        context = normalize_values(context, normalizer)
    with torch.inference_mode():
        prediction = model(context.unsqueeze(0).to(device)).squeeze(0).cpu()
    prediction = denormalize_values(prediction, normalizer).cpu()
    destination = resolve_prediction_output(output_path, runtime_config, sequence.embedding_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"predictions": prediction, "context": sequence.values[-context_len:].cpu(), "source_embedding": str(sequence.embedding_path), "checkpoint": str(resolve_ssm_path(checkpoint_path)), "prediction_shape": list(prediction.shape), "config": runtime_config}
    torch.save(payload, destination)
    return {"prediction_file": str(destination), "prediction_shape": list(prediction.shape), "source_embedding": str(sequence.embedding_path), "checkpoint": str(resolve_ssm_path(checkpoint_path))}


def resolve_prediction_config(config: str | Path | dict | None, checkpoint: dict) -> dict:
    if config is None:
        return checkpoint["config"]
    return load_ssm_config(config) if not isinstance(config, dict) else config


def resolve_prediction_output(output_path: str | Path | None, config: dict, embedding_path: Path) -> Path:
    if output_path is not None:
        return resolve_ssm_path(output_path)
    return resolve_ssm_path(config["predictions_dir"]) / f"{embedding_path.stem}_future.pt"
