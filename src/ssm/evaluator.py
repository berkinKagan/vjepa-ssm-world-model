from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.run_manager import load_ssm_config, resolve_device, resolve_ssm_path, save_json
from ssm.dataset import build_train_val_datasets, normalizer_from_checkpoint
from ssm.losses import detach_metrics, forecasting_metrics
from ssm.model import build_forecaster


def evaluate_forecaster(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> dict:
    model.eval()
    totals = None
    count = 0
    with torch.inference_mode():
        for context, target in dataloader:
            context = context.to(device)
            target = target.to(device)
            prediction = model(context)
            metrics = forecasting_metrics(prediction, target)
            batch_count = context.shape[0]
            totals = accumulate_metrics(totals, metrics, batch_count)
            count += batch_count
    return finalize_metrics(totals, count)


def evaluate_ssm_checkpoint(checkpoint_path: str | Path, config: str | Path | dict) -> dict:
    loaded_config = load_ssm_config(config) if not isinstance(config, dict) else config
    checkpoint = torch.load(resolve_ssm_path(checkpoint_path), map_location="cpu")
    runtime_config = {**loaded_config, "model_type": checkpoint.get("actual_model_type", loaded_config.get("model_type", "gru"))}
    normalizer = normalizer_from_checkpoint(checkpoint)
    train_dataset, val_dataset, _ = build_train_val_datasets(runtime_config, normalizer)
    dataset = val_dataset if len(val_dataset) else train_dataset
    dataloader = DataLoader(dataset, batch_size=int(runtime_config["batch_size"]), shuffle=False)
    latent_dim = int(checkpoint.get("latent_dim", dataset.latent_dim))
    device = resolve_device(runtime_config["device"])
    model = build_forecaster(runtime_config, latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate_forecaster(model, dataloader, device)
    metrics_path = resolve_ssm_path(runtime_config["checkpoints_dir"]) / "evaluation_metrics.json"
    save_json(metrics, metrics_path)
    return metrics


def accumulate_metrics(totals: dict | None, metrics: dict, batch_count: int) -> dict:
    if totals is None:
        totals = {key: value.detach().cpu() * batch_count for key, value in metrics.items()}
    else:
        for key, value in metrics.items():
            totals[key] += value.detach().cpu() * batch_count
    return totals


def finalize_metrics(totals: dict | None, count: int) -> dict:
    if totals is None or count == 0:
        return {}
    return detach_metrics({key: value / count for key, value in totals.items()})
