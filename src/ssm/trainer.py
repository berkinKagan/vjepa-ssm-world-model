from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from experiments.run_manager import load_ssm_config, resolve_device, resolve_ssm_path, save_json, save_yaml, set_seed
from ssm.dataset import NormalizationStats, build_train_val_datasets, compute_normalization_stats
from ssm.evaluator import evaluate_forecaster
from ssm.losses import forecasting_loss
from ssm.model import LatentForecaster, build_forecaster

TrainingProgress = Callable[[int, int, dict], None]


@dataclass
class TrainingResult:
    best_checkpoint: Path
    latest_checkpoint: Path
    training_metrics: list[dict]
    validation_metrics: list[dict]
    latent_dim: int
    model_type: str


def train_ssm_forecaster(config: str | Path | dict, progress_callback: TrainingProgress | None = None) -> TrainingResult:
    loaded_config = load_ssm_config(config) if not isinstance(config, dict) else config
    set_seed(int(loaded_config["seed"]))
    checkpoints_dir = resolve_ssm_path(loaded_config["checkpoints_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    train_dataset, val_dataset, _ = build_train_val_datasets(loaded_config)
    normalizer = compute_normalization_stats(train_dataset) if bool(loaded_config["normalize"]) else None
    train_dataset.normalizer = normalizer
    val_dataset.normalizer = normalizer
    latent_dim = infer_latent_dim(loaded_config, train_dataset.latent_dim)
    device = resolve_device(loaded_config["device"])
    model = build_forecaster(loaded_config, latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(loaded_config["learning_rate"]), weight_decay=float(loaded_config["weight_decay"]))
    train_loader = DataLoader(train_dataset, batch_size=int(loaded_config["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(loaded_config["batch_size"]), shuffle=False) if len(val_dataset) else None
    best_checkpoint = checkpoints_dir / "best.pt"
    latest_checkpoint = checkpoints_dir / "latest.pt"
    save_yaml(loaded_config, checkpoints_dir / "config.yaml")
    save_normalization_stats(normalizer, checkpoints_dir / "normalization.pt")
    training_history = []
    validation_history = []
    best_score = float("inf")
    for epoch in range(1, int(loaded_config["epochs"]) + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, loaded_config)
        val_metrics = evaluate_forecaster(model, val_loader, device) if val_loader is not None else train_metrics
        train_record = {"epoch": epoch, **train_metrics}
        val_record = {"epoch": epoch, **val_metrics}
        training_history.append(train_record)
        validation_history.append(val_record)
        save_checkpoint(latest_checkpoint, model, optimizer, loaded_config, normalizer, epoch, latent_dim, train_record, val_record)
        score = float(val_record.get("mse", val_record.get("loss", 0.0)))
        if score < best_score:
            best_score = score
            save_checkpoint(best_checkpoint, model, optimizer, loaded_config, normalizer, epoch, latent_dim, train_record, val_record)
        save_json(training_history, checkpoints_dir / "training_metrics.json")
        save_json(validation_history, checkpoints_dir / "validation_metrics.json")
        if progress_callback is not None:
            progress_callback(epoch, int(loaded_config["epochs"]), val_record)
    return TrainingResult(best_checkpoint=best_checkpoint, latest_checkpoint=latest_checkpoint, training_metrics=training_history, validation_metrics=validation_history, latent_dim=latent_dim, model_type=model.model_type)


def train_one_epoch(model: LatentForecaster, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, config: dict) -> dict:
    model.train()
    totals = {"loss": 0.0, "mse_loss": 0.0, "cosine_loss": 0.0}
    count = 0
    for context, target in dataloader:
        context = context.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(context)
        losses = forecasting_loss(prediction, target, float(config["mse_weight"]), float(config["cosine_weight"]))
        losses["loss"].backward()
        optimizer.step()
        batch_count = context.shape[0]
        count += batch_count
        for key in totals:
            totals[key] += float(losses[key].detach().cpu()) * batch_count
    return {key: value / max(1, count) for key, value in totals.items()}


def infer_latent_dim(config: dict, dataset_latent_dim: int) -> int:
    configured = config.get("latent_dim")
    if configured is None:
        return int(dataset_latent_dim)
    configured_dim = int(configured)
    if configured_dim != int(dataset_latent_dim):
        raise ValueError(f"Configured latent_dim {configured_dim} does not match dataset latent_dim {dataset_latent_dim}")
    return configured_dim


def save_checkpoint(path: Path, model: LatentForecaster, optimizer: torch.optim.Optimizer, config: dict, normalizer: NormalizationStats | None, epoch: int, latent_dim: int, train_metrics: dict, val_metrics: dict) -> None:
    payload = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "config": config, "normalization": normalizer.to_dict() if normalizer is not None else {"enabled": False}, "epoch": epoch, "latent_dim": latent_dim, "requested_model_type": model.requested_model_type, "actual_model_type": model.model_type, "train_metrics": train_metrics, "validation_metrics": val_metrics}
    torch.save(payload, path)


def save_normalization_stats(normalizer: NormalizationStats | None, path: Path) -> None:
    payload = normalizer.to_dict() if normalizer is not None else {"enabled": False}
    torch.save(payload, path)
