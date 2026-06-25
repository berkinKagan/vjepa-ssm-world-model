from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.run_manager import resolve_device, set_seed
from utils.paths import resolve_project_path


class TextToLatentAligner(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


@dataclass
class AlignerState:
    status: str
    checkpoint: str | None
    input_dim: int | None
    output_dim: int | None


def ensure_text_latent_aligner(config: dict, caption_embeddings: np.ndarray, latent_path: str | None) -> dict:
    output_dir = resolve_project_path("outputs/aligner")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not latent_path:
        return AlignerState("missing_latents", None, None, None).__dict__
    latents = np.load(resolve_project_path(latent_path)).astype("float32")
    pairs = min(len(caption_embeddings), len(latents))
    if pairs < int(config["min_alignment_pairs"]):
        return AlignerState("not_enough_pairs", None, int(caption_embeddings.shape[-1]), int(latents.shape[-1])).__dict__
    latest = output_dir / "text_to_vjepa_latest.pt"
    best = output_dir / "text_to_vjepa_best.pt"
    if reusable_checkpoint(best, config, caption_embeddings.shape[-1], latents.shape[-1]):
        return AlignerState("loaded", str(best), int(caption_embeddings.shape[-1]), int(latents.shape[-1])).__dict__
    result = train_text_latent_aligner(config, caption_embeddings[:pairs], latents[:pairs], output_dir)
    return AlignerState("trained", result["best_checkpoint"], int(caption_embeddings.shape[-1]), int(latents.shape[-1])).__dict__


def train_text_latent_aligner(config: dict, caption_embeddings: np.ndarray, latents: np.ndarray, output_dir: Path) -> dict:
    set_seed(int(config["seed"]))
    device = resolve_device(config["device"])
    input_dim = int(caption_embeddings.shape[-1])
    output_dim = int(latents.shape[-1])
    model = TextToLatentAligner(input_dim, output_dim, int(config["aligner_hidden_dim"]), float(config["aligner_dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["aligner_learning_rate"]))
    text = torch.from_numpy(caption_embeddings).float()
    target = torch.from_numpy(latents).float()
    target = F.normalize(target, dim=-1)
    dataset = TensorDataset(text, target)
    loader = DataLoader(dataset, batch_size=min(64, len(dataset)), shuffle=True)
    metrics = []
    best_loss = float("inf")
    best_path = output_dir / "text_to_vjepa_best.pt"
    latest_path = output_dir / "text_to_vjepa_latest.pt"
    for epoch in range(1, int(config["aligner_epochs"]) + 1):
        loss_value = train_epoch(model, loader, optimizer, device, config)
        metrics.append({"epoch": epoch, "loss": loss_value})
        checkpoint = {"model_state_dict": model.state_dict(), "input_dim": input_dim, "output_dim": output_dim, "hidden_dim": int(config["aligner_hidden_dim"]), "dropout": float(config["aligner_dropout"]), "config": config}
        torch.save(checkpoint, latest_path)
        if loss_value < best_loss:
            best_loss = loss_value
            torch.save(checkpoint, best_path)
    (output_dir / "metrics.json").write_text(json_dumps(metrics), encoding="utf-8")
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return {"best_checkpoint": str(best_path), "latest_checkpoint": str(latest_path), "metrics": metrics}


def train_epoch(model, loader, optimizer, device, config: dict) -> float:
    model.train()
    total = 0.0
    count = 0
    for text, target in loader:
        text = text.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = F.normalize(model(text), dim=-1)
        cosine_loss = 1.0 - F.cosine_similarity(prediction, target, dim=-1).mean()
        mse_loss = F.mse_loss(prediction, target)
        loss = float(config["align_cosine_weight"]) * cosine_loss + float(config["align_mse_weight"]) * mse_loss
        loss.backward()
        optimizer.step()
        batch = text.shape[0]
        total += float(loss.detach().cpu()) * batch
        count += batch
    return total / max(1, count)


def load_aligner(checkpoint_path: str | Path, device_value: str) -> tuple[TextToLatentAligner, dict]:
    checkpoint = torch.load(resolve_project_path(checkpoint_path), map_location="cpu")
    model = TextToLatentAligner(int(checkpoint["input_dim"]), int(checkpoint["output_dim"]), int(checkpoint["hidden_dim"]), float(checkpoint["dropout"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolve_device(device_value))
    model.eval()
    return model, checkpoint


def reusable_checkpoint(path: Path, config: dict, input_dim: int, output_dim: int) -> bool:
    if not path.exists():
        return False
    try:
        checkpoint = torch.load(path, map_location="cpu")
        return int(checkpoint["input_dim"]) == int(input_dim) and int(checkpoint["output_dim"]) == int(output_dim) and int(checkpoint["hidden_dim"]) == int(config["aligner_hidden_dim"])
    except Exception:
        return False


def json_dumps(value) -> str:
    import json

    return json.dumps(value, indent=2)
