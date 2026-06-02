from pathlib import Path
from time import perf_counter

import torch

from experiments.run_manager import resolve_device
from ssm.dataset import build_window_samples, load_embedding_sequence
from ssm.losses import forecasting_loss
from ssm.model import build_forecaster


def run_latent_benchmark(embedding_path: str | Path, config: dict, steps: int = 5) -> dict:
    device = resolve_device(config["device"])
    load_start = perf_counter()
    sequence = load_embedding_sequence(embedding_path, config["metadata_dir"], config.get("pooling_mode", "mean"))
    load_seconds = perf_counter() - load_start
    context_len = int(config["context_len"])
    pred_horizon = int(config["pred_horizon"])
    samples = build_window_samples([sequence], context_len, pred_horizon)
    if not samples:
        raise ValueError("Embedding sequence is too short for benchmark context_len and pred_horizon")
    latent_dim = int(sequence.values.shape[-1])
    build_start = perf_counter()
    model = build_forecaster(config, latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    build_seconds = perf_counter() - build_start
    context, target = make_benchmark_batch(sequence.values, samples, context_len, pred_horizon, int(config["batch_size"]), device)
    train_seconds = run_training_steps(model, optimizer, context, target, config, max(1, steps))
    inference_seconds = run_inference_steps(model, context, max(1, steps))
    artifact_size_mb = Path(sequence.embedding_path).stat().st_size / 1024 / 1024
    return {"embedding": str(sequence.embedding_path), "artifact_size_mb": artifact_size_mb, "sequence_shape": list(sequence.values.shape), "latent_dim": latent_dim, "num_windows": len(samples), "device": str(device), "requested_model_type": config.get("model_type", "gru"), "actual_model_type": model.model_type, "load_seconds": load_seconds, "model_build_seconds": build_seconds, "train_step_seconds_avg": train_seconds, "inference_seconds_avg": inference_seconds, "batch_size": min(int(config["batch_size"]), len(samples)), "context_len": context_len, "pred_horizon": pred_horizon}


def make_benchmark_batch(values: torch.Tensor, samples: list, context_len: int, pred_horizon: int, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    selected = samples[:max(1, min(batch_size, len(samples)))]
    contexts = []
    targets = []
    for sample in selected:
        context_end = sample.start + context_len
        target_end = context_end + pred_horizon
        contexts.append(values[sample.start:context_end])
        targets.append(values[context_end:target_end])
    return torch.stack(contexts).to(device), torch.stack(targets).to(device)


def run_training_steps(model, optimizer, context: torch.Tensor, target: torch.Tensor, config: dict, steps: int) -> float:
    model.train()
    start = perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(context)
        losses = forecasting_loss(prediction, target, float(config["mse_weight"]), float(config["cosine_weight"]))
        losses["loss"].backward()
        optimizer.step()
    if context.device.type == "cuda":
        torch.cuda.synchronize()
    return (perf_counter() - start) / steps


def run_inference_steps(model, context: torch.Tensor, steps: int) -> float:
    model.eval()
    start = perf_counter()
    with torch.inference_mode():
        for _ in range(steps):
            model(context)
    if context.device.type == "cuda":
        torch.cuda.synchronize()
    return (perf_counter() - start) / steps
