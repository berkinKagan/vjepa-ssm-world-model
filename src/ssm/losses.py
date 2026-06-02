import torch
import torch.nn.functional as F


def forecasting_loss(prediction: torch.Tensor, target: torch.Tensor, mse_weight: float, cosine_weight: float) -> dict:
    mse = F.mse_loss(prediction, target)
    cosine = 1.0 - F.cosine_similarity(prediction.reshape(-1, prediction.shape[-1]), target.reshape(-1, target.shape[-1]), dim=-1).mean()
    total = mse_weight * mse + cosine_weight * cosine
    return {"loss": total, "mse_loss": mse, "cosine_loss": cosine}


def forecasting_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    errors = prediction - target
    mse = errors.pow(2).mean()
    mae = errors.abs().mean()
    cosine = F.cosine_similarity(prediction.reshape(-1, prediction.shape[-1]), target.reshape(-1, target.shape[-1]), dim=-1).mean()
    per_horizon_mse = errors.pow(2).mean(dim=(0, 2))
    per_horizon_cosine = torch.stack([F.cosine_similarity(prediction[:, index, :], target[:, index, :], dim=-1).mean() for index in range(prediction.shape[1])])
    return {"mse": mse, "mae": mae, "cosine_similarity": cosine, "per_horizon_mse": per_horizon_mse, "per_horizon_cosine_similarity": per_horizon_cosine}


def detach_metrics(metrics: dict) -> dict:
    result = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu().tolist() if value.ndim > 0 else float(value.detach().cpu())
        else:
            result[key] = value
    return result
