import torch
from torch import nn


class GRUEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=recurrent_dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(values)
        return output


class MambaEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        from mamba_ssm import Mamba

        self.layers = nn.ModuleList([Mamba(d_model=hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = values
        for layer, norm in zip(self.layers, self.norms):
            hidden = hidden + self.dropout(layer(norm(hidden)))
        return self.final_norm(hidden)


class LatentForecaster(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, num_layers: int, dropout: float, context_len: int, pred_horizon: int, model_type: str):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len
        self.pred_horizon = pred_horizon
        self.requested_model_type = model_type
        self.model_type = model_type
        self.input_projection = nn.Linear(latent_dim, hidden_dim)
        self.sequence_model = self.build_sequence_model(model_type, hidden_dim, num_layers, dropout)
        self.dropout = nn.Dropout(dropout)
        self.future_tokens = nn.Parameter(torch.randn(pred_horizon, hidden_dim) * 0.02)
        self.output_projection = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, latent_dim))

    def build_sequence_model(self, model_type: str, hidden_dim: int, num_layers: int, dropout: float) -> nn.Module:
        if model_type == "mamba":
            try:
                return MambaEncoder(hidden_dim, num_layers, dropout)
            except Exception:
                self.model_type = "gru"
                return GRUEncoder(hidden_dim, num_layers, dropout)
        if model_type == "gru":
            return GRUEncoder(hidden_dim, num_layers, dropout)
        raise ValueError(f"Unsupported model_type: {model_type}")

    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(context)
        encoded = self.sequence_model(projected)
        return encoded[:, -1, :]

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        state = self.encode_context(context)
        future = state.unsqueeze(1) + self.future_tokens.unsqueeze(0)
        future = self.dropout(future)
        return self.output_projection(future)


def build_forecaster(config: dict, latent_dim: int) -> LatentForecaster:
    return LatentForecaster(latent_dim=latent_dim, hidden_dim=int(config["hidden_dim"]), num_layers=int(config["num_layers"]), dropout=float(config["dropout"]), context_len=int(config["context_len"]), pred_horizon=int(config["pred_horizon"]), model_type=config.get("model_type", "gru"))
