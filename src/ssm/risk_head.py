import torch
from torch import nn


class RiskHead(nn.Module):
    def __init__(self, hidden_dim: int, num_outputs: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, num_outputs))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)
