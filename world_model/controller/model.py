"""Linear controller C(z, h) -> a, tanh-bounded (Ha & Schmidhuber 2018)."""

from __future__ import annotations

import torch
import torch.nn as nn


class Controller(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim + hidden_dim, action_dim)

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc(torch.cat([z, h], dim=-1)))

    def flat_params(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self.parameters()])

    def set_flat_params(self, vec: torch.Tensor) -> None:
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(vec[offset : offset + n].view_as(p))
            offset += n
