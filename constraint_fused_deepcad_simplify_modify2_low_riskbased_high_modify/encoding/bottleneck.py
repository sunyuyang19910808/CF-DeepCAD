from __future__ import annotations

import torch
import torch.nn as nn


class Bottleneck512(nn.Module):
    """Single-token latent bottleneck; the public decoder contract remains z only."""

    def __init__(self, pooled_dim: int = 512, dim_z: int = 512):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(pooled_dim, dim_z),
            nn.LayerNorm(dim_z),
            nn.Tanh(),
        )

    def forward(self, z_pre: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(z_pre)


class DeepCADBottleneck(nn.Module):
    """Aligns with model.autoencoder.Bottleneck: Linear(d_model, dim_z) + Tanh."""

    def __init__(self, d_model: int = 256, dim_z: int = 256):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(d_model, dim_z),
            nn.Tanh(),
        )

    def forward(self, z_pre: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(z_pre)
