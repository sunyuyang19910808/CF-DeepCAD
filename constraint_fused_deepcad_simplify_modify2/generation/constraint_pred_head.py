from __future__ import annotations

import torch
import torch.nn as nn


class ConstraintPredHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int = 4):
        super().__init__()
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)

