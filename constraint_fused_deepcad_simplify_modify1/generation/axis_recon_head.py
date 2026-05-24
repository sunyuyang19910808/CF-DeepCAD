from __future__ import annotations

import torch
import torch.nn as nn


class AxisReconHead(nn.Module):
    def __init__(self, d_model: int, max_lines: int):
        super().__init__()
        self.max_lines = max_lines
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, max_lines * 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.out(z.squeeze(0))
        return logits.view(z.shape[1], self.max_lines, 2)
