from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstraintReconHead(nn.Module):
    def __init__(self, dim_z: int = 256, max_lines: int = 64):
        super().__init__()
        self.max_lines = max_lines
        self.unary_head = nn.Sequential(
            nn.Linear(dim_z, 256),
            nn.GELU(),
            nn.Linear(256, max_lines * 2),
        )
        self.pair_head = nn.Sequential(
            nn.Linear(dim_z, 512),
            nn.GELU(),
            nn.Linear(512, max_lines * max_lines * 2),
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = z.size(0)
        unary_logits = self.unary_head(z).view(batch_size, self.max_lines, 2)
        pair_logits = self.pair_head(z).view(batch_size, self.max_lines, self.max_lines, 2)
        return unary_logits, pair_logits


def weighted_bce_logits(logits: torch.Tensor, target: torch.Tensor, pos_weight: float = 5.0) -> torch.Tensor:
    pos_weight_tensor = torch.full(
        (target.size(-1),),
        float(pos_weight),
        device=target.device,
        dtype=target.dtype,
    )
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight_tensor)

