from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstraintReconHead(nn.Module):
    def __init__(
        self,
        max_lines: int = 64,
        d_model: int = 256,
        hidden_dim: int = 256,
        **_unused,
    ):
        super().__init__()
        self.max_lines = max_lines
        self.unary_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.line_proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.pair_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, decoder_line_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        line = self.line_proj(decoder_line_features)
        left = line.unsqueeze(2)
        right = line.unsqueeze(1)
        n_lines = line.size(1)
        pair_in = torch.cat(
            [
                left.expand(-1, -1, n_lines, -1),
                right.expand(-1, n_lines, -1, -1),
                torch.abs(left - right).expand(-1, -1, n_lines, -1),
                (left * right).expand(-1, -1, n_lines, -1),
            ],
            dim=-1,
        )
        unary_logits = self.unary_head(decoder_line_features)
        pair_logits = self.pair_head(pair_in)
        pair_logits = 0.5 * (pair_logits + pair_logits.transpose(1, 2))
        return unary_logits, pair_logits


def weighted_bce_logits(logits: torch.Tensor, target: torch.Tensor, pos_weight: float = 5.0) -> torch.Tensor:
    pos_weight_tensor = torch.full(
        (target.size(-1),),
        float(pos_weight),
        device=target.device,
        dtype=target.dtype,
    )
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight_tensor)
