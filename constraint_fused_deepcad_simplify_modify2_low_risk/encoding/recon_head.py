from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinePairReconScorer(nn.Module):
    def __init__(self, d_model: int = 256, dim_z: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.line_proj = nn.Linear(d_model, hidden_dim)
        self.global_proj = nn.Linear(dim_z, hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, line_feats: torch.Tensor, z_global: torch.Tensor) -> torch.Tensor:
        left = self.line_proj(line_feats).unsqueeze(2)
        right = self.line_proj(line_feats).unsqueeze(1)
        z_proj = self.global_proj(z_global).unsqueeze(1).unsqueeze(2)
        pair_in = torch.cat(
            [
                left.expand(-1, -1, line_feats.size(1), -1),
                right.expand(-1, line_feats.size(1), -1, -1),
                z_proj.expand(-1, line_feats.size(1), line_feats.size(1), -1),
            ],
            dim=-1,
        )
        logits = self.out(pair_in)
        return 0.5 * (logits + logits.transpose(1, 2))


class ConstraintReconHead(nn.Module):
    def __init__(
        self,
        dim_z: int = 256,
        max_lines: int = 64,
        d_model: int = 256,
        hidden_dim: int = 256,
        enable_line_pair_scorer: bool = True,
    ):
        super().__init__()
        self.max_lines = max_lines
        self.enable_line_pair_scorer = enable_line_pair_scorer
        self.unary_head = nn.Sequential(
            nn.Linear(dim_z, 256),
            nn.GELU(),
            nn.Linear(256, max_lines * 2),
        )
        self.pair_scorer = LinePairReconScorer(d_model=d_model, dim_z=dim_z, hidden_dim=hidden_dim)
        self.pair_head = nn.Sequential(
            nn.Linear(dim_z, 512),
            nn.GELU(),
            nn.Linear(512, max_lines * max_lines * 2),
        )

    def forward(self, z: torch.Tensor, line_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = z.size(0)
        unary_logits = self.unary_head(z).view(batch_size, self.max_lines, 2)
        if self.enable_line_pair_scorer:
            pair_logits = self.pair_scorer(line_features, z)
        else:
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
