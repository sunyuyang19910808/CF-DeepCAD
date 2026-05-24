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
            nn.Linear(512, max_lines * max_lines * 3),
        )

    def forward(self, z: torch.Tensor):
        n = z.size(0)
        unary = self.unary_head(z).view(n, self.max_lines, 2)
        pair = self.pair_head(z).view(n, self.max_lines, self.max_lines, 3)
        return torch.sigmoid(unary), torch.sigmoid(pair)


def weighted_bce(pred: torch.Tensor, target: torch.Tensor, pos_weight: float = 5.0) -> torch.Tensor:
    w = torch.where(target > 0.5, pos_weight, 1.0)
    return F.binary_cross_entropy(pred, target, weight=w)
