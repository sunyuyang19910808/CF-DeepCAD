from __future__ import annotations

import torch
import torch.nn.functional as F


def build_line_mask(line_count: torch.Tensor, max_lines: int) -> torch.Tensor:
    if line_count.dim() == 0:
        line_count = line_count.unsqueeze(0)
    line_indices = torch.arange(max_lines, device=line_count.device).unsqueeze(0)
    return line_indices < line_count.unsqueeze(1)


def compose_loss(
    cmd_loss: torch.Tensor,
    unary_pred: torch.Tensor,
    unary_gt: torch.Tensor,
    line_mask: torch.Tensor,
    beta: float,
    pos_weight: float = 1.0,
):
    pos_weight_tensor = torch.full((2,), float(pos_weight), device=unary_pred.device, dtype=unary_pred.dtype)
    bce = F.binary_cross_entropy_with_logits(
        unary_pred,
        unary_gt,
        reduction="none",
        pos_weight=pos_weight_tensor,
    )
    masked = bce * line_mask.unsqueeze(-1).float()
    axis_loss = masked.sum() / line_mask.sum().clamp(min=1).float()
    total = cmd_loss + beta * axis_loss
    return total, axis_loss
