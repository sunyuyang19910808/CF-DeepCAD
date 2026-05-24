from __future__ import annotations

import torch
import torch.nn as nn


class ConstraintEvaluator(nn.Module):
    def horizontal_residual(self, unit: torch.Tensor) -> torch.Tensor:
        return unit[..., 1].pow(2)

    def vertical_residual(self, unit: torch.Tensor) -> torch.Tensor:
        return unit[..., 0].pow(2)

    def forward(self, soft_lines, unary_gt: torch.Tensor, line_mask: torch.Tensor | None = None) -> torch.Tensor:
        unit = soft_lines["unit"]
        valid = soft_lines["valid"]
        if line_mask is not None:
            valid = valid * line_mask.float()
        r_h = self.horizontal_residual(unit) * unary_gt[..., 0] * valid
        r_v = self.vertical_residual(unit) * unary_gt[..., 1] * valid
        denom = (unary_gt.sum(dim=-1) * valid).sum().clamp_min(1.0)
        return (r_h.sum() + r_v.sum()) / denom
