from __future__ import annotations

import torch
import torch.nn as nn


class DifferentiableConstraintEvaluator(nn.Module):
    def horizontal_residual(self, unit: torch.Tensor) -> torch.Tensor:
        return unit[..., 1].pow(2)

    def vertical_residual(self, unit: torch.Tensor) -> torch.Tensor:
        return unit[..., 0].pow(2)

    def parallel_residual(self, unit: torch.Tensor, pair_unit: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.abs((unit * pair_unit).sum(dim=-1))

    def perpendicular_residual(self, unit: torch.Tensor, pair_unit: torch.Tensor) -> torch.Tensor:
        return torch.abs((unit * pair_unit).sum(dim=-1))

    def forward(
        self,
        soft_lines: dict,
        unary_gt: torch.Tensor,
        pair_gt: torch.Tensor,
        line_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        unit = soft_lines["unit"]
        valid = soft_lines["valid"]
        if line_mask is not None:
            valid = valid * line_mask.float()

        r_h = self.horizontal_residual(unit) * unary_gt[..., 0] * valid
        r_v = self.vertical_residual(unit) * unary_gt[..., 1] * valid

        unit_i = unit.unsqueeze(2)
        unit_j = unit.unsqueeze(1)
        pair_valid = valid.unsqueeze(2) * valid.unsqueeze(1)
        r_parallel = self.parallel_residual(unit_i, unit_j) * pair_gt[..., 0] * pair_valid
        r_perpendicular = self.perpendicular_residual(unit_i, unit_j) * pair_gt[..., 1] * pair_valid

        pair_denom = (pair_gt[..., 0] * pair_valid).sum() + (pair_gt[..., 1] * pair_valid).sum()
        total_denom = (unary_gt[..., 0] * valid).sum() + (unary_gt[..., 1] * valid).sum() + pair_denom
        total = (r_h.sum() + r_v.sum() + r_parallel.sum() + r_perpendicular.sum()) / total_denom.clamp_min(1.0)
        metrics = {
            "geom_horizontal": r_h.sum() / (unary_gt[..., 0] * valid).sum().clamp_min(1.0),
            "geom_vertical": r_v.sum() / (unary_gt[..., 1] * valid).sum().clamp_min(1.0),
            "geom_parallel": r_parallel.sum() / (pair_gt[..., 0] * pair_valid).sum().clamp_min(1.0),
            "geom_perpendicular": r_perpendicular.sum() / (pair_gt[..., 1] * pair_valid).sum().clamp_min(1.0),
        }
        return total, metrics
