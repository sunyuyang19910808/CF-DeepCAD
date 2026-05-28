from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositiveRelationConstraintEvaluator(nn.Module):
    """Positive-only geometry relation recovery loss.

    Only GT=1 positions contribute to loss. GT=0 is not treated as a hard negative.
    """

    def __init__(self, bce_scale: float = 4.0, negative_weight: float = 0.0):
        super().__init__()
        self.bce_scale = float(bce_scale)
        self.negative_weight = float(negative_weight)

    def _positive_bce(self, score: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
        pos_count = positive_mask.sum()
        if pos_count <= 0:
            return score.sum() * 0.0
        logit = (score - 0.5) * self.bce_scale
        target = torch.ones_like(score)
        per_elem = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
        per_elem = per_elem * positive_mask
        return per_elem.sum() / pos_count.clamp_min(1.0)

    def forward(
        self,
        soft_lines: dict,
        unary_gt: torch.Tensor,
        pair_gt: torch.Tensor,
        line_mask: torch.Tensor | None = None,
        compute_metrics: bool = True,
    ) -> tuple[dict, dict, dict]:
        unit = soft_lines["unit"]
        valid = soft_lines["valid"]
        if line_mask is not None:
            valid = valid * line_mask.float()

        pair_valid = valid.unsqueeze(2) * valid.unsqueeze(1)
        eye = torch.eye(valid.size(1), device=valid.device, dtype=valid.dtype).unsqueeze(0)
        pair_valid = pair_valid * (1.0 - eye)

        score_h = 1.0 - unit[..., 1].pow(2)
        score_v = 1.0 - unit[..., 0].pow(2)
        unit_i = unit.unsqueeze(2)
        unit_j = unit.unsqueeze(1)
        dot_abs = torch.abs((unit_i * unit_j).sum(dim=-1))
        score_par = dot_abs
        score_perp = 1.0 - dot_abs

        mask_h = unary_gt[..., 0] * valid
        mask_v = unary_gt[..., 1] * valid
        mask_par = pair_gt[..., 0] * pair_valid
        mask_perp = pair_gt[..., 1] * pair_valid

        loss_h = self._positive_bce(score_h, mask_h)
        loss_v = self._positive_bce(score_v, mask_v)
        loss_par = self._positive_bce(score_par, mask_par)
        loss_perp = self._positive_bce(score_perp, mask_perp)

        components = {
            "geom_h": loss_h,
            "geom_v": loss_v,
            "geom_parallel": loss_par,
            "geom_perpendicular": loss_perp,
            "loss_geom": loss_h + loss_v + loss_par + loss_perp,
        }

        counts = {
            "positive_count_h": mask_h.sum(),
            "positive_count_v": mask_v.sum(),
            "positive_count_parallel": mask_par.sum(),
            "positive_count_perpendicular": mask_perp.sum(),
        }

        metrics = {}
        if compute_metrics:
            with torch.no_grad():
                metrics = {
                    "geom_horizontal": self._masked_mean(score_h, mask_h),
                    "geom_vertical": self._masked_mean(score_v, mask_v),
                    "geom_parallel": self._masked_mean(score_par, mask_par),
                    "geom_perpendicular": self._masked_mean(score_perp, mask_perp),
                }

        return components, metrics, counts

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum().clamp_min(1.0)
        return (values * mask).sum() / denom
