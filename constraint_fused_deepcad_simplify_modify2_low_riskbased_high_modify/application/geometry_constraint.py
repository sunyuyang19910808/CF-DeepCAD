from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableConstraintEvaluator(nn.Module):
    """Compute four geometry constraint losses (horizontal / vertical / parallel / perpendicular).

    Soft path (legacy): each component is a unary residual masked by the GT label, normalised by
    the combined valid count so that summing the four components reproduces the legacy single
    ``geom_loss`` value (used for backward compatibility with H1/A2/A2b/A2c).

    Hard-BCE path (A2d): each component is a binary cross entropy against the GT label, applied to
    both positive and negative samples so that softly satisfying every pair (``score -> 1``) is no
    longer a viable shortcut.
    """

    def __init__(
        self,
        use_hard_geom_bce: bool = False,
        bce_scale: float = 6.0,
        pos_weight: float = 5.0,
    ) -> None:
        super().__init__()
        self.use_hard_geom_bce = bool(use_hard_geom_bce)
        self.bce_scale = float(bce_scale)
        self.pos_weight = float(pos_weight)

    def horizontal_residual(self, unit: torch.Tensor) -> torch.Tensor:
        return unit[..., 1].pow(2)

    def vertical_residual(self, unit: torch.Tensor) -> torch.Tensor:
        return unit[..., 0].pow(2)

    def parallel_residual(self, unit: torch.Tensor, pair_unit: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.abs((unit * pair_unit).sum(dim=-1))

    def perpendicular_residual(self, unit: torch.Tensor, pair_unit: torch.Tensor) -> torch.Tensor:
        return torch.abs((unit * pair_unit).sum(dim=-1))

    def _bce_with_mask(
        self,
        score: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        logit = (score - 0.5) * self.bce_scale
        pos_weight = torch.tensor(self.pos_weight, device=score.device, dtype=score.dtype)
        per_elem = F.binary_cross_entropy_with_logits(
            logit, target, pos_weight=pos_weight, reduction="none"
        )
        per_elem = per_elem * mask
        return per_elem.sum() / mask.sum().clamp_min(1.0)

    def forward(
        self,
        soft_lines: dict,
        unary_gt: torch.Tensor,
        pair_gt: torch.Tensor,
        line_mask: torch.Tensor | None = None,
    ) -> tuple[dict, dict]:
        unit = soft_lines["unit"]
        valid = soft_lines["valid"]
        if line_mask is not None:
            valid = valid * line_mask.float()

        pair_valid = valid.unsqueeze(2) * valid.unsqueeze(1)

        if self.use_hard_geom_bce:
            score_h = 1.0 - unit[..., 1].pow(2)
            score_v = 1.0 - unit[..., 0].pow(2)
            unit_i = unit.unsqueeze(2)
            unit_j = unit.unsqueeze(1)
            dot_abs = torch.abs((unit_i * unit_j).sum(dim=-1))
            score_par = dot_abs
            score_perp = 1.0 - dot_abs

            l_h = self._bce_with_mask(score_h, unary_gt[..., 0], valid)
            l_v = self._bce_with_mask(score_v, unary_gt[..., 1], valid)
            l_par = self._bce_with_mask(score_par, pair_gt[..., 0], pair_valid)
            l_perp = self._bce_with_mask(score_perp, pair_gt[..., 1], pair_valid)
        else:
            r_h = self.horizontal_residual(unit) * unary_gt[..., 0] * valid
            r_v = self.vertical_residual(unit) * unary_gt[..., 1] * valid
            unit_i = unit.unsqueeze(2)
            unit_j = unit.unsqueeze(1)
            r_par = self.parallel_residual(unit_i, unit_j) * pair_gt[..., 0] * pair_valid
            r_perp = self.perpendicular_residual(unit_i, unit_j) * pair_gt[..., 1] * pair_valid

            total_denom = (
                (unary_gt[..., 0] * valid).sum()
                + (unary_gt[..., 1] * valid).sum()
                + (pair_gt[..., 0] * pair_valid).sum()
                + (pair_gt[..., 1] * pair_valid).sum()
            ).clamp_min(1.0)
            l_h = r_h.sum() / total_denom
            l_v = r_v.sum() / total_denom
            l_par = r_par.sum() / total_denom
            l_perp = r_perp.sum() / total_denom

        components = {
            "geom_h": l_h,
            "geom_v": l_v,
            "geom_para": l_par,
            "geom_perp": l_perp,
        }

        with torch.no_grad():
            unit_i_m = unit.unsqueeze(2)
            unit_j_m = unit.unsqueeze(1)
            denom_h = (unary_gt[..., 0] * valid).sum().clamp_min(1.0)
            denom_v = (unary_gt[..., 1] * valid).sum().clamp_min(1.0)
            denom_par = (pair_gt[..., 0] * pair_valid).sum().clamp_min(1.0)
            denom_perp = (pair_gt[..., 1] * pair_valid).sum().clamp_min(1.0)
            metrics = {
                "geom_horizontal": (
                    self.horizontal_residual(unit) * unary_gt[..., 0] * valid
                ).sum() / denom_h,
                "geom_vertical": (
                    self.vertical_residual(unit) * unary_gt[..., 1] * valid
                ).sum() / denom_v,
                "geom_parallel": (
                    self.parallel_residual(unit_i_m, unit_j_m) * pair_gt[..., 0] * pair_valid
                ).sum() / denom_par,
                "geom_perpendicular": (
                    self.perpendicular_residual(unit_i_m, unit_j_m) * pair_gt[..., 1] * pair_valid
                ).sum() / denom_perp,
            }

        return components, metrics
