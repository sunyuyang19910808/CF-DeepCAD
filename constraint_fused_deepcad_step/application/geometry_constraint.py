from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_RAD2DEG = 180.0 / math.pi
_ATAN2_EPS = 1e-8


def _rad_to_deg(rad: torch.Tensor) -> torch.Tensor:
    return rad * _RAD2DEG


def angle_from_x_deg(ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """Undirected angle between unit direction and +x axis (matches test ``undirected_angle_deg(u, ex)``).

    Uses ``atan2`` instead of ``asin`` to avoid gradient singularities at |uy| -> 1.
    """
    return _rad_to_deg(torch.atan2(uy.abs(), ux.abs().clamp_min(_ATAN2_EPS)))


def angle_from_y_deg(ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """Undirected angle between unit direction and +y axis (matches test ``undirected_angle_deg(u, ey)``)."""
    return _rad_to_deg(torch.atan2(ux.abs(), uy.abs().clamp_min(_ATAN2_EPS)))


def undirected_angle_deg(u_i: torch.Tensor, u_j: torch.Tensor) -> torch.Tensor:
    """Undirected angle in degrees between two unit directions.

    Equivalent to ``min(acos(dot), 180 - acos(dot))`` for unit vectors but uses a stable
    ``atan2(|cross|, |dot|)`` formulation to avoid ``acos`` singularities at |dot| -> 1.
    """
    cross = (u_i[..., 0] * u_j[..., 1] - u_i[..., 1] * u_j[..., 0]).abs()
    dot = (u_i * u_j).sum(dim=-1).abs().clamp_min(_ATAN2_EPS)
    return _rad_to_deg(torch.atan2(cross, dot))


class PositiveRelationConstraintEvaluator(nn.Module):
    """Positive-only geometry relation recovery loss.

    Only GT=1 positions contribute to loss. GT=0 is not treated as a hard negative.

    ``geom_loss_mode``:
    - ``bce``: legacy soft-score positive BCE (saturates before 0.1° test threshold).
    - ``angle_hinge``: ``relu(angle_deg - angle_thresh)`` aligned with test hard recall.
    """

    VALID_LOSS_MODES = ("bce", "angle_hinge")

    def __init__(
        self,
        geom_loss_mode: str = "angle_hinge",
        angle_thresh: float = 0.1,
        bce_scale: float = 4.0,
        negative_weight: float = 0.0,
    ):
        super().__init__()
        if geom_loss_mode not in self.VALID_LOSS_MODES:
            raise ValueError(
                "geom_loss_mode must be one of {}, got {!r}".format(self.VALID_LOSS_MODES, geom_loss_mode)
            )
        self.geom_loss_mode = geom_loss_mode
        self.angle_thresh = float(angle_thresh)
        self.bce_scale = float(bce_scale)
        self.negative_weight = float(negative_weight)

    def _positive_bce(self, score: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
        pos_count = positive_mask.sum()
        if pos_count <= 0:
            return positive_mask.new_zeros(())
        logit = (score - 0.5) * self.bce_scale
        target = torch.ones_like(score)
        per_elem = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
        per_elem = per_elem * positive_mask
        return per_elem.sum() / pos_count.clamp_min(1.0)

    def _positive_hinge(self, angle_deg: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
        pos_count = positive_mask.sum()
        if pos_count <= 0:
            return positive_mask.new_zeros(())
        excess = F.relu(angle_deg - self.angle_thresh)
        return (excess * positive_mask).sum() / pos_count.clamp_min(1.0)

    def _compute_losses(
        self,
        score_h: torch.Tensor,
        score_v: torch.Tensor,
        score_par: torch.Tensor,
        score_perp: torch.Tensor,
        angle_h: torch.Tensor,
        angle_v: torch.Tensor,
        angle_ij: torch.Tensor,
        angle_perp_err: torch.Tensor,
        mask_h: torch.Tensor,
        mask_v: torch.Tensor,
        mask_par: torch.Tensor,
        mask_perp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.geom_loss_mode == "angle_hinge":
            return (
                self._positive_hinge(angle_h, mask_h),
                self._positive_hinge(angle_v, mask_v),
                self._positive_hinge(angle_ij, mask_par),
                self._positive_hinge(angle_perp_err, mask_perp),
            )
        return (
            self._positive_bce(score_h, mask_h),
            self._positive_bce(score_v, mask_v),
            self._positive_bce(score_par, mask_par),
            self._positive_bce(score_perp, mask_perp),
        )

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

        angle_h = angle_from_x_deg(unit[..., 0], unit[..., 1])
        angle_v = angle_from_y_deg(unit[..., 0], unit[..., 1])
        angle_ij = undirected_angle_deg(unit_i, unit_j)
        angle_perp_err = (angle_ij - 90.0).abs()

        mask_h = unary_gt[..., 0] * valid
        mask_v = unary_gt[..., 1] * valid
        mask_par = pair_gt[..., 0] * pair_valid
        mask_perp = pair_gt[..., 1] * pair_valid

        loss_h, loss_v, loss_par, loss_perp = self._compute_losses(
            score_h,
            score_v,
            score_par,
            score_perp,
            angle_h,
            angle_v,
            angle_ij,
            angle_perp_err,
            mask_h,
            mask_v,
            mask_par,
            mask_perp,
        )

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
                if self.geom_loss_mode == "angle_hinge":
                    metrics.update(
                        {
                            "geom_h_angle_deg": self._masked_mean(angle_h, mask_h),
                            "geom_v_angle_deg": self._masked_mean(angle_v, mask_v),
                            "geom_parallel_angle_deg": self._masked_mean(angle_ij, mask_par),
                            "geom_perpendicular_angle_err_deg": self._masked_mean(angle_perp_err, mask_perp),
                        }
                    )

        return components, metrics, counts

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum().clamp_min(1.0)
        return (values * mask).sum() / denom


def unit_from_angle_deg(angle_deg: float) -> torch.Tensor:
    """Helper for tests: unit vector at ``angle_deg`` from +x axis."""
    rad = math.radians(angle_deg)
    return torch.tensor([math.cos(rad), math.sin(rad)], dtype=torch.float32)
