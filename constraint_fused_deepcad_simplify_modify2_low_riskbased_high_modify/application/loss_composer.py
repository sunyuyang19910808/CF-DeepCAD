from __future__ import annotations

import torch
import torch.nn.functional as F

from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.encoding.recon_head import weighted_bce_logits


def constraint_pred_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    cmd_padding_mask: torch.Tensor,
    line_cmd_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = ~cmd_padding_mask
    if line_cmd_mask is not None:
        valid = valid & line_cmd_mask.bool()
    valid = valid.unsqueeze(-1).float()
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = loss * valid
    return loss.sum() / (valid.sum() * logits.size(-1) + 1e-6)


def compose_recon_loss(
    unary_logits: torch.Tensor,
    pair_logits: torch.Tensor,
    unary_gt: torch.Tensor,
    pair_gt: torch.Tensor,
    line_mask: torch.Tensor | None,
    pos_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    unary_valid_count = None
    pair_valid_count = None
    if line_mask is not None:
        unary_mask = line_mask.unsqueeze(-1).float()
        pair_mask = (line_mask.unsqueeze(2) * line_mask.unsqueeze(1)).unsqueeze(-1).float()
        unary_valid_count = unary_mask.sum() * unary_logits.size(-1)
        pair_valid_count = pair_mask.sum() * pair_logits.size(-1)
        unary_loss = F.binary_cross_entropy_with_logits(
            unary_logits, unary_gt, reduction="none", pos_weight=torch.full((unary_gt.size(-1),), float(pos_weight), device=unary_gt.device, dtype=unary_gt.dtype)
        )
        pair_loss = F.binary_cross_entropy_with_logits(
            pair_logits, pair_gt, reduction="none", pos_weight=torch.full((pair_gt.size(-1),), float(pos_weight), device=pair_gt.device, dtype=pair_gt.dtype)
        )
        unary_loss = (unary_loss * unary_mask).sum() / (unary_valid_count + 1e-6)
        pair_loss = (pair_loss * pair_mask).sum() / (pair_valid_count + 1e-6)
    else:
        unary_loss = weighted_bce_logits(unary_logits, unary_gt, pos_weight=pos_weight)
        pair_loss = weighted_bce_logits(pair_logits, pair_gt, pos_weight=pos_weight)
    return unary_loss + pair_loss, unary_loss, pair_loss


class LossComposer:
    """Combine the CAD reconstruction loss with auxiliary constraint losses.

    The geometry term is decomposed into four components -- ``geom_h``, ``geom_v``,
    ``geom_para`` and ``geom_perp`` -- weighted by ``gamma_h``, ``gamma_v``,
    ``gamma_para`` and ``gamma_perp`` respectively. In legacy mode (A1/A2/A2b/A2c)
    the four gammas are equal to a single ``gamma`` value so the total reduces to
    ``gamma * (L_h + L_v + L_para + L_perp)`` which matches the prior single-term
    ``self.gamma * geom_loss`` expression.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 0.5,
        gamma_h: float = 0.2,
        gamma_v: float = 0.2,
        gamma_para: float = 0.2,
        gamma_perp: float = 0.2,
        pos_weight: float = 5.0,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma_h = gamma_h
        self.gamma_v = gamma_v
        self.gamma_para = gamma_para
        self.gamma_perp = gamma_perp
        self.pos_weight = pos_weight

    def compose(
        self,
        cmd_loss: torch.Tensor,
        pred_loss: torch.Tensor,
        unary_logits: torch.Tensor,
        pair_logits: torch.Tensor,
        unary_gt: torch.Tensor,
        pair_gt: torch.Tensor,
        line_mask: torch.Tensor,
        geom_components: dict,
    ) -> dict:
        recon_loss, unary_loss, pair_loss = compose_recon_loss(
            unary_logits,
            pair_logits,
            unary_gt,
            pair_gt,
            line_mask=line_mask,
            pos_weight=self.pos_weight,
        )
        geom_total = (
            self.gamma_h * geom_components["geom_h"]
            + self.gamma_v * geom_components["geom_v"]
            + self.gamma_para * geom_components["geom_para"]
            + self.gamma_perp * geom_components["geom_perp"]
        )
        total = cmd_loss + self.alpha * pred_loss + self.beta * recon_loss + geom_total
        return {
            "loss": total,
            "recon_loss": recon_loss,
            "unary_recon_loss": unary_loss,
            "pair_recon_loss": pair_loss,
            "geom_total": geom_total,
        }
