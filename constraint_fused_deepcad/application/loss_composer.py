from __future__ import annotations

import torch
import torch.nn.functional as F

from constraint_fused_deepcad.encoding.recon_head import weighted_bce


class LossComposer:
    def __init__(self, alpha: float = 0.1, beta: float = 0.5, gamma: float = 0.0, pos_weight: float = 5.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.pos_weight = pos_weight

    def compose(
        self,
        cmd_loss: torch.Tensor,
        pred_loss: torch.Tensor | None,
        unary_pred: torch.Tensor,
        pair_pred: torch.Tensor,
        unary_gt: torch.Tensor,
        pair_gt: torch.Tensor,
        geom_loss: torch.Tensor | None = None,
        line_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if line_mask is not None:
            if line_mask.sum() == 0:
                z = cmd_loss
                if pred_loss is not None:
                    z = z + self.alpha * pred_loss
                if geom_loss is not None:
                    z = z + self.gamma * geom_loss
                return z
            m = line_mask.unsqueeze(-1).float()
            u_p = unary_pred * m
            u_t = unary_gt * m
            recon_u = weighted_bce(u_p, u_t, self.pos_weight)
            mm = (line_mask.unsqueeze(2) * line_mask.unsqueeze(1)).unsqueeze(-1).float()
            p_p = pair_pred * mm
            p_t = pair_gt * mm
            recon_p = weighted_bce(p_p, p_t, self.pos_weight)
        else:
            recon_u = weighted_bce(unary_pred, unary_gt, self.pos_weight)
            recon_p = weighted_bce(pair_pred, pair_gt, self.pos_weight)
        recon_loss = recon_u + recon_p
        total = cmd_loss + self.beta * recon_loss
        if pred_loss is not None:
            total = total + self.alpha * pred_loss
        if geom_loss is not None:
            total = total + self.gamma * geom_loss
        return total

    @staticmethod
    def constraint_pred_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        cmd_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits, targets: (S, N, 5); cmd_padding_mask (N, S) True = masked (padding).
        """
        s, n, c = logits.shape
        valid = (~cmd_padding_mask).transpose(0, 1).unsqueeze(-1).float()
        logits = logits * valid
        targets = targets * valid
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="sum") / (valid.sum() * c + 1e-6)
