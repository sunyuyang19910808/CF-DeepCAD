from __future__ import annotations

import torch.nn as nn

from constraint_fused_deepcad.domain.entities import ConstraintAwareLatent


class ConstraintFusionDomainService:
    def __init__(self, encoder_fused: nn.Module, bottleneck: nn.Module):
        self.encoder_fused = encoder_fused
        self.bottleneck = bottleneck

    def fuse(self, **batch_tensors):
        z_pre = self.encoder_fused(**batch_tensors)
        z = self.bottleneck(z_pre)
        return ConstraintAwareLatent(z)


class ConstraintReconstructionDomainService:
    def __init__(self, recon_head: nn.Module):
        self.recon_head = recon_head

    def reconstruct(self, latent: ConstraintAwareLatent):
        z_sq = latent.tensor.squeeze(0)
        unary_pred, pair_pred = self.recon_head(z_sq)
        return unary_pred, pair_pred
