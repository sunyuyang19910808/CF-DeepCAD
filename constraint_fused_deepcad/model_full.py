from __future__ import annotations

import torch
import torch.nn as nn

from constraint_fused_deepcad.application.geometry_constraint import DifferentiableConstraintEvaluator
from constraint_fused_deepcad.application.loss_composer import LossComposer
from constraint_fused_deepcad.application.train_use_case import TrainConstraintFusedBatchUseCase
from constraint_fused_deepcad.domain.services import ConstraintFusionDomainService, ConstraintReconstructionDomainService
from constraint_fused_deepcad.encoding.encoder_fused import EncoderFused
from constraint_fused_deepcad.encoding.recon_head import ConstraintReconHead
from constraint_fused_deepcad.generation.decoder_adapter import (
    ConstraintAwareDecoderAdapter,
    ConstraintPredHead,
    OptionalConstraintCrossAttn,
)
from model.autoencoder import Bottleneck, Decoder
from trainer.loss import CADLoss


def build_train_use_case(
    cfg,
    device: str | torch.device = "cuda",
    use_dual_stream: bool = False,
    enable_decoder_cross_attn: bool = False,
    constraint_cross_attn_dropout: float = 0.5,
    use_constraint_pred: bool = True,
) -> TrainConstraintFusedBatchUseCase:
    enc = EncoderFused(cfg, use_dual_stream=use_dual_stream).to(device)
    bottleneck = Bottleneck(cfg).to(device)
    decoder = Decoder(cfg).to(device)
    pred_head = ConstraintPredHead(cfg.d_model, 5).to(device) if use_constraint_pred else None
    cross = None
    if enable_decoder_cross_attn:
        cross = OptionalConstraintCrossAttn(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, training_dropout=constraint_cross_attn_dropout
        ).to(device)
    adapter = ConstraintAwareDecoderAdapter(decoder, pred_head, cross).to(device)
    recon = ConstraintReconHead(cfg.dim_z, getattr(cfg, "max_lines", 64)).to(device)
    geom_eval = DifferentiableConstraintEvaluator(
        max_lines=getattr(cfg, "max_lines", 64),
        lambda_collinear_dist=getattr(cfg, "lambda_collinear_dist", 1.0),
    ).to(device)
    fusion = ConstraintFusionDomainService(enc, bottleneck)
    recon_svc = ConstraintReconstructionDomainService(recon)
    loss_c = LossComposer(
        alpha=getattr(cfg, "alpha", 0.1),
        beta=getattr(cfg, "beta", 0.5),
        gamma=getattr(cfg, "gamma", 0.0),
        pos_weight=getattr(cfg, "pos_weight", 5.0),
    )
    cad_loss = CADLoss(cfg).to(device)
    return TrainConstraintFusedBatchUseCase(
        fusion,
        adapter,
        recon_svc,
        geom_eval,
        loss_c,
        cad_loss,
        use_constraint_pred=use_constraint_pred,
    )
