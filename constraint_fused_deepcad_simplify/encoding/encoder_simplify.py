from __future__ import annotations

import torch
import torch.nn as nn

from model.autoencoder import Bottleneck
from model.layers.improved_transformer import LayerNorm, TransformerEncoderLayerImproved
from model.layers.transformer import TransformerEncoder

from constraint_fused_deepcad_simplify.encoding.embeddings import CADEmbeddingWithAxisTags
from constraint_fused_deepcad_simplify.encoding.pooling import MaskedMeanPooling


class EncoderSimplify(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = CADEmbeddingWithAxisTags(
            cfg,
            cfg.max_total_len,
            use_group=cfg.use_group_emb,
        )
        encoder_layer = TransformerEncoderLayerImproved(
            cfg.d_model,
            cfg.n_heads,
            cfg.dim_feedforward,
            cfg.dropout,
        )
        encoder_norm = LayerNorm(cfg.d_model)
        self.encoder = TransformerEncoder(encoder_layer, cfg.n_layers, encoder_norm)
        self.po
        
        oling = MaskedMeanPooling()
        self.bottleneck = Bottleneck(cfg)

    def forward(
        self,
        commands: torch.Tensor,
        args: torch.Tensor,
        constraint_tags: torch.Tensor,
        cmd_padding_mask: torch.Tensor,
        groups: torch.Tensor = None,
    ) -> torch.Tensor:
        embedding = self.embedding(commands, args, groups=groups, constraint_tags=constraint_tags)
        memory = self.encoder(embedding, mask=None, src_key_padding_mask=cmd_padding_mask)
        z_pre = self.pooling(memory, cmd_padding_mask)
        return self.bottleneck(z_pre)
