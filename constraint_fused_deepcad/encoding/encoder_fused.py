from __future__ import annotations

import torch
import torch.nn as nn

from model.layers.improved_transformer import TransformerEncoderLayerImproved
from model.layers.transformer import TransformerEncoder

from constraint_fused_deepcad.encoding.constraint_token_encoder import ConstraintTokenEncoder, SegmentEmbedding
from constraint_fused_deepcad.encoding.embeddings import CADEmbeddingFused
from constraint_fused_deepcad.encoding.pooling import DualStreamPooling, MaskedMeanPooling


class EncoderFused(nn.Module):
    def __init__(self, cfg, pooling_strategy: str = "masked_mean", use_dual_stream: bool = False):
        super().__init__()
        seq_len = cfg.max_total_len
        self.embedding = CADEmbeddingFused(cfg, seq_len)
        self.constraint_token_enc = ConstraintTokenEncoder(
            6, getattr(cfg, "max_lines", 64), cfg.d_model
        )
        self.segment_embed = SegmentEmbedding(cfg.d_model)
        enc_layer = TransformerEncoderLayerImproved(
            cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout
        )
        encoder_norm = nn.LayerNorm(cfg.d_model)
        self.encoder = TransformerEncoder(enc_layer, cfg.n_layers, encoder_norm)

        self.pooling_strategy = pooling_strategy
        self.use_dual_stream = use_dual_stream
        self.masked_mean = MaskedMeanPooling()
        self.dual_pool = DualStreamPooling(cfg.d_model) if use_dual_stream else None

    def forward(
        self,
        commands: torch.Tensor,
        args: torch.Tensor,
        constraint_tags: torch.Tensor,
        c_types: torch.Tensor,
        c_line_a: torch.Tensor,
        c_line_b: torch.Tensor,
        cmd_padding_mask: torch.Tensor,
        constraint_padding_mask: torch.Tensor,
        groups: torch.Tensor | None = None,
    ) -> torch.Tensor:
        e_cmd = self.embedding(commands, args, groups, constraint_tags)
        e_con = self.constraint_token_enc(c_types, c_line_a, c_line_b)
        s, n, _ = e_cmd.shape
        t_c = e_con.shape[0]

        seg = torch.cat(
            [
                torch.zeros(s, n, dtype=torch.long, device=e_cmd.device),
                torch.ones(t_c, n, dtype=torch.long, device=e_cmd.device),
            ],
            dim=0,
        )
        e_joint = torch.cat([e_cmd, e_con], dim=0) + self.segment_embed(seg)
        mask_joint = torch.cat([cmd_padding_mask, constraint_padding_mask], dim=1)
        memory = self.encoder(e_joint, src_key_padding_mask=mask_joint)

        if self.use_dual_stream and self.dual_pool is not None:
            return self.dual_pool(memory, mask_joint, s_cmd=s)
        return self.masked_mean(memory, mask_joint)
