from __future__ import annotations

import torch
import torch.nn as nn

from model.autoencoder import Decoder
from model.layers.attention import MultiheadAttention
from model.model_utils import _make_batch_first

from constraint_fused_deepcad_simplify_modify2_low_risk.generation.constraint_pred_head import ConstraintPredHead


class OptionalConstraintCrossAttn(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, training_dropout: float = 0.15):
        super().__init__()
        self.mha = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.training_dropout_p = training_dropout

    def forward(
        self,
        tgt: torch.Tensor,
        constraint_memory: torch.Tensor | None,
        constraint_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if constraint_memory is None:
            return tgt
        if self.training and self.training_dropout_p > 0:
            if torch.rand(1, device=tgt.device).item() < self.training_dropout_p:
                return tgt
        if constraint_mask is None:
            query = self.norm_q(tgt)
            key = self.norm_kv(constraint_memory)
            out, _ = self.mha(query, key, key)
            return tgt + self.dropout(out)

        valid_batch = ~constraint_mask.bool().all(dim=1)
        if not valid_batch.any():
            return tgt

        result = tgt.clone()
        query = self.norm_q(tgt[:, valid_batch])
        key = self.norm_kv(constraint_memory[:, valid_batch])
        out, _ = self.mha(query, key, key, key_padding_mask=constraint_mask[valid_batch])
        result[:, valid_batch] = tgt[:, valid_batch] + self.dropout(out)
        return result


class ConstraintAwareDecoderAdapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.decoder = Decoder(cfg)
        self.constraint_pred_head = ConstraintPredHead(cfg.d_model, cfg.constraint_pred_dim)
        self.enable_decoder_cross_attn = getattr(cfg, "enable_decoder_cross_attn", False)
        self.optional_cross_attn = OptionalConstraintCrossAttn(
            cfg.d_model,
            cfg.n_heads,
            dropout=cfg.dropout,
            training_dropout=getattr(cfg, "constraint_cross_attn_dropout", 0.15),
        )

    def forward(
        self,
        z: torch.Tensor,
        constraint_memory: torch.Tensor | None = None,
        constraint_mask: torch.Tensor | None = None,
    ) -> dict:
        src = self.decoder.embedding(z)
        hidden_states = self.decoder.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None)
        if self.enable_decoder_cross_attn:
            hidden_states = self.optional_cross_attn(hidden_states, constraint_memory, constraint_mask)
        command_logits, args_logits = self.decoder.fcn(hidden_states)
        command_logits, args_logits, hidden_states = _make_batch_first(command_logits, args_logits, hidden_states)
        return {
            "command_logits": command_logits,
            "args_logits": args_logits,
            "hidden_states": hidden_states,
            "constraint_pred_logits": self.constraint_pred_head(hidden_states),
        }
