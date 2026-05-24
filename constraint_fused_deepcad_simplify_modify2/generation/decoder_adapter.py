from __future__ import annotations

import torch
import torch.nn as nn

from model.autoencoder import Decoder
from model.layers.attention import MultiheadAttention
from model.model_utils import _make_batch_first

from constraint_fused_deepcad_simplify_modify2.generation.constraint_pred_head import ConstraintPredHead


class OptionalConstraintCrossAttn(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, training_dropout: float = 0.5):
        super().__init__()
        self.mha = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.training_dropout_p = training_dropout

    def forward(
        self,
        tgt: torch.Tensor,
        constraint_memory: torch.Tensor | None,
        constraint_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if constraint_memory is None or not self.training:
            return tgt
        if torch.rand(1, device=tgt.device).item() < self.training_dropout_p:
            return tgt
        query = self.norm(tgt)
        key = self.norm(constraint_memory)
        out, _ = self.mha(query, key, key, key_padding_mask=constraint_mask)
        return tgt + self.dropout(out)


class ConstraintAwareDecoderAdapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.decoder = Decoder(cfg)
        self.constraint_pred_head = ConstraintPredHead(cfg.d_model, cfg.constraint_pred_dim)
        self.optional_cross_attn = None
        if getattr(cfg, "enable_decoder_cross_attn", False):
            self.optional_cross_attn = OptionalConstraintCrossAttn(
                cfg.d_model,
                cfg.n_heads,
                dropout=cfg.dropout,
                training_dropout=getattr(cfg, "constraint_cross_attn_dropout", 0.5),
            )

    def forward(
        self,
        z: torch.Tensor,
        constraint_memory: torch.Tensor | None = None,
        constraint_mask: torch.Tensor | None = None,
    ) -> dict:
        src = self.decoder.embedding(z)
        hidden_states = self.decoder.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None)
        if self.optional_cross_attn is not None:
            hidden_states = self.optional_cross_attn(hidden_states, constraint_memory, constraint_mask)
        command_logits, args_logits = self.decoder.fcn(hidden_states)
        command_logits, args_logits, hidden_states = _make_batch_first(command_logits, args_logits, hidden_states)
        return {
            "command_logits": command_logits,
            "args_logits": args_logits,
            "hidden_states": hidden_states,
            "constraint_pred_logits": self.constraint_pred_head(hidden_states),
        }

