from __future__ import annotations

import torch
import torch.nn as nn

from model.layers.attention import MultiheadAttention


class ConstraintPredHead(nn.Module):
    def __init__(self, d_model: int, n_constraint_types: int = 5):
        super().__init__()
        self.proj = nn.Linear(d_model, n_constraint_types)

    def forward(self, h_step: torch.Tensor) -> torch.Tensor:
        return self.proj(h_step)


class OptionalConstraintCrossAttn(nn.Module):
    """Training-time optional cross-attention; eval returns tgt unchanged (latent-only)."""

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
        if constraint_memory is None:
            return tgt
        if not self.training:
            return tgt
        if torch.rand(1, device=tgt.device).item() < self.training_dropout_p:
            return tgt
        q = self.norm(tgt)
        k = self.norm(constraint_memory)
        out, _ = self.mha(q, k, k, key_padding_mask=constraint_mask)
        return tgt + self.dropout(out)


class ConstraintAwareDecoderAdapter(nn.Module):
    def __init__(
        self,
        decoder: nn.Module,
        constraint_pred_head: nn.Module | None = None,
        optional_cross_attn: nn.Module | None = None,
    ):
        super().__init__()
        self.decoder = decoder
        self.constraint_pred_head = constraint_pred_head
        self.optional_cross_attn = optional_cross_attn

    def forward(
        self,
        z: torch.Tensor,
        constraint_memory: torch.Tensor | None = None,
        constraint_mask: torch.Tensor | None = None,
    ):
        src = self.decoder.embedding(z)
        out = self.decoder.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None)
        if self.optional_cross_attn is not None and constraint_memory is not None:
            out = self.optional_cross_attn(out, constraint_memory, constraint_mask)
        command_logits, args_logits = self.decoder.fcn(out)
        constraint_pred_logits = None
        if self.constraint_pred_head is not None:
            constraint_pred_logits = self.constraint_pred_head(out)
        return command_logits, args_logits, constraint_pred_logits
