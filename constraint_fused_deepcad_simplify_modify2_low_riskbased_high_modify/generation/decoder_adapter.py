from __future__ import annotations

import torch
import torch.nn as nn

from model.autoencoder import Decoder
from model.model_utils import _make_batch_first

from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.generation.constraint_pred_head import ConstraintPredHead


class LatentOnlyDecoderAdapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.decoder = Decoder(cfg)
        self.constraint_pred_head = ConstraintPredHead(cfg.d_model, cfg.constraint_pred_dim)

    def forward(self, z: torch.Tensor) -> dict:
        src = self.decoder.embedding(z)
        hidden_states = self.decoder.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None)
        command_logits, args_logits = self.decoder.fcn(hidden_states)
        command_logits, args_logits, hidden_states = _make_batch_first(command_logits, args_logits, hidden_states)
        return {
            "command_logits": command_logits,
            "args_logits": args_logits,
            "hidden_states": hidden_states,
            "constraint_pred_logits": self.constraint_pred_head(hidden_states),
        }


ConstraintAwareDecoderAdapter = LatentOnlyDecoderAdapter
