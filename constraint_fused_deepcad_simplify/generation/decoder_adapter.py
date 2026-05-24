from __future__ import annotations

import torch.nn as nn

from model.autoencoder import Decoder
from model.model_utils import _make_batch_first


class DecoderAdapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.decoder = Decoder(cfg)

    def forward(self, z):
        command_logits, args_logits = self.decoder(z)
        command_logits, args_logits = _make_batch_first(command_logits, args_logits)
        return {
            "command_logits": command_logits,
            "args_logits": args_logits,
        }
