from __future__ import annotations

import torch
import torch.nn as nn


class MaskedMeanPooling(nn.Module):
    def forward(self, memory: torch.Tensor, cmd_padding_mask: torch.Tensor) -> torch.Tensor:
        valid_mask = (~cmd_padding_mask).transpose(0, 1).unsqueeze(-1).float()
        denom = valid_mask.sum(dim=0, keepdim=True).clamp(min=1.0)
        return (memory * valid_mask).sum(dim=0, keepdim=True) / denom
