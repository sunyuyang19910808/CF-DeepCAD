from __future__ import annotations

import torch
import torch.nn as nn


class MaskedMeanPooling(nn.Module):
    def forward(self, memory: torch.Tensor, mask_joint: torch.Tensor) -> torch.Tensor:
        valid = (~mask_joint).transpose(0, 1).unsqueeze(-1).float()
        denom = valid.sum(dim=0, keepdim=True).clamp(min=1.0)
        return (memory * valid).sum(dim=0, keepdim=True) / denom


class DualStreamPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def forward(self, memory: torch.Tensor, mask_joint: torch.Tensor, s_cmd: int) -> torch.Tensor:
        mem_cmd = memory[:s_cmd]
        mem_con = memory[s_cmd:]
        mask_cmd = mask_joint[:, :s_cmd]
        mask_con = mask_joint[:, s_cmd:]
        valid_cmd = (~mask_cmd).transpose(0, 1).unsqueeze(-1).float()
        valid_con = (~mask_con).transpose(0, 1).unsqueeze(-1).float()
        z_cmd = (mem_cmd * valid_cmd).sum(dim=0) / valid_cmd.sum(dim=0).clamp(min=1.0)
        z_con = (mem_con * valid_con).sum(dim=0) / valid_con.sum(dim=0).clamp(min=1.0)
        gate = self.gate(torch.cat([z_cmd, z_con], dim=-1))
        return (gate * z_cmd + (1.0 - gate) * z_con).unsqueeze(0)
