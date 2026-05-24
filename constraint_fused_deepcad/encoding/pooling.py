from __future__ import annotations

import torch
import torch.nn as nn


class MaskedMeanPooling(nn.Module):
    def forward(self, memory: torch.Tensor, mask_joint: torch.Tensor) -> torch.Tensor:
        valid = (~mask_joint).transpose(0, 1).unsqueeze(-1).float()
        return (memory * valid).sum(dim=0, keepdim=True) / valid.sum(dim=0, keepdim=True).clamp(min=1e-6)


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
        valid_c = (~mask_cmd).transpose(0, 1).unsqueeze(-1).float()
        valid_n = (~mask_con).transpose(0, 1).unsqueeze(-1).float()
        z_cmd = (mem_cmd * valid_c).sum(dim=0) / valid_c.sum(dim=0).clamp(min=1e-6)
        z_con = (mem_con * valid_n).sum(dim=0) / valid_n.sum(dim=0).clamp(min=1e-6)
        g = self.gate(torch.cat([z_cmd, z_con], dim=-1))
        z = g * z_cmd + (1.0 - g) * z_con
        return z.unsqueeze(0)


class BottleneckAdapter(nn.Module):
    def __init__(self, d_in: int, d_out: int | None = None):
        super().__init__()
        d_out = d_out or d_in
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)
