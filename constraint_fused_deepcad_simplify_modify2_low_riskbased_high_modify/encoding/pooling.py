from __future__ import annotations

import torch
import torch.nn as nn


class MaskedMeanPooling(nn.Module):
    def forward(self, memory: torch.Tensor, mask_joint: torch.Tensor) -> torch.Tensor:
        valid = (~mask_joint).transpose(0, 1).unsqueeze(-1).float()
        denom = valid.sum(dim=0, keepdim=True).clamp(min=1.0)
        return (memory * valid).sum(dim=0, keepdim=True) / denom


class SegmentSeparatedPooling(nn.Module):
    def __init__(self, d_model: int = 256, pooled_dim: int = 512):
        super().__init__()
        self.cmd_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.con_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(d_model * 3, pooled_dim),
            nn.LayerNorm(pooled_dim),
            nn.Tanh(),
        )

    @staticmethod
    def masked_mean(memory: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        valid = (~padding_mask).transpose(0, 1).unsqueeze(-1).float()
        denom = valid.sum(dim=0).clamp_min(1.0)
        return (memory * valid).sum(dim=0) / denom

    def forward(
        self,
        command_memory: torch.Tensor,
        constraint_memory: torch.Tensor,
        cmd_padding_mask: torch.Tensor,
        constraint_padding_mask: torch.Tensor,
    ) -> dict:
        z_cmd = self.cmd_proj(self.masked_mean(command_memory, cmd_padding_mask))
        z_con = self.con_proj(self.masked_mean(constraint_memory, constraint_padding_mask))
        gate = self.gate(torch.cat([z_cmd, z_con], dim=-1))
        z_mix = gate * z_cmd + (1.0 - gate) * z_con
        z_pre = self.out_proj(torch.cat([z_cmd, z_con, z_mix], dim=-1)).unsqueeze(0)
        return {
            "z_pre": z_pre,
            "z_cmd": z_cmd,
            "z_con": z_con,
            "z_gate": gate,
        }


class ProjectedMaskedMeanPooling(nn.Module):
    def __init__(self, d_model: int = 256, pooled_dim: int = 512):
        super().__init__()
        self.mean = MaskedMeanPooling()
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, pooled_dim),
            nn.LayerNorm(pooled_dim),
            nn.Tanh(),
        )

    def forward(self, memory: torch.Tensor, mask_joint: torch.Tensor) -> dict:
        z_pre = self.out_proj(self.mean(memory, mask_joint))
        return {"z_pre": z_pre}
