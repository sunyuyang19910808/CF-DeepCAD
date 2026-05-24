from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

from cadlib.macro import CMD_ARGS_MASK

from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.application.train_use_case import build_train_use_case
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.config.config_constraint_fused_high_modify import (
    ConfigConstraintFusedHighModify,
)


def resolve_device(cfg) -> torch.device:
    if cfg.device == "cpu":
        return torch.device("cpu")
    if cfg.device == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() and cfg.gpu_ids != "cpu" else "cpu")


def logits_to_vec(command_logits: torch.Tensor, args_logits: torch.Tensor) -> torch.Tensor:
    out_command = torch.argmax(torch.softmax(command_logits, dim=-1), dim=-1)
    out_args = torch.argmax(torch.softmax(args_logits, dim=-1), dim=-1) - 1
    mask = ~torch.tensor(CMD_ARGS_MASK, device=command_logits.device).bool()[out_command.long()]
    out_args = out_args.clone()
    out_args[mask] = -1
    return torch.cat([out_command.unsqueeze(-1), out_args], dim=-1)


class GenerateFromLatentUseCase:
    def __init__(self, use_case, device: torch.device):
        self.use_case = use_case
        self.device = device

    def load_checkpoint(self, ckpt_path: str):
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.use_case.load_state_dict(checkpoint["model_state_dict"])
        self.use_case.eval()

    def encode_batch(self, batch: dict) -> tuple[torch.Tensor, dict]:
        with torch.no_grad():
            latent, encoder_outputs = self.use_case.fusion_service.fuse(
                commands=batch["command"].to(self.device).transpose(0, 1),
                args=batch["args"].to(self.device).transpose(0, 1),
                groups=batch["groups"].to(self.device).transpose(0, 1),
                constraint_tags=batch["constraint_tags"].to(self.device).transpose(0, 1),
                c_types=batch["c_types"].to(self.device).transpose(0, 1),
                c_line_a=batch["c_line_a"].to(self.device).transpose(0, 1),
                c_line_b=batch["c_line_b"].to(self.device).transpose(0, 1),
                cmd_padding_mask=batch["cmd_padding_mask"].to(self.device),
                constraint_padding_mask=batch["constraint_padding_mask"].to(self.device),
            )
        return latent.tensor, encoder_outputs

    def decode_latent(self, z: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            decoded = self.use_case.decoder(z.to(self.device))
            out_vec = logits_to_vec(decoded["command_logits"], decoded["args_logits"]).detach().cpu().numpy()
        return out_vec

    def generate_from_random(self, batch_size: int, dim_z: int) -> np.ndarray:
        z = torch.randn(1, batch_size, dim_z, device=self.device)
        return self.decode_latent(z)


def build_generation_use_case(cfg: Optional[object] = None):
    cfg = cfg or ConfigConstraintFusedHighModify("test")
    device = resolve_device(cfg)
    use_case = build_train_use_case(cfg, device=device)
    gen = GenerateFromLatentUseCase(use_case, device)
    ckpt_path = getattr(cfg, "model_path", None)
    if ckpt_path is None:
        ckpt_name = cfg.ckpt if cfg.ckpt == "latest" else "ckpt_epoch{}.pth".format(cfg.ckpt)
        if not ckpt_name.endswith(".pth"):
            ckpt_name = "{}.pth".format(cfg.ckpt if cfg.ckpt == "latest" else "ckpt_epoch{}".format(cfg.ckpt))
        ckpt_path = os.path.join(cfg.model_dir, ckpt_name)
    gen.load_checkpoint(ckpt_path)
    return gen
