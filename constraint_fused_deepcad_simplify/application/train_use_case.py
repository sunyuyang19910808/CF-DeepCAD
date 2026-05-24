from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from cadlib.macro import CMD_ARGS_MASK, EOS_IDX

from constraint_fused_deepcad_simplify.application.loss_composer import build_line_mask, compose_loss
from constraint_fused_deepcad_simplify.encoding.encoder_simplify import EncoderSimplify
from constraint_fused_deepcad_simplify.generation.axis_recon_head import AxisReconHead
from constraint_fused_deepcad_simplify.generation.decoder_adapter import DecoderAdapter


@dataclass
class TrainArtifacts:
    encoder: EncoderSimplify
    decoder: DecoderAdapter
    axis_recon_head: AxisReconHead
    cad_loss: nn.Module


class CommandCadLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_commands = cfg.n_commands
        self.args_dim = cfg.args_dim + 1
        self.weights = cfg.loss_weights
        self.register_buffer("cmd_args_mask", torch.tensor(CMD_ARGS_MASK))

    def forward(self, output):
        tgt_commands = output["tgt_commands"]
        tgt_args = output["tgt_args"]
        command_logits = output["command_logits"]
        args_logits = output["args_logits"]

        visibility_mask = (tgt_commands == EOS_IDX).sum(dim=-1) < (tgt_commands.size(-1) - 1)
        padding_mask = ((tgt_commands == EOS_IDX).cumsum(dim=-1) == 0).float()
        if tgt_commands.size(-1) > 3:
            padding_mask = padding_mask.clone()
            padding_mask[..., 3:] = (padding_mask[..., 3:] + padding_mask[..., :-3]).clamp(max=1.0)
        padding_mask = padding_mask * visibility_mask.unsqueeze(-1).float()

        arg_mask = self.cmd_args_mask[tgt_commands.long()].bool()
        loss_cmd = F.cross_entropy(
            command_logits[padding_mask.bool()].reshape(-1, self.n_commands),
            tgt_commands[padding_mask.bool()].reshape(-1).long(),
        )
        loss_args = F.cross_entropy(
            args_logits[arg_mask].reshape(-1, self.args_dim),
            (tgt_args[arg_mask].reshape(-1).long() + 1),
        )

        return {
            "loss_cmd": self.weights["loss_cmd_weight"] * loss_cmd,
            "loss_args": self.weights["loss_args_weight"] * loss_args,
        }


class TrainConstraintFusedSimplifyBatchUseCase:
    def __init__(self, artifacts: TrainArtifacts, cfg, device: torch.device):
        self.encoder = artifacts.encoder
        self.decoder = artifacts.decoder
        self.axis_recon_head = artifacts.axis_recon_head
        self.cad_loss = artifacts.cad_loss
        self.cfg = cfg
        self.device = device

    def modules(self):
        return [self.encoder, self.decoder, self.axis_recon_head]

    def train(self):
        for module in self.modules():
            module.train()

    def eval(self):
        for module in self.modules():
            module.eval()

    def state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "axis_recon_head": self.axis_recon_head.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.decoder.load_state_dict(state_dict["decoder"])
        self.axis_recon_head.load_state_dict(state_dict["axis_recon_head"])

    def execute(self, batch):
        commands = batch["command"].to(self.device)
        args = batch["args"].to(self.device)
        groups = batch["groups"].to(self.device)
        constraint_tags = batch["constraint_tags"].to(self.device)
        unary_gt = batch["unary_gt"].to(self.device)
        cmd_padding_mask = batch["cmd_padding_mask"].to(self.device)
        line_count = batch["line_count"].to(self.device)

        z = self.encoder(
            commands=commands.transpose(0, 1),
            args=args.transpose(0, 1),
            groups=groups.transpose(0, 1),
            constraint_tags=constraint_tags.transpose(0, 1),
            cmd_padding_mask=cmd_padding_mask,
        )
        decoder_output = self.decoder(z)
        decoder_output["tgt_commands"] = commands
        decoder_output["tgt_args"] = args

        cad_losses = self.cad_loss(decoder_output)
        loss_cmd = cad_losses["loss_cmd"] + cad_losses["loss_args"]

        unary_pred = self.axis_recon_head(z)
        line_mask = build_line_mask(line_count, unary_pred.size(1))
        total_loss, axis_loss = compose_loss(
            cmd_loss=loss_cmd,
            unary_pred=unary_pred,
            unary_gt=unary_gt,
            line_mask=line_mask,
            beta=self.cfg.axis_loss_weight,
            pos_weight=self.cfg.axis_pos_weight,
        )
        return {
            "loss": total_loss,
            "loss_cmd": loss_cmd,
            "loss_cmd_only": cad_losses["loss_cmd"],
            "loss_args": cad_losses["loss_args"],
            "axis_loss": axis_loss,
            "unary_pred": unary_pred,
            "z": z,
            "decoder_output": decoder_output,
        }


def build_train_use_case(cfg, device: torch.device) -> TrainConstraintFusedSimplifyBatchUseCase:
    artifacts = TrainArtifacts(
        encoder=EncoderSimplify(cfg).to(device),
        decoder=DecoderAdapter(cfg).to(device),
        axis_recon_head=AxisReconHead(cfg.dim_z, cfg.max_lines).to(device),
        cad_loss=CommandCadLoss(cfg).to(device),
    )
    return TrainConstraintFusedSimplifyBatchUseCase(artifacts=artifacts, cfg=cfg, device=device)
