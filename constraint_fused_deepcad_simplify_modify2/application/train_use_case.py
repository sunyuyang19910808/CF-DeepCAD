from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from cadlib.macro import CMD_ARGS_MASK, EOS_IDX
from model.autoencoder import Bottleneck

from constraint_fused_deepcad_simplify_modify2.application.differentiable_sketch_interpreter import (
    DifferentiableSketchInterpreter,
)
from constraint_fused_deepcad_simplify_modify2.application.geometry_constraint import (
    DifferentiableConstraintEvaluator,
)
from constraint_fused_deepcad_simplify_modify2.application.loss_composer import (
    LossComposer,
    constraint_pred_loss,
)
from constraint_fused_deepcad_simplify_modify2.domain.services import (
    ConstraintFusionDomainService,
    ConstraintReconstructionDomainService,
    build_line_mask,
)
from constraint_fused_deepcad_simplify_modify2.encoding.encoder_fused import EncoderFused
from constraint_fused_deepcad_simplify_modify2.encoding.recon_head import ConstraintReconHead
from constraint_fused_deepcad_simplify_modify2.generation.decoder_adapter import ConstraintAwareDecoderAdapter


@dataclass
class TrainArtifacts:
    encoder: EncoderFused
    bottleneck: Bottleneck
    decoder: ConstraintAwareDecoderAdapter
    recon_head: ConstraintReconHead
    interpreter: DifferentiableSketchInterpreter
    constraint_evaluator: DifferentiableConstraintEvaluator
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
        loss_cmd = nn.functional.cross_entropy(
            command_logits[padding_mask.bool()].reshape(-1, self.n_commands),
            tgt_commands[padding_mask.bool()].reshape(-1).long(),
        )
        loss_args = nn.functional.cross_entropy(
            args_logits[arg_mask].reshape(-1, self.args_dim),
            (tgt_args[arg_mask].reshape(-1).long() + 1),
        )

        return {
            "loss_cmd": self.weights["loss_cmd_weight"] * loss_cmd,
            "loss_args": self.weights["loss_args_weight"] * loss_args,
        }


class TrainConstraintFusedSimplifyModify2BatchUseCase:
    def __init__(self, artifacts: TrainArtifacts, cfg, device: torch.device):
        self.encoder = artifacts.encoder
        self.bottleneck = artifacts.bottleneck
        self.decoder = artifacts.decoder
        self.recon_head = artifacts.recon_head
        self.interpreter = artifacts.interpreter
        self.constraint_evaluator = artifacts.constraint_evaluator
        self.cad_loss = artifacts.cad_loss
        self.cfg = cfg
        self.device = device

        self.fusion_service = ConstraintFusionDomainService(self.encoder, self.bottleneck)
        self.recon_service = ConstraintReconstructionDomainService(self.recon_head)

    def modules(self):
        return [
            self.encoder,
            self.bottleneck,
            self.decoder,
            self.recon_head,
            self.interpreter,
            self.constraint_evaluator,
        ]

    def train(self):
        for module in self.modules():
            module.train()

    def eval(self):
        for module in self.modules():
            module.eval()

    def state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "bottleneck": self.bottleneck.state_dict(),
            "decoder": self.decoder.state_dict(),
            "recon_head": self.recon_head.state_dict(),
            "interpreter": self.interpreter.state_dict(),
            "constraint_evaluator": self.constraint_evaluator.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.bottleneck.load_state_dict(state_dict["bottleneck"])
        self.decoder.load_state_dict(state_dict["decoder"])
        self.recon_head.load_state_dict(state_dict["recon_head"])
        self.interpreter.load_state_dict(state_dict["interpreter"])
        self.constraint_evaluator.load_state_dict(state_dict["constraint_evaluator"])

    def execute(self, batch):
        commands = batch["command"].to(self.device)
        args = batch["args"].to(self.device)
        groups = batch["groups"].to(self.device)
        constraint_tags = batch["constraint_tags"].to(self.device)
        c_types = batch["c_types"].to(self.device)
        c_line_a = batch["c_line_a"].to(self.device)
        c_line_b = batch["c_line_b"].to(self.device)
        unary_gt = batch["unary_gt"].to(self.device)
        pair_gt = batch["pair_gt"].to(self.device)
        cmd_padding_mask = batch["cmd_padding_mask"].to(self.device)
        constraint_padding_mask = batch["constraint_padding_mask"].to(self.device)
        line_count = batch["line_count"].to(self.device)
        line_cmd_mask = batch["line_cmd_mask"].to(self.device)
        line_index_map = batch["line_index_map"].to(self.device)

        latent, encoder_outputs = self.fusion_service.fuse(
            commands=commands.transpose(0, 1),
            args=args.transpose(0, 1),
            groups=groups.transpose(0, 1),
            constraint_tags=constraint_tags.transpose(0, 1),
            c_types=c_types.transpose(0, 1),
            c_line_a=c_line_a.transpose(0, 1),
            c_line_b=c_line_b.transpose(0, 1),
            cmd_padding_mask=cmd_padding_mask,
            constraint_padding_mask=constraint_padding_mask,
        )
        z = latent.tensor

        decoder_output = self.decoder(
            z,
            constraint_memory=encoder_outputs["constraint_memory"] if self.cfg.enable_decoder_cross_attn else None,
            constraint_mask=encoder_outputs["constraint_mask"] if self.cfg.enable_decoder_cross_attn else None,
        )
        decoder_output["tgt_commands"] = commands
        decoder_output["tgt_args"] = args

        cad_losses = self.cad_loss(decoder_output)
        loss_cmd = cad_losses["loss_cmd"] + cad_losses["loss_args"]
        unary_logits, pair_logits = self.recon_service.reconstruct(latent)
        pred_loss = constraint_pred_loss(
            decoder_output["constraint_pred_logits"],
            constraint_tags,
            cmd_padding_mask,
        )
        line_mask = build_line_mask(line_count, unary_logits.size(1))
        soft_lines = self.interpreter(
            decoder_output["args_logits"],
            line_cmd_mask=line_cmd_mask,
            line_index_map=line_index_map,
            max_lines=unary_logits.size(1),
        )
        geom_loss, geom_metrics = self.constraint_evaluator(soft_lines, unary_gt, pair_gt, line_mask=line_mask)
        composed = LossComposer(
            alpha=self.cfg.alpha,
            beta=self.cfg.beta,
            gamma=self.cfg.gamma,
            pos_weight=self.cfg.pos_weight,
        ).compose(
            cmd_loss=loss_cmd,
            pred_loss=pred_loss,
            unary_logits=unary_logits,
            pair_logits=pair_logits,
            unary_gt=unary_gt,
            pair_gt=pair_gt,
            line_mask=line_mask,
            geom_loss=geom_loss,
        )
        return {
            "loss": composed["loss"],
            "loss_cmd": loss_cmd,
            "loss_cmd_only": cad_losses["loss_cmd"],
            "loss_args": cad_losses["loss_args"],
            "pred_loss": pred_loss,
            "recon_loss": composed["recon_loss"],
            "unary_recon_loss": composed["unary_recon_loss"],
            "pair_recon_loss": composed["pair_recon_loss"],
            "geom_loss": geom_loss,
            "geom_horizontal": geom_metrics["geom_horizontal"],
            "geom_vertical": geom_metrics["geom_vertical"],
            "geom_parallel": geom_metrics["geom_parallel"],
            "geom_perpendicular": geom_metrics["geom_perpendicular"],
            "unary_logits": unary_logits,
            "pair_logits": pair_logits,
            "z": z,
            "decoder_output": decoder_output,
            "encoder_outputs": encoder_outputs,
        }


def build_train_use_case(cfg, device: torch.device) -> TrainConstraintFusedSimplifyModify2BatchUseCase:
    artifacts = TrainArtifacts(
        encoder=EncoderFused(cfg, pooling_strategy=cfg.pooling_strategy).to(device),
        bottleneck=Bottleneck(cfg).to(device),
        decoder=ConstraintAwareDecoderAdapter(cfg).to(device),
        recon_head=ConstraintReconHead(cfg.dim_z, cfg.max_lines).to(device),
        interpreter=DifferentiableSketchInterpreter(
            n_bins=cfg.args_dim + 1,
            coord_range=(cfg.coord_range_min, cfg.coord_range_max),
        ).to(device),
        constraint_evaluator=DifferentiableConstraintEvaluator().to(device),
        cad_loss=CommandCadLoss(cfg).to(device),
    )
    return TrainConstraintFusedSimplifyModify2BatchUseCase(artifacts=artifacts, cfg=cfg, device=device)

