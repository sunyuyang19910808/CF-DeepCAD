from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from cadlib.macro import CMD_ARGS_MASK, EOS_IDX

from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.application.differentiable_sketch_interpreter import (
    DifferentiableSketchInterpreter,
)
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.application.geometry_constraint import (
    DifferentiableConstraintEvaluator,
)
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.application.loss_composer import (
    LossComposer,
    constraint_pred_loss,
)
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.application.loss_schedule import (
    resolve_aux_weights,
)
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.domain.services import (
    ConstraintFusionDomainService,
    ConstraintReconstructionDomainService,
    build_line_mask,
    gather_decoder_line_features,
)
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.encoding.bottleneck import (
    Bottleneck512,
    DeepCADBottleneck,
)
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.encoding.encoder_fused import EncoderFused
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.encoding.recon_head import ConstraintReconHead
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.generation.decoder_adapter import (
    LatentOnlyDecoderAdapter,
)


@dataclass
class TrainArtifacts:
    encoder: EncoderFused
    bottleneck: nn.Module
    decoder: LatentOnlyDecoderAdapter
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
        cmd_valid = padding_mask.bool()
        arg_valid = arg_mask
        if cmd_valid.any():
            loss_cmd = nn.functional.cross_entropy(
                command_logits[cmd_valid].reshape(-1, self.n_commands),
                tgt_commands[cmd_valid].reshape(-1).long(),
            )
        else:
            loss_cmd = command_logits.sum() * 0.0
        if arg_valid.any():
            loss_args = nn.functional.cross_entropy(
                args_logits[arg_valid].reshape(-1, self.args_dim),
                (tgt_args[arg_valid].reshape(-1).long() + 1),
            )
        else:
            loss_args = args_logits.sum() * 0.0

        return {
            "loss_cmd": self.weights["loss_cmd_weight"] * loss_cmd,
            "loss_args": self.weights["loss_args_weight"] * loss_args,
        }


class TrainConstraintFusedHighModifyBatchUseCase:
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

    def execute(self, batch, aux_weights=None, compute_metrics: bool = True):
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

        decoder_output = self.decoder(z)
        decoder_output["tgt_commands"] = commands
        decoder_output["tgt_args"] = args

        cad_losses = self.cad_loss(decoder_output)
        loss_cmd = cad_losses["loss_cmd"] + cad_losses["loss_args"]
        decoder_line_features = gather_decoder_line_features(
            decoder_output["hidden_states"],
            line_cmd_mask=line_cmd_mask,
            line_index_map=line_index_map,
            max_lines=self.cfg.max_lines,
        )
        unary_logits, pair_logits = self.recon_service.reconstruct(decoder_line_features)
        pred_loss = constraint_pred_loss(
            decoder_output["constraint_pred_logits"],
            constraint_tags,
            
            cmd_padding_mask,
            line_cmd_mask=line_cmd_mask if self.cfg.line_only_pred_loss else None,
        )
        line_mask = build_line_mask(line_count, unary_logits.size(1))
        soft_lines = self.interpreter(
            decoder_output["args_logits"],
            line_cmd_mask=line_cmd_mask,
            line_index_map=line_index_map,
            max_lines=unary_logits.size(1),
            commands=commands,
        )
        geom_components, geom_metrics = self.constraint_evaluator(
            soft_lines,
            unary_gt,
            pair_gt,
            line_mask=line_mask,
            compute_metrics=compute_metrics,
        )
        if not getattr(self.cfg, "enable_soft_geometry", True):
            geom_components = {key: value * 0.0 for key, value in geom_components.items()}

        if aux_weights is None:
            aux_weights = resolve_aux_weights(self.cfg, getattr(self.cfg, "_current_epoch", 1))
        if len(aux_weights) == 3:
            alpha, beta, gamma_legacy = aux_weights
            gamma_h = gamma_v = gamma_para = gamma_perp = float(gamma_legacy)
        else:
            alpha, beta, gamma_h, gamma_v, gamma_para, gamma_perp = aux_weights

        composed = LossComposer(
            alpha=alpha,
            beta=beta,
            gamma_h=gamma_h,
            gamma_v=gamma_v,
            gamma_para=gamma_para,
            gamma_perp=gamma_perp,
            pos_weight=self.cfg.pos_weight,
        ).compose(
            cmd_loss=loss_cmd,
            pred_loss=pred_loss,
            unary_logits=unary_logits,
            pair_logits=pair_logits,
            unary_gt=unary_gt,
            pair_gt=pair_gt,
            line_mask=line_mask,
            geom_components=geom_components,
        )
        geom_loss_legacy = (
            geom_components["geom_h"]
            + geom_components["geom_v"]
            + geom_components["geom_para"]
            + geom_components["geom_perp"]
        )
        out = {
            "loss": composed["loss"],
            "loss_cmd": loss_cmd,
            "loss_cmd_only": cad_losses["loss_cmd"],
            "loss_args": cad_losses["loss_args"],
            "pred_loss": pred_loss,
            "recon_loss": composed["recon_loss"],
            "unary_recon_loss": composed["unary_recon_loss"],
            "pair_recon_loss": composed["pair_recon_loss"],
            "geom_loss": geom_loss_legacy,
            "geom_total": composed["geom_total"],
            "geom_h_loss": geom_components["geom_h"],
            "geom_v_loss": geom_components["geom_v"],
            "geom_para_loss": geom_components["geom_para"],
            "geom_perp_loss": geom_components["geom_perp"],
            "aux_alpha": alpha,
            "aux_beta": beta,
            "aux_gamma": gamma_h,
            "aux_gamma_h": gamma_h,
            "aux_gamma_v": gamma_v,
            "aux_gamma_para": gamma_para,
            "aux_gamma_perp": gamma_perp,
            "unary_logits": unary_logits,
            "pair_logits": pair_logits,
            "z": z,
            "decoder_output": decoder_output,
            "encoder_outputs": encoder_outputs,
            "decoder_line_features": decoder_line_features,
        }
        if compute_metrics and geom_metrics:
            out.update(
                geom_horizontal=geom_metrics["geom_horizontal"],
                geom_vertical=geom_metrics["geom_vertical"],
                geom_parallel=geom_metrics["geom_parallel"],
                geom_perpendicular=geom_metrics["geom_perpendicular"],
            )
        return out


def _build_bottleneck(cfg) -> nn.Module:
    if getattr(cfg, "bottleneck_type", "layernorm_tanh") == "deepcad_tanh":
        return DeepCADBottleneck(cfg.d_model, cfg.dim_z)
    return Bottleneck512(
        pooled_dim=getattr(cfg, "pooled_dim", cfg.dim_z),
        dim_z=cfg.dim_z,
    )


def build_train_use_case(cfg, device: torch.device) -> TrainConstraintFusedHighModifyBatchUseCase:
    artifacts = TrainArtifacts(
        encoder=EncoderFused(cfg, pooling_strategy=cfg.pooling_strategy).to(device),
        bottleneck=_build_bottleneck(cfg).to(device),
        decoder=LatentOnlyDecoderAdapter(cfg).to(device),
        recon_head=ConstraintReconHead(
            dim_z=cfg.dim_z,
            max_lines=cfg.max_lines,
            d_model=cfg.d_model,
            hidden_dim=getattr(cfg, "line_pair_hidden_dim", cfg.d_model),
            enable_line_pair_scorer=getattr(cfg, "enable_line_pair_scorer", True),
        ).to(device),
        interpreter=DifferentiableSketchInterpreter(
            n_bins=cfg.args_dim + 1,
            coord_range=(cfg.coord_range_min, cfg.coord_range_max),
            use_corrected_line_start=getattr(cfg, "use_corrected_line_start", False),
        ).to(device),
        constraint_evaluator=DifferentiableConstraintEvaluator(
            use_hard_geom_bce=getattr(cfg, "use_hard_geom_bce", False),
            bce_scale=getattr(cfg, "hard_geom_bce_scale", 6.0),
            pos_weight=getattr(cfg, "hard_geom_pos_weight", 5.0),
        ).to(device),
        cad_loss=CommandCadLoss(cfg).to(device),
    )
    return TrainConstraintFusedHighModifyBatchUseCase(artifacts=artifacts, cfg=cfg, device=device)
