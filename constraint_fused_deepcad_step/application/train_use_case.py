from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from model.autoencoder import CADTransformer
from trainer.loss import CADLoss

from constraint_fused_deepcad_step.application.differentiable_sketch_interpreter import (
    DifferentiableSketchInterpreter,
)
from constraint_fused_deepcad_step.application.geom_schedule import resolve_gamma_geom
from constraint_fused_deepcad_step.application.geometry_constraint import PositiveRelationConstraintEvaluator
from constraint_fused_deepcad_step.domain.services import build_line_mask


@dataclass
class TrainArtifacts:
    model: CADTransformer
    interpreter: DifferentiableSketchInterpreter
    constraint_evaluator: PositiveRelationConstraintEvaluator
    cad_loss: CADLoss


class TrainDeepCADStepBatchUseCase:
    """Original DeepCAD P(S|z) path + optional positive-relation L_geom."""

    def __init__(self, artifacts: TrainArtifacts, cfg, device: torch.device):
        self.model = artifacts.model
        self.interpreter = artifacts.interpreter
        self.constraint_evaluator = artifacts.constraint_evaluator
        self.cad_loss = artifacts.cad_loss
        self.cfg = cfg
        self.device = device

    def modules(self):
        return [self.model, self.interpreter, self.constraint_evaluator]

    def train(self):
        for module in self.modules():
            module.train()

    def eval(self):
        for module in self.modules():
            module.eval()

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "interpreter": self.interpreter.state_dict(),
            "constraint_evaluator": self.constraint_evaluator.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        if "interpreter" in state_dict:
            self.interpreter.load_state_dict(state_dict["interpreter"])
        if "constraint_evaluator" in state_dict:
            self.constraint_evaluator.load_state_dict(state_dict["constraint_evaluator"])

    def execute(self, batch, epoch: int | None = None, compute_metrics: bool = True):
        commands = batch["command"].to(self.device)
        args = batch["args"].to(self.device)
        unary_gt = batch["unary_gt"].to(self.device)
        pair_gt = batch["pair_gt"].to(self.device)
        line_count = batch["line_count"].to(self.device)
        line_cmd_mask = batch["line_cmd_mask"].to(self.device)
        line_index_map = batch["line_index_map"].to(self.device)

        outputs = self.model(commands, args)
        cad_losses = self.cad_loss(outputs)

        loss_cmd_weighted = cad_losses["loss_cmd"]
        loss_args_weighted = cad_losses["loss_args"]
        cmd_weight = float(self.cfg.loss_weights["loss_cmd_weight"])
        args_weight = float(self.cfg.loss_weights["loss_args_weight"])
        loss_cmd_raw = loss_cmd_weighted / cmd_weight
        loss_args_raw = loss_args_weighted / args_weight

        max_lines = unary_gt.size(1)
        line_mask = build_line_mask(line_count, max_lines)
        soft_lines = self.interpreter(
            outputs["args_logits"],
            line_cmd_mask=line_cmd_mask,
            line_index_map=line_index_map,
            max_lines=max_lines,
            commands=commands,
        )
        geom_components, geom_metrics, geom_counts = self.constraint_evaluator(
            soft_lines,
            unary_gt,
            pair_gt,
            line_mask=line_mask,
            compute_metrics=compute_metrics,
        )

        if epoch is None:
            epoch = int(getattr(self.cfg, "_current_epoch", 1))
        gamma_geom = resolve_gamma_geom(self.cfg, epoch)
        loss_geom = geom_components["loss_geom"]
        loss_total = loss_cmd_weighted + loss_args_weighted + gamma_geom * loss_geom

        return {
            "loss": loss_total,
            "loss_cmd": loss_cmd_weighted,
            "loss_cmd_raw": loss_cmd_raw,
            "loss_args": loss_args_weighted,
            "loss_args_raw": loss_args_raw,
            "loss_geom": loss_geom,
            "gamma_geom": torch.tensor(gamma_geom, device=self.device),
            "geom_h": geom_components["geom_h"],
            "geom_v": geom_components["geom_v"],
            "geom_parallel": geom_components["geom_parallel"],
            "geom_perpendicular": geom_components["geom_perpendicular"],
            "geom_horizontal": geom_metrics.get("geom_horizontal", torch.tensor(0.0, device=self.device)),
            "geom_vertical": geom_metrics.get("geom_vertical", torch.tensor(0.0, device=self.device)),
            "positive_count_h": geom_counts["positive_count_h"],
            "positive_count_v": geom_counts["positive_count_v"],
            "positive_count_parallel": geom_counts["positive_count_parallel"],
            "positive_count_perpendicular": geom_counts["positive_count_perpendicular"],
            "outputs": outputs,
            "soft_lines": soft_lines,
        }


def build_train_use_case(cfg, device: torch.device) -> TrainDeepCADStepBatchUseCase:
    artifacts = TrainArtifacts(
        model=CADTransformer(cfg).to(device),
        interpreter=DifferentiableSketchInterpreter(
            n_bins=cfg.args_dim + 1,
            coord_range=(cfg.coord_range_min, cfg.coord_range_max),
            use_corrected_line_start=getattr(cfg, "use_corrected_line_start", True),
        ).to(device),
        constraint_evaluator=PositiveRelationConstraintEvaluator(
            bce_scale=getattr(cfg, "geom_bce_scale", 4.0),
            negative_weight=getattr(cfg, "geom_negative_weight", 0.0),
        ).to(device),
        cad_loss=CADLoss(cfg).to(device),
    )
    return TrainDeepCADStepBatchUseCase(artifacts=artifacts, cfg=cfg, device=device)
