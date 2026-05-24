from __future__ import annotations

import torch

from constraint_fused_deepcad.application.geometry_constraint import DifferentiableConstraintEvaluator
from constraint_fused_deepcad.application.loss_composer import LossComposer
from constraint_fused_deepcad.domain.services import ConstraintFusionDomainService, ConstraintReconstructionDomainService
from constraint_fused_deepcad.generation.decoder_adapter import ConstraintAwareDecoderAdapter
from model.model_utils import _make_batch_first
from trainer.loss import CADLoss


class TrainConstraintFusedBatchUseCase:
    def __init__(
        self,
        fusion_service: ConstraintFusionDomainService,
        decoder_adapter: ConstraintAwareDecoderAdapter,
        recon_service: ConstraintReconstructionDomainService,
        geom_evaluator: DifferentiableConstraintEvaluator | None,
        loss_composer: LossComposer,
        cad_loss: CADLoss,
        use_constraint_pred: bool = True,
    ):
        self.fusion_service = fusion_service
        self.decoder_adapter = decoder_adapter
        self.recon_service = recon_service
        self.geom_evaluator = geom_evaluator
        self.loss_composer = loss_composer
        self.cad_loss = cad_loss
        self.use_constraint_pred = use_constraint_pred

    def execute(self, batch: dict) -> dict:
        commands = batch["command"]
        args = batch["args"]
        constraint_tags = batch["constraint_tags"]
        c_types = batch["c_types"]
        c_line_a = batch["c_line_a"]
        c_line_b = batch["c_line_b"]
        cmd_padding_mask = batch["cmd_padding_mask"]
        constraint_padding_mask = batch["constraint_padding_mask"]
        groups = batch.get("groups")

        cmds_sf = commands.transpose(0, 1)
        args_sf = args.transpose(0, 1)
        tags_sf = constraint_tags.transpose(0, 1)
        c_types_sf = c_types.transpose(0, 1)
        c_la_sf = c_line_a.transpose(0, 1)
        c_lb_sf = c_line_b.transpose(0, 1)

        enc_kw = dict(
            commands=cmds_sf,
            args=args_sf,
            constraint_tags=tags_sf,
            c_types=c_types_sf,
            c_line_a=c_la_sf,
            c_line_b=c_lb_sf,
            cmd_padding_mask=cmd_padding_mask,
            constraint_padding_mask=constraint_padding_mask,
            groups=groups.transpose(0, 1) if groups is not None else None,
        )

        latent = self.fusion_service.fuse(**enc_kw)
        z = latent.tensor

        constraint_memory = None
        constraint_mask = None
        if batch.get("use_cross_attn") and "constraint_memory" in batch:
            constraint_memory = batch["constraint_memory"]
            constraint_mask = batch.get("constraint_mask")

        cmd_logits_sf, args_logits_sf, pred_logits_sf = self.decoder_adapter(
            z, constraint_memory=constraint_memory, constraint_mask=constraint_mask
        )

        cmd_logits, args_logits = _make_batch_first(cmd_logits_sf, args_logits_sf)
        outputs = {
            "command_logits": cmd_logits,
            "args_logits": args_logits,
            "tgt_commands": commands,
            "tgt_args": args,
        }
        loss_cmd_dict = self.cad_loss(outputs)
        cmd_loss = loss_cmd_dict["loss_cmd"] + loss_cmd_dict["loss_args"]

        unary_pred, pair_pred = self.recon_service.reconstruct(latent)

        geom_loss = None
        geom_metrics = {}
        if self.geom_evaluator is not None:
            geom_loss, geom_metrics = self.geom_evaluator(
                commands,
                args,
                args_logits,
                batch["unary_gt"],
                batch["pair_gt"],
            )

        pred_loss = None
        if self.use_constraint_pred and pred_logits_sf is not None:
            pred_loss = LossComposer.constraint_pred_loss(
                pred_logits_sf,
                tags_sf.float(),
                cmd_padding_mask,
            )

        line_counts = batch.get("line_counts")
        if line_counts is not None:
            n, L = unary_pred.shape[0], unary_pred.shape[1]
            line_mask = torch.zeros(n, L, device=unary_pred.device, dtype=torch.bool)
            for i, lc in enumerate(line_counts):
                lc = min(int(lc), L)
                line_mask[i, :lc] = True
        else:
            line_mask = None

        total = self.loss_composer.compose(
            cmd_loss,
            pred_loss,
            unary_pred,
            pair_pred,
            batch["unary_gt"],
            batch["pair_gt"],
            geom_loss=geom_loss,
            line_mask=line_mask,
        )

        res = {
            "loss": total,
            "loss_cmd": cmd_loss.detach(),
            "loss_dict": loss_cmd_dict,
            "unary_pred": unary_pred.detach(),
            "pair_pred": pair_pred.detach(),
            "pred_loss": pred_loss.detach() if pred_loss is not None else None,
        }
        if geom_loss is not None:
            res["geom_loss"] = geom_loss.detach()
            for k, v in geom_metrics.items():
                res[k] = v.detach() if torch.is_tensor(v) else v
        return res
