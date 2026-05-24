from __future__ import annotations

import json

import torch

from constraint_fused_deepcad_simplify_modify2_low_risk.application.train_use_case import build_train_use_case
from constraint_fused_deepcad_simplify_modify2_low_risk.config.config_constraint_fused_simplify_modify2_low_risk import (
    ConfigConstraintFusedSimplifyModify2LowRisk,
)
from constraint_fused_deepcad_simplify_modify2_low_risk.infrastructure.dataset_simplify_modify2_low_risk import (
    get_simplify_modify2_low_risk_dataloader,
)


def main():
    cfg = ConfigConstraintFusedSimplifyModify2LowRisk("test")
    cfg.device = "cpu"
    cfg.gpu_ids = "cpu"
    cfg.batch_size = 2
    cfg.num_workers = 0
    device = torch.device("cpu")

    loader = get_simplify_modify2_low_risk_dataloader("train", cfg, shuffle=False)
    batch = next(iter(loader))

    use_case = build_train_use_case(cfg, device=device)
    use_case.train()
    out = use_case.execute(batch)
    out["loss"].backward()

    line_only_valid = ((~batch["cmd_padding_mask"]) & batch["line_cmd_mask"]).sum().item()
    non_line_valid = ((~batch["cmd_padding_mask"]) & (~batch["line_cmd_mask"])).sum().item()

    payload = {
        "line_features_shape": list(out["line_features"].shape),
        "command_memory_shape": list(out["encoder_outputs"]["command_memory"].shape),
        "constraint_memory_shape": list(out["encoder_outputs"]["constraint_memory"].shape),
        "unary_logits_shape": list(out["unary_logits"].shape),
        "pair_logits_shape": list(out["pair_logits"].shape),
        "pred_loss": float(out["pred_loss"].item()),
        "recon_loss": float(out["recon_loss"].item()),
        "pair_recon_loss": float(out["pair_recon_loss"].item()),
        "line_only_valid_positions": int(line_only_valid),
        "non_line_positions_excluded": int(non_line_valid),
        "enable_decoder_cross_attn": bool(cfg.enable_decoder_cross_attn),
        "line_only_pred_loss": bool(cfg.line_only_pred_loss),
        "backward_ok": True,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
