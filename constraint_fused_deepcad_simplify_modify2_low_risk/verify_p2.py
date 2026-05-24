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


def run_case(enable_cross_attn: bool, enable_line_pair_scorer: bool, line_only_pred_loss: bool):
    cfg = ConfigConstraintFusedSimplifyModify2LowRisk("test")
    cfg.device = "cpu"
    cfg.gpu_ids = "cpu"
    cfg.batch_size = 2
    cfg.num_workers = 0
    cfg.enable_decoder_cross_attn = enable_cross_attn
    cfg.enable_line_pair_scorer = enable_line_pair_scorer
    cfg.line_only_pred_loss = line_only_pred_loss
    device = torch.device("cpu")

    loader = get_simplify_modify2_low_risk_dataloader("train", cfg, shuffle=False)
    batch = next(iter(loader))
    use_case = build_train_use_case(cfg, device=device)
    use_case.train()
    out = use_case.execute(batch)
    out["loss"].backward()
    return {
        "enable_decoder_cross_attn": enable_cross_attn,
        "enable_line_pair_scorer": enable_line_pair_scorer,
        "line_only_pred_loss": line_only_pred_loss,
        "pair_logits_shape": list(out["pair_logits"].shape),
        "pred_loss": float(out["pred_loss"].item()),
        "pair_recon_loss": float(out["pair_recon_loss"].item()),
        "backward_ok": True,
    }


def main():
    payload = {
        "cases": [
            run_case(True, True, True),
            run_case(False, True, True),
            run_case(False, False, False),
        ]
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
