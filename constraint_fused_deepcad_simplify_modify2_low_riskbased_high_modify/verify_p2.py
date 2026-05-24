from __future__ import annotations

import json

import torch

from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.application.train_use_case import build_train_use_case
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.config.config_constraint_fused_high_modify import (
    ConfigConstraintFusedHighModify,
)
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.infrastructure.dataset_high_modify import (
    get_high_modify_dataloader,
)


def run_case(pooling_strategy: str, enable_soft_geometry: bool, line_only_pred_loss: bool):
    cfg = ConfigConstraintFusedHighModify("test")
    cfg.device = "cpu"
    cfg.gpu_ids = "cpu"
    cfg.batch_size = 2
    cfg.num_workers = 0
    cfg.pooling_strategy = pooling_strategy
    cfg.enable_soft_geometry = enable_soft_geometry
    cfg.line_only_pred_loss = line_only_pred_loss
    device = torch.device("cpu")

    loader = get_high_modify_dataloader("train", cfg, shuffle=False)
    batch = next(iter(loader))
    use_case = build_train_use_case(cfg, device=device)
    use_case.train()
    out = use_case.execute(batch)
    out["loss"].backward()
    return {
        "decoder_input": "z_only",
        "pooling_strategy": pooling_strategy,
        "enable_soft_geometry": enable_soft_geometry,
        "line_only_pred_loss": line_only_pred_loss,
        "pair_logits_shape": list(out["pair_logits"].shape),
        "pred_loss": float(out["pred_loss"].item()),
        "pair_recon_loss": float(out["pair_recon_loss"].item()),
        "backward_ok": True,
    }


def main():
    payload = {
        "cases": [
            run_case("segment_separated", True, True),
            run_case("masked_mean", True, True),
            run_case("segment_separated", False, False),
        ]
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
