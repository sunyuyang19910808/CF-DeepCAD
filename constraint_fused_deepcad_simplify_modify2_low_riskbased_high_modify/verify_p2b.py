from __future__ import annotations

import json

import torch

from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.application.train_use_case import (
    _build_bottleneck,
    build_train_use_case,
)
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.config.config_constraint_fused_high_modify import (
    ConfigConstraintFusedHighModify,
)
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.encoding.bottleneck import (
    Bottleneck512,
    DeepCADBottleneck,
)
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.encoding.encoder_fused import EncoderFused
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.infrastructure.dataset_high_modify import (
    get_high_modify_dataloader,
)


def run_a2b_stack_smoke() -> dict:
    cfg = ConfigConstraintFusedHighModify("test")
    cfg.device = "cpu"
    cfg.gpu_ids = "cpu"
    cfg.batch_size = 2
    cfg.num_workers = 0
    cfg.dim_z = 256
    cfg.pooling_strategy = "masked_mean_plain"
    cfg.bottleneck_type = "deepcad_tanh"
    cfg.enable_soft_geometry = False
    cfg.alpha = 0.0
    cfg.beta = 0.0
    cfg.gamma = 0.0
    device = torch.device("cpu")

    loader = get_high_modify_dataloader("train", cfg, shuffle=False)
    batch = next(iter(loader))
    use_case = build_train_use_case(cfg, device=device)
    use_case.train()
    out = use_case.execute(batch)
    out["loss"].backward()

    z = out["z"]
    encoder_outputs = out["encoder_outputs"]
    return {
        "pooling_strategy": cfg.pooling_strategy,
        "bottleneck_type": cfg.bottleneck_type,
        "dim_z": cfg.dim_z,
        "z_shape": list(z.shape),
        "z_pre_shape": list(encoder_outputs["z_pre"].shape),
        "bottleneck_class": type(use_case.bottleneck).__name__,
        "backward_ok": True,
    }


def run_component_checks() -> dict:
    cfg = ConfigConstraintFusedHighModify("test")
    cfg.dim_z = 256
    cfg.pooled_dim = 256
    cfg.pooling_strategy = "masked_mean_plain"
    cfg.bottleneck_type = "deepcad_tanh"

    encoder = EncoderFused(cfg, pooling_strategy=cfg.pooling_strategy)
    bottleneck = _build_bottleneck(cfg)
    assert isinstance(bottleneck, DeepCADBottleneck)

    cfg.pooling_strategy = "masked_mean"
    cfg.bottleneck_type = "layernorm_tanh"
    encoder_projected = EncoderFused(cfg, pooling_strategy=cfg.pooling_strategy)
    bottleneck_ln = _build_bottleneck(cfg)
    assert isinstance(bottleneck_ln, Bottleneck512)

    return {
        "a2b_bottleneck": type(bottleneck).__name__,
        "default_bottleneck": type(bottleneck_ln).__name__,
        "encoder_has_plain_pool": hasattr(encoder, "masked_mean_plain"),
        "encoder_has_projected_pool": hasattr(encoder_projected, "masked_mean"),
    }


def main() -> None:
    payload = {
        "component_checks": run_component_checks(),
        "a2b_smoke": run_a2b_stack_smoke(),
    }
    z_shape = payload["a2b_smoke"]["z_shape"]
    expected_rank3 = len(z_shape) == 3
    expected_dim_z = z_shape[-1] == payload["a2b_smoke"]["dim_z"]
    payload["checks_passed"] = expected_rank3 and expected_dim_z and payload["a2b_smoke"]["z_shape"][0] == 1
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
