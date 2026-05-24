"""P4-02: minimal two-config smoke (short steps) writing eval_metrics.json under proj_dir."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from config.configAE import ConfigAE


def run_steps(cfg, device, dual_stream: bool, tag: str, max_steps: int, data_root: str):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from constraint_fused_deepcad.infrastructure.dataset_fused import get_fused_dataloader
    from constraint_fused_deepcad.model_full import build_train_use_case

    cfg.data_root = data_root
    cfg.max_lines = 64
    cfg.max_constraints = 128
    cfg.batch_size = min(cfg.batch_size, 4)
    cfg.num_workers = 0
    dl = get_fused_dataloader("train", cfg, shuffle=True)
    uc = build_train_use_case(cfg, device=device, use_dual_stream=dual_stream)
    uc.fusion_service.encoder_fused.train()
    uc.fusion_service.bottleneck.train()
    uc.decoder_adapter.train()
    uc.recon_service.recon_head.train()
    opt = torch.optim.Adam(
        list(uc.fusion_service.encoder_fused.parameters())
        + list(uc.fusion_service.bottleneck.parameters())
        + list(uc.decoder_adapter.parameters())
        + list(uc.recon_service.recon_head.parameters()),
        lr=cfg.lr,
    )
    losses = []
    for i, batch in enumerate(dl):
        if i >= max_steps:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        opt.zero_grad()
        out = uc.execute(batch)
        out["loss"].backward()
        opt.step()
        losses.append(float(out["loss"].item()))
    metrics = {"tag": tag, "dual_stream": dual_stream, "mean_loss": sum(losses) / max(len(losses), 1)}
    out_dir = os.path.join(cfg.proj_dir, cfg.exp_name, tag)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--proj_dir", type=str, default="proj_log/constraint_fused_ablation")
    p.add_argument("--max_steps", type=int, default=3)
    args, _ = p.parse_known_args()

    backup = sys.argv
    sys.argv = [sys.argv[0], "--data_root", args.data_root, "--proj_dir", args.proj_dir, "--exp_name", "ablation", "--continue"]
    cfg = ConfigAE("train")
    sys.argv = backup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m1 = run_steps(cfg, device, False, "baseline_masked_mean", args.max_steps, args.data_root)
    m2 = run_steps(cfg, device, True, "dual_stream", args.max_steps, args.data_root)
    print(json.dumps({"baseline": m1, "dual_stream": m2}, indent=2))


if __name__ == "__main__":
    main()
