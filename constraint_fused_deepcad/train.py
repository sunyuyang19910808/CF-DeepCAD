"""
Short training entry for Constraint-Fused DeepCAD (uses DATA_ROOT / h5 like DeepCAD).
Run from repo root: python -m constraint_fused_deepcad.train --data_root data --proj_dir proj_log/constraint_fused --exp_name smoke
"""

import argparse
import os
import sys

import torch

from config.configAE import ConfigAE


def main():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--proj_dir", type=str, default="proj_log/constraint_fused")
    parser.add_argument("--exp_name", type=str, default="cf_train")
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--dual_stream", action="store_true")
    args, _rest = parser.parse_known_args()

    backup = sys.argv
    sys.argv = [
        sys.argv[0],
        "--data_root",
        args.data_root,
        "--proj_dir",
        args.proj_dir,
        "--exp_name",
        args.exp_name,
        "--batch_size",
        str(args.batch_size),
        "--continue",
    ]
    if args.gpu_ids is not None:
        sys.argv += ["--gpu_ids", args.gpu_ids]
    cfg = ConfigAE("train")
    sys.argv = backup

    cfg.max_lines = 64
    cfg.max_constraints = 128
    cfg.alpha = 0.1
    cfg.beta = 0.5
    cfg.pos_weight = 5.0
    cfg.batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu_ids != "cpu" else "cpu")

    from constraint_fused_deepcad.infrastructure.dataset_fused import get_fused_dataloader
    from constraint_fused_deepcad.infrastructure.experiment_tracker import ExperimentTracker
    from constraint_fused_deepcad.model_full import build_train_use_case

    dl = get_fused_dataloader("train", cfg, shuffle=True)
    use_case = build_train_use_case(cfg, device=device, use_dual_stream=args.dual_stream)
    opt = torch.optim.Adam(
        list(use_case.fusion_service.encoder_fused.parameters())
        + list(use_case.fusion_service.bottleneck.parameters())
        + list(use_case.decoder_adapter.parameters())
        + list(use_case.recon_service.recon_head.parameters()),
        lr=cfg.lr,
    )

    exp_dir = os.path.join(cfg.proj_dir, cfg.exp_name)
    tracker = ExperimentTracker(exp_dir, exp_id=cfg.exp_name, data_root=args.data_root)
    tracker.write_manifest({k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_") and isinstance(getattr(cfg, k), (int, float, str, bool, dict))})

    use_case.fusion_service.encoder_fused.train()
    use_case.fusion_service.bottleneck.train()
    use_case.decoder_adapter.train()
    use_case.recon_service.recon_head.train()

    step = 0
    for batch in dl:
        step += 1
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        opt.zero_grad()
        out = use_case.execute(batch)
        out["loss"].backward()
        opt.step()
        tracker.append_train_metrics(
            {
                "step": step,
                "loss": float(out["loss"].item()),
                "loss_cmd": float(out["loss_cmd"].item()),
            }
        )
        print(f"step {step} loss={out['loss'].item():.4f} loss_cmd={out['loss_cmd'].item():.4f}")
        if step >= args.max_steps:
            break

    print("done")


if __name__ == "__main__":
    main()
