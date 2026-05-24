"""Latent-only decode smoke: encode one batch then decode. Run: python -m constraint_fused_deepcad.infer --data_root data"""

import argparse
import os
import sys

import torch

from config.configAE import ConfigAE


def main():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--batch_size", type=int, default=2)
    args = p.parse_args()

    backup = sys.argv
    sys.argv = [sys.argv[0], "--data_root", args.data_root, "--batch_size", str(args.batch_size), "--continue"]
    cfg = ConfigAE("train")
    sys.argv = backup
    cfg.max_lines = 64
    cfg.max_constraints = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from constraint_fused_deepcad.application.generate_use_case import GenerateFromLatentUseCase
    from constraint_fused_deepcad.infrastructure.dataset_fused import get_fused_dataloader
    from constraint_fused_deepcad.model_full import build_train_use_case

    dl = get_fused_dataloader("val", cfg, shuffle=False)
    use_case = build_train_use_case(cfg, device=device, use_constraint_pred=False)
    gen = GenerateFromLatentUseCase(use_case.decoder_adapter)

    batch = next(iter(dl))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    fusion = use_case.fusion_service
    enc_kw = dict(
        commands=batch["command"].transpose(0, 1),
        args=batch["args"].transpose(0, 1),
        constraint_tags=batch["constraint_tags"].transpose(0, 1),
        c_types=batch["c_types"].transpose(0, 1),
        c_line_a=batch["c_line_a"].transpose(0, 1),
        c_line_b=batch["c_line_b"].transpose(0, 1),
        cmd_padding_mask=batch["cmd_padding_mask"],
        constraint_padding_mask=batch["constraint_padding_mask"],
        groups=batch["groups"].transpose(0, 1),
    )
    fusion.encoder_fused.eval()
    fusion.bottleneck.eval()
    with torch.no_grad():
        z = fusion.bottleneck(fusion.encoder_fused(**enc_kw))
        out = gen.execute(z)
    print("infer ok", out["command_logits"].shape, out["args_logits"].shape)


if __name__ == "__main__":
    main()
