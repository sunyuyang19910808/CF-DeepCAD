"""Evaluate constraint satisfaction on val (greedy decode). Run: python -m constraint_fused_deepcad.evaluate --data_root data --limit 120"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from config.configAE import ConfigAE


def main():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--limit", type=int, default=120)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    backup = sys.argv
    sys.argv = [
        sys.argv[0],
        "--data_root",
        args.data_root,
        "--batch_size",
        str(args.batch_size),
        "--continue",
    ]
    cfg = ConfigAE("train")
    sys.argv = backup
    cfg.max_lines = 64
    cfg.max_constraints = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from constraint_fused_deepcad.application.evaluate_constraints import EvaluateConstraintSatisfactionUseCase, logits_to_cad_vec
    from constraint_fused_deepcad.infrastructure.dataset_fused import get_fused_dataloader
    from constraint_fused_deepcad.model_full import build_train_use_case
    from constraint_fused_deepcad.sketch_preparation.constraint_extractor import ConstraintExtractor

    dl = get_fused_dataloader("val", cfg, shuffle=False)
    use_case = build_train_use_case(cfg, device=device, use_constraint_pred=False)
    eval_uc = EvaluateConstraintSatisfactionUseCase()
    ext = ConstraintExtractor()

    use_case.fusion_service.encoder_fused.eval()
    use_case.fusion_service.bottleneck.eval()
    use_case.decoder_adapter.eval()

    all_gt = []
    all_vecs = []
    n = 0
    with torch.no_grad():
        for batch in dl:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            bsz = batch["command"].size(0)
            gts = []
            for i in range(bsz):
                cmd = batch["command"][i].cpu().numpy()
                args = batch["args"][i].cpu().numpy()
                cad_vec = np.concatenate([cmd[:, None], args], axis=-1)
                _, rels, _ = ext.extract_from_cad_vec(cad_vec)
                gts.append(rels)
            all_gt.extend(gts)

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
            z = use_case.fusion_service.bottleneck(use_case.fusion_service.encoder_fused(**enc_kw))
            cmd_l, arg_l, _ = use_case.decoder_adapter(z)
            vecs = logits_to_cad_vec(cmd_l.transpose(0, 1), arg_l.transpose(0, 1))
            all_vecs.append(vecs)
            n += bsz
            if n >= args.limit:
                break

    pred = np.concatenate(all_vecs, axis=0)
    metrics = eval_uc.evaluate_batch(all_gt[: pred.shape[0]], pred)
    print(json.dumps(metrics, indent=2))
    out_path = args.out or os.path.join(cfg.proj_dir, cfg.exp_name, "eval_metrics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
