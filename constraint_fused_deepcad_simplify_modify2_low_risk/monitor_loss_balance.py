from __future__ import annotations

import argparse
import csv
import json
import os
from statistics import mean


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="proj_log/constraint_fused_deepcad_simplify_modify2_low_risk/cf_simplify_modify2_low_risk/artifacts/train_metrics.csv",
    )
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--gamma", type=float, default=2.0)
    return parser.parse_args()


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def main():
    args = parse_args()
    csv_path = os.path.abspath(args.csv)
    with open(csv_path, "r", encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    recent = rows[-args.window :]

    cmd_vals = [_safe_float(row["loss_cmd"]) for row in recent]
    pred_vals = [_safe_float(row["pred_loss"]) for row in recent]
    recon_vals = [_safe_float(row["recon_loss"]) for row in recent]
    geom_vals = [_safe_float(row["geom_loss"]) for row in recent]

    cmd = mean(v for v in cmd_vals if v is not None)
    pred = mean(v for v in pred_vals if v is not None) * args.alpha
    recon = mean(v for v in recon_vals if v is not None) * args.beta
    geom = mean(v for v in geom_vals if v is not None) * args.gamma
    total = cmd + pred + recon + geom

    payload = {
        "window": min(args.window, len(recent)),
        "weighted_components": {
            "cmd_loss": cmd,
            "alpha_pred_loss": pred,
            "beta_recon_loss": recon,
            "gamma_geom_loss": geom,
        },
        "ratio": {
            "cmd_loss": cmd / total if total else None,
            "alpha_pred_loss": pred / total if total else None,
            "beta_recon_loss": recon / total if total else None,
            "gamma_geom_loss": geom / total if total else None,
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
