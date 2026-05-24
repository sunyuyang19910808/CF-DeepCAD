from __future__ import annotations

import csv
import json
import math
import os
from typing import Dict, List


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return math.nan


def _load_csv_rows(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def _window_mean(rows: List[dict], key: str, window: int = 50):
    values = [_safe_float(row.get(key)) for row in rows[-window:]]
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return None
    return sum(values) / len(values)


def build_baseline_snapshot(summary_path: str, metrics_csv_path: str) -> Dict[str, object]:
    with open(summary_path, "r", encoding="utf-8") as file_obj:
        summary = json.load(file_obj)
    rows = _load_csv_rows(metrics_csv_path)
    return {
        "summary_path": os.path.abspath(summary_path),
        "metrics_csv_path": os.path.abspath(metrics_csv_path),
        "ratio_h": summary.get("ratio_h"),
        "ratio_v": summary.get("ratio_v"),
        "parallel_recall_index_aligned": summary.get("parallel_recall_index_aligned"),
        "perpendicular_recall_index_aligned": summary.get("perpendicular_recall_index_aligned"),
        "n_parse_fail_pred": summary.get("n_parse_fail_pred"),
        "n_samples_extrude_count_mismatch": summary.get("n_samples_extrude_count_mismatch"),
        "train_window_mean": {
            "pred_loss": _window_mean(rows, "pred_loss"),
            "recon_loss": _window_mean(rows, "recon_loss"),
            "unary_recon_loss": _window_mean(rows, "unary_recon_loss"),
            "pair_recon_loss": _window_mean(rows, "pair_recon_loss"),
        },
        "train_last_row": rows[-1] if rows else None,
    }


def main():
    package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    repo_root = os.path.abspath(os.path.join(package_root, os.pardir))
    summary_path = os.path.join(repo_root, "constraint_fused_deepcad_simplify_modify2", "summary.json")
    metrics_csv_path = os.path.join(
        repo_root,
        "proj_log",
        "constraint_fused_deepcad_simplify_modify2",
        "cf_simplify_modify2",
        "artifacts",
        "train_metrics.csv",
    )
    payload = build_baseline_snapshot(summary_path, metrics_csv_path)
    out_path = os.path.join(package_root, "baseline_snapshot.json")
    with open(out_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
