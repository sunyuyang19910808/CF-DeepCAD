from __future__ import annotations

import json
import os


def main():
    package_root = os.path.abspath(os.path.dirname(__file__))
    baseline_path = os.path.join(package_root, "baseline_snapshot.json")
    low_risk_summary_path = os.path.join(package_root, "summary.json")
    simplify_summary_path = os.path.abspath(
        os.path.join(package_root, os.pardir, "constraint_fused_deepcad_simplify", "summary.json")
    )
    original_summary_path = os.path.abspath(
        os.path.join(package_root, os.pardir, "论文尝试", "DeepCAD原始约束指标", "summary.json")
    )
    out_path = os.path.join(package_root, "comparison_summary.json")

    with open(baseline_path, "r", encoding="utf-8") as file_obj:
        baseline = json.load(file_obj)
    with open(low_risk_summary_path, "r", encoding="utf-8") as file_obj:
        low_risk = json.load(file_obj)
    with open(simplify_summary_path, "r", encoding="utf-8") as file_obj:
        simplify = json.load(file_obj)
    with open(original_summary_path, "r", encoding="utf-8") as file_obj:
        original = json.load(file_obj)

    payload = {
        "baselines": {
            "modify2": _summary_slice(baseline, include_train_window=True),
            "simplify": _summary_slice(simplify),
            "original_deepcad": _summary_slice(original),
            "low_risk": _summary_slice(low_risk),
        },
        "delta_low_risk_minus": {
            "modify2": _delta_block(low_risk, baseline),
            "simplify": _delta_block(low_risk, simplify),
            "original_deepcad": _delta_block(low_risk, original),
        },
    }

    with open(out_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)
    print("Wrote", out_path)


def _delta(current, baseline):
    if current is None or baseline is None:
        return None
    return current - baseline


def _summary_slice(summary, include_train_window=False):
    payload = {
        "ratio_h": summary.get("ratio_h"),
        "ratio_v": summary.get("ratio_v"),
        "parallel_recall_index_aligned": summary.get("parallel_recall_index_aligned"),
        "perpendicular_recall_index_aligned": summary.get("perpendicular_recall_index_aligned"),
        "n_parse_fail_pred": summary.get("n_parse_fail_pred"),
        "n_samples_extrude_count_mismatch": summary.get("n_samples_extrude_count_mismatch"),
    }
    if include_train_window:
        payload["train_window_mean"] = summary.get("train_window_mean")
    return payload


def _delta_block(current, baseline):
    return {
        "ratio_h": _delta(current.get("ratio_h"), baseline.get("ratio_h")),
        "ratio_v": _delta(current.get("ratio_v"), baseline.get("ratio_v")),
        "parallel_recall_index_aligned": _delta(
            current.get("parallel_recall_index_aligned"),
            baseline.get("parallel_recall_index_aligned"),
        ),
        "perpendicular_recall_index_aligned": _delta(
            current.get("perpendicular_recall_index_aligned"),
            baseline.get("perpendicular_recall_index_aligned"),
        ),
        "n_parse_fail_pred": _delta(current.get("n_parse_fail_pred"), baseline.get("n_parse_fail_pred")),
        "n_samples_extrude_count_mismatch": _delta(
            current.get("n_samples_extrude_count_mismatch"),
            baseline.get("n_samples_extrude_count_mismatch"),
        ),
    }


if __name__ == "__main__":
    main()
