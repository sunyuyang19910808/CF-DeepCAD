from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from typing import Dict, Iterable, List, Tuple


DEFAULT_KEYS = [
    "loss",
    "loss_cmd",
    "pred_loss",
    "recon_loss",
    "geom_loss",
    "geom_horizontal",
    "geom_vertical",
    "geom_parallel",
    "geom_perpendicular",
]


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_rows(csv_path: str) -> Tuple[List[dict], int]:
    with open(csv_path, "r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        rows = list(reader)
    deduped: Dict[Tuple[int, int], dict] = {}
    for row in rows:
        epoch = int(float(row["epoch"]))
        step = int(float(row["step"]))
        deduped[(epoch, step)] = row
    return [deduped[key] for key in sorted(deduped.keys())], len(rows)


def rolling_mean(values: Iterable[float]) -> float:
    cleaned = [value for value in values if not math.isnan(value)]
    if not cleaned:
        return math.nan
    return statistics.fmean(cleaned)


def summarize_rows(rows: List[dict], raw_count: int, window: int, keys: List[str]) -> dict:
    if not rows:
        raise ValueError("No rows available to summarize.")

    latest = rows[-1]
    recent = rows[-window:] if window > 0 else rows
    previous = rows[-2 * window : -window] if window > 0 and len(rows) > window else []

    metrics = {}
    for key in keys:
        recent_vals = [_safe_float(row[key]) for row in recent if key in row]
        prev_vals = [_safe_float(row[key]) for row in previous if key in row]
        latest_val = _safe_float(latest.get(key, "nan"))
        recent_mean = rolling_mean(recent_vals)
        prev_mean = rolling_mean(prev_vals)
        metrics[key] = {
            "latest": latest_val,
            "rolling_mean": recent_mean,
            "previous_window_mean": prev_mean,
            "delta_vs_previous_window": recent_mean - prev_mean if not math.isnan(prev_mean) else math.nan,
            "window_min": min(recent_vals) if recent_vals else math.nan,
            "window_max": max(recent_vals) if recent_vals else math.nan,
        }

    return {
        "n_rows_raw": raw_count,
        "n_rows_deduped": len(rows),
        "window": window,
        "latest_epoch": int(float(latest["epoch"])),
        "latest_step": int(float(latest["step"])),
        "metrics": metrics,
    }


def format_summary(summary: dict) -> str:
    lines = [
        "rows={} window={} latest=(epoch {}, step {})".format(
            summary["n_rows_deduped"],
            summary["window"],
            summary["latest_epoch"],
            summary["latest_step"],
        )
    ]
    for key, payload in summary["metrics"].items():
        delta = payload["delta_vs_previous_window"]
        delta_str = "nan" if math.isnan(delta) else "{:+.6f}".format(delta)
        lines.append(
            "{}: latest={:.6f} smooth={:.6f} delta={}".format(
                key,
                payload["latest"],
                payload["rolling_mean"],
                delta_str,
            )
        )
    return "\n".join(lines)


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Smooth training monitor for Modify2 experiments.")
    parser.add_argument(
        "--csv",
        type=str,
        default="proj_log/constraint_fused_deepcad_simplify_modify2/cf_simplify_modify2/artifacts/train_metrics.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="proj_log/constraint_fused_deepcad_simplify_modify2/cf_simplify_modify2/artifacts/smoothed_metrics.json",
    )
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--watch-seconds", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=0, help="0 means run forever in watch mode.")
    parser.add_argument("--keys", nargs="*", default=DEFAULT_KEYS)
    return parser.parse_args()


def run_once(csv_path: str, output_path: str, window: int, keys: List[str]) -> dict:
    rows, raw_count = load_rows(csv_path)
    summary = summarize_rows(rows, raw_count=raw_count, window=window, keys=keys)
    payload = {
        "source_csv": os.path.abspath(csv_path),
        "generated_at_epoch_step": [summary["latest_epoch"], summary["latest_step"]],
        **summary,
    }
    write_json(output_path, payload)
    print(format_summary(summary))
    print("wrote", os.path.abspath(output_path))
    return summary


def main() -> None:
    args = parse_args()
    if args.watch_seconds <= 0:
        run_once(args.csv, args.output, args.window, args.keys)
        return

    iteration = 0
    while True:
        iteration += 1
        print("=== monitor iteration {} ===".format(iteration))
        try:
            run_once(args.csv, args.output, args.window, args.keys)
        except Exception as exc:
            print("monitor_error:", exc)
        if args.max_iterations > 0 and iteration >= args.max_iterations:
            break
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    main()

