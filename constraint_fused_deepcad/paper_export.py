"""Aggregate eval_metrics.json from multiple exp dirs into a summary table (P4-03)."""

from __future__ import annotations

import argparse
import csv
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dirs", nargs="+", required=True, help="Experiment directories containing eval_metrics.json")
    p.add_argument("--out_csv", type=str, default="paper_table.csv")
    args = p.parse_args()

    rows = []
    for d in args.exp_dirs:
        path = os.path.join(d, "eval_metrics.json")
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        row = {"exp_dir": os.path.basename(d.rstrip(os.sep))}
        row.update(m)
        rows.append(row)
    if not rows:
        print("No eval_metrics.json found")
        return
    keys = list(rows[0].keys())
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("Wrote", args.out_csv, "rows", len(rows))


if __name__ == "__main__":
    main()
