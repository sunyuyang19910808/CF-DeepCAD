from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime


class ExperimentTracker:
    def __init__(self, exp_dir: str, exp_id: str, data_root: str):
        self.exp_dir = exp_dir
        self.exp_id = exp_id
        self.data_root = data_root
        os.makedirs(exp_dir, exist_ok=True)

    def write_manifest(self, config_snapshot: dict, dataset_name: str = "deepcad", dataset_split: str = "train"):
        rev = ""
        try:
            rev = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            rev = "unknown"
        manifest = {
            "exp_id": self.exp_id,
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "DATA_ROOT": os.path.abspath(self.data_root),
            "python": sys.version,
            "git_commit": rev,
            "config": config_snapshot,
        }
        with open(os.path.join(self.exp_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def append_train_metrics(self, row: dict):
        path = os.path.join(self.exp_dir, "train_metrics.csv")
        exists = os.path.isfile(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                w.writeheader()
            w.writerow(row)

    def write_eval_metrics(self, metrics: dict):
        with open(os.path.join(self.exp_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    def write_best_checkpoint(self, path: str):
        with open(os.path.join(self.exp_dir, "best_checkpoint.txt"), "w", encoding="utf-8") as f:
            f.write(path + "\n")

    def write_qualitative_cases(self, cases: list):
        with open(os.path.join(self.exp_dir, "qualitative_cases.json"), "w", encoding="utf-8") as f:
            json.dump(cases, f, indent=2)
