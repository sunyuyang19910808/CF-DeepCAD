from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

from cadlib.macro import CMD_ARGS_MASK, EOS_IDX
from dataset.cad_dataset import get_dataloader
from trainer.base import TrainClock
from utils import ensure_dir

from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.application.evaluate_constraints import (
    EvaluateConstraintSatisfactionUseCase,
)
from constraint_fused_deepcad_step.application.device import resolve_device
from constraint_fused_deepcad_step.application.train_use_case import build_train_use_case
from constraint_fused_deepcad_step.config.config_step import ConfigStep


def logits_to_vec(command_logits: torch.Tensor, args_logits: torch.Tensor) -> torch.Tensor:
    out_command = torch.argmax(torch.softmax(command_logits, dim=-1), dim=-1)
    out_args = torch.argmax(torch.softmax(args_logits, dim=-1), dim=-1) - 1
    mask = ~torch.tensor(CMD_ARGS_MASK, device=command_logits.device).bool()[out_command.long()]
    out_args = out_args.clone()
    out_args[mask] = -1
    return torch.cat([out_command.unsqueeze(-1), out_args], dim=-1)


def resolve_checkpoint_path(cfg) -> str:
    if getattr(cfg, "model_path", None):
        return os.path.abspath(cfg.model_path)
    ckpt_name = cfg.ckpt if cfg.ckpt == "latest" else "ckpt_epoch{}".format(cfg.ckpt)
    return os.path.join(cfg.model_dir, "{}.pth".format(ckpt_name))


def load_checkpoint(use_case, ckpt_path: str, device: torch.device) -> TrainClock:
    checkpoint = torch.load(ckpt_path, map_location=device)
    use_case.load_state_dict(checkpoint["model_state_dict"])
    clock = TrainClock()
    if "clock" in checkpoint:
        clock.restore_checkpoint(checkpoint["clock"])
    return clock


def build_reconstruction_use_case(cfg):
    device = resolve_device(cfg)
    use_case = build_train_use_case(cfg, device=device)
    load_checkpoint(use_case, resolve_checkpoint_path(cfg), device)
    use_case.eval()
    return use_case, device


def reconstruction_file_stem(data_id: str) -> str:
    return data_id.replace("\\", "/").split("/")[-1]


def first_eos_length(command_seq: np.ndarray) -> int:
    eos_hits = np.flatnonzero(command_seq == EOS_IDX)
    return int(eos_hits[0]) if eos_hits.size > 0 else int(command_seq.shape[0])


def reconstruct_batch(use_case, batch, device: torch.device):
    commands = batch["command"].to(device)
    args = batch["args"].to(device)
    with torch.no_grad():
        outputs = use_case.model(commands, args)
        out_vec = logits_to_vec(outputs["command_logits"], outputs["args_logits"]).detach().cpu().numpy()

    gt_vec = torch.cat([batch["command"].unsqueeze(-1), batch["args"]], dim=-1).detach().cpu().numpy()
    return out_vec, gt_vec


def reconstruct_test_split(cfg, reconstruction_dir: str) -> None:
    eval_split = getattr(cfg, "eval_split", "test")
    loader = get_dataloader(eval_split, cfg, shuffle=False)
    use_case, device = build_reconstruction_use_case(cfg)
    remaining = getattr(cfg, "sample_count", 0) or 0

    ensure_dir(reconstruction_dir)
    for batch in tqdm(loader, desc="reconstruct"):
        out_vec, gt_vec = reconstruct_batch(use_case, batch, device)
        for batch_idx, data_id in enumerate(batch["id"]):
            seq_len = first_eos_length(gt_vec[batch_idx, :, 0])
            save_path = os.path.join(reconstruction_dir, "{}_vec.h5".format(reconstruction_file_stem(data_id)))
            with h5py.File(save_path, "w") as file_obj:
                file_obj.create_dataset("out_vec", data=out_vec[batch_idx][:seq_len], dtype=int)
                file_obj.create_dataset("gt_vec", data=gt_vec[batch_idx][:seq_len], dtype=int)
            if remaining > 0:
                remaining -= 1
                if remaining == 0:
                    return


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def run_acc_evaluation(reconstruction_dir: str) -> Tuple[str, Dict[str, float]]:
    script_path = os.path.join(_repo_root(), "evaluation", "evaluate_ae_acc.py")
    env = os.environ.copy()
    repo_root = _repo_root()
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [sys.executable, script_path, "--src", os.path.abspath(reconstruction_dir)],
        check=True,
        cwd=repo_root,
        env=env,
    )
    acc_stat_path = os.path.abspath(reconstruction_dir) + "_acc_stat.txt"
    metrics: Dict[str, float] = {}
    with open(acc_stat_path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            cmd_match = re.search(r"avg command acc \(ACC_cmd\):\s*(\S+)", line)
            if cmd_match:
                val = cmd_match.group(1)
                metrics["ACC_cmd"] = None if val.lower() == "nan" else float(val)
            param_match = re.search(r"avg param acc \(ACC_param\):\s*(\S+)", line)
            if param_match:
                val = param_match.group(1)
                metrics["ACC_param"] = None if val.lower() == "nan" else float(val)
    return acc_stat_path, metrics


def write_constraint_metrics(out_dir: str, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    ensure_dir(out_dir)
    summary_path = os.path.join(out_dir, "summary.json")
    csv_path = os.path.join(out_dir, "per_sample_counts.csv")
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, ensure_ascii=False)
    with open(csv_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print("Wrote", summary_path)
    print("Wrote", csv_path)


def evaluate(cfg) -> None:
    out_dir = os.path.abspath(cfg.outputs or cfg.default_eval_dir)
    reconstruction_dir = os.path.abspath(cfg.reconstruction_dir or cfg.default_reconstruction_dir)
    ensure_dir(out_dir)

    if not cfg.skip_reconstruct:
        reconstruct_test_split(cfg, reconstruction_dir)
        print("Wrote reconstruction to", reconstruction_dir)

    acc_stat_path, acc_metrics = run_acc_evaluation(reconstruction_dir)

    evaluator = EvaluateConstraintSatisfactionUseCase(angle_thresh=cfg.angle_thresh, grid_size=cfg.grid_size)
    rows, summary = evaluator.aggregate_metrics(reconstruction_dir)
    summary.update(acc_metrics)
    summary["acc_stat_path"] = acc_stat_path
    summary["reconstruction_dir"] = reconstruction_dir
    summary["checkpoint_path"] = resolve_checkpoint_path(cfg)
    summary["data_root"] = os.path.abspath(cfg.data_root)
    summary["exp_name"] = cfg.exp_name
    summary["proj_dir"] = os.path.abspath(cfg.proj_dir)
    summary["eval_split"] = getattr(cfg, "eval_split", "test")
    summary["ratio_h_metric"] = "index_aligned"
    summary["ratio_v_metric"] = "index_aligned"
    write_constraint_metrics(out_dir, rows, summary)


def main() -> None:
    cfg = ConfigStep("test")
    evaluate(cfg)


if __name__ == "__main__":
    main()
