from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

from cadlib.curves import Line
from cadlib.extrude import CADSequence
from cadlib.macro import CMD_ARGS_MASK, EOS_IDX, LINE_IDX
from trainer.base import TrainClock
from utils import ensure_dir

from constraint_fused_deepcad_simplify_modify1.application.train_use_case import build_train_use_case
from constraint_fused_deepcad_simplify_modify1.config.config_constraint_fused_simplify_modify1 import (
    ConfigConstraintFusedSimplifyModify1,
)
from constraint_fused_deepcad_simplify_modify1.infrastructure.dataset_simplify_modify1 import (
    get_simplify_modify1_dataloader,
)
from constraint_fused_deepcad_simplify_modify1.sketch_preparation.constraint_extractor_simplify_modify1 import (
    ANGLE_THRESH,
    ConstraintExtractorSimplifyModify1,
    line_direction_xy,
    undirected_angle_deg,
)


def logits_to_vec(command_logits: torch.Tensor, args_logits: torch.Tensor) -> torch.Tensor:
    out_command = torch.argmax(torch.softmax(command_logits, dim=-1), dim=-1)
    out_args = torch.argmax(torch.softmax(args_logits, dim=-1), dim=-1) - 1
    mask = ~torch.tensor(CMD_ARGS_MASK, device=command_logits.device).bool()[out_command.long()]
    out_args = out_args.clone()
    out_args[mask] = -1
    return torch.cat([out_command.unsqueeze(-1), out_args], dim=-1)


def resolve_device(cfg) -> torch.device:
    if cfg.device == "cpu":
        return torch.device("cpu")
    if cfg.device == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() and cfg.gpu_ids != "cpu" else "cpu")


def load_checkpoint(use_case, ckpt_path: str, device: torch.device) -> TrainClock:
    checkpoint = torch.load(ckpt_path, map_location=device)
    use_case.load_state_dict(checkpoint["model_state_dict"])
    clock = TrainClock()
    if "clock" in checkpoint:
        clock.restore_checkpoint(checkpoint["clock"])
    return clock


def resolve_checkpoint_path(cfg) -> str:
    if getattr(cfg, "model_path", None):
        return os.path.abspath(cfg.model_path)
    ckpt_name = cfg.ckpt if cfg.ckpt == "latest" else "ckpt_epoch{}".format(cfg.ckpt)
    return os.path.join(cfg.model_dir, "{}.pth".format(ckpt_name))


def reconstruction_file_stem(data_id: str) -> str:
    return data_id.replace("\\", "/").split("/")[-1]


def build_reconstruction_use_case(cfg):
    device = resolve_device(cfg)
    use_case = build_train_use_case(cfg, device=device)
    load_checkpoint(use_case, resolve_checkpoint_path(cfg), device)
    use_case.eval()
    return use_case, device


def reconstruct_batch(use_case, batch, device: torch.device):
    with torch.no_grad():
        z = use_case.encoder(
            commands=batch["command"].to(device).transpose(0, 1),
            args=batch["args"].to(device).transpose(0, 1),
            groups=batch["groups"].to(device).transpose(0, 1),
            constraint_tags=batch["constraint_tags"].to(device).transpose(0, 1),
            cmd_padding_mask=batch["cmd_padding_mask"].to(device),
        )
        decoded = use_case.decoder(z)
        out_vec = logits_to_vec(decoded["command_logits"], decoded["args_logits"]).detach().cpu().numpy()

    gt_vec = torch.cat([batch["command"].unsqueeze(-1), batch["args"]], dim=-1).detach().cpu().numpy()
    return out_vec, gt_vec


def reconstruct_test_split(cfg, reconstruction_dir: str) -> None:
    eval_split = getattr(cfg, "eval_split", "test")
    loader = get_simplify_modify1_dataloader(eval_split, cfg, shuffle=False)
    use_case, device = build_reconstruction_use_case(cfg)

    ensure_dir(reconstruction_dir)
    for batch in tqdm(loader, desc="reconstruct"):
        out_vec, gt_vec = reconstruct_batch(use_case, batch, device)
        commands = gt_vec[:, :, 0]
        for batch_idx, data_id in enumerate(batch["id"]):
            seq_len = commands[batch_idx].tolist().index(EOS_IDX)
            save_path = os.path.join(reconstruction_dir, "{}_vec.h5".format(reconstruction_file_stem(data_id)))
            with h5py.File(save_path, "w") as file_obj:
                file_obj.create_dataset("out_vec", data=out_vec[batch_idx][:seq_len], dtype=int)
                file_obj.create_dataset("gt_vec", data=gt_vec[batch_idx][:seq_len], dtype=int)


def _lines_in_extrude(ext) -> List[Line]:
    out: List[Line] = []
    for loop in ext.profile.children:
        for curve in loop.children:
            if isinstance(curve, Line):
                out.append(curve)
    return out


def _parallel_perpendicular_pairs(lines: List[Line], angle_thresh: float) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    dirs = [line_direction_xy(line) for line in lines]
    parallel: List[Tuple[int, int]] = []
    perpendicular: List[Tuple[int, int]] = []
    for left in range(len(lines)):
        for right in range(left + 1, len(lines)):
            angle = undirected_angle_deg(dirs[left], dirs[right])
            if angle < angle_thresh:
                parallel.append((left, right))
            if abs(angle - 90.0) < angle_thresh:
                perpendicular.append((left, right))
    return parallel, perpendicular


def parallel_perpendicular_recall_index_aligned(
    gt_vec: np.ndarray,
    pred_vec: np.ndarray,
    angle_thresh: float,
    grid_size: int,
) -> Tuple[int, int, int, int, int, bool]:
    try:
        gt_seq = CADSequence.from_vector(gt_vec, is_numerical=True, n=grid_size)
        pred_seq = CADSequence.from_vector(pred_vec, is_numerical=True, n=grid_size)
    except Exception:
        return 0, 0, 0, 0, 0, False

    extrude_count_mismatch = len(gt_seq.seq) != len(pred_seq.seq)
    par_hits = par_denom = 0
    perp_hits = perp_denom = 0
    n_extrudes_skipped_line_mismatch = 0

    for extrude_idx in range(min(len(gt_seq.seq), len(pred_seq.seq))):
        gt_lines = _lines_in_extrude(gt_seq.seq[extrude_idx])
        pred_lines = _lines_in_extrude(pred_seq.seq[extrude_idx])
        if len(gt_lines) != len(pred_lines):
            n_extrudes_skipped_line_mismatch += 1
            continue
        if len(gt_lines) < 2:
            continue

        gt_parallel, gt_perpendicular = _parallel_perpendicular_pairs(gt_lines, angle_thresh)
        if not gt_parallel and not gt_perpendicular:
            continue
        pred_dirs = [line_direction_xy(line) for line in pred_lines]

        for left, right in gt_parallel:
            par_denom += 1
            if undirected_angle_deg(pred_dirs[left], pred_dirs[right]) < angle_thresh:
                par_hits += 1

        for left, right in gt_perpendicular:
            perp_denom += 1
            if abs(undirected_angle_deg(pred_dirs[left], pred_dirs[right]) - 90.0) < angle_thresh:
                perp_hits += 1

    return (
        par_hits,
        par_denom,
        perp_hits,
        perp_denom,
        n_extrudes_skipped_line_mismatch,
        extrude_count_mismatch,
    )


def _line_count_from_commands(commands: np.ndarray) -> int:
    return int(np.sum(commands == LINE_IDX))


def _constraint_counts_from_vec(
    cad_vec: np.ndarray,
    extractor: ConstraintExtractorSimplifyModify1,
) -> Tuple[int, int, int, bool]:
    try:
        cad_seq = CADSequence.from_vector(cad_vec, is_numerical=True, n=extractor.grid_size)
    except Exception:
        return 0, 0, 0, False

    lines = extractor.collect_lines_from_cad_sequence(cad_seq)
    raw_all = extractor.extract_raw_from_lines(lines)
    return len(raw_all.horizontal), len(raw_all.vertical), len(lines), True


def aggregate_metrics(reconstruction_dir: str, angle_thresh: float, grid_size: int):
    extractor = ConstraintExtractorSimplifyModify1(angle_thresh=angle_thresh, grid_size=grid_size)
    paths = sorted(
        file_name
        for file_name in os.listdir(reconstruction_dir)
        if file_name.endswith("_vec.h5") and os.path.isfile(os.path.join(reconstruction_dir, file_name))
    )
    if not paths:
        raise FileNotFoundError(
            "No *_vec.h5 under {}. Run reconstruction first or pass --reconstruction_dir.".format(reconstruction_dir)
        )

    rows: List[Dict[str, Any]] = []
    sum_gt_h = sum_pred_h = sum_gt_v = sum_pred_v = 0
    n_parse_fail_gt = n_parse_fail_pred = 0
    par_rec_hits_sum = par_rec_denom_sum = 0
    perp_rec_hits_sum = perp_rec_denom_sum = 0
    n_samples_extrude_count_mismatch = 0
    n_extrudes_skipped_line_mismatch_sum = 0

    for file_name in tqdm(paths, desc="metrics"):
        with h5py.File(os.path.join(reconstruction_dir, file_name), "r") as file_obj:
            gt_vec = file_obj["gt_vec"][:]
            out_vec = file_obj["out_vec"][:]

        gt_h, gt_v, n_lines_gt, ok_gt = _constraint_counts_from_vec(gt_vec, extractor)
        pred_h, pred_v, n_lines_pred, ok_pred = _constraint_counts_from_vec(out_vec, extractor)
        if not ok_gt:
            n_parse_fail_gt += 1
        if not ok_pred:
            n_parse_fail_pred += 1

        sum_gt_h += gt_h
        sum_pred_h += pred_h
        sum_gt_v += gt_v
        sum_pred_v += pred_v

        cmd_gt = gt_vec[:, 0]
        cmd_pred = out_vec[:, 0]
        n_line_cmds_gt = _line_count_from_commands(cmd_gt)
        n_line_cmds_pred = _line_count_from_commands(cmd_pred)

        ratio_h_sample: Optional[float] = None if gt_h == 0 else pred_h / gt_h
        ratio_v_sample: Optional[float] = None if gt_v == 0 else pred_v / gt_v

        ph, pd, qh, qd, pr_skip_ln, pr_ext_mis = parallel_perpendicular_recall_index_aligned(
            gt_vec,
            out_vec,
            angle_thresh=angle_thresh,
            grid_size=grid_size,
        )
        par_rec_hits_sum += ph
        par_rec_denom_sum += pd
        perp_rec_hits_sum += qh
        perp_rec_denom_sum += qd
        n_extrudes_skipped_line_mismatch_sum += pr_skip_ln
        if pr_ext_mis:
            n_samples_extrude_count_mismatch += 1

        rows.append(
            {
                "id": file_name[: -len("_vec.h5")],
                "n_lines_gt": n_lines_gt,
                "n_lines_pred": n_lines_pred,
                "n_line_cmds_gt": n_line_cmds_gt,
                "n_line_cmds_pred": n_line_cmds_pred,
                "gt_h": gt_h,
                "gt_v": gt_v,
                "pred_h": pred_h,
                "pred_v": pred_v,
                "ratio_h_sample": ratio_h_sample,
                "ratio_v_sample": ratio_v_sample,
                "par_recall_hits": ph,
                "par_recall_denom": pd,
                "par_recall_index_aligned": None if pd == 0 else ph / pd,
                "perp_recall_hits": qh,
                "perp_recall_denom": qd,
                "perp_recall_index_aligned": None if qd == 0 else qh / qd,
                "pair_recall_extrude_count_mismatch": pr_ext_mis,
                "pair_recall_extrudes_skipped_line_mismatch": pr_skip_ln,
                "parse_ok_gt": ok_gt,
                "parse_ok_pred": ok_pred,
            }
        )

    summary = {
        "n_samples": len(rows),
        "sum_gt_h": sum_gt_h,
        "sum_pred_h": sum_pred_h,
        "sum_gt_v": sum_gt_v,
        "sum_pred_v": sum_pred_v,
        "angle_thresh_deg": angle_thresh,
        "grid_size": grid_size,
        "n_parse_fail_gt": n_parse_fail_gt,
        "n_parse_fail_pred": n_parse_fail_pred,
        "ratio_h": None if sum_gt_h == 0 else sum_pred_h / sum_gt_h,
        "ratio_v": None if sum_gt_v == 0 else sum_pred_v / sum_gt_v,
        "parallel_recall_index_aligned_hits": par_rec_hits_sum,
        "parallel_recall_index_aligned_denominator": par_rec_denom_sum,
        "parallel_recall_index_aligned": None if par_rec_denom_sum == 0 else par_rec_hits_sum / par_rec_denom_sum,
        "perpendicular_recall_index_aligned_hits": perp_rec_hits_sum,
        "perpendicular_recall_index_aligned_denominator": perp_rec_denom_sum,
        "perpendicular_recall_index_aligned": None if perp_rec_denom_sum == 0 else perp_rec_hits_sum / perp_rec_denom_sum,
        "n_samples_extrude_count_mismatch": n_samples_extrude_count_mismatch,
        "n_extrudes_skipped_line_count_mismatch_total": n_extrudes_skipped_line_mismatch_sum,
        "pair_recall_index_aligned_note": (
            "Per extrude k up to min(n_gt_ext,n_pred_ext); only extrudes with equal GT/pred line counts; "
            "GT parallel pairs: pred undirected angle < angle_thresh; "
            "GT perpendicular pairs: |undirected angle - 90°| < angle_thresh; same line indices (i,j)."
        ),
    }
    summary["R_h"] = summary["ratio_h"]
    summary["R_v"] = summary["ratio_v"]
    summary["axis_precision_mean"] = summary["R_h"]
    summary["axis_recall_mean"] = summary["R_v"]
    return rows, summary


def main():
    cfg = ConfigConstraintFusedSimplifyModify1("test")
    out_dir = os.path.abspath(cfg.outputs or cfg.package_root)
    reconstruction_dir = os.path.abspath(cfg.reconstruction_dir or cfg.default_reconstruction_dir)
    ensure_dir(out_dir)

    if not cfg.skip_reconstruct:
        reconstruct_test_split(cfg, reconstruction_dir)

    rows, summary = aggregate_metrics(reconstruction_dir, cfg.angle_thresh or ANGLE_THRESH, cfg.grid_size)
    summary_path = os.path.join(out_dir, "summary.json")
    csv_path = os.path.join(out_dir, "per_sample_counts.csv")
    summary["checkpoint_path"] = resolve_checkpoint_path(cfg)
    summary["data_root"] = os.path.abspath(cfg.data_root)
    summary["exp_name"] = cfg.exp_name
    summary["proj_dir"] = os.path.abspath(cfg.proj_dir)
    summary["eval_split"] = getattr(cfg, "eval_split", "test")

    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, ensure_ascii=False)
    with open(csv_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print("Wrote", summary_path)
    print("Wrote", csv_path)
    print("R_h = sum_pred_h / sum_gt_h = {} / {} = {}".format(summary["sum_pred_h"], summary["sum_gt_h"], summary.get("R_h")))
    print("R_v = sum_pred_v / sum_gt_v = {} / {} = {}".format(summary["sum_pred_v"], summary["sum_gt_v"], summary.get("R_v")))
    print(
        "Parallel_recall_index = hits / denom = {} / {} = {}".format(
            summary.get("parallel_recall_index_aligned_hits"),
            summary.get("parallel_recall_index_aligned_denominator"),
            summary.get("parallel_recall_index_aligned"),
        )
    )
    print(
        "Perp_recall_index = hits / denom = {} / {} = {}".format(
            summary.get("perpendicular_recall_index_aligned_hits"),
            summary.get("perpendicular_recall_index_aligned_denominator"),
            summary.get("perpendicular_recall_index_aligned"),
        )
    )


if __name__ == "__main__":
    main()
