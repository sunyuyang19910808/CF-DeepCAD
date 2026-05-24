"""
Rebuild DeepCAD AE on the test split and compute:

  R_h, R_v: global counts of horizontal / vertical Line curves (unchanged).

  Parallel / perpendicular: **index-aligned recall only** — per Extrude ``k``, if GT and pred have the same
  line count and order, each GT pair from ``extract_raw_from_lines`` is checked on **pred lines (i,j)**:
  parallel if undirected angle ``< angle_thresh``; perpendicular if ``|angle - 90°| < angle_thresh``.
  Global recall = sum(hits) / sum(denom) over all evaluable extrudes.

Run from repository root (recommended):
  python 论文尝试/DeepCAD原始约束指标/eval_original_deepcad_axis_constraints.py ^
    --proj_dir proj_log --exp_name newDeepCAD --ckpt latest --data_root data -g 0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cadlib.curves import Line  # noqa: E402
from cadlib.extrude import CADSequence  # noqa: E402
from cadlib.macro import EOS_IDX, LINE_IDX  # noqa: E402
from constraint_fused_deepcad.sketch_preparation.constraint_extractor import (  # noqa: E402
    ANGLE_THRESH,
    ConstraintExtractor,
    _undirected_line_angle_deg,
)
from dataset.cad_dataset import get_dataloader  # noqa: E402
from trainer import TrainerAE  # noqa: E402
from utils import ensure_dir  # noqa: E402


def _build_config_ae_for_rec(
    proj_dir: str,
    data_root: str,
    exp_name: str,
    ckpt: str,
    gpu_ids: str,
    outputs: str,
    batch_size: int,
    num_workers: int,
    eval_split: str,
):
    saved = sys.argv[:]
    try:
        sys.argv = [
            "eval_original_deepcad_axis_constraints",
            "-m",
            "rec",
            "--proj_dir",
            proj_dir,
            "--data_root",
            data_root,
            "--exp_name",
            exp_name,
            "--ckpt",
            str(ckpt),
            "-g",
            str(gpu_ids),
            "-o",
            outputs,
            "--batch_size",
            str(batch_size),
            "--num_workers",
            str(num_workers),
            "--eval_split",
            str(eval_split),
        ]
        from config import ConfigAE  # noqa: WPS433

        return ConfigAE("test")
    finally:
        sys.argv = saved


def _lines_in_extrude(ext) -> List[Line]:
    out: List[Line] = []
    for loop in ext.profile.children:
        for curve in loop.children:
            if isinstance(curve, Line):
                out.append(curve)
    return out


def _unit_dir2d(line: Line) -> np.ndarray:
    d = line.direction(from_start=True).astype(np.float64)
    nd = float(np.linalg.norm(d[:2]) + 1e-12)
    return d[:2] / nd


def parallel_perpendicular_recall_index_aligned(
    gt_vec: np.ndarray, pred_vec: np.ndarray, ex: ConstraintExtractor
) -> Tuple[int, int, int, int, int, bool]:
    """GT-defined parallel and perpendicular pairs; pred checked at same line indices per Extrude.

    Returns:
        par_hits, par_denom, perp_hits, perp_denom, n_extrudes_skipped_line_count_mismatch,
        extrude_count_mismatch
    """
    try:
        gt_seq = CADSequence.from_vector(gt_vec, is_numerical=True, n=ex.grid_size)
        pred_seq = CADSequence.from_vector(pred_vec, is_numerical=True, n=ex.grid_size)
    except Exception:
        return 0, 0, 0, 0, 0, False

    ext_mismatch = len(gt_seq.seq) != len(pred_seq.seq)
    par_hits = par_denom = 0
    perp_hits = perp_denom = 0
    n_skip_line = 0
    ath = ex.angle_thresh

    for k in range(min(len(gt_seq.seq), len(pred_seq.seq))):
        gt_lines = _lines_in_extrude(gt_seq.seq[k])
        pred_lines = _lines_in_extrude(pred_seq.seq[k])
        if len(gt_lines) != len(pred_lines):
            n_skip_line += 1
            continue
        if len(gt_lines) < 2:
            continue
        raw_gt = ex.extract_raw_from_lines(gt_lines)
        if not raw_gt.parallel and not raw_gt.perpendicular:
            continue
        pred_dirs = [_unit_dir2d(ln) for ln in pred_lines]
        for i, j in raw_gt.parallel:
            par_denom += 1
            ua = _undirected_line_angle_deg(pred_dirs[i], pred_dirs[j])
            if ua < ath:
                par_hits += 1
        for i, j in raw_gt.perpendicular:
            perp_denom += 1
            ua = _undirected_line_angle_deg(pred_dirs[i], pred_dirs[j])
            if abs(ua - 90.0) < ath:
                perp_hits += 1

    return par_hits, par_denom, perp_hits, perp_denom, n_skip_line, ext_mismatch


def _constraint_counts_from_vec(
    cad_vec: np.ndarray, ex: ConstraintExtractor
) -> Tuple[int, int, int, bool]:
    """Returns (n_h, n_v, n_lines, parse_ok) — horizontal/vertical only (all Line, all extrudes)."""
    try:
        cad_seq = CADSequence.from_vector(cad_vec, is_numerical=True, n=ex.grid_size)
    except Exception:
        return 0, 0, 0, False

    lines = ex.collect_lines_from_cad_sequence(cad_seq)
    raw_all = ex.extract_raw_from_lines(lines)
    return len(raw_all.horizontal), len(raw_all.vertical), len(lines), True


def _line_count_from_commands(commands: np.ndarray) -> int:
    return int(np.sum(commands == LINE_IDX))


def reconstruct_eval_split(cfg) -> None:
    tr_agent = TrainerAE(cfg)
    tr_agent.load_ckpt(cfg.ckpt)
    tr_agent.net.eval()

    eval_split = getattr(cfg, "eval_split", "test")
    test_loader = get_dataloader(eval_split, cfg)
    print("Total number of {} data:".format(eval_split), len(test_loader))

    ensure_dir(cfg.outputs)

    pbar = tqdm(test_loader)
    for _, data in enumerate(pbar):
        batch_size = data["command"].shape[0]
        commands = data["command"]
        args = data["args"]
        gt_vec = torch.cat([commands.unsqueeze(-1), args], dim=-1).squeeze(1).detach().cpu().numpy()
        commands_ = gt_vec[:, :, 0]
        with torch.no_grad():
            outputs, _ = tr_agent.forward(data)
            batch_out_vec = tr_agent.logits2vec(outputs)

        for j in range(batch_size):
            out_vec = batch_out_vec[j]
            seq_len = commands_[j].tolist().index(EOS_IDX)

            data_id = data["id"][j].split("/")[-1]

            save_path = os.path.join(cfg.outputs, "{}_vec.h5".format(data_id))
            with h5py.File(save_path, "w") as fp:
                fp.create_dataset("out_vec", data=out_vec[:seq_len], dtype=int)
                fp.create_dataset("gt_vec", data=gt_vec[j][:seq_len], dtype=int)


def aggregate_metrics(
    reconstruction_dir: str,
    angle_thresh: float,
    grid_size: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ex = ConstraintExtractor(angle_thresh=angle_thresh, grid_size=grid_size)

    paths = sorted(
        f
        for f in os.listdir(reconstruction_dir)
        if f.endswith("_vec.h5") and os.path.isfile(os.path.join(reconstruction_dir, f))
    )
    if not paths:
        raise FileNotFoundError(
            "No *_vec.h5 under {}. Run without --skip_reconstruct first.".format(reconstruction_dir)
        )

    rows: List[Dict[str, Any]] = []
    sum_gt_h = sum_pred_h = sum_gt_v = sum_pred_v = 0
    n_parse_fail_gt = n_parse_fail_pred = 0
    par_rec_hits_sum = par_rec_denom_sum = 0
    perp_rec_hits_sum = perp_rec_denom_sum = 0
    n_samples_extrude_count_mismatch = 0
    n_extrudes_skipped_line_mismatch_sum = 0

    for name in tqdm(paths, desc="metrics"):
        h5_path = os.path.join(reconstruction_dir, name)
        with h5py.File(h5_path, "r") as fp:
            gt_vec = fp["gt_vec"][:]
            out_vec = fp["out_vec"][:]

        gt_h, gt_v, n_lines_gt, ok_gt = _constraint_counts_from_vec(gt_vec, ex)
        pred_h, pred_v, n_lines_pred, ok_pred = _constraint_counts_from_vec(out_vec, ex)

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

        sample_id = name[: -len("_vec.h5")]
        ratio_h_i: Optional[float]
        ratio_v_i: Optional[float]
        if gt_h > 0:
            ratio_h_i = pred_h / gt_h
        else:
            ratio_h_i = None
        if gt_v > 0:
            ratio_v_i = pred_v / gt_v
        else:
            ratio_v_i = None

        ph, pd, qh, qd, pr_skip_ln, pr_ext_mis = parallel_perpendicular_recall_index_aligned(
            gt_vec, out_vec, ex
        )
        par_rec_hits_sum += ph
        par_rec_denom_sum += pd
        perp_rec_hits_sum += qh
        perp_rec_denom_sum += qd
        n_extrudes_skipped_line_mismatch_sum += pr_skip_ln
        if pr_ext_mis:
            n_samples_extrude_count_mismatch += 1

        par_recall_i = ph / pd if pd > 0 else None
        perp_recall_i = qh / qd if qd > 0 else None

        rows.append(
            {
                "id": sample_id,
                "n_lines_gt": n_lines_gt,
                "n_lines_pred": n_lines_pred,
                "n_line_cmds_gt": n_line_cmds_gt,
                "n_line_cmds_pred": n_line_cmds_pred,
                "gt_h": gt_h,
                "gt_v": gt_v,
                "pred_h": pred_h,
                "pred_v": pred_v,
                "ratio_h_sample": ratio_h_i,
                "ratio_v_sample": ratio_v_i,
                "par_recall_hits": ph,
                "par_recall_denom": pd,
                "par_recall_index_aligned": par_recall_i,
                "perp_recall_hits": qh,
                "perp_recall_denom": qd,
                "perp_recall_index_aligned": perp_recall_i,
                "pair_recall_extrude_count_mismatch": pr_ext_mis,
                "pair_recall_extrudes_skipped_line_mismatch": pr_skip_ln,
                "parse_ok_gt": ok_gt,
                "parse_ok_pred": ok_pred,
            }
        )

    summary: Dict[str, Any] = {
        "n_samples": len(rows),
        "sum_gt_h": sum_gt_h,
        "sum_pred_h": sum_pred_h,
        "sum_gt_v": sum_gt_v,
        "sum_pred_v": sum_pred_v,
        "angle_thresh_deg": angle_thresh,
        "dist_thresh": ex.dist_thresh,
        "grid_size": grid_size,
        "n_parse_fail_gt": n_parse_fail_gt,
        "n_parse_fail_pred": n_parse_fail_pred,
    }

    if sum_gt_h > 0:
        summary["ratio_h"] = sum_pred_h / sum_gt_h
    else:
        summary["ratio_h"] = None
        summary["ratio_h_defined"] = False

    if sum_gt_v > 0:
        summary["ratio_v"] = sum_pred_v / sum_gt_v
    else:
        summary["ratio_v"] = None
        summary["ratio_v_defined"] = False

    summary["parallel_recall_index_aligned_hits"] = par_rec_hits_sum
    summary["parallel_recall_index_aligned_denominator"] = par_rec_denom_sum
    if par_rec_denom_sum > 0:
        summary["parallel_recall_index_aligned"] = par_rec_hits_sum / par_rec_denom_sum
    else:
        summary["parallel_recall_index_aligned"] = None
        summary["parallel_recall_index_aligned_defined"] = False

    summary["perpendicular_recall_index_aligned_hits"] = perp_rec_hits_sum
    summary["perpendicular_recall_index_aligned_denominator"] = perp_rec_denom_sum
    if perp_rec_denom_sum > 0:
        summary["perpendicular_recall_index_aligned"] = perp_rec_hits_sum / perp_rec_denom_sum
    else:
        summary["perpendicular_recall_index_aligned"] = None
        summary["perpendicular_recall_index_aligned_defined"] = False

    summary["n_samples_extrude_count_mismatch"] = n_samples_extrude_count_mismatch
    summary["n_extrudes_skipped_line_count_mismatch_total"] = n_extrudes_skipped_line_mismatch_sum
    summary["pair_recall_index_aligned_note"] = (
        "Per extrude k up to min(n_gt_ext,n_pred_ext); only extrudes with equal GT/pred line counts; "
        "GT parallel pairs: pred undirected angle < angle_thresh; "
        "GT perpendicular pairs: |undirected angle - 90°| < angle_thresh; same line indices (i,j)."
    )

    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepCAD test-set: R_h/R_v + parallel & perpendicular index-aligned recall."
    )
    parser.add_argument("--proj_dir", type=str, default="proj_log")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--exp_name", type=str, default="newDeepCAD")
    parser.add_argument("--ckpt", type=str, default="latest")
    parser.add_argument("-g", "--gpu_ids", type=str, default="0")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(_REPO_ROOT, "论文尝试", "DeepCAD原始约束指标"),
        help="Output directory for reconstruction/, CSV, summary (default: this folder under repo).",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        help="Dataset split used for reconstruction/evaluation, e.g. train, validation, or test.",
    )
    parser.add_argument(
        "--reconstruction_dir",
        type=str,
        default=None,
        help="Folder with *_vec.h5 (default: <out_dir>/reconstruction). Use with --skip_reconstruct to read from another path.",
    )
    parser.add_argument(
        "--skip_reconstruct",
        action="store_true",
        help="Only compute metrics from existing *_vec.h5 under reconstruction/.",
    )
    parser.add_argument(
        "--reconstruct_only",
        action="store_true",
        help="Only run AE reconstruction; skip CSV/summary.",
    )
    parser.add_argument(
        "--angle_thresh",
        type=float,
        default=ANGLE_THRESH,
        help="Degrees; same as ConstraintExtractor default unless overridden.",
    )
    parser.add_argument("--grid_size", type=int, default=256)
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    recon_dir = (
        os.path.abspath(args.reconstruction_dir)
        if args.reconstruction_dir
        else os.path.join(out_dir, "reconstruction")
    )
    csv_path = os.path.join(out_dir, "per_sample_counts.csv")
    summary_path = os.path.join(out_dir, "summary.json")

    ensure_dir(out_dir)

    if not args.skip_reconstruct:
        cfg = _build_config_ae_for_rec(
            proj_dir=args.proj_dir,
            data_root=os.path.abspath(args.data_root),
            exp_name=args.exp_name,
            ckpt=args.ckpt,
            gpu_ids=args.gpu_ids,
            outputs=recon_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            eval_split=args.eval_split,
        )
        ensure_dir(recon_dir)
        reconstruct_eval_split(cfg)

    if args.reconstruct_only:
        print("reconstruct_only: done. Outputs in", recon_dir)
        return

    rows, summary = aggregate_metrics(recon_dir, args.angle_thresh, args.grid_size)

    model_dir = os.path.join(os.path.abspath(args.proj_dir), args.exp_name, "model")
    ckpt_name = args.ckpt if args.ckpt == "latest" else "ckpt_epoch{}".format(args.ckpt)
    load_path = os.path.join(model_dir, "{}.pth".format(ckpt_name))
    summary["checkpoint_path"] = load_path
    summary["data_root"] = os.path.abspath(args.data_root)
    summary["exp_name"] = args.exp_name
    summary["proj_dir"] = os.path.abspath(args.proj_dir)
    summary["eval_split"] = args.eval_split

    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Wrote", csv_path)
    print("Wrote", summary_path)
    print(
        "R_h = sum_pred_h / sum_gt_h = {} / {} = {}".format(
            summary["sum_pred_h"], summary["sum_gt_h"], summary.get("ratio_h")
        )
    )
    print(
        "R_v = sum_pred_v / sum_gt_v = {} / {} = {}".format(
            summary["sum_pred_v"], summary["sum_gt_v"], summary.get("ratio_v")
        )
    )
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
