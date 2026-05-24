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

from constraint_fused_deepcad_simplify_modify2.application.train_use_case import build_train_use_case
from constraint_fused_deepcad_simplify_modify2.config.config_constraint_fused_simplify_modify2 import (
    ConfigConstraintFusedSimplifyModify2,
)
from constraint_fused_deepcad_simplify_modify2.infrastructure.dataset_simplify_modify2 import (
    get_simplify_modify2_dataloader,
)
from constraint_fused_deepcad_simplify_modify2.sketch_preparation.constraint_extractor_simplify_modify2 import (
    ANGLE_THRESH,
    ConstraintExtractorSimplifyModify2,
    undirected_angle_deg,
)
from trainer.base import TrainClock
from utils import ensure_dir


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


def first_eos_length(command_seq: np.ndarray) -> int:
    eos_hits = np.flatnonzero(command_seq == EOS_IDX)
    return int(eos_hits[0]) if eos_hits.size > 0 else int(command_seq.shape[0])


def build_reconstruction_use_case(cfg):
    device = resolve_device(cfg)
    use_case = build_train_use_case(cfg, device=device)
    load_checkpoint(use_case, resolve_checkpoint_path(cfg), device)
    use_case.eval()
    return use_case, device


def reconstruct_batch(use_case, batch, device: torch.device):
    with torch.no_grad():
        latent, _encoder_outputs = use_case.fusion_service.fuse(
            commands=batch["command"].to(device).transpose(0, 1),
            args=batch["args"].to(device).transpose(0, 1),
            groups=batch["groups"].to(device).transpose(0, 1),
            constraint_tags=batch["constraint_tags"].to(device).transpose(0, 1),
            c_types=batch["c_types"].to(device).transpose(0, 1),
            c_line_a=batch["c_line_a"].to(device).transpose(0, 1),
            c_line_b=batch["c_line_b"].to(device).transpose(0, 1),
            cmd_padding_mask=batch["cmd_padding_mask"].to(device),
            constraint_padding_mask=batch["constraint_padding_mask"].to(device),
        )
        decoded = use_case.decoder(latent.tensor)
        out_vec = logits_to_vec(decoded["command_logits"], decoded["args_logits"]).detach().cpu().numpy()

    gt_vec = torch.cat([batch["command"].unsqueeze(-1), batch["args"]], dim=-1).detach().cpu().numpy()
    return out_vec, gt_vec


def reconstruct_test_split(cfg, reconstruction_dir: str) -> None:
    eval_split = getattr(cfg, "eval_split", "test")
    loader = get_simplify_modify2_dataloader(eval_split, cfg, shuffle=False)
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


class EvaluateConstraintSatisfactionUseCase:
    def __init__(self, angle_thresh: float = ANGLE_THRESH, grid_size: int = 256):
        self.extractor = ConstraintExtractorSimplifyModify2(angle_thresh=angle_thresh, grid_size=grid_size)

    def _lines_in_extrude(self, ext) -> List[Line]:
        lines: List[Line] = []
        for loop in ext.profile.children:
            for curve in loop.children:
                if isinstance(curve, Line):
                    lines.append(curve)
        return lines

    def _unit_dir2d(self, line: Line) -> np.ndarray:
        direction = line.direction(from_start=True).astype(np.float64)
        norm = float(np.linalg.norm(direction[:2]) + 1e-12)
        return direction[:2] / norm

    def _line_count_from_commands(self, commands: np.ndarray) -> int:
        return int(np.sum(commands == LINE_IDX))

    def _axis_counts_from_vec(self, cad_vec: np.ndarray) -> Tuple[int, int, int, bool]:
        try:
            cad_seq = CADSequence.from_vector(cad_vec, is_numerical=True, n=self.extractor.grid_size)
        except Exception:
            return (0, 0, 0, False)
        lines = self.extractor.collect_lines_from_cad_sequence(cad_seq)
        raw = self.extractor.extract_raw_from_lines(lines)
        return len(raw.horizontal), len(raw.vertical), len(lines), True

    def parallel_perpendicular_recall_index_aligned(
        self, gt_vec: np.ndarray, pred_vec: np.ndarray
    ) -> Tuple[int, int, int, int, int, bool]:
        try:
            gt_seq = CADSequence.from_vector(gt_vec, is_numerical=True, n=self.extractor.grid_size)
            pred_seq = CADSequence.from_vector(pred_vec, is_numerical=True, n=self.extractor.grid_size)
        except Exception:
            return 0, 0, 0, 0, 0, False

        ext_mismatch = len(gt_seq.seq) != len(pred_seq.seq)
        par_hits = 0
        par_denom = 0
        perp_hits = 0
        perp_denom = 0
        skipped_line_mismatch = 0
        angle_thresh = self.extractor.angle_thresh

        for extrude_idx in range(min(len(gt_seq.seq), len(pred_seq.seq))):
            gt_lines = self._lines_in_extrude(gt_seq.seq[extrude_idx])
            pred_lines = self._lines_in_extrude(pred_seq.seq[extrude_idx])
            if len(gt_lines) != len(pred_lines):
                skipped_line_mismatch += 1
                continue
            if len(gt_lines) < 2:
                continue

            raw_gt = self.extractor.extract_raw_from_lines(gt_lines)
            if not raw_gt.parallel and not raw_gt.perpendicular:
                continue

            pred_dirs = [self._unit_dir2d(line) for line in pred_lines]
            for left, right in raw_gt.parallel:
                par_denom += 1
                if undirected_angle_deg(pred_dirs[left], pred_dirs[right]) < angle_thresh:
                    par_hits += 1

            for left, right in raw_gt.perpendicular:
                perp_denom += 1
                angle = undirected_angle_deg(pred_dirs[left], pred_dirs[right])
                if abs(angle - 90.0) < angle_thresh:
                    perp_hits += 1

        return par_hits, par_denom, perp_hits, perp_denom, skipped_line_mismatch, ext_mismatch

    def aggregate_metrics(self, reconstruction_dir: str) -> tuple[list[dict], dict]:
        paths = sorted(
            file_name
            for file_name in os.listdir(reconstruction_dir)
            if file_name.endswith("_vec.h5") and os.path.isfile(os.path.join(reconstruction_dir, file_name))
        )
        if not paths:
            raise FileNotFoundError("No *_vec.h5 found under {}.".format(reconstruction_dir))

        rows: List[Dict[str, Any]] = []
        sum_gt_h = 0
        sum_pred_h = 0
        sum_gt_v = 0
        sum_pred_v = 0
        parse_fail_gt = 0
        parse_fail_pred = 0
        par_rec_hits_sum = 0
        par_rec_denom_sum = 0
        perp_rec_hits_sum = 0
        perp_rec_denom_sum = 0
        n_samples_extrude_count_mismatch = 0
        n_extrudes_skipped_line_mismatch_sum = 0

        for file_name in tqdm(paths, desc="metrics"):
            with h5py.File(os.path.join(reconstruction_dir, file_name), "r") as file_obj:
                gt_vec = file_obj["gt_vec"][:]
                out_vec = file_obj["out_vec"][:]

            gt_h, gt_v, n_lines_gt, ok_gt = self._axis_counts_from_vec(gt_vec)
            pred_h, pred_v, n_lines_pred, ok_pred = self._axis_counts_from_vec(out_vec)
            if not ok_gt:
                parse_fail_gt += 1
            if not ok_pred:
                parse_fail_pred += 1

            sum_gt_h += gt_h
            sum_pred_h += pred_h
            sum_gt_v += gt_v
            sum_pred_v += pred_v

            n_line_cmds_gt = self._line_count_from_commands(gt_vec[:, 0])
            n_line_cmds_pred = self._line_count_from_commands(out_vec[:, 0])
            sample_id = file_name[: -len("_vec.h5")]
            ratio_h_sample = pred_h / gt_h if gt_h > 0 else None
            ratio_v_sample = pred_v / gt_v if gt_v > 0 else None
            par_hits, par_denom, perp_hits, perp_denom, skipped_line_mismatch, ext_mismatch = (
                self.parallel_perpendicular_recall_index_aligned(gt_vec, out_vec)
            )
            par_rec_hits_sum += par_hits
            par_rec_denom_sum += par_denom
            perp_rec_hits_sum += perp_hits
            perp_rec_denom_sum += perp_denom
            n_extrudes_skipped_line_mismatch_sum += skipped_line_mismatch
            if ext_mismatch:
                n_samples_extrude_count_mismatch += 1

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
                    "ratio_h_sample": ratio_h_sample,
                    "ratio_v_sample": ratio_v_sample,
                    "par_recall_hits": par_hits,
                    "par_recall_denom": par_denom,
                    "par_recall_index_aligned": (par_hits / par_denom) if par_denom > 0 else None,
                    "perp_recall_hits": perp_hits,
                    "perp_recall_denom": perp_denom,
                    "perp_recall_index_aligned": (perp_hits / perp_denom) if perp_denom > 0 else None,
                    "pair_recall_extrude_count_mismatch": ext_mismatch,
                    "pair_recall_extrudes_skipped_line_mismatch": skipped_line_mismatch,
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
            "angle_thresh_deg": self.extractor.angle_thresh,
            "dist_thresh": self.extractor.dist_thresh,
            "grid_size": self.extractor.grid_size,
            "n_parse_fail_gt": parse_fail_gt,
            "n_parse_fail_pred": parse_fail_pred,
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


def main():
    cfg = ConfigConstraintFusedSimplifyModify2("test")
    out_dir = os.path.abspath(cfg.outputs or cfg.package_root)
    reconstruction_dir = os.path.abspath(cfg.reconstruction_dir or cfg.default_reconstruction_dir)
    ensure_dir(out_dir)

    if not cfg.skip_reconstruct:
        reconstruct_test_split(cfg, reconstruction_dir)

    evaluator = EvaluateConstraintSatisfactionUseCase(angle_thresh=cfg.angle_thresh, grid_size=cfg.grid_size)
    rows, summary = evaluator.aggregate_metrics(reconstruction_dir)
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


if __name__ == "__main__":
    main()

