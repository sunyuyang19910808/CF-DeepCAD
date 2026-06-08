from __future__ import annotations

import argparse
import json
import os
import shutil

from cadlib.macro import ALL_COMMANDS, ARGS_DIM, MAX_N_CURVES, MAX_N_EXT, MAX_N_LOOPS, MAX_TOTAL_LEN, N_ARGS
from utils import ensure_dirs


class ConfigStep:
    """Configuration for original DeepCAD + positive-relation L_geom."""

    def __init__(self, phase: str):
        self.phase = phase
        self.is_train = phase == "train"
        self.set_configuration()
        _parser, args = self.parse()

        print("----Constraint-Fused DeepCAD Step Configuration-----")
        for key, value in args.__dict__.items():
            print("{0:32}".format(key), value)
            setattr(self, key, value)

        self.exp_dir = os.path.join(self.proj_dir, self.exp_name)
        if self.is_train and not self.cont and os.path.exists(self.exp_dir):
            if self.force_overwrite:
                shutil.rmtree(self.exp_dir)
            else:
                model_dir = os.path.join(self.exp_dir, "model")
                has_checkpoint = os.path.isdir(model_dir) and any(
                    name.endswith(".pth") for name in os.listdir(model_dir)
                )
                if has_checkpoint:
                    raise FileExistsError(
                        "Experiment directory already exists: {}. Pass --force_overwrite or --continue.".format(
                            self.exp_dir
                        )
                    )

        self.log_dir = os.path.join(self.exp_dir, "log")
        self.model_dir = os.path.join(self.exp_dir, "model")
        self.artifact_dir = os.path.join(self.exp_dir, "artifacts")
        self.default_eval_dir = os.path.join(self.artifact_dir, "{}_eval_latest".format(self.eval_split))
        self.default_reconstruction_dir = os.path.join(
            self.artifact_dir,
            "reconstruction_{}_latest".format(self.eval_split),
        )
        ensure_dirs([self.log_dir, self.model_dir, self.artifact_dir])

        if self.gpu_ids is not None and self.gpu_ids != "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_ids)

        if self.is_train:
            with open(os.path.join(self.exp_dir, "config.txt"), "w", encoding="utf-8") as file_obj:
                json.dump(args.__dict__, file_obj, indent=2, ensure_ascii=False)

    def set_configuration(self) -> None:
        self.args_dim = ARGS_DIM
        self.n_args = N_ARGS
        self.n_commands = len(ALL_COMMANDS)

        self.n_layers = 4
        self.n_layers_decode = 4
        self.n_heads = 8
        self.dim_feedforward = 512
        self.d_model = 256
        self.dropout = 0.1
        self.dim_z = 256
        self.use_group_emb = True

        self.max_n_ext = MAX_N_EXT
        self.max_n_loops = MAX_N_LOOPS
        self.max_n_curves = MAX_N_CURVES
        self.max_num_groups = 30
        self.max_total_len = MAX_TOTAL_LEN
        self.max_lines = MAX_TOTAL_LEN
        self.max_constraints = 128

        self.loss_weights = {
            "loss_cmd_weight": 1.0,
            "loss_args_weight": 2.0,
        }
        self.augment = False

    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--proj_dir", type=str, default="proj_log/constraint_fused_deepcad_step")
        parser.add_argument("--data_root", type=str, default="data")
        parser.add_argument("--exp_name", type=str, default="deepcad_step_s0_origin")
        parser.add_argument("-g", "--gpu_ids", type=str, default="0")
        parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            help="Use 0 with --dataset_cache disk to avoid duplicate cache builds.",
        )
        parser.add_argument("--nr_epochs", type=int, default=100)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--grad_clip", type=float, default=1.0)
        parser.add_argument("--warmup_step", type=int, default=2000)
        parser.add_argument("--save_frequency", type=int, default=5)
        parser.add_argument("--val_frequency", type=int, default=100)
        parser.add_argument("--log_frequency", type=int, default=10)
        parser.add_argument("--continue", dest="cont", action="store_true")
        parser.add_argument("--ckpt", type=str, default="latest")
        parser.add_argument("--force_overwrite", action="store_true")
        parser.add_argument("--max_steps", type=int, default=0, help="0 means full epoch training.")

        parser.add_argument("--angle_thresh", type=float, default=0.1)
        parser.add_argument("--dist_thresh", type=float, default=1e-3)
        parser.add_argument("--grid_size", type=int, default=256)
        parser.add_argument("--coord_range_min", type=float, default=-1.0)
        parser.add_argument("--coord_range_max", type=float, default=1.0)
        parser.add_argument("--eval_split", type=str, default="test")
        parser.add_argument(
            "--dataset_cache",
            type=str,
            default="off",
            choices=["off", "memory", "disk"],
            help="Cache parsed samples (incl. unary_gt/pair_gt). Long training should use disk.",
        )
        parser.add_argument(
            "--dataset_cache_dir",
            type=str,
            default="",
            help="Optional cache root override; default is {data_root}/.cache/deepcad_step/",
        )

        parser.add_argument("--enable_geom_loss", action="store_true")
        parser.add_argument(
            "--geom_log_only",
            action="store_true",
            help="Compute geom metrics but do not backprop L_geom (S1).",
        )
        parser.add_argument("--geom_positive_only", action="store_true", default=True)
        parser.add_argument("--gamma_geom", type=float, default=0.1)
        parser.add_argument(
            "--geom_target_ratio",
            type=float,
            default=0.0,
            help="If >0, adapt gamma_geom each step so geom_effective_ratio tracks this target "
            "(warmup uses geom_warmup_* epochs). gamma_geom becomes the upper cap.",
        )
        parser.add_argument(
            "--geom_ratio_ema",
            type=float,
            default=0.99,
            help="EMA decay for main/geom losses in adaptive gamma mode; 0 disables EMA.",
        )
        parser.add_argument(
            "--geom_gamma_min",
            type=float,
            default=1e-6,
            help="Lower clamp for adaptively resolved gamma_geom.",
        )
        parser.add_argument(
            "--geom_loss_mode",
            type=str,
            default="angle_hinge",
            choices=["bce", "angle_hinge"],
            help="bce=legacy soft-score BCE; angle_hinge=aligned with test angle_thresh recall.",
        )
        parser.add_argument("--geom_bce_scale", type=float, default=4.0)
        parser.add_argument("--geom_negative_weight", type=float, default=0.0)
        parser.add_argument("--geom_warmup_start_epoch", type=int, default=1)
        parser.add_argument("--geom_warmup_end_epoch", type=int, default=1)
        parser.add_argument("--use_corrected_line_start", dest="use_corrected_line_start", action="store_true")
        parser.add_argument("--no-use_corrected_line_start", dest="use_corrected_line_start", action="store_false")
        parser.set_defaults(use_corrected_line_start=True)
        parser.add_argument(
            "--legacy_line_start",
            action="store_true",
            help="Alias for --no-use_corrected_line_start.",
        )

        if not self.is_train:
            parser.add_argument("--model_path", type=str, default=None)
            parser.add_argument("--reconstruction_dir", type=str, default=None)
            parser.add_argument("--outputs", type=str, default=None, help="Constraint/ACC eval output dir.")
            parser.add_argument("--skip_reconstruct", action="store_true")
            parser.add_argument(
                "--sample_count",
                type=int,
                default=0,
                help="0 evaluates the full split; use a small value only for debugging.",
            )

        args = parser.parse_args()
        if args.legacy_line_start:
            args.use_corrected_line_start = False
        return parser, args
