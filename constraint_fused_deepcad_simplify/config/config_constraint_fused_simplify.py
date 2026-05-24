from __future__ import annotations

import argparse
import json
import os
import shutil

from cadlib.macro import ALL_COMMANDS, ARGS_DIM, MAX_N_CURVES, MAX_N_EXT, MAX_N_LOOPS, MAX_TOTAL_LEN, N_ARGS
from utils import ensure_dirs


class ConfigConstraintFusedSimplify:
    def __init__(self, phase: str):
        self.phase = phase
        self.is_train = phase == "train"
        self.set_configuration()
        parser, args = self.parse()

        print("----Constraint-Fused Simplify Configuration-----")
        for key, value in args.__dict__.items():
            print("{0:24}".format(key), value)
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
        self.package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        self.default_reconstruction_dir = os.path.join(self.package_root, "reconstruction")
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

        self.loss_weights = {
            "loss_cmd_weight": 1.0,
            "loss_args_weight": 2.0,
        }

    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--proj_dir", type=str, default="proj_log/constraint_fused_deepcad_simplify")
        parser.add_argument("--data_root", type=str, default="data")
        parser.add_argument("--exp_name", type=str, default="cf_simplify")
        parser.add_argument("-g", "--gpu_ids", type=str, default="0")
        parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--nr_epochs", type=int, default=50)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--grad_clip", type=float, default=1.0)
        parser.add_argument("--warmup_step", type=int, default=2000)
        parser.add_argument("--save_frequency", type=int, default=5)
        parser.add_argument("--val_frequency", type=int, default=100)
        parser.add_argument("--log_frequency", type=int, default=10)
        parser.add_argument("--augment", action="store_true")
        parser.add_argument("--continue", dest="cont", action="store_true")
        parser.add_argument("--ckpt", type=str, default="latest")
        parser.add_argument("--force_overwrite", action="store_true")
        parser.add_argument("--max_steps", type=int, default=0, help="0 means full epoch training.")

        parser.add_argument("--axis_loss_weight", type=float, default=0.5)
        parser.add_argument("--axis_pos_weight", type=float, default=5.0)
        parser.add_argument("--angle_thresh", type=float, default=0.1)
        parser.add_argument("--grid_size", type=int, default=256)

        if not self.is_train:
            parser.add_argument("--reconstruction_dir", type=str, default=None)
            parser.add_argument("--outputs", type=str, default=None)
            parser.add_argument("--skip_reconstruct", action="store_true")

        args = parser.parse_args()
        return parser, args
