from __future__ import annotations

import argparse
import json
import os
import shutil

from cadlib.macro import ALL_COMMANDS, ARGS_DIM, MAX_N_CURVES, MAX_N_EXT, MAX_N_LOOPS, MAX_TOTAL_LEN, N_ARGS
from utils import ensure_dirs


class ConfigConstraintFusedHighModify:
    def __init__(self, phase: str):
        self.phase = phase
        self.is_train = phase == "train"
        self.set_configuration()
        _parser, args = self.parse()

        print("----Constraint-Fused High Modify Configuration-----")
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
        self.package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
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
        self.dim_z = 512
        self.pooled_dim = 512
        self.use_group_emb = True

        self.max_n_ext = MAX_N_EXT
        self.max_n_loops = MAX_N_LOOPS
        self.max_n_curves = MAX_N_CURVES
        self.max_num_groups = 30
        self.max_total_len = MAX_TOTAL_LEN
        self.max_lines = MAX_TOTAL_LEN
        self.max_constraints = 128

        self.n_tag_bits = 4
        self.n_constraint_types = 5
        self.constraint_pred_dim = 4

        self.loss_weights = {
            "loss_cmd_weight": 1.0,
            "loss_args_weight": 2.0,
        }

    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--proj_dir", type=str, default="proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify")
        parser.add_argument("--data_root", type=str, default="data")
        parser.add_argument("--exp_name", type=str, default="cf_high_modify")
        parser.add_argument("-g", "--gpu_ids", type=str, default="0")
        parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--nr_epochs", type=int, default=100)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--grad_clip", type=float, default=1.0)
        parser.add_argument("--warmup_step", type=int, default=2000)
        parser.add_argument("--save_frequency", type=int, default=5)
        parser.add_argument("--val_frequency", type=int, default=100)
        parser.add_argument("--log_frequency", type=int, default=10)
        parser.add_argument("--continue", dest="cont", action="store_true")
        parser.add_argument(
            "--init_weights_from",
            type=str,
            default="",
            help="Load model_state_dict only from this checkpoint; does not restore optimizer or epoch.",
        )
        parser.add_argument("--ckpt", type=str, default="latest")
        parser.add_argument("--force_overwrite", action="store_true")
        parser.add_argument("--max_steps", type=int, default=0, help="0 means full epoch training.")

        parser.add_argument("--alpha", type=float, default=3.0)
        parser.add_argument("--beta", type=float, default=1.0)
        parser.add_argument("--gamma", type=float, default=3.0)
        parser.add_argument(
            "--gamma_horizontal",
            type=float,
            default=None,
            help="Per-component gamma override for horizontal residual (A2d). "
            "Defaults to --gamma when not set, preserving legacy behaviour.",
        )
        parser.add_argument(
            "--gamma_vertical",
            type=float,
            default=None,
            help="Per-component gamma override for vertical residual (A2d).",
        )
        parser.add_argument(
            "--gamma_parallel",
            type=float,
            default=None,
            help="Per-component gamma override for parallel residual (A2d, recommended 3.0).",
        )
        parser.add_argument(
            "--gamma_perpendicular",
            type=float,
            default=None,
            help="Per-component gamma override for perpendicular residual (A2d).",
        )
        parser.add_argument(
            "--use_corrected_line_start",
            action="store_true",
            help="A2d: derive unit = (end - prev_curve_end) / ||.|| with SOL boundary reset. "
            "Default off keeps the legacy unit = (args[:2] - args[2:4]) behaviour.",
        )
        parser.add_argument(
            "--use_hard_geom_bce",
            action="store_true",
            help="A2d: replace single-sided geom soft residual with bidirectional hard BCE. "
            "Default off keeps the legacy soft path.",
        )
        parser.add_argument(
            "--hard_geom_bce_scale",
            type=float,
            default=6.0,
            help="Scale factor mapping the satisfaction score in [0,1] into a BCE logit.",
        )
        parser.add_argument(
            "--hard_geom_pos_weight",
            type=float,
            default=5.0,
            help="pos_weight argument for the hard geom BCE (default mirrors recon_loss).",
        )
        parser.add_argument("--aux_schedule", type=str, default="constant", choices=["constant", "warmup"])
        parser.add_argument("--aux_warmup_start_epoch", type=int, default=10)
        parser.add_argument("--aux_warmup_end_epoch", type=int, default=30)
        parser.add_argument("--pos_weight", type=float, default=5.0)
        parser.add_argument("--angle_thresh", type=float, default=0.1)
        parser.add_argument("--dist_thresh", type=float, default=1e-3)
        parser.add_argument("--grid_size", type=int, default=256)
        parser.add_argument("--coord_range_min", type=float, default=-1.0)
        parser.add_argument("--coord_range_max", type=float, default=1.0)
        parser.add_argument("--dim_z", type=int, default=self.dim_z)
        parser.add_argument("--pooled_dim", type=int, default=self.pooled_dim)
        parser.add_argument(
            "--pooling_strategy",
            type=str,
            default="segment_separated",
            choices=["segment_separated", "masked_mean", "masked_mean_plain"],
        )
        parser.add_argument(
            "--bottleneck_type",
            type=str,
            default="layernorm_tanh",
            choices=["layernorm_tanh", "deepcad_tanh"],
        )
        parser.add_argument("--recon_input", type=str, default="decoder_hidden", choices=["decoder_hidden"])
        parser.add_argument("--disable_soft_geometry", dest="enable_soft_geometry", action="store_false")
        parser.add_argument("--eval_split", type=str, default="test")
        parser.add_argument("--enable_line_pair_scorer", dest="enable_line_pair_scorer", action="store_true")
        parser.add_argument("--disable_line_pair_scorer", dest="enable_line_pair_scorer", action="store_false")
        parser.add_argument("--line_pair_hidden_dim", type=int, default=256)
        parser.add_argument("--line_only_pred_loss", dest="line_only_pred_loss", action="store_true")
        parser.add_argument("--disable_line_only_pred_loss", dest="line_only_pred_loss", action="store_false")
        parser.set_defaults(
            enable_soft_geometry=True,
            enable_line_pair_scorer=True,
            line_only_pred_loss=True,
        )

        if not self.is_train:
            parser.add_argument("--model_path", type=str, default=None)
            parser.add_argument("--reconstruction_dir", type=str, default=None)
            parser.add_argument("--outputs", type=str, default=None)
            parser.add_argument("--skip_reconstruct", action="store_true")
            parser.add_argument("--sample_count", type=int, default=0, help="0 evaluates the full split; use a small value only for debugging.")

        args = parser.parse_args()
        return parser, args
