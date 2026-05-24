import math
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch

from cadlib.curves import Line
from cadlib.macro import EOS_IDX, LINE_IDX, N_ARGS

from constraint_fused_deepcad_simplify.application.evaluate_axis_constraints import aggregate_metrics
from constraint_fused_deepcad_simplify.application.reconstruct_test_split import (
    reconstruction_file_stem,
    resolve_checkpoint_path,
)
from constraint_fused_deepcad_simplify.application.train_use_case import build_train_use_case
from constraint_fused_deepcad_simplify.domain.entities import ConstraintRelationSimplify, ConstraintTypeSimplify
from constraint_fused_deepcad_simplify.infrastructure.dataset_simplify import CADDatasetSimplify
from constraint_fused_deepcad_simplify.sketch_preparation.batch_assembler_simplify import (
    ConstraintBatchAssemblerSimplify,
    build_constraint_tags,
)
from constraint_fused_deepcad_simplify.sketch_preparation.constraint_extractor_simplify import (
    ConstraintExtractorSimplify,
)


class ConstraintFusedDeepCadSimplifyTests(unittest.TestCase):
    def test_constraint_extractor_finds_horizontal_and_vertical(self):
        extractor = ConstraintExtractorSimplify(angle_thresh=0.1)
        lines = [
            Line(start_point=torch.tensor([0.0, 0.0]).numpy(), end_point=torch.tensor([3.0, 0.0]).numpy()),
            Line(start_point=torch.tensor([1.0, 1.0]).numpy(), end_point=torch.tensor([1.0, 4.0]).numpy()),
            Line(start_point=torch.tensor([0.0, 0.0]).numpy(), end_point=torch.tensor([2.0, 2.0]).numpy()),
        ]
        raw = extractor.extract_raw_from_lines(lines)
        self.assertEqual(raw.horizontal, [0])
        self.assertEqual(raw.vertical, [1])

    def test_batch_assembler_rejects_invalid_line_idx(self):
        assembler = ConstraintBatchAssemblerSimplify(max_lines=4, seq_len=4)
        cad_vec = torch.tensor(
            [
                [LINE_IDX] + [0] * N_ARGS,
                [EOS_IDX] + [-1] * N_ARGS,
                [EOS_IDX] + [-1] * N_ARGS,
                [EOS_IDX] + [-1] * N_ARGS,
            ],
            dtype=torch.long,
        ).numpy()
        with self.assertRaises(ValueError):
            assembler.assemble_from_vec(
                cad_vec=cad_vec,
                relations=[ConstraintRelationSimplify(ConstraintTypeSimplify.HORIZONTAL, 1)],
                geometry_line_count=1,
            )

    def test_constraint_tags_only_mark_line_positions(self):
        commands = torch.tensor([LINE_IDX, EOS_IDX, LINE_IDX, EOS_IDX], dtype=torch.long).numpy()
        tags = build_constraint_tags(
            4,
            commands,
            [
                ConstraintRelationSimplify(ConstraintTypeSimplify.HORIZONTAL, 0),
                ConstraintRelationSimplify(ConstraintTypeSimplify.VERTICAL, 1),
            ],
        )
        self.assertTrue(torch.equal(tags[0], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.equal(tags[2], torch.tensor([0.0, 1.0])))
        self.assertTrue(torch.equal(tags[1], torch.tensor([0.0, 0.0])))

    def test_train_use_case_runs_single_batch(self):
        cfg = SimpleNamespace(
            n_commands=6,
            args_dim=255,
            n_args=N_ARGS,
            d_model=32,
            dim_z=32,
            n_heads=8,
            dim_feedforward=64,
            dropout=0.1,
            n_layers=1,
            n_layers_decode=1,
            use_group_emb=True,
            max_total_len=6,
            max_num_groups=4,
            max_lines=4,
            loss_weights={"loss_cmd_weight": 1.0, "loss_args_weight": 2.0},
            axis_loss_weight=0.5,
            axis_pos_weight=5.0,
        )
        use_case = build_train_use_case(cfg, device=torch.device("cpu"))
        batch = {
            "command": torch.tensor([[LINE_IDX, LINE_IDX, EOS_IDX, EOS_IDX, EOS_IDX, EOS_IDX]], dtype=torch.long),
            "args": torch.zeros(1, 6, N_ARGS, dtype=torch.long),
            "groups": torch.zeros(1, 6, dtype=torch.long),
            "constraint_tags": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]],
                dtype=torch.float32,
            ),
            "unary_gt": torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32),
            "cmd_padding_mask": torch.tensor([[False, False, True, True, True, True]]),
            "line_count": torch.tensor([2], dtype=torch.long),
        }
        out = use_case.execute(batch)
        self.assertEqual(tuple(out["unary_pred"].shape), (1, 4, 2))
        self.assertTrue(math.isfinite(float(out["loss"].item())))
        self.assertTrue(math.isfinite(float(out["axis_loss"].item())))

    def test_evaluate_axis_metrics_script_logic(self):
        gt_vec = np.zeros((2, 1 + N_ARGS), dtype=np.int64)
        out_vec = np.ones((2, 1 + N_ARGS), dtype=np.int64)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "sample_vec.h5")
            with h5py.File(path, "w") as file_obj:
                file_obj.create_dataset("gt_vec", data=gt_vec, dtype="i8")
                file_obj.create_dataset("out_vec", data=out_vec, dtype="i8")

            side_effect = [
                (SimpleNamespace(horizontal=[0, 2], vertical=[1]), [], []),
                (SimpleNamespace(horizontal=[0, 2], vertical=[1]), [], []),
            ]
            with patch(
                "constraint_fused_deepcad_simplify.sketch_preparation.constraint_extractor_simplify.ConstraintExtractorSimplify.extract_from_cad_vec",
                side_effect=side_effect,
            ):
                rows, summary = aggregate_metrics(temp_dir, angle_thresh=0.1, grid_size=256)
        self.assertEqual(len(rows), 1)
        self.assertEqual(summary["R_h"], 1.0)
        self.assertEqual(summary["R_v"], 1.0)

    def test_dataset_skips_corrupted_sample_and_falls_back(self):
        dataset = CADDatasetSimplify.__new__(CADDatasetSimplify)
        dataset.phase = "train"
        dataset.all_data = ["bad_sample", "good_sample"]
        dataset.max_total_len = 6
        dataset._warned_bad_samples = set()

        class Repo:
            def load_cad_vec(self, data_id):
                if data_id == "bad_sample":
                    raise OSError("truncated file")
                return np.zeros((2, 1 + N_ARGS), dtype=np.int64)

        dataset.repository = Repo()

        data_id, parse_source = dataset._load_parse_source(0)
        self.assertEqual(data_id, "good_sample")
        self.assertEqual(tuple(parse_source.shape), (2, 1 + N_ARGS))

    def test_reconstruction_file_stem_matches_test_py_behavior(self):
        self.assertEqual(reconstruction_file_stem("0001/00000093"), "00000093")
        self.assertEqual(reconstruction_file_stem(r"0001\00000093"), "00000093")

    def test_resolve_checkpoint_path_prefers_explicit_model_path(self):
        cfg = SimpleNamespace(
            model_path=r"proj_log/constraint_fused_deepcad_simplify/cf_simplify/model/latest.pth",
            ckpt="latest",
            model_dir=r"proj_log/constraint_fused_deepcad_simplify/cf_simplify/model",
        )
        path = resolve_checkpoint_path(cfg)
        self.assertTrue(path.endswith(os.path.join("proj_log", "constraint_fused_deepcad_simplify", "cf_simplify", "model", "latest.pth")))


if __name__ == "__main__":
    unittest.main()
