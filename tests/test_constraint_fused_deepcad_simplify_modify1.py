import math
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
import h5py
import numpy as np

from cadlib.macro import EOS_IDX, LINE_IDX, N_ARGS

from constraint_fused_deepcad_simplify_modify1.application.differentiable_sketch_interpreter import (
    DifferentiableSketchInterpreter,
)
from constraint_fused_deepcad_simplify_modify1.application.evaluate_axis_constraints import (
    aggregate_metrics,
    parallel_perpendicular_recall_index_aligned,
    reconstruction_file_stem,
    resolve_checkpoint_path,
)
from constraint_fused_deepcad_simplify_modify1.application.geometry_constraint import ConstraintEvaluator
from constraint_fused_deepcad_simplify_modify1.application.loss_composer import (
    build_line_mask,
    compose_loss,
    constraint_pred_loss,
)
from constraint_fused_deepcad_simplify_modify1.application.train_use_case import build_train_use_case
from constraint_fused_deepcad_simplify_modify1.domain.entities import (
    ConstraintRelationSimplifyModify1,
    ConstraintTypeSimplifyModify1,
)
from constraint_fused_deepcad_simplify_modify1.infrastructure.dataset_simplify_modify1 import (
    fused_simplify_modify1_collate_fn,
)
from constraint_fused_deepcad_simplify_modify1.sketch_preparation.batch_assembler_simplify_modify1 import (
    ConstraintBatchAssemblerSimplifyModify1,
    build_constraint_tags,
)


class ConstraintFusedDeepCadSimplifyModify1Tests(unittest.TestCase):
    def test_batch_assembler_exposes_modify1_fields(self):
        assembler = ConstraintBatchAssemblerSimplifyModify1(max_lines=4, seq_len=4)
        cad_vec = torch.tensor(
            [
                [LINE_IDX] + [0] * N_ARGS,
                [EOS_IDX] + [-1] * N_ARGS,
                [EOS_IDX] + [-1] * N_ARGS,
                [EOS_IDX] + [-1] * N_ARGS,
            ],
            dtype=torch.long,
        ).numpy()
        aggregate = assembler.assemble_from_vec(
            cad_vec=cad_vec,
            relations=[ConstraintRelationSimplifyModify1(ConstraintTypeSimplifyModify1.HORIZONTAL, 0)],
            geometry_line_count=1,
        )
        self.assertEqual(tuple(aggregate.constraint_tags.shape), (4, 2))
        self.assertEqual(tuple(aggregate.unary_gt.shape), (4, 2))
        self.assertEqual(tuple(aggregate.line_cmd_mask.shape), (4,))
        self.assertEqual(tuple(aggregate.line_index_map.shape), (4,))
        self.assertTrue(bool(aggregate.line_cmd_mask[0]))
        self.assertEqual(int(aggregate.line_index_map[0].item()), 0)

    def test_constraint_tags_match_line_positions(self):
        commands = torch.tensor([LINE_IDX, EOS_IDX, LINE_IDX, EOS_IDX], dtype=torch.long).numpy()
        tags = build_constraint_tags(
            4,
            commands,
            [
                ConstraintRelationSimplifyModify1(ConstraintTypeSimplifyModify1.HORIZONTAL, 0),
                ConstraintRelationSimplifyModify1(ConstraintTypeSimplifyModify1.VERTICAL, 1),
            ],
        )
        self.assertTrue(torch.equal(tags[0], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.equal(tags[2], torch.tensor([0.0, 1.0])))
        self.assertTrue(torch.equal(tags[1], torch.tensor([0.0, 0.0])))

    def test_collate_keeps_modify1_fields(self):
        sample = {
            "command": torch.tensor([LINE_IDX, EOS_IDX, EOS_IDX], dtype=torch.long),
            "args": torch.zeros(3, N_ARGS, dtype=torch.long),
            "groups": torch.zeros(3, dtype=torch.long),
            "constraint_tags": torch.zeros(3, 2, dtype=torch.float32),
            "unary_gt": torch.zeros(3, 2, dtype=torch.float32),
            "cmd_padding_mask": torch.tensor([False, True, True]),
            "line_count": torch.tensor(1, dtype=torch.long),
            "line_cmd_mask": torch.tensor([True, False, False]),
            "line_index_map": torch.tensor([0, -1, -1], dtype=torch.long),
            "id": "sample_1",
        }
        batch = fused_simplify_modify1_collate_fn([sample])
        self.assertIn("line_cmd_mask", batch)
        self.assertIn("line_index_map", batch)
        self.assertEqual(tuple(batch["line_cmd_mask"].shape), (1, 3))
        self.assertEqual(tuple(batch["line_index_map"].shape), (1, 3))

    def test_constraint_pred_loss_masks_padding(self):
        logits = torch.tensor([[[2.0, -2.0], [3.0, -3.0]]], dtype=torch.float32)
        targets = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
        cmd_padding_mask = torch.tensor([[False, True]])
        loss = constraint_pred_loss(logits, targets, cmd_padding_mask)
        self.assertTrue(math.isfinite(float(loss.item())))
        self.assertGreaterEqual(float(loss.item()), 0.0)

    def test_interpreter_returns_valid_soft_lines(self):
        interpreter = DifferentiableSketchInterpreter(n_bins=8, coord_range=(-1.0, 1.0))
        arg_logits = torch.zeros(1, 3, N_ARGS, 8, dtype=torch.float32)
        arg_logits[0, 0, 0, 1] = 5.0
        arg_logits[0, 0, 1, 1] = 5.0
        arg_logits[0, 0, 2, 6] = 5.0
        arg_logits[0, 0, 3, 1] = 5.0
        line_cmd_mask = torch.tensor([[True, False, True]])
        line_index_map = torch.tensor([[0, -1, 1]], dtype=torch.long)
        soft_lines = interpreter(arg_logits, line_cmd_mask, line_index_map, max_lines=4)
        self.assertEqual(tuple(soft_lines["start"].shape), (1, 4, 2))
        self.assertEqual(tuple(soft_lines["unit"].shape), (1, 4, 2))
        self.assertEqual(float(soft_lines["valid"][0, 0].item()), 1.0)

    def test_constraint_evaluator_prefers_horizontal_line(self):
        evaluator = ConstraintEvaluator()
        soft_lines = {
            "unit": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32),
            "valid": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        }
        unary_gt = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
        line_mask = torch.tensor([[True, True]])
        loss = evaluator(soft_lines, unary_gt, line_mask)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)

    def test_compose_loss_modify1_is_finite(self):
        total, axis_loss = compose_loss(
            cmd_loss=torch.tensor(1.0),
            pred_loss=torch.tensor(0.5),
            unary_pred=torch.zeros(1, 2, 2),
            unary_gt=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            line_mask=torch.tensor([[True, True]]),
            geom_loss=torch.tensor(0.25),
            alpha=0.1,
            beta=0.5,
            gamma=0.2,
            pos_weight=5.0,
        )
        self.assertTrue(math.isfinite(float(total.item())))
        self.assertTrue(math.isfinite(float(axis_loss.item())))

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
            alpha=0.1,
            beta=0.5,
            gamma=0.2,
            axis_pos_weight=5.0,
            coord_range_min=-1.0,
            coord_range_max=1.0,
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
            "line_cmd_mask": torch.tensor([[True, True, False, False, False, False]]),
            "line_index_map": torch.tensor([[0, 1, -1, -1, -1, -1]], dtype=torch.long),
        }
        out = use_case.execute(batch)
        self.assertEqual(tuple(out["unary_pred"].shape), (1, 4, 2))
        self.assertEqual(tuple(out["decoder_output"]["constraint_pred_logits"].shape), (1, 6, 2))
        self.assertTrue(math.isfinite(float(out["loss"].item())))
        self.assertTrue(math.isfinite(float(out["axis_loss"].item())))
        self.assertTrue(math.isfinite(float(out["pred_loss"].item())))
        self.assertTrue(math.isfinite(float(out["geom_loss"].item())))

    def test_evaluate_axis_metrics_exposes_modify1_summary_fields(self):
        gt_vec = np.zeros((2, 1 + N_ARGS), dtype=np.int64)
        out_vec = np.ones((2, 1 + N_ARGS), dtype=np.int64)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "sample_vec.h5")
            with h5py.File(path, "w") as file_obj:
                file_obj.create_dataset("gt_vec", data=gt_vec, dtype="i8")
                file_obj.create_dataset("out_vec", data=out_vec, dtype="i8")

            rows, summary = aggregate_metrics(temp_dir, angle_thresh=0.1, grid_size=256)
        self.assertEqual(len(rows), 1)
        self.assertIn("R_h", summary)
        self.assertIn("R_v", summary)
        self.assertIn("parallel_recall_index_aligned", summary)
        self.assertIn("perpendicular_recall_index_aligned", summary)
        self.assertIn("n_samples_extrude_count_mismatch", summary)

    def test_parallel_perpendicular_recall_handles_parse_failure(self):
        gt_vec = np.zeros((2, 1 + N_ARGS), dtype=np.int64)
        out_vec = np.ones((2, 1 + N_ARGS), dtype=np.int64)
        result = parallel_perpendicular_recall_index_aligned(
            gt_vec,
            out_vec,
            angle_thresh=0.1,
            grid_size=256,
        )
        self.assertEqual(result, (0, 0, 0, 0, 0, False))

    def test_reconstruction_file_stem_matches_test_py_behavior(self):
        self.assertEqual(reconstruction_file_stem("0001/00000093"), "00000093")
        self.assertEqual(reconstruction_file_stem(r"0001\00000093"), "00000093")

    def test_resolve_checkpoint_path_prefers_explicit_model_path(self):
        cfg = SimpleNamespace(
            model_path=r"proj_log/constraint_fused_deepcad_simplify_modify1/cf_simplify_modify1/model/latest.pth",
            ckpt="latest",
            model_dir=r"proj_log/constraint_fused_deepcad_simplify_modify1/cf_simplify_modify1/model",
        )
        path = resolve_checkpoint_path(cfg)
        self.assertTrue(
            path.endswith(
                os.path.join(
                    "proj_log",
                    "constraint_fused_deepcad_simplify_modify1",
                    "cf_simplify_modify1",
                    "model",
                    "latest.pth",
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
