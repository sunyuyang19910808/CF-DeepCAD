from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch

from cadlib.macro import EOS_IDX, N_ARGS, PAD_VAL

from constraint_fused_deepcad_simplify_modify2_low_risk.domain.entities import (
    ConstraintRelation,
    ConstraintType,
    SketchSequenceAggregate,
)
from constraint_fused_deepcad_simplify_modify2_low_risk.domain.services import (
    build_cad_commands,
    build_cmd_padding_mask,
    build_line_cmd_mask,
    build_line_index_map,
    iter_line_command_positions,
    validate_relations,
)


def _commands_args_from_vec(cad_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return cad_vec[:, 0].astype(np.int64), cad_vec[:, 1:].astype(np.int64)


def _tag_dim_for_relation(rel: ConstraintRelation) -> int:
    mapping = {
        ConstraintType.HORIZONTAL: 0,
        ConstraintType.VERTICAL: 1,
        ConstraintType.PARALLEL: 2,
        ConstraintType.PERPENDICULAR: 3,
    }
    return mapping[rel.type_id]


class ConstraintBatchAssemblerSimplifyModify2LowRisk:
    def __init__(self, max_lines: int, max_constraints: int, seq_len: int):
        self.max_lines = max_lines
        self.max_constraints = max_constraints
        self.seq_len = seq_len

    def _resolve_line_count(self, commands_np: np.ndarray, geometry_line_count: int) -> int:
        n_line_cmds = len(iter_line_command_positions(commands_np))
        if geometry_line_count <= 0:
            return 0
        return min(n_line_cmds, geometry_line_count, self.max_lines)

    def build_constraint_tags(self, seq_len: int, commands_np: np.ndarray, relations: Iterable[ConstraintRelation]) -> torch.Tensor:
        tags = torch.zeros(seq_len, 4, dtype=torch.float32)
        line_positions = iter_line_command_positions(commands_np)
        for rel in relations:
            dim = _tag_dim_for_relation(rel)
            for line_idx in {rel.line_a, rel.line_b}:
                if line_idx >= len(line_positions):
                    raise ValueError(
                        "Constraint line {} exceeds discovered line positions {}.".format(line_idx, len(line_positions))
                    )
                tags[line_positions[line_idx], dim] = 1.0
        return tags

    def build_constraint_tokens(
        self,
        relations: List[ConstraintRelation],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        real = relations[: self.max_constraints]
        c_types = torch.full((self.max_constraints,), ConstraintType.NONE, dtype=torch.long)
        c_line_a = torch.zeros(self.max_constraints, dtype=torch.long)
        c_line_b = torch.zeros(self.max_constraints, dtype=torch.long)
        padding_mask = torch.ones(self.max_constraints, dtype=torch.bool)
        for index, rel in enumerate(real):
            c_types[index] = rel.type_id
            c_line_a[index] = rel.line_a
            c_line_b[index] = rel.line_b
            padding_mask[index] = False
        tokens = torch.stack([c_types.float(), c_line_a.float(), c_line_b.float()], dim=-1)
        return tokens, c_types, c_line_a, c_line_b, padding_mask

    def build_recon_targets(self, relations: Iterable[ConstraintRelation], line_count: int) -> Tuple[torch.Tensor, torch.Tensor]:
        unary_gt = torch.zeros(self.max_lines, 2, dtype=torch.float32)
        pair_gt = torch.zeros(self.max_lines, self.max_lines, 2, dtype=torch.float32)
        for rel in relations:
            if rel.line_a >= line_count or rel.line_b >= line_count:
                continue
            if rel.type_id == ConstraintType.HORIZONTAL:
                unary_gt[rel.line_a, 0] = 1.0
            elif rel.type_id == ConstraintType.VERTICAL:
                unary_gt[rel.line_a, 1] = 1.0
            elif rel.type_id == ConstraintType.PARALLEL:
                pair_gt[rel.line_a, rel.line_b, 0] = 1.0
                pair_gt[rel.line_b, rel.line_a, 0] = 1.0
            elif rel.type_id == ConstraintType.PERPENDICULAR:
                pair_gt[rel.line_a, rel.line_b, 1] = 1.0
                pair_gt[rel.line_b, rel.line_a, 1] = 1.0
        return unary_gt, pair_gt

    def assemble_from_vec(
        self,
        cad_vec: np.ndarray,
        relations: List[ConstraintRelation],
        geometry_line_count: int,
        sample_id: Optional[str] = None,
    ) -> SketchSequenceAggregate:
        if cad_vec.shape[0] < self.seq_len:
            pad_rows = self.seq_len - cad_vec.shape[0]
            tail = np.tile(np.array([EOS_IDX] + [PAD_VAL] * N_ARGS, dtype=np.int64), (pad_rows, 1))
            cad_vec = np.concatenate([cad_vec, tail], axis=0)
        elif cad_vec.shape[0] > self.seq_len:
            cad_vec = cad_vec[: self.seq_len]

        commands_np, args_np = _commands_args_from_vec(cad_vec)
        line_count = self._resolve_line_count(commands_np, geometry_line_count)
        relations = validate_relations(relations, line_count) if line_count > 0 else []

        constraint_tags = self.build_constraint_tags(self.seq_len, commands_np, relations)
        constraint_tokens, c_types, c_line_a, c_line_b, constraint_padding_mask = self.build_constraint_tokens(relations)
        unary_gt, pair_gt = self.build_recon_targets(relations, line_count)

        return SketchSequenceAggregate(
            commands=build_cad_commands(commands_np, args_np),
            constraints=relations,
            constraint_tags=constraint_tags,
            constraint_tokens=constraint_tokens,
            unary_gt=unary_gt,
            pair_gt=pair_gt,
            c_types=c_types,
            c_line_a=c_line_a,
            c_line_b=c_line_b,
            cmd_padding_mask=build_cmd_padding_mask(commands_np),
            constraint_padding_mask=constraint_padding_mask,
            line_count=line_count,
            line_cmd_mask=build_line_cmd_mask(commands_np),
            line_index_map=build_line_index_map(commands_np),
            sample_id=sample_id,
        )
