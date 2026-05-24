from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import torch

from cadlib.macro import EOS_IDX

from constraint_fused_deepcad_simplify_modify1.domain.entities import (
    ConstraintRelationSimplifyModify1,
    SketchSequenceAggregateSimplifyModify1,
)
from constraint_fused_deepcad_simplify_modify1.domain.services import (
    build_cad_commands,
    build_line_cmd_mask,
    build_line_index_map,
    iter_line_command_positions,
    validate_relations,
)


def build_constraint_tags(
    seq_len: int,
    commands_np: np.ndarray,
    relations: Iterable[ConstraintRelationSimplifyModify1],
) -> torch.Tensor:
    tags = torch.zeros(seq_len, 2, dtype=torch.float32)
    line_positions = iter_line_command_positions(commands_np)
    for rel in relations:
        if rel.line_idx >= len(line_positions):
            raise ValueError(
                "Constraint line_idx {} exceeds discovered line positions {}.".format(
                    rel.line_idx,
                    len(line_positions),
                )
            )
        tags[line_positions[rel.line_idx], rel.type_id] = 1.0
    return tags


def build_unary_gt(
    max_lines: int,
    relations: Iterable[ConstraintRelationSimplifyModify1],
    line_count: int,
) -> torch.Tensor:
    unary_gt = torch.zeros(max_lines, 2, dtype=torch.float32)
    for rel in relations:
        if rel.line_idx >= line_count:
            raise ValueError(
                "Constraint line_idx {} exceeds line_count {}.".format(rel.line_idx, line_count)
            )
        unary_gt[rel.line_idx, rel.type_id] = 1.0
    return unary_gt


def build_cmd_padding_mask(commands_np: np.ndarray) -> torch.Tensor:
    commands = torch.as_tensor(commands_np, dtype=torch.long)
    padding_mask = (commands == EOS_IDX).cumsum(dim=0) > 0
    return padding_mask.bool()


class ConstraintBatchAssemblerSimplifyModify1:
    def __init__(self, max_lines: int, seq_len: int):
        self.max_lines = max_lines
        self.seq_len = seq_len

    def _resolve_line_count(self, commands_np: np.ndarray, geometry_line_count: int) -> int:
        n_line_cmds = len(iter_line_command_positions(commands_np))
        if geometry_line_count <= 0:
            return 0
        return min(n_line_cmds, geometry_line_count)

    def assemble_from_vec(
        self,
        cad_vec: np.ndarray,
        relations: List[ConstraintRelationSimplifyModify1],
        geometry_line_count: int,
        sample_id: Optional[str] = None,
    ) -> SketchSequenceAggregateSimplifyModify1:
        commands_np = np.asarray(cad_vec[:, 0], dtype=np.int64)
        args_np = np.asarray(cad_vec[:, 1:], dtype=np.int64)
        if len(commands_np) != self.seq_len:
            raise ValueError("Expected padded cad_vec of length {}, got {}.".format(self.seq_len, len(commands_np)))
        line_count = self._resolve_line_count(commands_np, geometry_line_count)
        relations = validate_relations(relations, line_count) if line_count > 0 else []
        return SketchSequenceAggregateSimplifyModify1(
            commands=build_cad_commands(commands_np, args_np),
            constraints=relations,
            constraint_tags=build_constraint_tags(self.seq_len, commands_np, relations),
            unary_gt=build_unary_gt(self.max_lines, relations, line_count),
            cmd_padding_mask=build_cmd_padding_mask(commands_np),
            line_count=line_count,
            line_cmd_mask=build_line_cmd_mask(commands_np),
            line_index_map=build_line_index_map(commands_np),
            sample_id=sample_id,
        )
