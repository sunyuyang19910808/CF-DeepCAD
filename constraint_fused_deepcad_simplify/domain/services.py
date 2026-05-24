from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from cadlib.macro import EXT_IDX, LINE_IDX

from .entities import CadCommand, ConstraintRelationSimplify


def iter_line_command_positions(commands_np: Sequence[int]) -> List[int]:
    return [idx for idx, command_id in enumerate(commands_np) if int(command_id) == LINE_IDX]


def build_cad_commands(commands_np: np.ndarray, args_np: np.ndarray) -> List[CadCommand]:
    line_positions = iter_line_command_positions(commands_np)
    line_ref_by_pos = {pos: line_idx for line_idx, pos in enumerate(line_positions)}
    commands: List[CadCommand] = []
    group_id = 0
    for pos, command_id in enumerate(commands_np.tolist()):
        if int(command_id) == EXT_IDX:
            group_id += 1
        commands.append(
            CadCommand(
                command_id=int(command_id),
                args=[int(v) for v in args_np[pos].tolist()],
                group_id=group_id,
                line_ref=line_ref_by_pos.get(pos),
            )
        )
    return commands


def validate_relations(relations: Iterable[ConstraintRelationSimplify], line_count: int) -> List[ConstraintRelationSimplify]:
    validated = []
    for rel in relations:
        if rel.line_idx >= line_count:
            raise ValueError(
                "Constraint line_idx {} is outside [0, {}).".format(rel.line_idx, line_count)
            )
        validated.append(rel)
    return validated
