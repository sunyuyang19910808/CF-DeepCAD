from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


class ConstraintTypeSimplify:
    HORIZONTAL = 0
    VERTICAL = 1

    ALL = (HORIZONTAL, VERTICAL)
    NAME_BY_ID = {
        HORIZONTAL: "HORIZONTAL",
        VERTICAL: "VERTICAL",
    }


@dataclass(frozen=True)
class AxisTagVector:
    horizontal: float = 0.0
    vertical: float = 0.0

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.horizontal, self.vertical], dtype=torch.float32)


@dataclass
class CadCommand:
    command_id: int
    args: List[int]
    group_id: Optional[int] = None
    line_ref: Optional[int] = None


@dataclass(frozen=True)
class ConstraintRelationSimplify:
    type_id: int
    line_idx: int

    def __post_init__(self) -> None:
        if self.type_id not in ConstraintTypeSimplify.ALL:
            raise ValueError("ConstraintRelationSimplify only supports HORIZONTAL or VERTICAL.")
        if self.line_idx < 0:
            raise ValueError("line_idx must be non-negative.")

    @property
    def type_name(self) -> str:
        return ConstraintTypeSimplify.NAME_BY_ID[self.type_id]


@dataclass
class SketchSequenceAggregateSimplify:
    commands: List[CadCommand]
    constraints: List[ConstraintRelationSimplify]
    constraint_tags: torch.Tensor
    unary_gt: torch.Tensor
    cmd_padding_mask: torch.Tensor
    line_count: int
    sample_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.line_count < 0:
            raise ValueError("line_count must be non-negative.")
        if self.constraint_tags.dim() != 2 or self.constraint_tags.size(-1) != 2:
            raise ValueError("constraint_tags must have shape (S, 2).")
        if self.unary_gt.dim() != 2 or self.unary_gt.size(-1) != 2:
            raise ValueError("unary_gt must have shape (L, 2).")
        if self.cmd_padding_mask.dim() != 1:
            raise ValueError("cmd_padding_mask must have shape (S,).")
        if len(self.commands) != int(self.constraint_tags.size(0)):
            raise ValueError("commands and constraint_tags length mismatch.")
        if len(self.commands) != int(self.cmd_padding_mask.size(0)):
            raise ValueError("commands and cmd_padding_mask length mismatch.")
        if self.line_count > int(self.unary_gt.size(0)):
            raise ValueError("line_count exceeds unary_gt capacity.")
        for rel in self.constraints:
            if rel.line_idx >= self.line_count:
                raise ValueError(
                    "Constraint line_idx {} out of range for line_count {}.".format(
                        rel.line_idx,
                        self.line_count,
                    )
                )
