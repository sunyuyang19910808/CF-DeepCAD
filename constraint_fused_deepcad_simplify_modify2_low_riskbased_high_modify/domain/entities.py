from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


class ConstraintType:
    HORIZONTAL = 0
    VERTICAL = 1
    PARALLEL = 2
    PERPENDICULAR = 3
    NONE = 4

    REAL_TYPES = (HORIZONTAL, VERTICAL, PARALLEL, PERPENDICULAR)
    UNARY_TYPES = (HORIZONTAL, VERTICAL)
    PAIR_TYPES = (PARALLEL, PERPENDICULAR)
    NAME_BY_ID = {
        HORIZONTAL: "HORIZONTAL",
        VERTICAL: "VERTICAL",
        PARALLEL: "PARALLEL",
        PERPENDICULAR: "PERPENDICULAR",
        NONE: "NONE",
    }


@dataclass
class CadCommand:
    command_id: int
    args: List[int]
    group_id: Optional[int] = None
    line_ref: Optional[int] = None

    @property
    def is_line_command(self) -> bool:
        return self.line_ref is not None


@dataclass(frozen=True)
class ConstraintRelation:
    type_id: int
    line_a: int
    line_b: int = 0

    def __post_init__(self) -> None:
        if self.type_id not in ConstraintType.REAL_TYPES:
            raise ValueError("ConstraintRelation only supports four real constraint types.")
        if self.line_a < 0 or self.line_b < 0:
            raise ValueError("line indices must be non-negative.")
        if self.is_unary() and self.line_a != self.line_b:
            object.__setattr__(self, "line_b", self.line_a)

    def is_unary(self) -> bool:
        return self.type_id in ConstraintType.UNARY_TYPES

    def is_pair(self) -> bool:
        return self.type_id in ConstraintType.PAIR_TYPES

    @property
    def type_name(self) -> str:
        return ConstraintType.NAME_BY_ID[self.type_id]


@dataclass(frozen=True)
class ConstraintTagVector:
    horizontal: float = 0.0
    vertical: float = 0.0
    parallel: float = 0.0
    perpendicular: float = 0.0

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [self.horizontal, self.vertical, self.parallel, self.perpendicular],
            dtype=torch.float32,
        )


@dataclass
class ConstraintAwareLatent:
    tensor: torch.Tensor

    def __post_init__(self) -> None:
        if self.tensor.dim() != 3 or self.tensor.size(0) != 1:
            raise ValueError("ConstraintAwareLatent.tensor must be shape (1, N, d_model).")


@dataclass
class SketchSequenceAggregate:
    commands: List[CadCommand]
    constraints: List[ConstraintRelation]
    constraint_tags: torch.Tensor
    constraint_tokens: torch.Tensor
    unary_gt: torch.Tensor
    pair_gt: torch.Tensor
    c_types: torch.Tensor
    c_line_a: torch.Tensor
    c_line_b: torch.Tensor
    cmd_padding_mask: torch.Tensor
    constraint_padding_mask: torch.Tensor
    line_count: int
    line_cmd_mask: torch.Tensor
    line_index_map: torch.Tensor
    sample_id: Optional[str] = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        seq_len = len(self.commands)
        if self.line_count < 0:
            raise ValueError("line_count must be non-negative.")
        if self.constraint_tags.dim() != 2 or self.constraint_tags.size(-1) != 4:
            raise ValueError("constraint_tags must have shape (S, 4).")
        if self.unary_gt.dim() != 2 or self.unary_gt.size(-1) != 2:
            raise ValueError("unary_gt must have shape (L, 2).")
        if self.pair_gt.dim() != 3 or self.pair_gt.size(-1) != 2:
            raise ValueError("pair_gt must have shape (L, L, 2).")
        if self.c_types.dim() != 1 or self.c_line_a.dim() != 1 or self.c_line_b.dim() != 1:
            raise ValueError("constraint token fields must be 1D tensors.")
        if self.constraint_tokens.dim() != 2 or self.constraint_tokens.size(-1) != 3:
            raise ValueError("constraint_tokens must have shape (T, 3).")
        if self.cmd_padding_mask.dim() != 1 or self.constraint_padding_mask.dim() != 1:
            raise ValueError("padding masks must be 1D.")
        if self.line_cmd_mask.dim() != 1 or self.line_index_map.dim() != 1:
            raise ValueError("line helpers must be 1D.")
        if self.constraint_tags.size(0) != seq_len:
            raise ValueError("commands and constraint_tags length mismatch.")
        if self.cmd_padding_mask.size(0) != seq_len:
            raise ValueError("commands and cmd_padding_mask length mismatch.")
        if self.line_cmd_mask.size(0) != seq_len or self.line_index_map.size(0) != seq_len:
            raise ValueError("commands and line helpers length mismatch.")
        if self.c_types.size(0) != self.constraint_padding_mask.size(0):
            raise ValueError("constraint token size mismatch.")
        if self.c_line_a.size(0) != self.c_types.size(0) or self.c_line_b.size(0) != self.c_types.size(0):
            raise ValueError("constraint token line index size mismatch.")
        if self.line_count > self.unary_gt.size(0) or self.line_count > self.pair_gt.size(0):
            raise ValueError("line_count exceeds target tensor capacity.")
        for rel in self.constraints:
            if rel.type_id not in ConstraintType.REAL_TYPES:
                raise ValueError("Unexpected constraint type {}.".format(rel.type_id))
            if rel.line_a >= self.line_count or rel.line_b >= self.line_count:
                raise ValueError(
                    "Constraint line indices ({}, {}) out of range for line_count {}.".format(
                        rel.line_a, rel.line_b, self.line_count
                    )
                )
