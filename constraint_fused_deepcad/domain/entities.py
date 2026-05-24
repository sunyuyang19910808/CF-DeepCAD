from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import torch

if TYPE_CHECKING:
    pass


class ConstraintType:
    HORIZONTAL = 0
    VERTICAL = 1
    PARALLEL = 2
    PERPENDICULAR = 3
    COLLINEAR = 4
    NONE = 5


@dataclass
class CadCommand:
    command_id: int
    args: List[int]
    group_id: Optional[int] = None
    line_ref: Optional[int] = None

    @property
    def is_line_command(self) -> bool:
        return self.line_ref is not None


@dataclass
class ConstraintRelation:
    type_id: int
    line_a: int
    line_b: int = 0

    def is_unary(self) -> bool:
        return self.type_id in (ConstraintType.HORIZONTAL, ConstraintType.VERTICAL)


@dataclass(frozen=True)
class ConstraintTagVector:
    horizontal: int
    vertical: int
    parallel: int
    perpendicular: int
    collinear: int


@dataclass
class ConstraintAwareLatent:
    tensor: torch.Tensor

    def __post_init__(self):
        if self.tensor.dim() != 3 or self.tensor.size(0) != 1:
            raise ValueError("ConstraintAwareLatent.tensor must be shape (1, N, d_model)")


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
    cmd_padding_mask: Optional[torch.Tensor] = None
    constraint_padding_mask: Optional[torch.Tensor] = None
    line_count: int = 0
    sample_id: Optional[str] = None

    def validate(self, max_lines: int) -> None:
        for rel in self.constraints:
            if rel.type_id == ConstraintType.NONE:
                continue
            if not (0 <= rel.line_a < max_lines):
                raise ValueError(f"line_a out of range: {rel.line_a} >= {max_lines}")
            if not (0 <= rel.line_b < max_lines):
                raise ValueError(f"line_b out of range: {rel.line_b} >= {max_lines}")
        if self.line_count > max_lines:
            raise ValueError(f"line_count {self.line_count} > max_lines {max_lines}")
