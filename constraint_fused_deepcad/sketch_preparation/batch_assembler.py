from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from cadlib.macro import EOS_IDX, EXT_IDX, LINE_IDX, N_ARGS, PAD_VAL, SOL_IDX

from constraint_fused_deepcad.domain.entities import (
    CadCommand,
    ConstraintRelation,
    ConstraintType,
    SketchSequenceAggregate,
)


def _commands_args_from_vec(cad_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return cad_vec[:, 0].astype(np.int64), cad_vec[:, 1:].astype(np.int64)


class ConstraintBatchAssembler:
    def __init__(self, max_lines: int, max_constraints: int, seq_len: int):
        self.max_lines = max_lines
        self.max_constraints = max_constraints
        self.seq_len = seq_len

    def build_cad_commands(self, commands: np.ndarray, args: np.ndarray) -> List[CadCommand]:
        cmds: List[CadCommand] = []
        line_id = 0
        ext_cum = 0
        for t in range(commands.shape[0]):
            cid = int(commands[t])
            if cid == EOS_IDX:
                break
            g = None
            if cid == EXT_IDX:
                ext_cum += 1
            elif cid == SOL_IDX:
                g = ext_cum
            a = args[t].tolist()
            lr = None
            if cid == LINE_IDX:
                lr = line_id
                line_id += 1
            cmds.append(CadCommand(command_id=cid, args=a, group_id=g, line_ref=lr))
        return cmds

    def build_constraint_tags(
        self,
        seq_len: int,
        commands_np: np.ndarray,
        relations: List[ConstraintRelation],
    ) -> torch.Tensor:
        tags = torch.zeros(seq_len, 5, dtype=torch.float32)
        line_to_pos: dict = {}
        lid = 0
        for t in range(min(seq_len, commands_np.shape[0])):
            if int(commands_np[t]) == LINE_IDX:
                line_to_pos[lid] = t
                lid += 1

        def set_tag(line_idx: int, dim: int) -> None:
            if line_idx not in line_to_pos:
                return
            pos = line_to_pos[line_idx]
            if pos < seq_len:
                tags[pos, dim] = 1.0

        for rel in relations:
            if rel.type_id == ConstraintType.NONE:
                continue
            if rel.is_unary():
                if rel.type_id == ConstraintType.HORIZONTAL:
                    set_tag(rel.line_a, 0)
                elif rel.type_id == ConstraintType.VERTICAL:
                    set_tag(rel.line_a, 1)
            else:
                for li in (rel.line_a, rel.line_b):
                    set_tag(li, 2)
                    set_tag(li, 3)
                    set_tag(li, 4)
        return tags

    def build_constraint_tokens(
        self, relations: List[ConstraintRelation]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        real = [r for r in relations if r.type_id != ConstraintType.NONE]
        real = real[: self.max_constraints]
        t = len(real)
        c_types = torch.full((self.max_constraints,), ConstraintType.NONE, dtype=torch.long)
        c_la = torch.zeros(self.max_constraints, dtype=torch.long)
        c_lb = torch.zeros(self.max_constraints, dtype=torch.long)
        mask = torch.ones(self.max_constraints, dtype=torch.bool)
        for i, rel in enumerate(real):
            c_types[i] = rel.type_id
            c_la[i] = rel.line_a
            c_lb[i] = rel.line_b if not rel.is_unary() else rel.line_a
            mask[i] = False
        constraint_padding_mask = mask
        return c_types, c_la, c_lb, constraint_padding_mask

    def build_recon_targets(
        self, relations: List[ConstraintRelation], num_lines: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        L = self.max_lines
        unary = torch.zeros(L, 2, dtype=torch.float32)
        pair = torch.zeros(L, L, 3, dtype=torch.float32)
        n = min(num_lines, L)
        for rel in relations:
            if rel.type_id == ConstraintType.NONE:
                continue
            if rel.is_unary():
                if rel.line_a < n:
                    if rel.type_id == ConstraintType.HORIZONTAL:
                        unary[rel.line_a, 0] = 1.0
                    elif rel.type_id == ConstraintType.VERTICAL:
                        unary[rel.line_a, 1] = 1.0
            else:
                a, b = rel.line_a, rel.line_b
                if a < n and b < n:
                    if rel.type_id == ConstraintType.PARALLEL:
                        pair[a, b, 0] = 1.0
                        pair[b, a, 0] = 1.0
                    elif rel.type_id == ConstraintType.PERPENDICULAR:
                        pair[a, b, 1] = 1.0
                        pair[b, a, 1] = 1.0
                    elif rel.type_id == ConstraintType.COLLINEAR:
                        pair[a, b, 2] = 1.0
                        pair[b, a, 2] = 1.0
        return unary, pair

    def assemble_from_vec(
        self,
        cad_vec: np.ndarray,
        relations: List[ConstraintRelation],
        sample_id: str | None = None,
    ) -> SketchSequenceAggregate:
        commands_np, args_np = _commands_args_from_vec(cad_vec)
        cmds = self.build_cad_commands(commands_np, args_np)
        line_count = sum(1 for c in cmds if c.line_ref is not None)

        if cad_vec.shape[0] < self.seq_len:
            pad_rows = self.seq_len - cad_vec.shape[0]
            tail = np.tile(
                np.array([EOS_IDX] + [PAD_VAL] * N_ARGS, dtype=np.int64),
                (pad_rows, 1),
            )
            cad_vec = np.concatenate([cad_vec, tail], axis=0)
        elif cad_vec.shape[0] > self.seq_len:
            cad_vec = cad_vec[: self.seq_len]

        commands_np, args_np = _commands_args_from_vec(cad_vec)

        constraint_tags = self.build_constraint_tags(self.seq_len, commands_np, relations)
        c_types, c_la, c_lb, con_pad = self.build_constraint_tokens(relations)
        unary_gt, pair_gt = self.build_recon_targets(relations, line_count)

        cmd_padding_mask = torch.zeros(self.seq_len, dtype=torch.bool)
        eos_hit = False
        for t in range(self.seq_len):
            if int(commands_np[t]) == EOS_IDX:
                eos_hit = True
            cmd_padding_mask[t] = eos_hit

        return SketchSequenceAggregate(
            commands=cmds,
            constraints=relations,
            constraint_tags=constraint_tags,
            constraint_tokens=torch.stack([c_types.float(), c_la.float(), c_lb.float()], dim=-1),
            unary_gt=unary_gt,
            pair_gt=pair_gt,
            c_types=c_types,
            c_line_a=c_la,
            c_line_b=c_lb,
            cmd_padding_mask=cmd_padding_mask,
            constraint_padding_mask=con_pad,
            line_count=line_count,
            sample_id=sample_id,
        )
