from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import torch

from cadlib.macro import EOS_IDX, EXT_IDX, LINE_IDX

from .entities import CadCommand, ConstraintAwareLatent, ConstraintRelation


def iter_line_command_positions(commands_np: Sequence[int]) -> List[int]:
    return [idx for idx, command_id in enumerate(commands_np) if int(command_id) == LINE_IDX]


def build_line_cmd_mask(commands_np: Sequence[int]) -> torch.Tensor:
    return torch.tensor([int(command_id) == LINE_IDX for command_id in commands_np], dtype=torch.bool)


def build_line_index_map(commands_np: Sequence[int]) -> torch.Tensor:
    mapping = []
    line_idx = 0
    for command_id in commands_np:
        if int(command_id) == LINE_IDX:
            mapping.append(line_idx)
            line_idx += 1
        else:
            mapping.append(-1)
    return torch.tensor(mapping, dtype=torch.long)


def build_cmd_padding_mask(commands_np: Sequence[int]) -> torch.Tensor:
    commands = torch.as_tensor(commands_np, dtype=torch.long)
    return ((commands == EOS_IDX).cumsum(dim=0) > 0).bool()


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


def validate_relations(relations: Iterable[ConstraintRelation], line_count: int) -> List[ConstraintRelation]:
    validated = []
    for rel in relations:
        if rel.line_a >= line_count or rel.line_b >= line_count:
            raise ValueError(
                "Constraint relation ({}, {}) is outside [0, {}).".format(rel.line_a, rel.line_b, line_count)
            )
        validated.append(rel)
    return validated


def build_line_mask(line_count: torch.Tensor, max_lines: int) -> torch.Tensor:
    if line_count.dim() == 0:
        line_count = line_count.unsqueeze(0)
    line_indices = torch.arange(max_lines, device=line_count.device).unsqueeze(0)
    return line_indices < line_count.unsqueeze(1)


def gather_line_features(command_memory: torch.Tensor, line_cmd_mask: torch.Tensor, line_index_map: torch.Tensor, max_lines: int):
    batch_size = line_cmd_mask.size(0)
    d_model = command_memory.size(-1)
    out = torch.zeros(batch_size, max_lines, d_model, device=command_memory.device, dtype=command_memory.dtype)
    counts = torch.zeros(batch_size, max_lines, 1, device=command_memory.device, dtype=command_memory.dtype)

    batch_first_memory = command_memory.transpose(0, 1)
    line_positions = line_cmd_mask.bool()
    for batch_idx in range(batch_size):
        active_positions = torch.nonzero(line_positions[batch_idx], as_tuple=False).flatten()
        for pos in active_positions.tolist():
            line_idx = int(line_index_map[batch_idx, pos].item())
            if 0 <= line_idx < max_lines:
                out[batch_idx, line_idx] += batch_first_memory[batch_idx, pos]
                counts[batch_idx, line_idx] += 1.0
    return out / counts.clamp_min(1.0)


def gather_decoder_line_features(
    hidden_states: torch.Tensor,
    line_cmd_mask: torch.Tensor,
    line_index_map: torch.Tensor,
    max_lines: int,
) -> torch.Tensor:
    batch_size, _seq_len, d_model = hidden_states.shape
    out = torch.zeros(batch_size, max_lines, d_model, device=hidden_states.device, dtype=hidden_states.dtype)
    counts = torch.zeros(batch_size, max_lines, 1, device=hidden_states.device, dtype=hidden_states.dtype)

    line_positions = line_cmd_mask.bool()
    for batch_idx in range(batch_size):
        active_positions = torch.nonzero(line_positions[batch_idx], as_tuple=False).flatten()
        for pos in active_positions.tolist():
            line_idx = int(line_index_map[batch_idx, pos].item())
            if 0 <= line_idx < max_lines:
                out[batch_idx, line_idx] += hidden_states[batch_idx, pos]
                counts[batch_idx, line_idx] += 1.0
    return out / counts.clamp_min(1.0)


class ConstraintFusionDomainService:
    def __init__(self, encoder_fused, bottleneck):
        self.encoder_fused = encoder_fused
        self.bottleneck = bottleneck

    def fuse(self, **batch_tensors):
        encoder_outputs = self.encoder_fused(**batch_tensors)
        z = self.bottleneck(encoder_outputs["z_pre"])
        return ConstraintAwareLatent(z), encoder_outputs


class ConstraintReconstructionDomainService:
    def __init__(self, recon_head):
        self.recon_head = recon_head

    def reconstruct(self, decoder_line_features: torch.Tensor):
        return self.recon_head(decoder_line_features)
