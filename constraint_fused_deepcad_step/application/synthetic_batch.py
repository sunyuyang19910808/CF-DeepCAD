from __future__ import annotations

import torch

from cadlib.macro import EOS_IDX, EXT_IDX, LINE_IDX, MAX_TOTAL_LEN, N_ARGS, PAD_VAL, SOL_IDX


def make_synthetic_batch(batch_size: int = 2, seq_len: int = MAX_TOTAL_LEN, max_lines: int = 8) -> dict:
    """Minimal CAD batch for smoke tests when real data is unavailable."""
    commands = torch.full((batch_size, seq_len), EOS_IDX, dtype=torch.long)
    args = torch.full((batch_size, seq_len, N_ARGS), PAD_VAL, dtype=torch.long)
    groups = torch.zeros(batch_size, seq_len, dtype=torch.long)

    for b in range(batch_size):
        commands[b, 0] = SOL_IDX
        commands[b, 1] = LINE_IDX
        commands[b, 2] = LINE_IDX
        commands[b, 3] = EXT_IDX
        commands[b, 4] = EOS_IDX
        args[b, 1, 0] = 128
        args[b, 1, 1] = 120
        args[b, 2, 0] = 200
        args[b, 2, 1] = 120

    line_cmd_mask = commands == LINE_IDX
    line_index_map = torch.full((batch_size, seq_len), -1, dtype=torch.long)
    for b in range(batch_size):
        line_positions = torch.nonzero(line_cmd_mask[b], as_tuple=False).flatten()
        for line_idx, pos in enumerate(line_positions.tolist()):
            line_index_map[b, pos] = line_idx

    line_count = torch.tensor([2] * batch_size, dtype=torch.long)
    unary_gt = torch.zeros(batch_size, max_lines, 2, dtype=torch.float32)
    pair_gt = torch.zeros(batch_size, max_lines, max_lines, 2, dtype=torch.float32)
    unary_gt[:, 0, 0] = 1.0
    unary_gt[:, 1, 0] = 1.0
    pair_gt[:, 0, 1, 0] = 1.0
    pair_gt[:, 1, 0, 0] = 1.0

    cmd_padding_mask = (commands == EOS_IDX).cumsum(dim=-1) > 0

    return {
        "command": commands,
        "args": args,
        "groups": groups,
        "unary_gt": unary_gt,
        "pair_gt": pair_gt,
        "cmd_padding_mask": cmd_padding_mask,
        "line_count": line_count,
        "line_cmd_mask": line_cmd_mask,
        "line_index_map": line_index_map,
        "id": ["synthetic_{}".format(i) for i in range(batch_size)],
    }
