from __future__ import annotations

import numpy as np

from cadlib.macro import ARC_IDX, CIRCLE_IDX, EXT_IDX, LINE_IDX, SOL_IDX

from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.domain.services import (
    iter_line_command_positions,
)

CURVE_CMD_IDS = (LINE_IDX, ARC_IDX, CIRCLE_IDX)


def _last_curve_pos_per_sol_segment_numpy(
    cand_idx: np.ndarray,
    curve_mask: np.ndarray,
    seg_id: np.ndarray,
) -> np.ndarray:
    length = cand_idx.shape[0]
    max_seg = int(seg_id.max()) + 1 if length else 1
    last_curve_per_seg = np.full(max_seg, -1, dtype=np.int64)
    for pos in range(length):
        if not curve_mask[pos]:
            continue
        seg = int(seg_id[pos])
        last_curve_per_seg[seg] = max(last_curve_per_seg[seg], cand_idx[pos])
    return last_curve_per_seg[seg_id]


def corrected_start_per_token_numpy(commands: np.ndarray, end_per_token: np.ndarray) -> np.ndarray:
    """Chain + loop-close starts aligned with ``DifferentiableSketchInterpreter``."""
    length = commands.shape[0]
    starts = np.zeros((length, 2), dtype=np.float64)

    curve_mask = np.isin(commands, CURVE_CMD_IDS)
    sol_mask = commands == SOL_IDX
    seg_id = np.cumsum(sol_mask, dtype=np.int64)

    seq_idx = np.arange(length, dtype=np.int64)
    neg_ones = np.full(length, -1, dtype=np.int64)
    cand_idx = np.where(curve_mask, seq_idx, neg_ones)
    sol_idx_only = np.where(sol_mask, seq_idx, neg_ones)

    cand_shifted = np.concatenate([neg_ones[:1], cand_idx[:-1]])
    sol_shifted = np.concatenate([neg_ones[:1], sol_idx_only[:-1]])

    prev_curve_pos = np.maximum.accumulate(cand_shifted)
    prev_sol_pos = np.maximum.accumulate(sol_shifted)

    valid_chain = prev_curve_pos > prev_sol_pos
    loop_last_curve_pos = _last_curve_pos_per_sol_segment_numpy(cand_idx, curve_mask, seg_id)
    in_loop = seg_id > 0
    use_loop_close = (~valid_chain) & curve_mask & in_loop & (loop_last_curve_pos >= 0)

    prev_source = np.where(
        valid_chain,
        np.clip(prev_curve_pos, 0, None),
        np.where(use_loop_close, np.clip(loop_last_curve_pos, 0, None), 0),
    )
    valid_prev = valid_chain | use_loop_close

    for pos in range(length):
        if valid_prev[pos]:
            starts[pos] = end_per_token[int(prev_source[pos])]
    return starts


def extrude_block_id_per_position(commands: np.ndarray) -> np.ndarray:
    """Block index for each token: lines after the k-th EXT belong to block k."""
    return np.cumsum(commands == EXT_IDX, dtype=np.int64)


def collect_token_order_line_directions(
    cad_vec: np.ndarray,
) -> tuple[list[int], list[np.ndarray], list[int]]:
    """Return Line token positions and unit directions in token order (line_index_map)."""
    commands = cad_vec[:, 0].astype(np.int64)
    end_per_token = cad_vec[:, 1:3].astype(np.float64)
    starts = corrected_start_per_token_numpy(commands, end_per_token)
    block_ids = extrude_block_id_per_position(commands)

    line_positions = iter_line_command_positions(commands.tolist())
    directions: list[np.ndarray] = []
    extrude_ids: list[int] = []
    for pos in line_positions:
        delta = end_per_token[pos] - starts[pos]
        norm = float(np.linalg.norm(delta) + 1e-12)
        directions.append(delta / norm)
        extrude_ids.append(int(block_ids[pos]))
    return line_positions, directions, extrude_ids
