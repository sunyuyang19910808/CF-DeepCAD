from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cadlib.macro import ARC_IDX, CIRCLE_IDX, LINE_IDX, SOL_IDX


class DifferentiableSketchInterpreter(nn.Module):
    """Soft-dequantize args_logits into per-line unit directions."""

    def __init__(
        self,
        n_bins: int,
        coord_range=(-1.0, 1.0),
        eps: float = 1e-6,
        use_corrected_line_start: bool = True,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.coord_range = coord_range
        self.eps = eps
        self.use_corrected_line_start = bool(use_corrected_line_start)
        self._curve_cmd_ids_cache: torch.Tensor | None = None

    def _curve_ids_on(self, device: torch.device) -> torch.Tensor:
        cache = self._curve_cmd_ids_cache
        if cache is None or cache.device != device:
            cache = torch.tensor([LINE_IDX, ARC_IDX, CIRCLE_IDX], dtype=torch.long, device=device)
            self._curve_cmd_ids_cache = cache
        return cache

    def soft_dequantize(self, arg_logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(arg_logits, dim=-1)
        bins = torch.arange(self.n_bins, device=arg_logits.device, dtype=arg_logits.dtype)
        soft_idx = (probs * bins).sum(dim=-1)
        lo, hi = self.coord_range
        return lo + (hi - lo) * soft_idx / max(self.n_bins - 1, 1)

    def _last_curve_pos_per_sol_segment(
        self,
        cand_idx: torch.Tensor,
        curve_mask: torch.Tensor,
        seg_id: torch.Tensor,
    ) -> torch.Tensor:
        """For each token, index of the last curve in the same SOL segment (DeepCAD loop span).

        Mirrors ``Loop.from_vector`` when ``start_point=None``: the first curve in a loop uses
        the end point (``vec[1:3]``) of the last curve before ``EOS`` in that segment.
        """
        B, L = cand_idx.shape
        device = cand_idx.device
        max_seg = int(seg_id.max().item()) + 1
        last_curve_per_seg = torch.full((B, max_seg), -1, dtype=cand_idx.dtype, device=device)
        if curve_mask.any():
            b_idx, pos_idx = torch.where(curve_mask)
            seg_ids = seg_id[b_idx, pos_idx]
            curve_pos = cand_idx[b_idx, pos_idx]
            flat = last_curve_per_seg.view(-1)
            flat.scatter_reduce_(
                0,
                b_idx * max_seg + seg_ids,
                curve_pos,
                reduce="amax",
                include_self=False,
            )
            last_curve_per_seg = flat.view(B, max_seg)
        seg_lookup = seg_id.clamp_min(0).clamp_max(max_seg - 1)
        return last_curve_per_seg.gather(1, seg_lookup)

    def _corrected_start_per_token(
        self,
        end_per_token: torch.Tensor,
        commands: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token line starts following ``Loop.from_vector`` chain semantics.

        * Same-loop chain: start = previous curve token's ``vec[1:3]`` (stored as end here).
        * First curve after ``SOL``: start = last curve in the segment before ``EOS`` (loop close).
        """
        B, L, _ = end_per_token.shape
        device = end_per_token.device
        long_dtype = torch.long

        curve_ids = self._curve_ids_on(device)
        curve_mask = (commands.unsqueeze(-1) == curve_ids).any(dim=-1)
        sol_mask = commands == SOL_IDX
        seg_id = sol_mask.long().cumsum(dim=1)

        seq_idx = torch.arange(L, device=device, dtype=long_dtype).expand(B, L)
        neg_ones = torch.full_like(seq_idx, -1)
        cand_idx = torch.where(curve_mask, seq_idx, neg_ones)
        sol_idx_only = torch.where(sol_mask, seq_idx, neg_ones)

        pad = torch.full((B, 1), -1, dtype=long_dtype, device=device)
        cand_shifted = torch.cat([pad, cand_idx[:, :-1]], dim=1)
        sol_shifted = torch.cat([pad, sol_idx_only[:, :-1]], dim=1)

        prev_curve_pos = cand_shifted.cummax(dim=1).values
        prev_sol_pos = sol_shifted.cummax(dim=1).values

        valid_chain = prev_curve_pos > prev_sol_pos
        loop_last_curve_pos = self._last_curve_pos_per_sol_segment(cand_idx, curve_mask, seg_id)
        in_loop = seg_id > 0
        use_loop_close = (~valid_chain) & curve_mask & in_loop & (loop_last_curve_pos >= 0)

        prev_source = torch.where(
            valid_chain,
            prev_curve_pos.clamp_min(0),
            torch.where(use_loop_close, loop_last_curve_pos.clamp_min(0), torch.zeros_like(prev_curve_pos)),
        )
        valid_prev = valid_chain | use_loop_close

        gathered = end_per_token.gather(1, prev_source.unsqueeze(-1).expand(B, L, 2))
        start_per_token = gathered * valid_prev.unsqueeze(-1).to(end_per_token.dtype)
        return start_per_token, valid_prev

    def forward(
        self,
        arg_logits: torch.Tensor,
        line_cmd_mask: torch.Tensor,
        line_index_map: torch.Tensor,
        max_lines: int,
        commands: torch.Tensor | None = None,
    ) -> dict:
        arg_cont = self.soft_dequantize(arg_logits)
        B, L = line_cmd_mask.shape
        device = arg_logits.device
        dtype = arg_cont.dtype

        if self.use_corrected_line_start:
            if commands is None:
                raise ValueError("use_corrected_line_start=True requires commands.")
            end_per_token = arg_cont[..., 0:2]
            start_per_token, _ = self._corrected_start_per_token(end_per_token, commands)
        else:
            start_per_token = arg_cont[..., 0:2]
            end_per_token = arg_cont[..., 2:4]

        direction = end_per_token - start_per_token
        norm = torch.norm(direction, dim=-1, keepdim=True).clamp_min(self.eps)
        unit_per_token = direction / norm

        valid_scatter = line_cmd_mask.bool() & (line_index_map >= 0) & (line_index_map < max_lines)
        b_idx, pos_idx = torch.where(valid_scatter)
        l_idx = line_index_map[b_idx, pos_idx]

        start = torch.zeros(B, max_lines, 2, device=device, dtype=dtype)
        end = torch.zeros(B, max_lines, 2, device=device, dtype=dtype)
        unit = torch.zeros(B, max_lines, 2, device=device, dtype=dtype)
        valid = torch.zeros(B, max_lines, device=device, dtype=dtype)

        if b_idx.numel() > 0:
            start[b_idx, l_idx] = start_per_token[b_idx, pos_idx]
            end[b_idx, l_idx] = end_per_token[b_idx, pos_idx]
            unit[b_idx, l_idx] = unit_per_token[b_idx, pos_idx]
            valid[b_idx, l_idx] = 1.0

        return {
            "start": start,
            "end": end,
            "unit": unit,
            "valid": valid,
        }
