from __future__ import annotations

import torch
import torch.nn as nn


class DifferentiableSketchInterpreter(nn.Module):
    def __init__(self, n_bins: int, coord_range=(-1.0, 1.0), eps: float = 1e-6):
        super().__init__()
        self.n_bins = n_bins
        self.coord_range = coord_range
        self.eps = eps

    def soft_dequantize(self, arg_logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(arg_logits, dim=-1)
        bins = torch.arange(self.n_bins, device=arg_logits.device, dtype=arg_logits.dtype)
        soft_idx = (probs * bins).sum(dim=-1)
        lo, hi = self.coord_range
        return lo + (hi - lo) * soft_idx / max(self.n_bins - 1, 1)

    def forward(
        self,
        arg_logits: torch.Tensor,
        line_cmd_mask: torch.Tensor,
        line_index_map: torch.Tensor,
        max_lines: int,
    ) -> dict:
        arg_cont = self.soft_dequantize(arg_logits)
        batch_size = line_cmd_mask.shape[0]
        device = arg_logits.device
        dtype = arg_cont.dtype

        start = torch.zeros(batch_size, max_lines, 2, device=device, dtype=dtype)
        end = torch.zeros(batch_size, max_lines, 2, device=device, dtype=dtype)
        unit = torch.zeros(batch_size, max_lines, 2, device=device, dtype=dtype)
        valid = torch.zeros(batch_size, max_lines, device=device, dtype=dtype)

        line_arg = arg_cont[..., :4]
        p1_all = line_arg[..., 0:2]
        p2_all = line_arg[..., 2:4]
        d_all = p2_all - p1_all
        norm_all = torch.norm(d_all, dim=-1, keepdim=True).clamp_min(self.eps)
        unit_all = d_all / norm_all

        for batch_idx in range(batch_size):
            line_positions = torch.nonzero(line_cmd_mask[batch_idx], as_tuple=False).flatten()
            for pos in line_positions.tolist():
                line_idx = int(line_index_map[batch_idx, pos].item())
                if line_idx < 0 or line_idx >= max_lines:
                    continue
                start[batch_idx, line_idx] = p1_all[batch_idx, pos]
                end[batch_idx, line_idx] = p2_all[batch_idx, pos]
                unit[batch_idx, line_idx] = unit_all[batch_idx, pos]
                valid[batch_idx, line_idx] = 1.0

        return {
            "start": start,
            "end": end,
            "unit": unit,
            "valid": valid,
        }
