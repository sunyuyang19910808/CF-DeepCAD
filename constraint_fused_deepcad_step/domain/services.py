from __future__ import annotations

import torch


def build_line_mask(line_count: torch.Tensor, max_lines: int) -> torch.Tensor:
    if line_count.dim() == 0:
        line_count = line_count.unsqueeze(0)
    line_indices = torch.arange(max_lines, device=line_count.device).unsqueeze(0)
    return line_indices < line_count.unsqueeze(1)
