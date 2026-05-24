from __future__ import annotations

import torch
import torch.nn as nn

from cadlib.macro import ARC_IDX, CIRCLE_IDX, LINE_IDX, SOL_IDX


class DifferentiableSketchInterpreter(nn.Module):
    """Soft-dequantize ``args_logits`` into per-line geometric features.

    Two interpretations of a line's ``(start, end)`` are supported:

    * Legacy (``use_corrected_line_start=False``): take ``args[..., 0:2]`` as ``p1`` and
      ``args[..., 2:4]`` as ``p2``. This matches the original A2b/A2c behaviour but ``p2`` falls
      on Line's PAD dimensions (see ``cadlib.macro.CMD_ARGS_MASK``), so ``unit`` is not the
      real line direction.
    * A2d (``use_corrected_line_start=True``): ``end = args[pos, 0:2]`` and ``start`` is the
      previous curve token's ``args[prev_pos, 0:2]`` within the same sketch (segment between
      consecutive ``SOL`` tokens). The first line in a sketch and any line crossing a ``SOL``
      boundary uses ``(0, 0)`` as the start (sketch origin).
    """

    def __init__(
        self,
        n_bins: int,
        coord_range=(-1.0, 1.0),
        eps: float = 1e-6,
        use_corrected_line_start: bool = False,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.coord_range = coord_range
        self.eps = eps
        self.use_corrected_line_start = bool(use_corrected_line_start)

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
        commands: torch.Tensor | None = None,
    ) -> dict:
        arg_cont = self.soft_dequantize(arg_logits)
        batch_size = line_cmd_mask.shape[0]
        device = arg_logits.device
        dtype = arg_cont.dtype

        start = torch.zeros(batch_size, max_lines, 2, device=device, dtype=dtype)
        end = torch.zeros(batch_size, max_lines, 2, device=device, dtype=dtype)
        unit = torch.zeros(batch_size, max_lines, 2, device=device, dtype=dtype)
        valid = torch.zeros(batch_size, max_lines, device=device, dtype=dtype)

        if self.use_corrected_line_start:
            if commands is None:
                raise ValueError(
                    "use_corrected_line_start=True requires the `commands` tensor to "
                    "be passed to DifferentiableSketchInterpreter.forward."
                )
            curve_ids = {LINE_IDX, ARC_IDX, CIRCLE_IDX}
            for batch_idx in range(batch_size):
                line_positions = torch.nonzero(line_cmd_mask[batch_idx], as_tuple=False).flatten()
                cmds_b = commands[batch_idx].tolist()
                for pos in line_positions.tolist():
                    line_idx = int(line_index_map[batch_idx, pos].item())
                    if line_idx < 0 or line_idx >= max_lines:
                        continue
                    end_pt = arg_cont[batch_idx, pos, 0:2]
                    prev_pos = -1
                    for scan in range(pos - 1, -1, -1):
                        cmd_id = int(cmds_b[scan])
                        if cmd_id == SOL_IDX:
                            break
                        if cmd_id in curve_ids:
                            prev_pos = scan
                            break
                    if prev_pos >= 0:
                        start_pt = arg_cont[batch_idx, prev_pos, 0:2]
                    else:
                        start_pt = torch.zeros(2, device=device, dtype=dtype)
                    direction = end_pt - start_pt
                    norm = torch.norm(direction, dim=-1, keepdim=True).clamp_min(self.eps)
                    unit_pt = direction / norm
                    start[batch_idx, line_idx] = start_pt
                    end[batch_idx, line_idx] = end_pt
                    unit[batch_idx, line_idx] = unit_pt
                    valid[batch_idx, line_idx] = 1.0
        else:
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
