import torch
import torch.nn as nn


class ConstraintTokenEncoder(nn.Module):
    def __init__(self, n_types: int = 6, max_lines: int = 64, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        half = max(1, d_model // 2)
        self.type_embed = nn.Embedding(n_types, d_model)
        self.line_embed = nn.Embedding(max_lines, half)
        self.pair_fuse = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, c_types: torch.Tensor, c_line_a: torch.Tensor, c_line_b: torch.Tensor) -> torch.Tensor:
        e_type = self.type_embed(c_types.long())
        e_a = self.line_embed(c_line_a.long())
        e_b = self.line_embed(c_line_b.long())
        e_lines = self.pair_fuse(torch.cat([e_a, e_b], dim=-1))
        return self.norm(self.out_proj(e_type + e_lines))


class SegmentEmbedding(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.embed = nn.Embedding(2, d_model)

    def forward(self, seg_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(seg_ids.long())
