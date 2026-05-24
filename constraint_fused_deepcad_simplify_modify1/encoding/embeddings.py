from __future__ import annotations

import torch
import torch.nn as nn

from model.layers.positional_encoding import PositionalEncodingLUT


class AxisTagEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, tags: torch.Tensor) -> torch.Tensor:
        return self.proj(tags.float())


class CADEmbeddingWithAxisTags(nn.Module):
    def __init__(self, cfg, seq_len: int, use_group: bool = False, group_len: int = None):
        super().__init__()
        self.command_embed = nn.Embedding(cfg.n_commands, cfg.d_model)
        self.arg_embed = nn.Embedding(cfg.args_dim + 1, 64, padding_idx=0)
        self.embed_fcn = nn.Linear(64 * cfg.n_args, cfg.d_model)
        self.axis_tag_embed = AxisTagEmbedding(cfg.d_model)

        self.use_group = use_group
        if use_group:
            if group_len is None:
                group_len = cfg.max_num_groups
            self.group_embed = nn.Embedding(group_len + 2, cfg.d_model)

        self.pos_encoding = PositionalEncodingLUT(cfg.d_model, max_len=seq_len + 2)

    def forward(
        self,
        commands: torch.Tensor,
        args: torch.Tensor,
        groups: torch.Tensor = None,
        constraint_tags: torch.Tensor = None,
    ) -> torch.Tensor:
        src = self.command_embed(commands.long())
        src = src + self.embed_fcn(self.arg_embed((args + 1).long()).view(commands.size(0), commands.size(1), -1))
        if self.use_group and groups is not None:
            src = src + self.group_embed(groups.long())
        if constraint_tags is not None:
            src = src + self.axis_tag_embed(constraint_tags)
        return self.pos_encoding(src)
