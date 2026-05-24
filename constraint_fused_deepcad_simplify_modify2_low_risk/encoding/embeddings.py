from __future__ import annotations

import torch
import torch.nn as nn

from model.layers.positional_encoding import PositionalEncodingLUT


class ConstraintTagEmbedding(nn.Module):
    def __init__(self, n_constraint_types: int = 4, d_model: int = 256):
        super().__init__()
        self.tag_proj = nn.Sequential(
            nn.Linear(n_constraint_types, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, constraint_tags: torch.Tensor) -> torch.Tensor:
        return self.tag_proj(constraint_tags.float())


class CADEmbeddingFused(nn.Module):
    def __init__(self, cfg, seq_len: int):
        super().__init__()
        self.n_args = cfg.n_args
        self.args_dim = cfg.args_dim + 1
        self.use_group = getattr(cfg, "use_group_emb", True)

        self.command_embed = nn.Embedding(cfg.n_commands, cfg.d_model)
        self.arg_embed = nn.Embedding(self.args_dim, 64, padding_idx=0)
        self.embed_fcn = nn.Linear(64 * cfg.n_args, cfg.d_model)
        self.pos_encoding = PositionalEncodingLUT(cfg.d_model, max_len=seq_len + 2)
        self.constraint_tag = ConstraintTagEmbedding(4, cfg.d_model)

        if self.use_group:
            group_len = getattr(cfg, "max_num_groups", 30)
            self.group_embed = nn.Embedding(group_len + 2, cfg.d_model)

    def forward(
        self,
        commands: torch.Tensor,
        args: torch.Tensor,
        groups: torch.Tensor | None = None,
        constraint_tags: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src = self.command_embed(commands.long()) + self.embed_fcn(
            self.arg_embed((args + 1).long()).view(commands.shape[0], commands.shape[1], -1)
        )
        if self.use_group and groups is not None:
            src = src + self.group_embed(groups.long())
        if constraint_tags is not None:
            src = src + self.constraint_tag(constraint_tags)
        return self.pos_encoding(src)
