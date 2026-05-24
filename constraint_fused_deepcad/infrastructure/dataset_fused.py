import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from cadlib.macro import EOS_IDX, EOS_VEC, EXT_IDX, LINE_IDX, N_ARGS, PAD_VAL

from constraint_fused_deepcad.sketch_preparation.batch_assembler import ConstraintBatchAssembler
from constraint_fused_deepcad.sketch_preparation.constraint_extractor import ConstraintExtractor
from dataset.cad_dataset import CADDataset
from model.model_utils import _get_group_mask, _get_key_padding_mask


class FusedCADDataset(CADDataset):
    """DeepCAD h5 samples + on-the-fly constraint tensors (aug disabled for label consistency)."""

    def __init__(self, phase, config):
        super().__init__(phase, config)
        self.aug = False
        self.max_lines = getattr(config, "max_lines", 64)
        self.max_constraints = getattr(config, "max_constraints", 128)
        self.extractor = ConstraintExtractor(grid_size=self.size)
        self.assembler = ConstraintBatchAssembler(
            max_lines=self.max_lines,
            max_constraints=self.max_constraints,
            seq_len=self.max_total_len,
        )

    def __getitem__(self, index):
        data_id = self.all_data[index]
        h5_path = os.path.join(self.raw_data, data_id + ".h5")
        with h5py.File(h5_path, "r") as fp:
            cad_vec = fp["vec"][:]

        pad_len = self.max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        command = torch.tensor(cad_vec[:, 0], dtype=torch.long)
        args = torch.tensor(cad_vec[:, 1:], dtype=torch.long)

        _, relations, lines = self.extractor.extract_from_cad_vec(cad_vec)
        n_line_cmds = int((cad_vec[:, 0] == LINE_IDX).sum())
        n_geom = len(lines)
        n_lines = min(n_line_cmds, n_geom) if n_geom else n_line_cmds

        def ok(rel):
            if rel.type_id >= 5:
                return False
            mx = max(rel.line_a, rel.line_b)
            return mx < n_lines

        relations = [r for r in relations if ok(r)]

        agg = self.assembler.assemble_from_vec(cad_vec, relations, sample_id=data_id)

        groups = _get_group_mask(command.unsqueeze(1), seq_dim=0).squeeze(1).long()
        cmd_padding_mask = _get_key_padding_mask(command.unsqueeze(1), seq_dim=0).squeeze(0)

        return {
            "command": command,
            "args": args,
            "id": data_id,
            "constraint_tags": agg.constraint_tags,
            "c_types": agg.c_types,
            "c_line_a": agg.c_line_a,
            "c_line_b": agg.c_line_b,
            "unary_gt": agg.unary_gt,
            "pair_gt": agg.pair_gt,
            "cmd_padding_mask": cmd_padding_mask,
            "constraint_padding_mask": agg.constraint_padding_mask,
            "line_count": agg.line_count,
            "groups": groups,
        }


def fused_collate_fn(batch):
    commands = torch.stack([b["command"] for b in batch], dim=0)
    args = torch.stack([b["args"] for b in batch], dim=0)
    groups = torch.stack([b["groups"] for b in batch], dim=0)
    constraint_tags = torch.stack([b["constraint_tags"] for b in batch], dim=0)
    c_types = torch.stack([b["c_types"] for b in batch], dim=0)
    c_line_a = torch.stack([b["c_line_a"] for b in batch], dim=0)
    c_line_b = torch.stack([b["c_line_b"] for b in batch], dim=0)
    unary_gt = torch.stack([b["unary_gt"] for b in batch], dim=0)
    pair_gt = torch.stack([b["pair_gt"] for b in batch], dim=0)
    cmd_padding_mask = torch.stack([b["cmd_padding_mask"] for b in batch], dim=0)
    constraint_padding_mask = torch.stack([b["constraint_padding_mask"] for b in batch], dim=0)
    line_counts = [int(b["line_count"]) for b in batch]
    return {
        "command": commands,
        "args": args,
        "groups": groups,
        "constraint_tags": constraint_tags,
        "c_types": c_types,
        "c_line_a": c_line_a,
        "c_line_b": c_line_b,
        "unary_gt": unary_gt,
        "pair_gt": pair_gt,
        "cmd_padding_mask": cmd_padding_mask,
        "constraint_padding_mask": constraint_padding_mask,
        "line_counts": line_counts,
    }


def get_fused_dataloader(phase, config, shuffle=None):
    is_shuffle = phase == "train" if shuffle is None else shuffle
    ds = FusedCADDataset(phase, config)
    return DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=is_shuffle,
        num_workers=getattr(config, "num_workers", 0),
        collate_fn=fused_collate_fn,
    )
