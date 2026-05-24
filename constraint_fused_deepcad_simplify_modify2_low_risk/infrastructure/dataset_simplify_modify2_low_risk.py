from __future__ import annotations

import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cadlib.macro import EOS_VEC
from model.model_utils import _get_group_mask

from constraint_fused_deepcad_simplify_modify2_low_risk.domain.services import iter_line_command_positions
from constraint_fused_deepcad_simplify_modify2_low_risk.infrastructure.repository import CadVectorRepository
from constraint_fused_deepcad_simplify_modify2_low_risk.sketch_preparation.batch_assembler_simplify_modify2_low_risk import (
    ConstraintBatchAssemblerSimplifyModify2LowRisk,
)
from constraint_fused_deepcad_simplify_modify2_low_risk.sketch_preparation.constraint_extractor_simplify_modify2_low_risk import (
    ConstraintExtractorSimplifyModify2LowRisk,
)


class CADDatasetSimplifyModify2LowRisk(Dataset):
    def __init__(self, phase: str, config):
        super().__init__()
        self.phase = phase
        self.config = config
        self.repository = CadVectorRepository(config.data_root)
        self.all_data = self.repository.list_split_ids(phase)
        self.max_total_len = config.max_total_len
        self.extractor = ConstraintExtractorSimplifyModify2LowRisk(
            angle_thresh=config.angle_thresh,
            dist_thresh=config.dist_thresh,
            grid_size=config.grid_size,
        )
        self.assembler = ConstraintBatchAssemblerSimplifyModify2LowRisk(
            max_lines=config.max_lines,
            max_constraints=config.max_constraints,
            seq_len=config.max_total_len,
        )
        self._warned_bad_samples = set()

    def __len__(self) -> int:
        return len(self.all_data)

    def _load_parse_source(self, index: int):
        last_error = None
        total = len(self.all_data)
        for offset in range(total):
            candidate_index = (index + offset) % total
            data_id = self.all_data[candidate_index]
            try:
                cad_vec = self.repository.load_cad_vec(data_id)
                parse_source = np.asarray(cad_vec, dtype=np.int64)
                if parse_source.shape[0] > self.max_total_len:
                    raise ValueError(
                        "cad_vec length {} exceeds max_total_len {}.".format(
                            parse_source.shape[0],
                            self.max_total_len,
                        )
                    )
                if offset > 0:
                    warnings.warn(
                        "Sample {} is unavailable; falling back to {}.".format(self.all_data[index], data_id),
                        RuntimeWarning,
                    )
                return data_id, parse_source
            except (OSError, FileNotFoundError, KeyError, ValueError) as exc:
                last_error = exc
                if data_id not in self._warned_bad_samples:
                    warnings.warn("Skipping invalid cad_vec sample {}: {}".format(data_id, exc), RuntimeWarning)
                    self._warned_bad_samples.add(data_id)
                continue
        raise RuntimeError("No valid cad_vec samples could be loaded for phase {}.".format(self.phase)) from last_error

    def __getitem__(self, index: int):
        data_id, parse_source = self._load_parse_source(index)
        pad_len = self.max_total_len - parse_source.shape[0]
        cad_vec = np.concatenate([parse_source, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        raw, relations, lines = self.extractor.extract_from_cad_vec(cad_vec)
        del raw
        max_valid_line_idx = min(len(iter_line_command_positions(cad_vec[:, 0])), len(lines))
        relations = [rel for rel in relations if rel.line_a < max_valid_line_idx and rel.line_b < max_valid_line_idx]
        aggregate = self.assembler.assemble_from_vec(
            cad_vec=cad_vec,
            relations=relations,
            geometry_line_count=len(lines),
            sample_id=data_id,
        )

        command = torch.tensor(cad_vec[:, 0], dtype=torch.long)
        args = torch.tensor(cad_vec[:, 1:], dtype=torch.long)
        groups = _get_group_mask(command.unsqueeze(1), seq_dim=0).squeeze(1).long()

        return {
            "command": command,
            "args": args,
            "groups": groups,
            "constraint_tags": aggregate.constraint_tags,
            "constraint_tokens": aggregate.constraint_tokens,
            "c_types": aggregate.c_types,
            "c_line_a": aggregate.c_line_a,
            "c_line_b": aggregate.c_line_b,
            "unary_gt": aggregate.unary_gt,
            "pair_gt": aggregate.pair_gt,
            "cmd_padding_mask": aggregate.cmd_padding_mask,
            "constraint_padding_mask": aggregate.constraint_padding_mask,
            "line_count": torch.tensor(aggregate.line_count, dtype=torch.long),
            "line_cmd_mask": aggregate.line_cmd_mask,
            "line_index_map": aggregate.line_index_map,
            "id": data_id,
        }


def fused_simplify_modify2_low_risk_collate_fn(batch):
    return {
        "command": torch.stack([item["command"] for item in batch], dim=0),
        "args": torch.stack([item["args"] for item in batch], dim=0),
        "groups": torch.stack([item["groups"] for item in batch], dim=0),
        "constraint_tags": torch.stack([item["constraint_tags"] for item in batch], dim=0),
        "constraint_tokens": torch.stack([item["constraint_tokens"] for item in batch], dim=0),
        "c_types": torch.stack([item["c_types"] for item in batch], dim=0),
        "c_line_a": torch.stack([item["c_line_a"] for item in batch], dim=0),
        "c_line_b": torch.stack([item["c_line_b"] for item in batch], dim=0),
        "unary_gt": torch.stack([item["unary_gt"] for item in batch], dim=0),
        "pair_gt": torch.stack([item["pair_gt"] for item in batch], dim=0),
        "cmd_padding_mask": torch.stack([item["cmd_padding_mask"] for item in batch], dim=0),
        "constraint_padding_mask": torch.stack([item["constraint_padding_mask"] for item in batch], dim=0),
        "line_count": torch.stack([item["line_count"] for item in batch], dim=0),
        "line_cmd_mask": torch.stack([item["line_cmd_mask"] for item in batch], dim=0),
        "line_index_map": torch.stack([item["line_index_map"] for item in batch], dim=0),
        "id": [item["id"] for item in batch],
    }


def get_simplify_modify2_low_risk_dataloader(phase: str, config, shuffle=None):
    is_shuffle = phase == "train" if shuffle is None else shuffle
    dataset = CADDatasetSimplifyModify2LowRisk(phase, config)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=is_shuffle,
        num_workers=config.num_workers,
        collate_fn=fused_simplify_modify2_low_risk_collate_fn,
    )
