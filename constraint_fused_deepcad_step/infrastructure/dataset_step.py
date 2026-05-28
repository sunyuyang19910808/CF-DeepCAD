from __future__ import annotations

import os
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cadlib.macro import EOS_VEC
from model.model_utils import _get_group_mask

from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.domain.services import iter_line_command_positions
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.sketch_preparation.batch_assembler_high_modify import (
    ConstraintBatchAssemblerHighModify,
)
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.sketch_preparation.constraint_extractor_high_modify import (
    ConstraintExtractorHighModify,
)

from constraint_fused_deepcad_step.infrastructure.dataset_cache_step import (
    build_source_stat,
    clone_sample,
    load_cached_sample,
    resolve_cached_sample_path,
    resolve_dataset_cache_dir,
    save_cached_sample,
    source_stat_matches,
)
from constraint_fused_deepcad_step.infrastructure.repository import CadVectorRepository


class CADDatasetStep(Dataset):
    """Original DeepCAD vectors + GT geometry relation labels for L_geom."""

    def __init__(self, phase: str, config):
        super().__init__()
        self.phase = phase
        self.config = config
        self.repository = CadVectorRepository(config.data_root)
        self.all_data = self.repository.list_split_ids(phase)
        self.max_total_len = config.max_total_len
        self.extractor = ConstraintExtractorHighModify(
            angle_thresh=config.angle_thresh,
            dist_thresh=config.dist_thresh,
            grid_size=config.grid_size,
        )
        self.assembler = ConstraintBatchAssemblerHighModify(
            max_lines=config.max_lines,
            max_constraints=config.max_constraints,
            seq_len=config.max_total_len,
        )
        self._warned_bad_samples = set()
        self.cache_mode = getattr(config, "dataset_cache", "off")
        self._memory_cache: dict[str, dict] = {}
        self._memory_cache_source_stat: dict[str, dict] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_stale = 0
        if self.cache_mode == "disk":
            self._disk_cache_dir = resolve_dataset_cache_dir(config, phase)
            print(
                "Dataset cache enabled: mode=disk phase={} dir={} samples={}".format(
                    phase,
                    self._disk_cache_dir,
                    len(self.all_data),
                )
            )
        elif self.cache_mode == "memory":
            print(
                "Dataset cache enabled: mode=memory phase={} samples={}".format(
                    phase,
                    len(self.all_data),
                )
            )

    def __len__(self) -> int:
        return len(self.all_data)

    def cache_stats(self) -> dict:
        total = self._cache_hits + self._cache_misses
        return {
            "cache_mode": self.cache_mode,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_stale": self._cache_stale,
            "cache_hit_rate": (self._cache_hits / total) if total else 0.0,
        }

    def _source_stat_for_id(self, data_id: str):
        return build_source_stat(self.repository.resolve_h5_path(data_id))

    def _load_cad_vec_for_id(self, data_id: str) -> np.ndarray:
        cad_vec = self.repository.load_cad_vec(data_id)
        parse_source = np.asarray(cad_vec, dtype=np.int64)
        if parse_source.shape[0] > self.max_total_len:
            raise ValueError(
                "cad_vec length {} exceeds max_total_len {}.".format(parse_source.shape[0], self.max_total_len)
            )
        return parse_source

    def _warn_fallback(self, index: int, data_id: str) -> None:
        if self.all_data[index] != data_id:
            warnings.warn(
                "Sample {} is unavailable; falling back to {}.".format(self.all_data[index], data_id),
                RuntimeWarning,
            )

    def _warn_bad_sample(self, data_id: str, exc: Exception) -> None:
        if data_id not in self._warned_bad_samples:
            warnings.warn("Skipping invalid cad_vec sample {}: {}".format(data_id, exc), RuntimeWarning)
            self._warned_bad_samples.add(data_id)

    def _build_sample(self, data_id: str, parse_source: np.ndarray) -> dict:
        pad_len = self.max_total_len - parse_source.shape[0]
        cad_vec = np.concatenate([parse_source, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        _raw, relations, lines = self.extractor.extract_from_cad_vec(cad_vec)
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
            "unary_gt": aggregate.unary_gt,
            "pair_gt": aggregate.pair_gt,
            "cmd_padding_mask": aggregate.cmd_padding_mask,
            "line_count": torch.tensor(aggregate.line_count, dtype=torch.long),
            "line_cmd_mask": aggregate.line_cmd_mask,
            "line_index_map": aggregate.line_index_map,
            "id": data_id,
        }

    def _load_from_cache(self, data_id: str) -> dict | None:
        if self.cache_mode == "off":
            return None
        current_stat = self._source_stat_for_id(data_id)
        if self.cache_mode == "memory":
            cached = self._memory_cache.get(data_id)
            if cached is not None:
                if source_stat_matches(self._memory_cache_source_stat.get(data_id), current_stat):
                    self._cache_hits += 1
                    return cached
                self._memory_cache.pop(data_id, None)
                self._memory_cache_source_stat.pop(data_id, None)
                self._cache_stale += 1
            return None
        cache_path = resolve_cached_sample_path(self.config, self.phase, data_id)
        clone_on_load = int(getattr(self.config, "num_workers", 0)) > 0
        had_file = os.path.isfile(cache_path)
        cached = load_cached_sample(
            cache_path,
            clone=clone_on_load,
            current_source_stat=current_stat,
        )
        if cached is not None:
            self._cache_hits += 1
            return cached
        if had_file:
            self._cache_stale += 1
        return None

    def _store_in_cache(self, data_id: str, sample: dict) -> None:
        if self.cache_mode == "off":
            return
        self._cache_misses += 1
        source_stat = self._source_stat_for_id(data_id)
        if source_stat is None:
            return
        if self.cache_mode == "memory":
            self._memory_cache[data_id] = clone_sample(sample)
            self._memory_cache_source_stat[data_id] = source_stat
            return
        cache_path = resolve_cached_sample_path(self.config, self.phase, data_id)
        save_cached_sample(cache_path, sample, source_stat)

    def __getitem__(self, index: int):
        last_error = None
        total = len(self.all_data)
        for offset in range(total):
            candidate_index = (index + offset) % total
            data_id = self.all_data[candidate_index]

            cached = self._load_from_cache(data_id)
            if cached is not None:
                if offset > 0:
                    self._warn_fallback(index, data_id)
                return cached

            try:
                parse_source = self._load_cad_vec_for_id(data_id)
            except (OSError, FileNotFoundError, KeyError, ValueError) as exc:
                last_error = exc
                self._warn_bad_sample(data_id, exc)
                continue

            if offset > 0:
                self._warn_fallback(index, data_id)
            sample = self._build_sample(data_id, parse_source)
            self._store_in_cache(data_id, sample)
            return sample

        raise RuntimeError("No valid cad_vec samples could be loaded for phase {}.".format(self.phase)) from last_error


def get_step_dataloader(phase: str, config, shuffle: bool) -> DataLoader:
    dataset = CADDatasetStep(phase, config)
    num_workers = int(getattr(config, "num_workers", 0))
    cache_mode = getattr(config, "dataset_cache", "off")
    loader_kwargs = {
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    if num_workers > 0 and cache_mode == "memory":
        loader_kwargs["persistent_workers"] = True
    return DataLoader(dataset, **loader_kwargs)
