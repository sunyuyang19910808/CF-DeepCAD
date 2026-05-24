from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict

import torch

from utils import ensure_dir

CACHE_FORMAT_VERSION = 2
CACHE_KEY_SCHEMA = 1
TENSOR_FIELDS = (
    "command",
    "args",
    "groups",
    "constraint_tags",
    "constraint_tokens",
    "c_types",
    "c_line_a",
    "c_line_b",
    "unary_gt",
    "pair_gt",
    "cmd_padding_mask",
    "constraint_padding_mask",
    "line_count",
    "line_cmd_mask",
    "line_index_map",
)


def build_dataset_cache_key(config) -> str:
    payload = {
        "version": CACHE_KEY_SCHEMA,
        "angle_thresh": float(config.angle_thresh),
        "dist_thresh": float(config.dist_thresh),
        "grid_size": int(config.grid_size),
        "max_total_len": int(config.max_total_len),
        "max_lines": int(config.max_lines),
        "max_constraints": int(config.max_constraints),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]


def resolve_dataset_cache_dir(config, phase: str) -> str:
    override = (getattr(config, "dataset_cache_dir", "") or "").strip()
    cache_key = build_dataset_cache_key(config)
    if override:
        return os.path.join(os.path.abspath(override), cache_key, phase)
    data_root = os.path.abspath(config.data_root)
    return os.path.join(data_root, ".cache", "high_modify", cache_key, phase)


def _safe_cache_filename(data_id: str) -> str:
    safe = data_id.replace("\\", "__").replace("/", "__")
    return "{}.pt".format(safe)


def build_source_stat(h5_path: str) -> Dict[str, int] | None:
    try:
        st = os.stat(h5_path)
    except OSError:
        return None
    return {"mtime_ns": int(st.st_mtime_ns), "size": int(st.st_size)}


def source_stat_matches(stored: Dict[str, Any] | None, current: Dict[str, int] | None) -> bool:
    if not stored or current is None:
        return False
    return int(stored.get("mtime_ns", -1)) == current["mtime_ns"] and int(stored.get("size", -1)) == current["size"]


def _invalidate_cached_file(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def resolve_cached_sample_path(config, phase: str, data_id: str) -> str:
    return os.path.join(resolve_dataset_cache_dir(config, phase), _safe_cache_filename(data_id))


def clone_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    cloned = {}
    for key in TENSOR_FIELDS:
        value = sample[key]
        cloned[key] = value.clone() if torch.is_tensor(value) else torch.tensor(value)
    cloned["id"] = sample["id"]
    return cloned


def load_cached_sample(
    path: str,
    *,
    clone: bool = True,
    current_source_stat: Dict[str, int] | None,
) -> Dict[str, Any] | None:
    if not os.path.isfile(path):
        return None
    try:
        payload = torch.load(path, map_location="cpu")
        if payload.get("version") != CACHE_FORMAT_VERSION:
            _invalidate_cached_file(path)
            return None
        if not source_stat_matches(payload.get("source_stat"), current_source_stat):
            _invalidate_cached_file(path)
            return None
        sample = payload["sample"]
        return clone_sample(sample) if clone else sample
    except (OSError, RuntimeError, KeyError, TypeError, ValueError):
        _invalidate_cached_file(path)
        return None


def save_cached_sample(path: str, sample: Dict[str, Any], source_stat: Dict[str, int] | None) -> None:
    if os.path.isfile(path):
        return
    if source_stat is None:
        return
    ensure_dir(os.path.dirname(path))
    tmp_path = "{}.tmp.{}".format(path, os.getpid())
    payload = {
        "version": CACHE_FORMAT_VERSION,
        "source_stat": source_stat,
        "sample": clone_sample(sample),
    }
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
