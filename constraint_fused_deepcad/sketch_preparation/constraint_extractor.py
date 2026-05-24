from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from cadlib.curves import Line
from cadlib.extrude import CADSequence
from cadlib.macro import EOS_VEC, LINE_IDX

from constraint_fused_deepcad.domain.entities import ConstraintRelation, ConstraintType


ANGLE_THRESH = 0.1
DIST_THRESH = 1e-3
EPS = 1e-5

_DEFAULT_EXTRACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ExtractData")

_TYPE_ID_TO_NAME: Dict[int, str] = {
    ConstraintType.HORIZONTAL: "HORIZONTAL",
    ConstraintType.VERTICAL: "VERTICAL",
    ConstraintType.PARALLEL: "PARALLEL",
    ConstraintType.PERPENDICULAR: "PERPENDICULAR",
    ConstraintType.COLLINEAR: "COLLINEAR",
    ConstraintType.NONE: "NONE",
}


def _normalize_split_phase(phase: str) -> str:
    """Map CLI aliases to keys in ``train_val_test_split.json`` (uses ``validation``, not ``val``)."""
    p = phase.strip().lower()
    if p == "val":
        return "validation"
    if p in ("train", "validation", "test"):
        return p
    raise ValueError(f"Unknown phase {phase!r}; expected train, val, validation, or test")


def _safe_sample_filename(data_id: str) -> str:
    """Split IDs may contain ``/``; flatten for a single path component."""
    return data_id.replace("\\", "_").replace("/", "_")


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u[:2]) + 1e-12)
    nv = float(np.linalg.norm(v[:2]) + 1e-12)
    c = float(np.dot(u[:2], v[:2]) / (nu * nv))
    c = min(1.0, max(-1.0, c))
    return math.degrees(math.acos(c))


def _undirected_line_angle_deg(d1: np.ndarray, d2: np.ndarray) -> float:
    a = _angle_deg(d1, d2)
    return min(a, 180.0 - a)


def _point_to_line_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b[:2] - a[:2]
    ap = p[:2] - a[:2]
    lab2 = float(np.dot(ab, ab) + 1e-12)
    t = max(0.0, min(1.0, float(np.dot(ap, ab) / lab2)))
    proj = a[:2] + t * ab
    return float(np.linalg.norm(p[:2] - proj))


@dataclass
class RawConstraintDict:
    horizontal: List[int] = field(default_factory=list)
    vertical: List[int] = field(default_factory=list)
    parallel: List[Tuple[int, int]] = field(default_factory=list)
    perpendicular: List[Tuple[int, int]] = field(default_factory=list)
    collinear: List[Tuple[int, int]] = field(default_factory=list)

    def to_jsonish(self) -> Dict[str, Any]:
        return {
            "horizontal": list(self.horizontal),
            "vertical": list(self.vertical),
            "parallel": [list(p) for p in self.parallel],
            "perpendicular": [list(p) for p in self.perpendicular],
            "collinear": [list(p) for p in self.collinear],
        }


def _line_unit_dir_xy(ln: Line) -> np.ndarray:
    d = ln.direction(from_start=True).astype(np.float64)
    nd = float(np.linalg.norm(d[:2]) + 1e-12)
    return d[:2] / nd


class ConstraintExtractor:
    def __init__(
        self,
        angle_thresh: float = ANGLE_THRESH,
        dist_thresh: float = DIST_THRESH,
        grid_size: int = 256,
    ):
        self.angle_thresh = angle_thresh
        self.dist_thresh = dist_thresh
        self.grid_size = grid_size

    def collect_lines_by_extrude(self, cad_seq: CADSequence) -> List[List[Line]]:
        """Profile → Loop → curve order within each Extrude (same as eval ``_lines_in_extrude``)."""
        out: List[List[Line]] = []
        for ext in cad_seq.seq:
            block: List[Line] = []
            for loop in ext.profile.children:
                for curve in loop.children:
                    if isinstance(curve, Line):
                        block.append(curve)
            out.append(block)
        return out

    def _axis_indices_from_dirs(self, dirs: List[np.ndarray]) -> Tuple[List[int], List[int]]:
        ex = np.array([1.0, 0.0], dtype=np.float64)
        ey = np.array([0.0, 1.0], dtype=np.float64)
        horizontal: List[int] = []
        vertical: List[int] = []
        for i, di in enumerate(dirs):
            if _undirected_line_angle_deg(di, ex) < self.angle_thresh:
                horizontal.append(i)
            if _undirected_line_angle_deg(di, ey) < self.angle_thresh:
                vertical.append(i)
        return horizontal, vertical

    def _pair_constraints_from_lines(
        self, lines: List[Line], dirs: List[np.ndarray], index_offset: int = 0
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Parallel / perpendicular / collinear only among ``lines``; indices shifted by ``index_offset``."""
        n = len(lines)
        parallel: List[Tuple[int, int]] = []
        perpendicular: List[Tuple[int, int]] = []
        collinear: List[Tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                ua = _undirected_line_angle_deg(dirs[i], dirs[j])
                if ua < self.angle_thresh:
                    parallel.append((index_offset + i, index_offset + j))
                    p1s, p1e = lines[i].start_point, lines[i].end_point
                    p2s, p2e = lines[j].start_point, lines[j].end_point
                    d1 = min(
                        _point_to_line_dist(p2s, p1s, p1e),
                        _point_to_line_dist(p2e, p1s, p1e),
                        _point_to_line_dist(p1s, p2s, p2e),
                        _point_to_line_dist(p1e, p2s, p2e),
                    )
                    if d1 < self.dist_thresh:
                        collinear.append((index_offset + i, index_offset + j))
                if abs(ua - 90.0) < self.angle_thresh:
                    perpendicular.append((index_offset + i, index_offset + j))
        return parallel, perpendicular, collinear

    def extract_raw_from_lines(self, lines: List[Line]) -> RawConstraintDict:
        """Extract constraints for one **contiguous** line list (typically one Extrude).

        Horizontal/vertical use the same angle test as the paper eval. Parallel, perpendicular, and
        collinear pairs are **only** between lines in this list — matching per-Extrude recall in
        ``论文尝试/DeepCAD原始约束指标/评估口径说明.md``. For a full ``CADSequence``, use
        :meth:`extract_raw_from_cad_sequence` so pair constraints are not formed across extrudes.
        """
        n = len(lines)
        if n == 0:
            return RawConstraintDict()

        dirs = [_line_unit_dir_xy(ln) for ln in lines]
        horizontal, vertical = self._axis_indices_from_dirs(dirs)
        parallel, perpendicular, collinear = self._pair_constraints_from_lines(lines, dirs, index_offset=0)

        return RawConstraintDict(
            horizontal=horizontal,
            vertical=vertical,
            parallel=parallel,
            perpendicular=perpendicular,
            collinear=collinear,
        )

    def extract_raw_from_cad_sequence(self, cad_seq: CADSequence) -> RawConstraintDict:
        """Same rules as eval: axis (h/v) over **all** Lines; pair constraints **per Extrude** with global indices."""
        by_ext = self.collect_lines_by_extrude(cad_seq)
        all_lines = [ln for block in by_ext for ln in block]
        if not all_lines:
            return RawConstraintDict()

        dirs_all = [_line_unit_dir_xy(ln) for ln in all_lines]
        horizontal, vertical = self._axis_indices_from_dirs(dirs_all)

        parallel: List[Tuple[int, int]] = []
        perpendicular: List[Tuple[int, int]] = []
        collinear: List[Tuple[int, int]] = []
        offset = 0
        for ext_lines in by_ext:
            if ext_lines:
                dirs = [_line_unit_dir_xy(ln) for ln in ext_lines]
                p, pp, c = self._pair_constraints_from_lines(ext_lines, dirs, index_offset=offset)
                parallel.extend(p)
                perpendicular.extend(pp)
                collinear.extend(c)
            offset += len(ext_lines)

        return RawConstraintDict(
            horizontal=horizontal,
            vertical=vertical,
            parallel=parallel,
            perpendicular=perpendicular,
            collinear=collinear,
        )

    def raw_to_relations(self, raw: RawConstraintDict) -> List[ConstraintRelation]:
        rels: List[ConstraintRelation] = []
        for lid in raw.horizontal:
            rels.append(ConstraintRelation(ConstraintType.HORIZONTAL, lid, lid))
        for lid in raw.vertical:
            rels.append(ConstraintRelation(ConstraintType.VERTICAL, lid, lid))
        for a, b in raw.parallel:
            rels.append(ConstraintRelation(ConstraintType.PARALLEL, a, b))
        for a, b in raw.perpendicular:
            rels.append(ConstraintRelation(ConstraintType.PERPENDICULAR, a, b))
        for a, b in raw.collinear:
            rels.append(ConstraintRelation(ConstraintType.COLLINEAR, a, b))
        return rels

    def collect_lines_from_cad_sequence(self, cad_seq: CADSequence) -> List[Line]:
        return [ln for block in self.collect_lines_by_extrude(cad_seq) for ln in block]

    def line_extrude_indices_from_cad_sequence(self, cad_seq: CADSequence) -> List[int]:
        """0-based Extrude index for each global line index (same order as ``collect_lines_from_cad_sequence``)."""
        return [k for k, block in enumerate(self.collect_lines_by_extrude(cad_seq)) for _ in block]

    def extrude_line_spans_from_cad_sequence(self, cad_seq: CADSequence) -> List[Dict[str, int]]:
        """Per Extrude: global ``first_line_index`` and ``n_lines`` (empty extrudes have ``n_lines`` 0)."""
        by_ext = self.collect_lines_by_extrude(cad_seq)
        spans: List[Dict[str, int]] = []
        offset = 0
        for k, block in enumerate(by_ext):
            n = len(block)
            spans.append({"extrude_idx": k, "first_line_index": offset, "n_lines": n})
            offset += n
        return spans

    def extract_from_cad_vec(self, cad_vec: np.ndarray) -> Tuple[RawConstraintDict, List[ConstraintRelation], List[Line]]:
        try:
            cad_seq = CADSequence.from_vector(cad_vec, is_numerical=True, n=self.grid_size)
        except Exception:
            return RawConstraintDict(), [], []
        lines = self.collect_lines_from_cad_sequence(cad_seq)
        raw = self.extract_raw_from_cad_sequence(cad_seq)
        relations = self.raw_to_relations(raw)
        return raw, relations, lines

    def line_indices_from_commands(self, commands: np.ndarray) -> List[int]:
        return [int(i) for i, c in enumerate(commands) if int(c) == LINE_IDX]


def constraint_dict_from_deepcad_json_sketch(all_stat: dict) -> RawConstraintDict:
    """Optional path: reuse CADSequence.from_dict for JSON-aligned samples."""
    cad_seq = CADSequence.from_dict(all_stat)
    ex = ConstraintExtractor()
    return ex.extract_raw_from_cad_sequence(cad_seq)


def _np_xy_to_list(p: np.ndarray) -> List[float]:
    a = np.asarray(p).reshape(-1)
    return [float(a[0]), float(a[1])] if a.size >= 2 else [float(a[0]), 0.0]


def relations_to_jsonish(
    relations: List[ConstraintRelation],
    line_extrude_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in relations:
        d: Dict[str, Any] = {
            "type_id": int(r.type_id),
            "type_name": _TYPE_ID_TO_NAME.get(int(r.type_id), "UNKNOWN"),
            "line_a": int(r.line_a),
            "line_b": int(r.line_b),
            "is_unary": bool(r.is_unary()),
        }
        if line_extrude_ids is not None and 0 <= int(r.line_a) < len(line_extrude_ids):
            d["extrude_idx"] = int(line_extrude_ids[int(r.line_a)])
        out.append(d)
    return out


def lines_to_jsonish(
    lines: List[Line],
    line_extrude_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, ln in enumerate(lines):
        d: Dict[str, Any] = {
            "index": i,
            "start_xy": _np_xy_to_list(ln.start_point),
            "end_xy": _np_xy_to_list(ln.end_point),
        }
        if line_extrude_ids is not None and i < len(line_extrude_ids):
            d["extrude_idx"] = int(line_extrude_ids[i])
        out.append(d)
    return out


def extract_from_cad_vec_with_error(
    extractor: ConstraintExtractor, cad_vec: np.ndarray
) -> Tuple[
    RawConstraintDict,
    List[ConstraintRelation],
    List[Line],
    Optional[str],
    List[int],
    List[Dict[str, int]],
]:
    """Like ``extract_from_cad_vec`` but returns a parse error string instead of swallowing exceptions.

    On success, also returns ``line_extrude_ids`` (parallel to ``lines``) and ``extrude_line_spans``
    for JSON dumps (see ``offline_extract_constraints``).
    """
    try:
        cad_seq = CADSequence.from_vector(cad_vec, is_numerical=True, n=extractor.grid_size)
    except Exception as e:
        return RawConstraintDict(), [], [], f"{type(e).__name__}: {e}", [], []
    lines = extractor.collect_lines_from_cad_sequence(cad_seq)
    raw = extractor.extract_raw_from_cad_sequence(cad_seq)
    relations = extractor.raw_to_relations(raw)
    line_extrude_ids = extractor.line_extrude_indices_from_cad_sequence(cad_seq)
    extrude_line_spans = extractor.extrude_line_spans_from_cad_sequence(cad_seq)
    return raw, relations, lines, None, line_extrude_ids, extrude_line_spans


def _maybe_pad_cad_vec(cad_vec: np.ndarray, pad_to: int) -> np.ndarray:
    if pad_to <= 0 or cad_vec.shape[0] >= pad_to:
        return cad_vec
    pad_len = pad_to - cad_vec.shape[0]
    return np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)


def offline_extract_constraints(
    data_root: str,
    out_dir: Optional[str] = None,
    phase: str = "train",
    limit: Optional[int] = None,
    angle_thresh: float = ANGLE_THRESH,
    dist_thresh: float = DIST_THRESH,
    grid_size: int = 256,
    pad_to: int = 0,
) -> Dict[str, Any]:
    """
    Read ``cad_vec/*.h5`` under ``data_root`` (split from ``train_val_test_split.json``),
    run constraint extraction, and write one JSON per sample under ``out_dir``.

    Default ``out_dir`` is ``sketch_preparation/ExtractData`` next to this module.

    ``phase`` may be ``val`` (alias for split key ``validation``).
    """
    split_phase = _normalize_split_phase(phase)
    out_dir = out_dir or _DEFAULT_EXTRACT_DIR
    cad_vec_root = os.path.join(data_root, "cad_vec")
    split_path = os.path.join(data_root, "train_val_test_split.json")
    os.makedirs(out_dir, exist_ok=True)
    phase_dir = os.path.join(out_dir, split_phase)
    os.makedirs(phase_dir, exist_ok=True)

    with open(split_path, "r", encoding="utf-8") as fp:
        ids: List[str] = list(json.load(fp)[split_phase])
    if limit is not None:
        ids = ids[: max(0, int(limit))]

    extractor = ConstraintExtractor(
        angle_thresh=angle_thresh, dist_thresh=dist_thresh, grid_size=grid_size
    )

    errors: List[Dict[str, str]] = []
    sample_relpaths: List[str] = []

    for data_id in ids:
        file_stem = _safe_sample_filename(data_id)
        h5_path = os.path.join(cad_vec_root, data_id + ".h5")
        relpath = os.path.join(split_phase, f"{file_stem}.json")
        sample_relpaths.append(relpath)
        out_path = os.path.join(out_dir, relpath)

        if not os.path.isfile(h5_path):
            err = f"missing_h5: {h5_path}"
            errors.append({"id": data_id, "error": err})
            record = {
                "sample_id": data_id,
                "phase": split_phase,
                "h5_path": h5_path,
                "parse_error": err,
                "raw_constraints": RawConstraintDict().to_jsonish(),
                "relations": [],
                "lines": [],
                "line_extrude_ids": [],
                "extrude_line_spans": [],
                "cad_vec_shape": None,
                "n_line_commands": None,
                "n_geometry_lines": 0,
                "extractor_config": {
                    "angle_thresh": angle_thresh,
                    "dist_thresh": dist_thresh,
                    "grid_size": grid_size,
                    "pad_to": pad_to,
                },
            }
            with open(out_path, "w", encoding="utf-8") as fp:
                json.dump(record, fp, ensure_ascii=False, indent=2)
            continue

        try:
            with h5py.File(h5_path, "r") as fp:
                cad_vec = fp["vec"][:]
        except OSError as e:
            err = f"h5_read_error: {e}"
            errors.append({"id": data_id, "error": err})
            record = {
                "sample_id": data_id,
                "phase": split_phase,
                "h5_path": h5_path,
                "parse_error": err,
                "raw_constraints": RawConstraintDict().to_jsonish(),
                "relations": [],
                "lines": [],
                "line_extrude_ids": [],
                "extrude_line_spans": [],
                "cad_vec_shape": None,
                "n_line_commands": None,
                "n_geometry_lines": 0,
                "extractor_config": {
                    "angle_thresh": angle_thresh,
                    "dist_thresh": dist_thresh,
                    "grid_size": grid_size,
                    "pad_to": pad_to,
                },
            }
            with open(out_path, "w", encoding="utf-8") as fp:
                json.dump(record, fp, ensure_ascii=False, indent=2)
            continue
        cad_vec_used = _maybe_pad_cad_vec(np.asarray(cad_vec, dtype=np.int64), pad_to)
        n_line_cmds = int((cad_vec_used[:, 0] == LINE_IDX).sum())

        raw, relations, lines, parse_error, line_extrude_ids, extrude_line_spans = extract_from_cad_vec_with_error(
            extractor, cad_vec_used
        )

        record = {
            "sample_id": data_id,
            "phase": split_phase,
            "h5_path": h5_path,
            "parse_error": parse_error,
            "cad_vec_shape": list(cad_vec.shape),
            "cad_vec_shape_used": list(cad_vec_used.shape),
            "n_line_commands": n_line_cmds,
            "n_geometry_lines": len(lines),
            "line_extrude_ids": line_extrude_ids,
            "extrude_line_spans": extrude_line_spans,
            "raw_constraints": raw.to_jsonish(),
            "relations": relations_to_jsonish(relations, line_extrude_ids),
            "lines": lines_to_jsonish(lines, line_extrude_ids),
            "extractor_config": {
                "angle_thresh": angle_thresh,
                "dist_thresh": dist_thresh,
                "grid_size": grid_size,
                "pad_to": pad_to,
            },
        }
        if parse_error:
            errors.append({"id": data_id, "error": parse_error})

        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(record, fp, ensure_ascii=False, indent=2)

    manifest = {
        "data_root": os.path.abspath(data_root),
        "out_dir": os.path.abspath(out_dir),
        "phase": split_phase,
        "limit": limit,
        "processed": len(ids),
        "sample_files": sample_relpaths,
        "errors": errors,
        "extractor_config": {
            "angle_thresh": angle_thresh,
            "dist_thresh": dist_thresh,
            "grid_size": grid_size,
            "pad_to": pad_to,
        },
    }
    manifest_path = os.path.join(out_dir, f"manifest_{split_phase}.json")
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline constraint extraction: dump raw constraints, relations, and line geometry to ExtractData."
    )
    p.add_argument(
        "data_root",
        type=str,
        help="DeepCAD data root containing cad_vec/ and train_val_test_split.json",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=_DEFAULT_EXTRACT_DIR,
        help=f"Output directory (default: {_DEFAULT_EXTRACT_DIR})",
    )
    p.add_argument(
        "--phase",
        type=str,
        default="train",
        help="Split: train | val | validation | test (val = validation in split JSON)",
    )
    p.add_argument(
        "--all-phases",
        action="store_true",
        help="Run train, validation, and test sequentially (ignores --phase)",
    )
    p.add_argument("--limit", type=int, default=None, help="Max number of samples from the split (default: all)")
    p.add_argument("--angle-thresh", type=float, default=ANGLE_THRESH)
    p.add_argument("--dist-thresh", type=float, default=DIST_THRESH)
    p.add_argument("--grid-size", type=int, default=256)
    p.add_argument(
        "--pad-to",
        type=int,
        default=0,
        help="If > 0, pad cad_vec with EOS to this length (e.g. 60 to match training max_total_len)",
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    phases = ("train", "validation", "test") if args.all_phases else (_normalize_split_phase(args.phase),)
    summaries: List[Dict[str, Any]] = []
    for sp in phases:
        m = offline_extract_constraints(
            data_root=args.data_root,
            out_dir=args.out_dir,
            phase=sp,
            limit=args.limit,
            angle_thresh=args.angle_thresh,
            dist_thresh=args.dist_thresh,
            grid_size=args.grid_size,
            pad_to=args.pad_to,
        )
        summaries.append(
            {
                "phase": m["phase"],
                "manifest": os.path.join(m["out_dir"], f"manifest_{m['phase']}.json"),
                "processed": m["processed"],
                "n_errors": len(m["errors"]),
            }
        )
    print(
        json.dumps(
            {"out_dir": os.path.abspath(args.out_dir), "runs": summaries},
            ensure_ascii=False,
            indent=2,
        )
    )
