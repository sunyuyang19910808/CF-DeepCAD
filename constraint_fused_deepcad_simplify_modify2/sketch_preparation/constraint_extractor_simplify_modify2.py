from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from cadlib.curves import Line
from cadlib.extrude import CADSequence

from constraint_fused_deepcad_simplify_modify2.domain.entities import ConstraintRelation, ConstraintType

ANGLE_THRESH = 0.1
DIST_THRESH = 1e-3


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u[:2]) + 1e-12)
    nv = float(np.linalg.norm(v[:2]) + 1e-12)
    cosine = float(np.dot(u[:2], v[:2]) / (nu * nv))
    cosine = min(1.0, max(-1.0, cosine))
    return math.degrees(math.acos(cosine))


def undirected_angle_deg(d1: np.ndarray, d2: np.ndarray) -> float:
    angle = _angle_deg(d1, d2)
    return min(angle, 180.0 - angle)


def line_direction_xy(line: Line) -> np.ndarray:
    direction = line.direction(from_start=True).astype(np.float64)
    norm = float(np.linalg.norm(direction[:2]) + 1e-12)
    return direction[:2] / norm


@dataclass
class RawConstraintDict:
    horizontal: List[int] = field(default_factory=list)
    vertical: List[int] = field(default_factory=list)
    parallel: List[Tuple[int, int]] = field(default_factory=list)
    perpendicular: List[Tuple[int, int]] = field(default_factory=list)

    def to_jsonish(self) -> Dict[str, List]:
        return {
            "horizontal": list(self.horizontal),
            "vertical": list(self.vertical),
            "parallel": [list(p) for p in self.parallel],
            "perpendicular": [list(p) for p in self.perpendicular],
        }


class ConstraintExtractorSimplifyModify2:
    def __init__(self, angle_thresh: float = ANGLE_THRESH, dist_thresh: float = DIST_THRESH, grid_size: int = 256):
        self.angle_thresh = angle_thresh
        self.dist_thresh = dist_thresh
        self.grid_size = grid_size

    def collect_lines_by_extrude(self, cad_seq: CADSequence) -> List[List[Line]]:
        blocks: List[List[Line]] = []
        for ext in cad_seq.seq:
            current: List[Line] = []
            for loop in ext.profile.children:
                for curve in loop.children:
                    if isinstance(curve, Line):
                        current.append(curve)
            blocks.append(current)
        return blocks

    def collect_lines_from_cad_sequence(self, cad_seq: CADSequence) -> List[Line]:
        return [line for block in self.collect_lines_by_extrude(cad_seq) for line in block]

    def _axis_indices_from_dirs(self, dirs: List[np.ndarray]) -> Tuple[List[int], List[int]]:
        horizontal = []
        vertical = []
        ex = np.array([1.0, 0.0], dtype=np.float64)
        ey = np.array([0.0, 1.0], dtype=np.float64)
        for line_idx, direction in enumerate(dirs):
            if undirected_angle_deg(direction, ex) < self.angle_thresh:
                horizontal.append(line_idx)
            if undirected_angle_deg(direction, ey) < self.angle_thresh:
                vertical.append(line_idx)
        return horizontal, vertical

    def _pair_constraints_from_lines(
        self,
        lines: List[Line],
        dirs: List[np.ndarray],
        index_offset: int = 0,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        parallel: List[Tuple[int, int]] = []
        perpendicular: List[Tuple[int, int]] = []
        for left in range(len(lines)):
            for right in range(left + 1, len(lines)):
                angle = undirected_angle_deg(dirs[left], dirs[right])
                if angle < self.angle_thresh:
                    parallel.append((index_offset + left, index_offset + right))
                if abs(angle - 90.0) < self.angle_thresh:
                    perpendicular.append((index_offset + left, index_offset + right))
        return parallel, perpendicular

    def extract_raw_from_lines(self, lines: List[Line]) -> RawConstraintDict:
        if not lines:
            return RawConstraintDict()
        dirs = [line_direction_xy(line) for line in lines]
        horizontal, vertical = self._axis_indices_from_dirs(dirs)
        parallel, perpendicular = self._pair_constraints_from_lines(lines, dirs)
        return RawConstraintDict(
            horizontal=horizontal,
            vertical=vertical,
            parallel=parallel,
            perpendicular=perpendicular,
        )

    def extract_raw_from_cad_sequence(self, cad_seq: CADSequence) -> RawConstraintDict:
        by_ext = self.collect_lines_by_extrude(cad_seq)
        all_lines = [line for block in by_ext for line in block]
        if not all_lines:
            return RawConstraintDict()
        dirs_all = [line_direction_xy(line) for line in all_lines]
        horizontal, vertical = self._axis_indices_from_dirs(dirs_all)
        parallel: List[Tuple[int, int]] = []
        perpendicular: List[Tuple[int, int]] = []
        offset = 0
        for block in by_ext:
            if block:
                dirs = [line_direction_xy(line) for line in block]
                p, pp = self._pair_constraints_from_lines(block, dirs, index_offset=offset)
                parallel.extend(p)
                perpendicular.extend(pp)
            offset += len(block)
        return RawConstraintDict(
            horizontal=horizontal,
            vertical=vertical,
            parallel=parallel,
            perpendicular=perpendicular,
        )

    def raw_to_relations(self, raw: RawConstraintDict) -> List[ConstraintRelation]:
        relations = [ConstraintRelation(ConstraintType.HORIZONTAL, line_idx, line_idx) for line_idx in raw.horizontal]
        relations.extend(ConstraintRelation(ConstraintType.VERTICAL, line_idx, line_idx) for line_idx in raw.vertical)
        relations.extend(ConstraintRelation(ConstraintType.PARALLEL, a, b) for a, b in raw.parallel)
        relations.extend(
            ConstraintRelation(ConstraintType.PERPENDICULAR, a, b) for a, b in raw.perpendicular
        )
        return relations

    def extract_from_cad_vec(self, cad_vec: np.ndarray) -> Tuple[RawConstraintDict, List[ConstraintRelation], List[Line]]:
        try:
            cad_seq = CADSequence.from_vector(cad_vec, is_numerical=True, n=self.grid_size)
        except Exception:
            return RawConstraintDict(), [], []
        lines = self.collect_lines_from_cad_sequence(cad_seq)
        raw = self.extract_raw_from_cad_sequence(cad_seq)
        return raw, self.raw_to_relations(raw), lines

    def extract_from_json(self, json_path: str) -> Tuple[RawConstraintDict, List[ConstraintRelation], List[Line]]:
        with open(json_path, "r", encoding="utf-8") as file_obj:
            all_stat = json.load(file_obj)
        cad_seq = CADSequence.from_dict(all_stat)
        lines = self.collect_lines_from_cad_sequence(cad_seq)
        raw = self.extract_raw_from_cad_sequence(cad_seq)
        return raw, self.raw_to_relations(raw), lines

