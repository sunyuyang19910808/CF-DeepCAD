from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from cadlib.curves import Line
from cadlib.extrude import CADSequence

from constraint_fused_deepcad_simplify.domain.entities import ConstraintRelationSimplify, ConstraintTypeSimplify


ANGLE_THRESH = 0.1


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


def extract_axis_constraints(lines: List[Line], angle_thresh: float) -> Tuple[List[int], List[int]]:
    horizontal = []
    vertical = []
    ex = np.array([1.0, 0.0], dtype=np.float64)
    ey = np.array([0.0, 1.0], dtype=np.float64)

    for line_idx, line in enumerate(lines):
        direction = line_direction_xy(line)
        if undirected_angle_deg(direction, ex) < angle_thresh:
            horizontal.append(line_idx)
        if undirected_angle_deg(direction, ey) < angle_thresh:
            vertical.append(line_idx)
    return horizontal, vertical


@dataclass
class RawAxisConstraintDict:
    horizontal: List[int] = field(default_factory=list)
    vertical: List[int] = field(default_factory=list)

    def to_jsonish(self) -> Dict[str, List[int]]:
        return {
            "horizontal": list(self.horizontal),
            "vertical": list(self.vertical),
        }


class ConstraintExtractorSimplify:
    def __init__(self, angle_thresh: float = ANGLE_THRESH, grid_size: int = 256):
        self.angle_thresh = angle_thresh
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

    def extract_raw_from_lines(self, lines: List[Line]) -> RawAxisConstraintDict:
        horizontal, vertical = extract_axis_constraints(lines, self.angle_thresh)
        return RawAxisConstraintDict(horizontal=horizontal, vertical=vertical)

    def extract_raw_from_cad_sequence(self, cad_seq: CADSequence) -> RawAxisConstraintDict:
        return self.extract_raw_from_lines(self.collect_lines_from_cad_sequence(cad_seq))

    def raw_to_relations(self, raw: RawAxisConstraintDict) -> List[ConstraintRelationSimplify]:
        relations = [
            ConstraintRelationSimplify(ConstraintTypeSimplify.HORIZONTAL, line_idx)
            for line_idx in raw.horizontal
        ]
        relations.extend(
            ConstraintRelationSimplify(ConstraintTypeSimplify.VERTICAL, line_idx)
            for line_idx in raw.vertical
        )
        return relations

    def extract_from_cad_vec(self, cad_vec: np.ndarray) -> Tuple[RawAxisConstraintDict, List[ConstraintRelationSimplify], List[Line]]:
        try:
            cad_seq = CADSequence.from_vector(cad_vec, is_numerical=True, n=self.grid_size)
        except Exception:
            return RawAxisConstraintDict(), [], []
        lines = self.collect_lines_from_cad_sequence(cad_seq)
        raw = self.extract_raw_from_lines(lines)
        return raw, self.raw_to_relations(raw), lines

    def extract_from_json(self, json_path: str) -> Tuple[RawAxisConstraintDict, List[ConstraintRelationSimplify], List[Line]]:
        with open(json_path, "r", encoding="utf-8") as file_obj:
            all_stat = json.load(file_obj)
        cad_seq = CADSequence.from_dict(all_stat)
        lines = self.collect_lines_from_cad_sequence(cad_seq)
        raw = self.extract_raw_from_lines(lines)
        return raw, self.raw_to_relations(raw), lines
