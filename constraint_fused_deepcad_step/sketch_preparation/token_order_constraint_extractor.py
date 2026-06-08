from __future__ import annotations

from typing import List, Tuple

import numpy as np

from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.domain.entities import (
    ConstraintRelation,
)
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.sketch_preparation.constraint_extractor_high_modify import (
    ANGLE_THRESH,
    DIST_THRESH,
    RawConstraintDict,
    ConstraintExtractorHighModify,
    undirected_angle_deg,
)
from constraint_fused_deepcad_step.domain.token_line_geometry import collect_token_order_line_directions


class TokenOrderConstraintExtractorStep:
    """Extract h/v/parallel/perpendicular GT in Line token order (``line_index_map``)."""

    def __init__(
        self,
        angle_thresh: float = ANGLE_THRESH,
        dist_thresh: float = DIST_THRESH,
        grid_size: int = 256,
    ):
        self.angle_thresh = angle_thresh
        self.dist_thresh = dist_thresh
        self.grid_size = grid_size
        self._relation_builder = ConstraintExtractorHighModify(
            angle_thresh=angle_thresh,
            dist_thresh=dist_thresh,
            grid_size=grid_size,
        )

    def _axis_indices_from_dirs(self, dirs: List[np.ndarray]) -> tuple[list[int], list[int]]:
        horizontal: list[int] = []
        vertical: list[int] = []
        ex = np.array([1.0, 0.0], dtype=np.float64)
        ey = np.array([0.0, 1.0], dtype=np.float64)
        for line_idx, direction in enumerate(dirs):
            if undirected_angle_deg(direction, ex) < self.angle_thresh:
                horizontal.append(line_idx)
            if undirected_angle_deg(direction, ey) < self.angle_thresh:
                vertical.append(line_idx)
        return horizontal, vertical

    def _pair_constraints_token_order(
        self,
        dirs: List[np.ndarray],
        extrude_ids: List[int],
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        parallel: list[tuple[int, int]] = []
        perpendicular: list[tuple[int, int]] = []
        if len(dirs) < 2:
            return parallel, perpendicular

        blocks: dict[int, list[int]] = {}
        for line_idx, block_id in enumerate(extrude_ids):
            blocks.setdefault(block_id, []).append(line_idx)

        for indices in blocks.values():
            for left_pos in range(len(indices)):
                for right_pos in range(left_pos + 1, len(indices)):
                    left = indices[left_pos]
                    right = indices[right_pos]
                    angle = undirected_angle_deg(dirs[left], dirs[right])
                    if angle < self.angle_thresh:
                        parallel.append((left, right))
                    if abs(angle - 90.0) < self.angle_thresh:
                        perpendicular.append((left, right))
        return parallel, perpendicular

    def extract_from_cad_vec(
        self,
        cad_vec: np.ndarray,
    ) -> Tuple[RawConstraintDict, List[ConstraintRelation], int]:
        try:
            _positions, directions, extrude_ids = collect_token_order_line_directions(cad_vec)
        except Exception:
            return RawConstraintDict(), [], 0

        line_count = len(directions)
        if line_count == 0:
            return RawConstraintDict(), [], 0

        horizontal, vertical = self._axis_indices_from_dirs(directions)
        parallel, perpendicular = self._pair_constraints_token_order(directions, extrude_ids)
        raw = RawConstraintDict(
            horizontal=horizontal,
            vertical=vertical,
            parallel=parallel,
            perpendicular=perpendicular,
        )
        relations = self._relation_builder.raw_to_relations(raw)
        return raw, relations, line_count
