from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np
import torch

from cadlib.macro import CMD_ARGS_MASK

from constraint_fused_deepcad.domain.entities import ConstraintRelation, ConstraintType
from constraint_fused_deepcad.sketch_preparation.constraint_extractor import ConstraintExtractor


def logits_to_cad_vec(command_logits: torch.Tensor, args_logits: torch.Tensor) -> np.ndarray:
    """Greedy decode (batch 1..N) to cad_vec numpy (N, S, 1+N_ARGS)."""
    out_command = torch.argmax(torch.softmax(command_logits, dim=-1), dim=-1)
    out_args = torch.argmax(torch.softmax(args_logits, dim=-1), dim=-1) - 1
    dev = out_command.device
    mask = ~torch.tensor(CMD_ARGS_MASK).bool().to(dev)[out_command.long()]
    out_args = out_args.clone()
    out_args[mask] = -1
    return torch.cat([out_command.unsqueeze(-1), out_args], dim=-1).detach().cpu().numpy()


class EvaluateConstraintSatisfactionUseCase:
    def __init__(self, angle_thresh: float | None = None):
        self.extractor = ConstraintExtractor() if angle_thresh is None else ConstraintExtractor(angle_thresh=angle_thresh)

    @staticmethod
    def _pair_key(t: int, a: int, b: int) -> Tuple[int, int, int]:
        x, y = (a, b) if a <= b else (b, a)
        return (t, x, y)

    def _rel_set(self, relations: List[ConstraintRelation]) -> Set[Tuple[int, int, int]]:
        s = set()
        for r in relations:
            if r.type_id in (ConstraintType.HORIZONTAL, ConstraintType.VERTICAL):
                continue
            if r.type_id >= ConstraintType.NONE:
                continue
            s.add(self._pair_key(r.type_id, r.line_a, r.line_b))
        return s

    def evaluate_batch(self, gt_relations: List[List[ConstraintRelation]], pred_cad_vec: np.ndarray) -> Dict[str, float]:
        precisions = []
        recalls = []
        for gt, vec in zip(gt_relations, pred_cad_vec):
            _, pred_rels, _ = self.extractor.extract_from_cad_vec(vec)
            g = self._rel_set(gt)
            p = self._rel_set(pred_rels)
            if not g:
                continue
            inter = len(g & p)
            recalls.append(inter / len(g))
            precisions.append(inter / len(p) if p else 0.0)
        return {
            "constraint_pair_recall_mean": float(np.mean(recalls)) if recalls else 0.0,
            "constraint_pair_precision_mean": float(np.mean(precisions)) if precisions else 0.0,
            "constraint_satisfaction_rate": float(np.mean(recalls)) if recalls else 0.0,
        }
