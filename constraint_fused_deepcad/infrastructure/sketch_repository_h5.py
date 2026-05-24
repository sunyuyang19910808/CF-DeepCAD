import os

import h5py
import numpy as np

from cadlib.macro import EOS_VEC

from constraint_fused_deepcad.domain.entities import SketchSequenceAggregate
from constraint_fused_deepcad.sketch_preparation.batch_assembler import ConstraintBatchAssembler
from constraint_fused_deepcad.sketch_preparation.constraint_extractor import ConstraintExtractor


class SketchRepositoryH5:
    def __init__(self, data_root: str, max_lines: int = 64, max_constraints: int = 128, seq_len: int = 60, grid_size: int = 256):
        self.cad_vec_root = os.path.join(data_root, "cad_vec")
        split_path = os.path.join(data_root, "train_val_test_split.json")
        import json

        with open(split_path, "r") as fp:
            self._splits = json.load(fp)
        self.extractor = ConstraintExtractor(grid_size=grid_size)
        self.assembler = ConstraintBatchAssembler(max_lines=max_lines, max_constraints=max_constraints, seq_len=seq_len)

    def load(self, sample_id: str, phase: str = "train") -> SketchSequenceAggregate:
        h5_path = os.path.join(self.cad_vec_root, sample_id + ".h5")
        with h5py.File(h5_path, "r") as fp:
            cad_vec = fp["vec"][:]
        pad_len = self.assembler.seq_len - cad_vec.shape[0]
        if pad_len > 0:
            cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        elif cad_vec.shape[0] > self.assembler.seq_len:
            cad_vec = cad_vec[: self.assembler.seq_len]
        _, relations, _ = self.extractor.extract_from_cad_vec(cad_vec)
        return self.assembler.assemble_from_vec(cad_vec, relations, sample_id=sample_id)

    def list_ids(self, phase: str):
        return list(self._splits[phase])
