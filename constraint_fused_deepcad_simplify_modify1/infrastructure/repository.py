from __future__ import annotations

import json
import os
from typing import List

import h5py
import numpy as np


class CadVectorRepository:
    def __init__(self, data_root: str):
        self.data_root = os.path.abspath(data_root)
        self.cad_vec_root = os.path.join(self.data_root, "cad_vec")
        self.split_path = os.path.join(self.data_root, "train_val_test_split.json")

    def list_split_ids(self, phase: str) -> List[str]:
        with open(self.split_path, "r", encoding="utf-8") as file_obj:
            return list(json.load(file_obj)[phase])

    def resolve_h5_path(self, data_id: str) -> str:
        return os.path.join(self.cad_vec_root, data_id + ".h5")

    def load_cad_vec(self, data_id: str) -> np.ndarray:
        h5_path = self.resolve_h5_path(data_id)
        with h5py.File(h5_path, "r") as file_obj:
            return file_obj["vec"][:]
