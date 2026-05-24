from __future__ import annotations

import os

import torch


class ModelCheckpointRepositoryFs:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def save(self, state: dict, name: str = "latest.pth") -> str:
        path = os.path.join(self.model_dir, name)
        torch.save(state, path)
        return path

    def load(self, name: str = "latest.pth", map_location="cpu") -> dict:
        path = os.path.join(self.model_dir, name)
        return torch.load(path, map_location=map_location)
