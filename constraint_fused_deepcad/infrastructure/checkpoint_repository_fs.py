from __future__ import annotations

import os

import torch


class ModelCheckpointRepositoryFs:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def save(self, epoch: int, state: dict, name: str | None = None) -> str:
        fn = name or f"epoch_{epoch}.pth"
        path = os.path.join(self.model_dir, fn)
        torch.save(state, path)
        return path

    def load_latest(self, name: str = "latest.pth") -> dict:
        path = os.path.join(self.model_dir, name)
        return torch.load(path, map_location="cpu")
