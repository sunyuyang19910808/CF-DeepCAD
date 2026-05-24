from __future__ import annotations

import os

import h5py
from tqdm import tqdm

from cadlib.macro import EOS_IDX
from utils import ensure_dir

from constraint_fused_deepcad_simplify.application.evaluate_axis_constraints import (
    build_reconstruction_use_case,
    reconstruct_batch,
)
from constraint_fused_deepcad_simplify.config.config_constraint_fused_simplify import ConfigConstraintFusedSimplify
from constraint_fused_deepcad_simplify.infrastructure.dataset_simplify import get_simplify_dataloader


def reconstruct(cfg) -> None:
    loader = get_simplify_dataloader("test", cfg, shuffle=False)
    use_case, device = build_reconstruction_use_case(cfg)
    reconstruction_dir = os.path.abspath(cfg.reconstruction_dir or cfg.default_reconstruction_dir)
    ensure_dir(reconstruction_dir)

    for batch in tqdm(loader, desc="reconstruct"):
        out_vec, gt_vec = reconstruct_batch(use_case, batch, device)
        commands = gt_vec[:, :, 0]
        for batch_idx, data_id in enumerate(batch["id"]):
            seq_len = commands[batch_idx].tolist().index(EOS_IDX)
            file_stem = data_id.split("/")[-1]
            save_path = os.path.join(reconstruction_dir, "{}_vec.h5".format(file_stem))
            with h5py.File(save_path, "w") as file_obj:
                file_obj.create_dataset("out_vec", data=out_vec[batch_idx][:seq_len], dtype=int)
                file_obj.create_dataset("gt_vec", data=gt_vec[batch_idx][:seq_len], dtype=int)

    print("Wrote reconstruction to", reconstruction_dir)


def main() -> None:
    cfg = ConfigConstraintFusedSimplify("test")
    reconstruct(cfg)


if __name__ == "__main__":
    main()
