from __future__ import annotations

import json
import os

from utils import ensure_dir

from constraint_fused_deepcad_simplify_modify2.application.generate_use_case import build_generation_use_case
from constraint_fused_deepcad_simplify_modify2.config.config_constraint_fused_simplify_modify2 import (
    ConfigConstraintFusedSimplifyModify2,
)
from constraint_fused_deepcad_simplify_modify2.infrastructure.dataset_simplify_modify2 import (
    get_simplify_modify2_dataloader,
)


def main() -> None:
    cfg = ConfigConstraintFusedSimplifyModify2("test")
    out_dir = os.path.abspath(cfg.outputs or cfg.artifact_dir)
    ensure_dir(out_dir)

    generator = build_generation_use_case(cfg)
    loader = get_simplify_modify2_dataloader(cfg.eval_split, cfg, shuffle=False)

    cases = []
    for batch_idx, batch in enumerate(loader):
        z = generator.encode_batch(batch)
        out_vec = generator.decode_latent(z)
        for sample_idx, sample_id in enumerate(batch["id"]):
            cases.append(
                {
                    "id": sample_id,
                    "mode": "encoded_latent",
                    "decoded_shape": list(out_vec[sample_idx].shape),
                }
            )
            if len(cases) >= cfg.sample_count:
                break
        if len(cases) >= cfg.sample_count:
            break

    random_vec = generator.generate_from_random(batch_size=min(cfg.sample_count, 4), dim_z=cfg.dim_z)
    payload = {
        "sample_count": len(cases),
        "cases": cases,
        "random_generation_shape": list(random_vec.shape),
    }
    out_path = os.path.join(out_dir, "latent_generation_summary.json")
    with open(out_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()

