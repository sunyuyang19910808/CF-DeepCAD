from __future__ import annotations

import json
import os

from utils import ensure_dir

from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.application.generate_use_case import build_generation_use_case
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.config.config_constraint_fused_high_modify import (
    ConfigConstraintFusedHighModify,
)
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.infrastructure.dataset_high_modify import (
    get_high_modify_dataloader,
)


def main() -> None:
    cfg = ConfigConstraintFusedHighModify("test")
    out_dir = os.path.abspath(cfg.outputs or cfg.artifact_dir)
    ensure_dir(out_dir)

    generator = build_generation_use_case(cfg)
    loader = get_high_modify_dataloader(cfg.eval_split, cfg, shuffle=False)

    cases = []
    for batch in loader:
        z, _encoder_outputs = generator.encode_batch(batch)
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
