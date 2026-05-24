from __future__ import annotations

import os

from utils import ensure_dir

from constraint_fused_deepcad_simplify_modify1.application.evaluate_axis_constraints import (
    reconstruct_test_split,
)
from constraint_fused_deepcad_simplify_modify1.config.config_constraint_fused_simplify_modify1 import (
    ConfigConstraintFusedSimplifyModify1,
)


def reconstruct(cfg) -> None:
    reconstruction_dir = os.path.abspath(cfg.reconstruction_dir or cfg.default_reconstruction_dir)
    ensure_dir(reconstruction_dir)
    reconstruct_test_split(cfg, reconstruction_dir)
    print("Wrote reconstruction to", reconstruction_dir)


def main() -> None:
    cfg = ConfigConstraintFusedSimplifyModify1("test")
    reconstruct(cfg)


if __name__ == "__main__":
    main()
