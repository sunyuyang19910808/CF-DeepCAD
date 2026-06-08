from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import torch

from constraint_fused_deepcad_step.application.synthetic_batch import make_synthetic_batch
from constraint_fused_deepcad_step.application.device import resolve_device
from constraint_fused_deepcad_step.application.train_use_case import build_train_use_case
from constraint_fused_deepcad_step.config.config_step import ConfigStep
from constraint_fused_deepcad_step.infrastructure.dataset_step import get_step_dataloader


def _make_cfg(**overrides) -> SimpleNamespace:
    cfg = ConfigStep.__new__(ConfigStep)
    cfg.set_configuration()
    base = {
        "proj_dir": "proj_log/constraint_fused_deepcad_step",
        "data_root": "data",
        "exp_name": "verify_gates",
        "gpu_ids": "cpu",
        "device": "cpu",
        "batch_size": 2,
        "num_workers": 0,
        "nr_epochs": 1,
        "lr": 1e-3,
        "grad_clip": 1.0,
        "warmup_step": 1,
        "save_frequency": 1,
        "val_frequency": 1000,
        "log_frequency": 1,
        "cont": False,
        "ckpt": "latest",
        "force_overwrite": False,
        "max_steps": 0,
        "angle_thresh": 0.1,
        "dist_thresh": 1e-3,
        "grid_size": 256,
        "coord_range_min": -1.0,
        "coord_range_max": 1.0,
        "eval_split": "test",
        "enable_geom_loss": False,
        "geom_log_only": False,
        "geom_positive_only": True,
        "gamma_geom": 0.0,
        "geom_loss_mode": "angle_hinge",
        "geom_bce_scale": 4.0,
        "geom_negative_weight": 0.0,
        "geom_warmup_start_epoch": 1,
        "geom_warmup_end_epoch": 1,
        "geom_target_ratio": 0.0,
        "geom_ratio_ema": 0.99,
        "geom_gamma_min": 1e-6,
        "use_corrected_line_start": True,
    }
    base.update(overrides)
    for key, value in base.items():
        setattr(cfg, key, value)
    return cfg


def _resolve_data_root(preferred: str | None = None) -> str | None:
    candidates = [
        preferred or "",
        os.environ.get("CF_DEEPCAD_DATA_ROOT", ""),
        "data",
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data")),
        r"D:\DeepCAD\DeepCAD\data",
    ]
    for path in candidates:
        if not path:
            continue
        split_path = os.path.join(path, "train_val_test_split.json")
        if os.path.isfile(split_path):
            return path
    return None


def _load_batch(cfg):
    data_root = _resolve_data_root(getattr(cfg, "data_root", None))
    if data_root is not None:
        cfg.data_root = data_root
        loader = get_step_dataloader("train", cfg, shuffle=False)
        return next(iter(loader))
    return make_synthetic_batch(batch_size=2, max_lines=cfg.max_lines)


def _assert_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise AssertionError("{} contains non-finite values".format(name))


def verify_g1(cfg, device) -> None:
    use_case = build_train_use_case(cfg, device)
    batch = _load_batch(cfg)
    use_case.train()
    out = use_case.execute(batch, epoch=1)
    out["loss"].backward()
    assert out["gamma_geom"].item() == 0.0
    assert abs(out["loss_args"].item() - 2.0 * out["loss_args_raw"].item()) < 1e-5
    _assert_finite("loss", out["loss"])
    has_grad = any(
        param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
        for param in use_case.model.parameters()
    )
    assert has_grad, "expected gradients on model parameters"
    print("[G1] PASS: original DeepCAD path forward/backward with L_cmd + 2.0*L_args")


def verify_g2(cfg, device) -> None:
    cfg_geom = _make_cfg(enable_geom_loss=True, geom_log_only=True, gamma_geom=0.1)
    use_case = build_train_use_case(cfg_geom, device)
    batch = _load_batch(cfg_geom)
    use_case.eval()
    with torch.no_grad():
        out = use_case.execute(batch, epoch=1)
    unit = out["soft_lines"]["unit"]
    valid = out["soft_lines"]["valid"]
    active = valid > 0
    _assert_finite("pred_unit", unit)
    if active.any():
        norms = torch.norm(unit[active], dim=-1)
        assert (norms > 0.9).all() and (norms < 1.1).all()
    assert out["positive_count_h"] >= 0
    assert out["positive_count_parallel"] >= 0
    print("[G2] PASS: GT relations + predicted line geometry parsing")


def verify_g4(device) -> None:
    cfg = _make_cfg(
        enable_geom_loss=True,
        geom_log_only=False,
        gamma_geom=0.1,
        geom_target_ratio=0.1,
        geom_ratio_ema=0.0,
        geom_warmup_start_epoch=1,
        geom_warmup_end_epoch=1,
        batch_size=2,
        num_workers=0,
    )
    batch = _load_batch(cfg)
    use_case = build_train_use_case(cfg, device)
    use_case.train()
    out = use_case.execute(batch, epoch=10)
    out["loss"].backward()
    main = out["loss_cmd"].item() + out["loss_args"].item()
    geom = out["loss_geom"].item()
    expected_gamma = min(0.1, max(1e-6, 0.1 * main / max(geom, 1e-8)))
    assert abs(out["gamma_geom"].item() - expected_gamma) < 1e-4
    assert abs(out["geom_target_ratio"].item() - 0.1) < 1e-6
    ratio = out["geom_effective_ratio"].item()
    assert abs(ratio - 0.1) < 0.02
    _assert_finite("loss", out["loss"])
    print("[G4] PASS: adaptive gamma_geom tracks geom_target_ratio={}".format(out["geom_target_ratio"].item()))


def verify_g3(device) -> None:
    scenarios = {
        "S0": {"enable_geom_loss": False, "geom_log_only": False, "gamma_geom": 0.0},
        "S1": {"enable_geom_loss": True, "geom_log_only": True, "gamma_geom": 0.1},
        "S2": {"enable_geom_loss": True, "geom_log_only": False, "gamma_geom": 0.1},
    }
    batch_cfg = _make_cfg(batch_size=2, num_workers=0)
    batch = _load_batch(batch_cfg)
    for name, overrides in scenarios.items():
        cfg = _make_cfg(**overrides, batch_size=2, num_workers=0)
        use_case = build_train_use_case(cfg, device)
        use_case.train()
        out = use_case.execute(batch, epoch=1)
        out["loss"].backward()
        expected_gamma = 0.0 if name in ("S0", "S1") else 0.1
        assert abs(out["gamma_geom"].item() - expected_gamma) < 1e-6
        _assert_finite("loss", out["loss"])
        print("[G3:{}] PASS: loss={}, gamma_geom={}".format(name, out["loss"].item(), out["gamma_geom"].item()))
    print("[G3] PASS: S0/S1/S2 configs run on one real batch")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify P0-01..G3 gates for constraint_fused_deepcad_step")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    args = parser.parse_args()

    device = resolve_device(SimpleNamespace(device=args.device, gpu_ids="cpu"))
    cfg = _make_cfg(data_root=args.data_root, device=args.device, gpu_ids="cpu", batch_size=2, num_workers=0)
    data_root = _resolve_data_root(args.data_root)
    if data_root is None:
        print("Note: real data not found; using synthetic batch for gate checks.")
    else:
        print("Using data_root:", data_root)

    try:
        verify_g1(cfg, device)
        verify_g2(cfg, device)
        verify_g3(device)
        verify_g4(device)
    except Exception as exc:
        print("Gate verification failed:", exc, file=sys.stderr)
        return 1

    print("All gates P0-01..G4 verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
